"""Supervised pre-training for the TFT encoder.

Pre-trains the TFT on two auxiliary tasks before RL training begins:
1. **Direction** (classification): predict whether next bar closes up or down
2. **Volatility** (regression): predict absolute log-return of the next bar

This gives the TFT meaningful initial weights so that RL doesn't start
from a randomly-initialized encoder — dramatically improving early
training stability and sample efficiency.

Usage in Trainer::

    pretrainer = TFTPretrainer(tft_model, device="cpu")
    pretrainer.train(features_df, n_market_features=15, lookback=60)
    # TFT weights are updated in-place

Standalone::

    python scripts/pretrain_tft.py --epochs 30 --synthetic-only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from apexfx.models.tft.tft_model import TemporalFusionTransformer
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class _SupervisedTFTDataset(Dataset):
    """Dataset that produces (market_seq, time_seq) → (direction, volatility) pairs."""

    def __init__(
        self,
        features: pd.DataFrame,
        market_cols: list[str],
        lookback: int,
        forward_bars: int = 1,
    ) -> None:
        self._features = features
        self._market_cols = market_cols
        self._lookback = lookback
        self._forward = forward_bars

        close = features["close"].values
        self._log_returns = np.zeros(len(close))
        self._log_returns[1:] = np.diff(np.log(np.maximum(close, 1e-10)))

        # Valid indices: need lookback history + forward_bars for label
        self._indices = list(range(lookback, len(features) - forward_bars))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        i = self._indices[idx]
        start = i - self._lookback

        # Market features: (lookback, n_features)
        market = self._features[self._market_cols].iloc[start:i].values.astype(np.float32)
        market = np.nan_to_num(market, nan=0.0, posinf=5.0, neginf=-5.0)

        # Time features: (lookback, 5) — sin/cos hour, sin/cos dow, session
        time_feat = np.zeros((self._lookback, 5), dtype=np.float32)
        if "time" in self._features.columns:
            for j, row_idx in enumerate(range(start, i)):
                try:
                    dt = pd.Timestamp(self._features.iloc[row_idx]["time"])
                    hour = dt.hour + dt.minute / 60.0
                    dow = dt.dayofweek
                    time_feat[j, 0] = np.sin(2 * np.pi * hour / 24)
                    time_feat[j, 1] = np.cos(2 * np.pi * hour / 24)
                    time_feat[j, 2] = np.sin(2 * np.pi * dow / 5)
                    time_feat[j, 3] = np.cos(2 * np.pi * dow / 5)
                    time_feat[j, 4] = self._session_id(dt.hour) / 5.0
                except Exception:
                    pass

        # Labels
        future_return = self._log_returns[i + self._forward]
        direction = 1.0 if future_return > 0 else 0.0
        volatility = abs(future_return)

        return {
            "market": torch.from_numpy(market),
            "time": torch.from_numpy(time_feat),
            "direction": torch.tensor(direction, dtype=torch.float32),
            "volatility": torch.tensor(volatility, dtype=torch.float32),
        }

    @staticmethod
    def _session_id(hour: int) -> float:
        if 12 <= hour <= 16:
            return 4.0  # overlap
        if 7 <= hour <= 16:
            return 3.0  # london
        if 12 <= hour <= 21:
            return 2.0  # ny
        if 0 <= hour <= 9:
            return 1.0  # tokyo
        return 0.0


class _SupervisedHead(nn.Module):
    """Lightweight classification + regression heads on top of TFT encoded_state."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # volatility is always positive
        )

    def forward(
        self, encoded_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction_logit = self.direction_head(encoded_state).squeeze(-1)
        volatility_pred = self.volatility_head(encoded_state).squeeze(-1)
        return direction_logit, volatility_pred


class TFTPretrainer:
    """Pre-trains a TFT encoder with supervised auxiliary tasks.

    The TFT weights are modified **in-place** — after calling ``train()``,
    the same TFT object (inside HiveMind) has better initial weights.

    Parameters
    ----------
    tft : TemporalFusionTransformer
        The TFT module to pre-train (typically ``hive_mind.tft``).
    device : str
        Torch device ("cpu", "cuda", "mps").
    lr : float
        Learning rate for Adam.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience (epochs without val loss improvement).
    direction_weight : float
        Loss weight for direction classification task.
    volatility_weight : float
        Loss weight for volatility regression task.
    """

    def __init__(
        self,
        tft: TemporalFusionTransformer,
        device: str = "cpu",
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128,
        patience: int = 5,
        direction_weight: float = 1.0,
        volatility_weight: float = 0.5,
    ) -> None:
        self._tft = tft
        self._device = device
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = patience
        self._dir_w = direction_weight
        self._vol_w = volatility_weight

    def train(
        self,
        features: pd.DataFrame,
        n_market_features: int,
        lookback: int,
        market_cols: list[str] | None = None,
    ) -> dict[str, float]:
        """Run supervised pre-training.

        Parameters
        ----------
        features : pd.DataFrame
            Full feature DataFrame with OHLCV + extracted features.
        n_market_features : int
            Number of market feature columns the TFT expects.
        lookback : int
            Sequence length (number of historical bars).
        market_cols : list[str] | None
            Explicit list of market feature columns. If None, auto-detected.

        Returns
        -------
        dict with final train/val losses and direction accuracy.
        """
        if market_cols is None:
            exclude = {"time", "open", "high", "low", "close", "volume", "tick_count",
                        "regime_label", "hurst_regime", "in_liquidity_zone",
                        "delta_divergence", "poc_price"}
            market_cols = [c for c in features.columns if c not in exclude
                           and features[c].dtype.kind in ("f", "i")]
            market_cols = market_cols[:n_market_features]

        if len(market_cols) < n_market_features:
            logger.warning(
                "Fewer market columns than expected",
                found=len(market_cols),
                expected=n_market_features,
            )

        # Train/val split (time-ordered, 80/20)
        split = int(len(features) * 0.8)
        train_ds = _SupervisedTFTDataset(features.iloc[:split].reset_index(drop=True),
                                          market_cols, lookback)
        val_ds = _SupervisedTFTDataset(features.iloc[split:].reset_index(drop=True),
                                        market_cols, lookback)

        train_loader = DataLoader(train_ds, batch_size=self._batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self._batch_size, shuffle=False,
                                 num_workers=0, drop_last=False)

        logger.info(
            "TFT pre-training",
            train_samples=len(train_ds),
            val_samples=len(val_ds),
            epochs=self._epochs,
            lookback=lookback,
            n_features=len(market_cols),
        )

        # Move TFT to device and create supervised head
        self._tft.to(self._device)
        head = _SupervisedHead(self._tft.d_model).to(self._device)

        # Optimizer: TFT + head parameters
        optimizer = torch.optim.Adam(
            list(self._tft.parameters()) + list(head.parameters()),
            lr=self._lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self._epochs):
            # ── Train ──
            self._tft.train()
            head.train()
            train_losses = []

            for batch in train_loader:
                market = batch["market"].to(self._device)
                time_feat = batch["time"].to(self._device)
                direction = batch["direction"].to(self._device)
                volatility = batch["volatility"].to(self._device)

                tft_out = self._tft(x_past=market, x_future=time_feat)
                dir_logit, vol_pred = head(tft_out.encoded_state)

                loss_dir = bce_loss(dir_logit, direction)
                loss_vol = mse_loss(vol_pred, volatility)
                loss = self._dir_w * loss_dir + self._vol_w * loss_vol

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._tft.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)

            # ── Validate ──
            self._tft.eval()
            head.eval()
            val_losses = []
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    market = batch["market"].to(self._device)
                    time_feat = batch["time"].to(self._device)
                    direction = batch["direction"].to(self._device)
                    volatility = batch["volatility"].to(self._device)

                    tft_out = self._tft(x_past=market, x_future=time_feat)
                    dir_logit, vol_pred = head(tft_out.encoded_state)

                    loss_dir = bce_loss(dir_logit, direction)
                    loss_vol = mse_loss(vol_pred, volatility)
                    loss = self._dir_w * loss_dir + self._vol_w * loss_vol
                    val_losses.append(loss.item())

                    preds = (torch.sigmoid(dir_logit) > 0.5).float()
                    correct += (preds == direction).sum().item()
                    total += len(direction)

            avg_val = np.mean(val_losses)
            val_acc = correct / max(total, 1)
            scheduler.step(avg_val)

            logger.info(
                "Pretrain epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_dir_acc=%.4f",
                epoch + 1, self._epochs, avg_train, avg_val, val_acc,
            )

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in self._tft.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best TFT weights
        if best_state is not None:
            self._tft.load_state_dict(best_state)
            logger.info("Restored best TFT weights (val_loss=%.4f)", best_val_loss)

        # Move TFT back to original device (trainer will move it later)
        self._tft.cpu()

        return {
            "best_val_loss": best_val_loss,
            "final_val_acc": val_acc,
            "epochs_trained": epoch + 1,
        }

    def save_pretrained(self, path: str | Path) -> None:
        """Save pre-trained TFT weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._tft.state_dict(), path)
        logger.info("Saved pre-trained TFT weights to %s", path)

    def load_pretrained(self, path: str | Path) -> None:
        """Load pre-trained TFT weights from disk."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._tft.load_state_dict(state)
        logger.info("Loaded pre-trained TFT weights from %s", path)
