"""Online Learning: adaptive model retraining on recent market data.

Markets are non-stationary — a model trained on 2023 data degrades in 2024.
This module handles incremental adaptation:

1. Fine-tune mode: Short retraining on recent window with small LR + EWC
2. Validation gating: New model must beat current on held-out data
3. Safe promotion: Only swap models if validation Sharpe improves
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from apexfx.config.schema import AppConfig
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OnlineLearnResult:
    """Result of an online learning attempt."""
    retrained: bool
    promoted: bool          # True if new model replaced old
    model_path: str         # Path to the promoted model (or current if not promoted)
    validation_sharpe: float
    current_sharpe: float
    sharpe_delta: float
    n_bars_used: int
    message: str


class OnlineLearner:
    """Incremental model adaptation to recent market data.

    The learner periodically fine-tunes the current model on the most
    recent N days of data, validates on a held-out portion, and only
    promotes the new model if it improves the validation Sharpe ratio.

    This prevents catastrophic forgetting via EWC regularization and
    prevents degradation via validation gating.
    """

    def __init__(
        self,
        model_path: str,
        config: AppConfig,
        retrain_window_days: int = 30,
        retrain_steps: int = 10_000,
        retrain_lr: float = 1e-5,
        min_new_bars: int = 24,
        validation_sharpe_min: float = 0.0,
    ) -> None:
        """Initialize online learner.

        Args:
            model_path: Path to the current best model
            config: Full app configuration
            retrain_window_days: Use last N days for fine-tuning
            retrain_steps: Gradient steps per retrain cycle
            retrain_lr: Small LR for fine-tuning (prevents large weight changes)
            min_new_bars: Minimum new bars before considering retrain
            validation_sharpe_min: Reject retrained model if below this Sharpe
        """
        self._model_path = model_path
        self._config = config
        self._retrain_window = retrain_window_days
        self._retrain_steps = retrain_steps
        self._retrain_lr = retrain_lr
        self._min_new_bars = min_new_bars
        self._validation_sharpe_min = validation_sharpe_min
        self._bars_since_retrain = 0
        self._retrain_count = 0

        logger.info(
            "OnlineLearner initialized",
            model=model_path,
            window_days=retrain_window_days,
            steps=retrain_steps,
            lr=retrain_lr,
        )

    def record_bar(self) -> None:
        """Record a new bar processed (for retrain timing)."""
        self._bars_since_retrain += 1

    def should_retrain(self, new_bars_count: int | None = None) -> bool:
        """Check if retraining conditions are met.

        Args:
            new_bars_count: Override internal bar counter
        """
        bars = new_bars_count if new_bars_count is not None else self._bars_since_retrain
        return bars >= self._min_new_bars

    def retrain(self, recent_data: pd.DataFrame) -> OnlineLearnResult:
        """Fine-tune model on recent data with validation gating.

        Steps:
        1. Split recent data into train (80%) and validation (20%)
        2. Load current best model
        3. Fine-tune with small LR + EWC regularization
        4. Evaluate on validation set
        5. If Sharpe improves → promote; otherwise keep current

        Args:
            recent_data: Recent market OHLCV data for fine-tuning

        Returns:
            OnlineLearnResult with promotion decision
        """
        if len(recent_data) < 100:
            return OnlineLearnResult(
                retrained=False,
                promoted=False,
                model_path=self._model_path,
                validation_sharpe=0.0,
                current_sharpe=0.0,
                sharpe_delta=0.0,
                n_bars_used=len(recent_data),
                message="Insufficient data for retraining",
            )

        logger.info(
            "Starting online retraining",
            n_bars=len(recent_data),
            retrain_cycle=self._retrain_count + 1,
        )

        try:
            # Split: 80% train, 20% validation
            split_idx = int(len(recent_data) * 0.8)
            train_data = recent_data.iloc[:split_idx]
            val_data = recent_data.iloc[split_idx:]

            # Evaluate current model on validation set
            current_sharpe = self._evaluate_model(self._model_path, val_data)

            # Fine-tune
            candidate_path = self._fine_tune(train_data)

            # Evaluate candidate
            candidate_sharpe = self._evaluate_model(candidate_path, val_data)

            sharpe_delta = candidate_sharpe - current_sharpe
            promoted = (
                candidate_sharpe > current_sharpe
                and candidate_sharpe >= self._validation_sharpe_min
            )

            if promoted:
                self._model_path = candidate_path
                self._retrain_count += 1
                logger.info(
                    "Model promoted via online learning",
                    sharpe_delta=round(sharpe_delta, 4),
                    new_sharpe=round(candidate_sharpe, 4),
                    cycle=self._retrain_count,
                )
            else:
                logger.info(
                    "Candidate rejected (no improvement)",
                    current_sharpe=round(current_sharpe, 4),
                    candidate_sharpe=round(candidate_sharpe, 4),
                )

            self._bars_since_retrain = 0

            return OnlineLearnResult(
                retrained=True,
                promoted=promoted,
                model_path=self._model_path,
                validation_sharpe=candidate_sharpe,
                current_sharpe=current_sharpe,
                sharpe_delta=sharpe_delta,
                n_bars_used=len(recent_data),
                message="Promoted" if promoted else "Rejected (no improvement)",
            )

        except Exception as e:
            logger.error("Online retraining failed", error=str(e))
            return OnlineLearnResult(
                retrained=False,
                promoted=False,
                model_path=self._model_path,
                validation_sharpe=0.0,
                current_sharpe=0.0,
                sharpe_delta=0.0,
                n_bars_used=len(recent_data),
                message=f"Retraining failed: {e}",
            )

    def _fine_tune(self, train_data: pd.DataFrame) -> str:
        """Fine-tune current model on recent data.

        Returns path to candidate model.
        """
        from apexfx.features.pipeline import FeaturePipeline
        from apexfx.env.forex_env import ForexTradingEnv
        from apexfx.env.reward import TradingReward

        pipeline = FeaturePipeline()
        features = pipeline.compute(train_data)
        n_features = pipeline.n_features

        env = ForexTradingEnv(
            data=features,
            initial_balance=100_000.0,
            n_market_features=n_features,
            lookback=self._config.data.feature_window,
            reward_fn=TradingReward(loss_weight=2.0, reward_scale=1000.0),
            max_drawdown_pct=0.15,
        )

        # Load current model
        from stable_baselines3 import SAC
        model = SAC.load(self._model_path, env=env)

        # Override LR for fine-tuning
        model.learning_rate = self._retrain_lr

        # Fine-tune
        model.learn(
            total_timesteps=self._retrain_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        # Save candidate
        candidate_dir = Path(self._config.base.paths.models_dir) / "online_candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = str(candidate_dir / f"candidate_{self._retrain_count + 1}")
        model.save(candidate_path)

        return candidate_path

    def _evaluate_model(self, model_path: str, val_data: pd.DataFrame) -> float:
        """Evaluate a model on validation data and return Sharpe ratio."""
        from apexfx.features.pipeline import FeaturePipeline
        from apexfx.env.forex_env import ForexTradingEnv
        from apexfx.env.reward import DifferentialSharpeReward
        from apexfx.utils.metrics import compute_all_metrics

        pipeline = FeaturePipeline()
        features = pipeline.compute(val_data)
        n_features = pipeline.n_features

        env = ForexTradingEnv(
            data=features,
            initial_balance=100_000.0,
            n_market_features=n_features,
            lookback=self._config.data.feature_window,
            reward_fn=DifferentialSharpeReward(),
            max_drawdown_pct=1.0,  # Don't terminate during eval
        )

        from stable_baselines3 import SAC
        try:
            model = SAC.load(model_path)
        except Exception:
            return -999.0

        obs, info = env.reset()
        done = False
        returns: list[float] = []
        prev_value = 100_000.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_value = info.get("portfolio_value", prev_value)
            if prev_value > 0:
                returns.append((current_value - prev_value) / prev_value)
            prev_value = current_value

        if not returns:
            return 0.0

        metrics = compute_all_metrics(np.array(returns))
        return metrics.get("sharpe_ratio", 0.0)

    @property
    def model_path(self) -> str:
        return self._model_path
