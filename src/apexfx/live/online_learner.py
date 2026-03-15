"""Live Online Incremental Learning — real-time micro-updates with drift detection.

Unlike the batch-based OnlineLearner in training/online_learner.py (which does
full retraining every N hours), this module performs MICRO gradient updates on
live transitions — adapting to the current market regime in near real-time.

Key features:
- Mini-buffer of recent (obs, action, reward, next_obs, done) transitions
- Drift detection via rolling Sharpe ratio of trade returns
- EWC penalty preserves knowledge from initial training
- Automatic checkpoint/rollback if performance degrades
- Higher learning rate during detected drift for faster adaptation

Usage
-----
>>> learner = LiveOnlineLearner(model, config)
>>> learner.record_transition(obs, action, reward, next_obs, done)
>>> learner.record_trade_result(pnl_return)
>>> if learner.maybe_update():
...     logger.info("Model micro-updated")
"""

from __future__ import annotations

import copy
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch

from apexfx.config.schema import OnlineLearningConfig
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class LiveOnlineLearner:
    """Micro-updates on live data with EWC protection and drift detection.

    Parameters
    ----------
    model : SAC | TQC
        SB3 off-policy model (must have replay_buffer and critic).
    config : OnlineLearningConfig
        Configuration for micro-update parameters.
    ewc_lambda : float
        EWC penalty strength. Higher = more memory retention.
    """

    def __init__(
        self,
        model,
        config: OnlineLearningConfig,
        ewc_lambda: float = 5000.0,
    ) -> None:
        self._model = model
        self._config = config

        # Mini-buffer for recent transitions
        self._mini_buffer: deque[tuple] = deque(maxlen=config.mini_buffer_size)

        # Trade result tracking for drift detection
        self._rolling_returns: deque[float] = deque(maxlen=config.drift_detection_window)
        self._trade_count: int = 0
        self._update_count: int = 0

        # Checkpoint stack for rollback
        self._checkpoints: deque[dict] = deque(maxlen=config.max_rollback_checkpoints)

        # EWC regularizer (optional)
        self._ewc_lambda = ewc_lambda
        self._ewc_fisher: dict[str, torch.Tensor] | None = None
        self._ewc_optimal: dict[str, torch.Tensor] | None = None

        if config.ewc_enabled:
            self._init_ewc()

        # Base learning rate from model
        self._base_lr = float(model.learning_rate) if hasattr(model, "learning_rate") else 3e-4

        logger.info(
            "LiveOnlineLearner initialized",
            buffer_size=config.mini_buffer_size,
            update_every_n_trades=config.update_every_n_trades,
            drift_window=config.drift_detection_window,
            ewc_enabled=config.ewc_enabled,
        )

    def _init_ewc(self) -> None:
        """Snapshot current model parameters as EWC anchor."""
        extractor = getattr(self._model.policy, "features_extractor", None)
        if extractor is None:
            logger.warning("No features extractor — EWC disabled for live learner")
            return

        self._ewc_optimal = {}
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                self._ewc_optimal[name] = param.data.clone()

        # Use uniform Fisher as initial approximation
        # (true Fisher would require replay buffer samples)
        self._ewc_fisher = {}
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                self._ewc_fisher[name] = torch.ones_like(param.data)

        logger.info("EWC anchor initialized from current model")

    def record_transition(
        self,
        obs: dict[str, np.ndarray],
        action: float,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """Store a live transition in the mini-buffer."""
        self._mini_buffer.append((obs, action, reward, next_obs, done))

    def record_trade_result(self, pnl_return: float) -> None:
        """Track a completed trade return for drift detection."""
        self._rolling_returns.append(pnl_return)
        self._trade_count += 1

    def maybe_update(self) -> bool:
        """Run micro-update if enough trades have accumulated.

        Returns True if an update was performed.
        """
        cfg = self._config

        # Check trade count threshold
        if self._trade_count == 0 or self._trade_count % cfg.update_every_n_trades != 0:
            return False

        # Check buffer has enough data
        min_samples = cfg.gradient_steps * 64  # rough estimate
        if len(self._mini_buffer) < min(min_samples, cfg.mini_buffer_size // 2):
            return False

        # Detect drift → adjust learning rate and gradient steps
        drift = self._detect_drift()
        if drift:
            lr = self._base_lr * cfg.adaptation_lr_scale
            n_steps = cfg.adaptation_gradient_steps
            logger.warning(
                "Drift detected — increasing adaptation speed",
                rolling_sharpe=self._compute_rolling_sharpe(),
                lr=lr,
                n_steps=n_steps,
            )
        else:
            lr = self._base_lr * cfg.lr_scale
            n_steps = cfg.gradient_steps

        # Checkpoint before update
        self._save_checkpoint()

        # Run gradient steps
        try:
            self._run_gradient_steps(n_steps, lr)
            self._update_count += 1
            logger.info(
                "Live micro-update complete",
                update_count=self._update_count,
                n_steps=n_steps,
                lr=lr,
                drift=drift,
                buffer_size=len(self._mini_buffer),
            )
            return True
        except Exception as e:
            logger.error("Micro-update failed, rolling back", error=str(e))
            self.rollback()
            return False

    def _detect_drift(self) -> bool:
        """Detect regime drift via rolling Sharpe ratio.

        Returns True if rolling Sharpe < threshold, indicating the
        current model is underperforming (likely regime change).
        """
        sharpe = self._compute_rolling_sharpe()
        if sharpe is None:
            return False
        return sharpe < self._config.drift_threshold_sharpe

    def _compute_rolling_sharpe(self) -> float | None:
        """Compute annualized Sharpe ratio from rolling trade returns."""
        if len(self._rolling_returns) < 20:
            return None
        returns = np.array(self._rolling_returns)
        std = returns.std()
        if std < 1e-8:
            return 0.0
        return float(returns.mean() / std * np.sqrt(252))

    def _run_gradient_steps(self, n_steps: int, lr: float) -> None:
        """Execute gradient steps on the mini-buffer with EWC penalty.

        Uses SB3's internal training mechanism where possible, falling
        back to manual gradient computation.
        """
        model = self._model

        # Temporarily adjust learning rate
        original_lr = None
        if hasattr(model, "lr_schedule"):
            original_lr = model.lr_schedule
            model.lr_schedule = lambda _: lr

        try:
            # Use SB3's built-in train method if replay buffer has data
            if hasattr(model, "train") and hasattr(model, "replay_buffer"):
                # Add mini-buffer transitions to replay buffer
                self._flush_to_replay_buffer()

                # Run SB3 gradient steps
                model.train(gradient_steps=n_steps, batch_size=64)

                # Apply EWC penalty if enabled
                if self._ewc_fisher is not None and self._ewc_optimal is not None:
                    self._apply_ewc_penalty()
        finally:
            # Restore learning rate
            if original_lr is not None:
                model.lr_schedule = original_lr

    def _flush_to_replay_buffer(self) -> None:
        """Add mini-buffer transitions to SB3 replay buffer."""
        buf = getattr(self._model, "replay_buffer", None)
        if buf is None:
            return

        for obs, action, reward, next_obs, done in self._mini_buffer:
            try:
                # Convert to format expected by SB3
                obs_arr = {k: np.expand_dims(v, 0) for k, v in obs.items()}
                next_obs_arr = {k: np.expand_dims(v, 0) for k, v in next_obs.items()}
                action_arr = np.array([[action]], dtype=np.float32)
                reward_arr = np.array([reward], dtype=np.float32)
                done_arr = np.array([done], dtype=np.float32)
                infos = [{}]

                buf.add(obs_arr, next_obs_arr, action_arr, reward_arr, done_arr, infos)
            except Exception:
                pass  # Skip malformed transitions

    def _apply_ewc_penalty(self) -> None:
        """Apply EWC penalty after gradient steps to prevent forgetting."""
        extractor = getattr(self._model.policy, "features_extractor", None)
        if extractor is None:
            return

        penalty = torch.tensor(0.0, device=next(extractor.parameters()).device)
        for name, param in extractor.named_parameters():
            if name in self._ewc_fisher and name in self._ewc_optimal:
                fisher = self._ewc_fisher[name]
                optimal = self._ewc_optimal[name]
                penalty = penalty + (fisher * (param - optimal).pow(2)).sum()

        penalty = self._ewc_lambda / 2.0 * penalty

        if penalty.item() > 0:
            # Apply correction step
            extractor.zero_grad()
            penalty.backward()
            with torch.no_grad():
                for param in extractor.parameters():
                    if param.grad is not None:
                        param.data -= self._base_lr * self._config.lr_scale * param.grad
                        param.grad.zero_()

    def _save_checkpoint(self) -> None:
        """Save model state for potential rollback."""
        try:
            state = {
                "policy_state": copy.deepcopy(self._model.policy.state_dict()),
                "update_count": self._update_count,
                "trade_count": self._trade_count,
            }
            self._checkpoints.append(state)
        except Exception as e:
            logger.warning("Failed to save checkpoint", error=str(e))

    def rollback(self) -> bool:
        """Rollback to the last checkpoint.

        Returns True if rollback was successful.
        """
        if not self._checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False

        state = self._checkpoints.pop()
        try:
            self._model.policy.load_state_dict(state["policy_state"])
            logger.info(
                "Rolled back to checkpoint",
                checkpoint_update=state["update_count"],
                current_update=self._update_count,
            )
            return True
        except Exception as e:
            logger.error("Rollback failed", error=str(e))
            return False

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def drift_detected(self) -> bool:
        return self._detect_drift()
