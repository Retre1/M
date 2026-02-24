"""SB3-compatible training callbacks for logging, checkpointing, and early stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from apexfx.utils.logging import get_logger
from apexfx.utils.metrics import compute_all_metrics

logger = get_logger(__name__)


class MetricsCallback(BaseCallback):
    """Log episode-level trading metrics to TensorBoard and structlog."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_returns: list[float] = []
        self._episode_portfolio_values: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self._episode_returns.append(ep["r"])

                if "portfolio_value" in info:
                    self._episode_portfolio_values.append(info["portfolio_value"])

                if self.logger:
                    self.logger.record("trading/episode_return", ep["r"])
                    self.logger.record("trading/episode_length", ep["l"])

                    if "drawdown" in info:
                        self.logger.record("trading/drawdown", info["drawdown"])
                    if "total_trades" in info:
                        self.logger.record("trading/total_trades", info["total_trades"])

                logger.info(
                    "Episode complete",
                    return_=round(ep["r"], 4),
                    length=ep["l"],
                    portfolio=info.get("portfolio_value", 0),
                    total_step=self.num_timesteps,
                )

        # Periodic aggregated metrics (every 10 episodes)
        if len(self._episode_returns) >= 10 and len(self._episode_returns) % 10 == 0:
            recent = np.array(self._episode_returns[-10:])
            if self.logger:
                self.logger.record("trading/mean_return_10ep", np.mean(recent))
                self.logger.record("trading/std_return_10ep", np.std(recent))

        return True


class TradingCheckpointCallback(BaseCallback):
    """Save model checkpoints at regular intervals, keeping the best N."""

    def __init__(
        self,
        save_freq: int = 50_000,
        save_dir: str = "models/checkpoints",
        keep_best_n: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._save_freq = save_freq
        self._save_dir = Path(save_dir)
        self._keep_best_n = keep_best_n
        self._best_scores: list[tuple[float, str]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq == 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_dir / f"model_step_{self.num_timesteps}"
        self.model.save(str(path))
        logger.info("Checkpoint saved", path=str(path), step=self.num_timesteps)


class EarlyStoppingCallback(BaseCallback):
    """Stop training if evaluation reward plateaus."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        eval_freq: int = 10_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._patience = patience
        self._min_delta = min_delta
        self._eval_freq = eval_freq
        self._best_reward: float = -np.inf
        self._no_improve_count: int = 0
        self._recent_returns: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._recent_returns.append(info["episode"]["r"])

        if self.num_timesteps % self._eval_freq == 0 and len(self._recent_returns) > 0:
            mean_return = np.mean(self._recent_returns[-20:])

            if mean_return > self._best_reward + self._min_delta:
                self._best_reward = mean_return
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            if self._no_improve_count >= self._patience:
                logger.warning(
                    "Early stopping triggered",
                    patience=self._patience,
                    best_reward=self._best_reward,
                )
                return False

        return True


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning stage transitions."""

    def __init__(
        self,
        stage_timesteps: list[int],
        on_stage_complete: Any = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._stage_timesteps = stage_timesteps
        self._current_stage = 0
        self._on_stage_complete = on_stage_complete

    def _on_step(self) -> bool:
        if self._current_stage < len(self._stage_timesteps):
            target = sum(self._stage_timesteps[: self._current_stage + 1])
            if self.num_timesteps >= target:
                logger.info(
                    "Curriculum stage complete",
                    stage=self._current_stage,
                    timesteps=self.num_timesteps,
                )
                self._current_stage += 1

                if self._on_stage_complete:
                    self._on_stage_complete(self._current_stage)

        return True
