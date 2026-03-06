"""SB3-compatible training callbacks for logging, checkpointing, early stopping, and diversity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    """Adaptive curriculum callback with performance-based stage transitions.

    Instead of fixed timestep budgets, stages advance when the agent reaches
    a target Sharpe ratio. If performance degrades below a threshold, the
    stage can regress (go back to easier data).
    """

    def __init__(
        self,
        stage_timesteps: list[int],
        on_stage_complete: Any = None,
        verbose: int = 0,
        target_sharpe: float = 0.3,
        regression_sharpe: float = -0.5,
        eval_window: int = 50,
        min_steps_per_stage: int = 50_000,
    ) -> None:
        super().__init__(verbose)
        self._stage_timesteps = stage_timesteps
        self._current_stage = 0
        self._on_stage_complete = on_stage_complete
        self._target_sharpe = target_sharpe
        self._regression_sharpe = regression_sharpe
        self._eval_window = eval_window
        self._min_steps_per_stage = min_steps_per_stage
        self._stage_start_step: int = 0
        self._episode_returns: list[float] = []

    def _on_step(self) -> bool:
        # Collect episode returns
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_returns.append(info["episode"]["r"])

        if self._current_stage >= len(self._stage_timesteps):
            return True

        steps_in_stage = self.num_timesteps - self._stage_start_step

        # Check fixed timestep budget (hard limit)
        target = sum(self._stage_timesteps[: self._current_stage + 1])
        if self.num_timesteps >= target:
            self._advance_stage("timestep_budget")
            return True

        # Performance-based early advancement (after minimum steps)
        if steps_in_stage >= self._min_steps_per_stage and len(self._episode_returns) >= self._eval_window:
            recent = np.array(self._episode_returns[-self._eval_window:])
            mean_ret = np.mean(recent)
            std_ret = np.std(recent) + 1e-8
            rolling_sharpe = mean_ret / std_ret

            if rolling_sharpe >= self._target_sharpe:
                self._advance_stage(f"target_sharpe_reached={rolling_sharpe:.3f}")
            elif rolling_sharpe < self._regression_sharpe and self._current_stage > 0:
                logger.warning(
                    "Performance regression detected — regressing stage",
                    rolling_sharpe=round(rolling_sharpe, 3),
                    threshold=self._regression_sharpe,
                    current_stage=self._current_stage,
                )
                self._current_stage -= 1
                self._stage_start_step = self.num_timesteps
                self._episode_returns = []
                if self._on_stage_complete:
                    self._on_stage_complete(self._current_stage)

        return True

    def _advance_stage(self, reason: str) -> None:
        logger.info(
            "Curriculum stage complete",
            stage=self._current_stage,
            timesteps=self.num_timesteps,
            reason=reason,
        )
        self._current_stage += 1
        self._stage_start_step = self.num_timesteps
        self._episode_returns = []
        if self._on_stage_complete:
            self._on_stage_complete(self._current_stage)


class EWCCallback(BaseCallback):
    """Apply Elastic Weight Consolidation penalty during RL training.

    After Fisher information has been computed (between curriculum stages),
    this callback periodically applies an EWC gradient step to the feature
    extractor.  Uses its own Adam optimiser with a small learning rate,
    same pattern as :class:`DiversityCallback`.

    The penalty ``λ/2 * Σ F_i * (θ_i - θ*_i)²`` pulls parameters back
    toward their previous-stage optimum, weighted by importance (Fisher).
    """

    def __init__(
        self,
        ewc_regularizer,
        update_freq: int = 10,
        lr: float = 1e-4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.ewc_reg = ewc_regularizer
        self.update_freq = update_freq
        self.lr = lr
        self._optimizer: torch.optim.Optimizer | None = None
        self._initialized: bool = False

    def _init_optimizer(self) -> None:
        """Create a dedicated Adam optimiser for feature extractor parameters."""
        from apexfx.training.ewc import EWCRegularizer

        extractor = EWCRegularizer._get_extractor(self.model)
        if extractor is None:
            logger.warning("No features extractor for EWC — disabling")
            self.ewc_reg = type("Dummy", (), {"has_fisher": False})()
            return
        params = [p for p in extractor.parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam(params, lr=self.lr)
        self._initialized = True
        logger.info(
            "EWCCallback initialised",
            n_params=sum(p.numel() for p in params),
            lambda_ewc=self.ewc_reg.lambda_ewc,
            update_freq=self.update_freq,
        )

    def _on_step(self) -> bool:
        # No Fisher yet → nothing to do
        if not self.ewc_reg.has_fisher:
            return True

        # Respect learning_starts
        learning_starts = getattr(self.model, "learning_starts", 0)
        if self.num_timesteps < learning_starts:
            return True

        if self.num_timesteps % self.update_freq != 0:
            return True

        if not self._initialized:
            self._init_optimizer()

        try:
            self._apply_ewc_step()
        except Exception as exc:
            # Never crash training — EWC is auxiliary
            logger.debug("EWC step skipped", reason=str(exc))

        return True

    def _apply_ewc_step(self) -> None:
        """Compute EWC penalty → backward → step."""
        from apexfx.training.ewc import EWCRegularizer

        extractor = EWCRegularizer._get_extractor(self.model)

        with torch.enable_grad():
            penalty = self.ewc_reg.penalty(extractor)

        if penalty.item() == 0.0:
            return

        self._optimizer.zero_grad()
        penalty.backward()

        # Gradient clipping for stability
        params = [p for p in extractor.parameters() if p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        self._optimizer.step()

        # Logging
        if self.logger:
            self.logger.record("ewc/penalty", penalty.item())


class DiversityCallback(BaseCallback):
    """Periodic ensemble diversity optimisation step.

    Runs a **separate forward pass** through the feature extractor every
    ``update_freq`` environment steps, computes a diversity loss from
    cached agent outputs and gating weights, and applies a small gradient
    update to agent + gating parameters only.

    This is an auxiliary objective independent of the main RL loss.  It uses
    its own Adam optimiser with a dedicated (smaller) learning rate so that
    the diversity signal does not destabilise the primary RL training.

    The loss consists of:

    * **-action_variance** — push agents to produce spread-out signals.
    * **+pairwise_correlation** — penalise linear correlation between pairs.
    * **-gating_entropy** — encourage the gating network to use all agents.
    """

    def __init__(
        self,
        diversity_weight: float = 0.01,
        entropy_weight: float = 0.03,
        update_freq: int = 100,
        batch_size: int = 256,
        lr: float = 1e-4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.lr = lr
        self._optimizer: torch.optim.Optimizer | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------

    def _init_optimizer(self) -> None:
        """Create a dedicated Adam optimiser for agent + gating parameters."""
        extractor = self.model.policy.features_extractor
        hm = extractor.hive_mind

        diversity_params = (
            list(hm.trend_agent.parameters())
            + list(hm.reversion_agent.parameters())
            + list(hm.breakout_agent.parameters())
            + list(hm.gating.parameters())
        )

        self._optimizer = torch.optim.Adam(diversity_params, lr=self.lr)
        self._initialized = True
        logger.info(
            "DiversityCallback initialised",
            n_params=sum(p.numel() for p in diversity_params),
            lr=self.lr,
            update_freq=self.update_freq,
        )

    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        # Guard: only works with off-policy algorithms that have a replay buffer
        if not hasattr(self.model, "replay_buffer") or self.model.replay_buffer is None:
            return True

        # Wait until enough data is collected
        if self.model.replay_buffer.size() < self.batch_size:
            return True

        learning_starts = getattr(self.model, "learning_starts", 0)
        if self.num_timesteps < learning_starts:
            return True

        if self.num_timesteps % self.update_freq != 0:
            return True

        if not self._initialized:
            self._init_optimizer()

        try:
            self._apply_diversity_step()
        except Exception as exc:
            # Never crash training — diversity is auxiliary
            logger.debug("Diversity step skipped", reason=str(exc))

        return True

    # ------------------------------------------------------------------

    def _apply_diversity_step(self) -> None:
        """Sample → forward → diversity loss → backward → step."""
        from apexfx.training.diversity import DiversityMetrics, compute_diversity_loss

        env = getattr(self.model, "_vec_normalize_env", None)
        replay_data = self.model.replay_buffer.sample(self.batch_size, env=env)
        obs = replay_data.observations

        extractor = self.model.policy.features_extractor

        # Forward pass in training mode — populates _cached_* attributes
        was_training = extractor.training
        extractor.train()
        with torch.enable_grad():
            _ = extractor(obs)

        # Restore mode
        if not was_training:
            extractor.eval()

        hm = extractor.hive_mind
        trend_act = getattr(hm, "_cached_trend_action", None)
        rev_act = getattr(hm, "_cached_reversion_action", None)
        brk_act = getattr(hm, "_cached_breakout_action", None)
        gating_w = getattr(hm, "_cached_gating_weights", None)

        if any(t is None for t in (trend_act, rev_act, brk_act, gating_w)):
            return

        loss = compute_diversity_loss(
            [trend_act, rev_act, brk_act],
            gating_w,
            diversity_weight=self.diversity_weight,
            entropy_weight=self.entropy_weight,
        )

        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        all_params = [
            p
            for pg in self._optimizer.param_groups
            for p in pg["params"]
            if p.grad is not None
        ]
        if all_params:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        self._optimizer.step()

        # --- Logging ---
        if self.logger:
            self.logger.record("diversity/loss", loss.item())
            metrics = DiversityMetrics.compute(
                [trend_act.detach(), rev_act.detach(), brk_act.detach()],
                gating_w.detach(),
            )
            for key, val in metrics.items():
                self.logger.record(f"diversity/{key}", val)
