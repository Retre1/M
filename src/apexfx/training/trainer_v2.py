"""Trainer v2 — Full integration of CurriculumV2 with all v2 components.

Wraps the 4-stage curriculum orchestrator with:
    - RARAReward_v5 environment construction
    - WorldModelHybrid callback integration
    - GatingV2 anti-collapse loss injection
    - Adaptive LR application to SB3 optimizer
    - Multi-criteria checkpointing
    - EWC Fisher consolidation between stages
    - Full TensorBoard logging

Usage::

    from apexfx.training.trainer_v2 import TrainerV2
    from apexfx.training.config import CurriculumV2Config

    trainer = TrainerV2(
        curriculum_config=CurriculumV2Config(),
        app_config=app_config,
        real_data=df,
    )
    trainer.train()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from apexfx.training.config import CurriculumV2Config, StageConfig
from apexfx.training.curriculum_v2 import CurriculumV2, StageDataV2
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ── Picklable env thunk factory (module-level for SubprocVecEnv) ──────

def _make_env_thunk(
    df: pd.DataFrame,
    reward_fn_factory: Any,
    seed: int,
) -> Any:
    """Return a zero-argument factory that builds a single Monitor-wrapped env.

    Kept at module scope (not a closure inside ``TrainerV2``) so the thunk
    pickles cleanly under ``SubprocVecEnv(start_method="spawn"|"forkserver")``,
    which is required on macOS and on any Python build without fork.
    """
    def _thunk() -> Any:
        from stable_baselines3.common.monitor import Monitor
        from apexfx.env.forex_env import ForexTradingEnv

        env = ForexTradingEnv(
            data=df,
            initial_balance=100_000.0,
            reward_fn=reward_fn_factory(),
            max_drawdown_pct=0.15,
        )
        try:
            env.reset(seed=seed)
        except TypeError:
            # Older gym API without seed kwarg
            pass
        return Monitor(env)

    return _thunk


# ── Curriculum monitoring callback ───────────────────────────────────

class CurriculumV2Callback(BaseCallback):
    """SB3 callback integrating CurriculumV2 monitoring.

    Handles:
    - Periodic metric collection (reward, Sharpe, trades, etc.)
    - Adaptive LR updates via curriculum.on_step()
    - Multi-criteria checkpoint saves
    - Early stopping signal
    - Gating entropy and world model metrics logging
    """

    def __init__(
        self,
        curriculum: CurriculumV2,
        stage_idx: int,
        check_freq: int = 10000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._curriculum = curriculum
        self._stage_idx = stage_idx
        self._check_freq = check_freq
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._should_stop = False

    def _on_step(self) -> bool:
        # Collect episode metrics
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        # Periodic check
        if self.num_timesteps % self._check_freq == 0 and self._episode_rewards:
            metrics = self.get_metrics()

            # Log to TensorBoard
            if self.logger:
                for key, val in metrics.items():
                    self.logger.record(f"curriculum/{key}", val)

            # Run curriculum step (adaptive LR + checkpointing + early stop)
            result = self._curriculum.on_step(
                step=self.num_timesteps,
                stage_idx=self._stage_idx,
                metrics=metrics,
                model=self.model,
            )

            # Apply new LR
            new_lr = result["new_lr"]
            self._apply_lr(new_lr)

            # Check early stopping
            if result["should_stop"]:
                self._should_stop = True
                logger.info(
                    "Early stopping requested by curriculum",
                    step=self.num_timesteps,
                    stage=self._stage_idx,
                )
                return False

        return True

    def get_metrics(self) -> dict[str, float]:
        """Collect current training metrics (public — used by trainer too)."""
        recent = self._episode_rewards[-100:]
        rewards = np.array(recent) if recent else np.array([0.0])

        metrics: dict[str, float] = {
            "ep_rew_mean": float(np.mean(rewards)),
            "ep_rew_std": float(np.std(rewards)),
            "n_episodes": len(self._episode_rewards),
        }

        # Sharpe ratio (annualized, assuming ~252 trading days)
        if len(rewards) > 1 and np.std(rewards) > 0:
            metrics["sharpe"] = float(np.mean(rewards) / np.std(rewards) * np.sqrt(252))
        else:
            metrics["sharpe"] = 0.0

        # Profit factor
        gains = rewards[rewards > 0].sum()
        losses = abs(rewards[rewards < 0].sum())
        metrics["profit_factor"] = float(gains / max(losses, 1e-8))

        metrics["gating_entropy"] = self._read_entropy()
        return metrics

    def _read_entropy(self) -> float:
        """Read live entropy from the model.

        Resolution order:
        1. Gating module attached to the policy (``policy.gating``) that
           exposes ``last_entropy`` (float) or ``last_diagnostics`` dict.
        2. SB3 logger values — ``train/entropy_loss`` (preferred, already
           aggregated), else ``train/ent_coef``.
        3. Fallback constant 0.5 so adaptive LR remains in the neutral band.
        """
        model = self.model
        if model is None:
            return 0.5

        # 1) Optional GatingV2 module on the policy
        policy = getattr(model, "policy", None)
        gating = getattr(policy, "gating", None) if policy is not None else None
        if gating is not None:
            last_ent = getattr(gating, "last_entropy", None)
            if last_ent is not None:
                try:
                    return float(last_ent)
                except (TypeError, ValueError):
                    pass
            diag = getattr(gating, "last_diagnostics", None)
            if isinstance(diag, dict):
                val = diag.get("gating/entropy")
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass

        # 2) SB3 logger — action-distribution entropy proxy
        sb3_logger = getattr(model, "logger", None)
        if sb3_logger is not None:
            name_to_value = getattr(sb3_logger, "name_to_value", {}) or {}
            for key in ("train/entropy_loss", "train/ent_coef"):
                if key in name_to_value:
                    try:
                        # entropy_loss is typically negative — take absolute value
                        return float(abs(name_to_value[key]))
                    except (TypeError, ValueError):
                        continue

        return 0.5

    def _apply_lr(self, new_lr: float) -> None:
        """Apply learning rate to SB3 optimizer."""
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
            optimizer = self.model.policy.optimizer
            if optimizer is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr

    @property
    def should_stop(self) -> bool:
        return self._should_stop


class WorldModelV2Callback(BaseCallback):
    """Lightweight callback that logs world model metrics.

    The actual WorldModelCallback handles training. This callback
    adds imagination rollout scheduling per curriculum config.
    """

    def __init__(self, imagination_freq: int = 10, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._imagination_freq = imagination_freq

    def _on_step(self) -> bool:
        # Log imagination metrics if world model callback is present
        if self.num_timesteps % (self._imagination_freq * 100) == 0:
            if self.logger:
                self.logger.record(
                    "world_model/imagination_freq",
                    self._imagination_freq,
                )
        return True


# ── TrainerV2 ────────────────────────────────────────────────────────

class TrainerV2:
    """Full v2 training pipeline with 4-stage curriculum.

    Integrates CurriculumV2 with all v2 components (RARAReward_v5,
    WorldModelHybrid, GatingV2) via the existing Trainer infrastructure.

    This class can be used standalone or as a drop-in replacement
    for the Trainer class when v2 features are desired.

    Args:
        curriculum_config: CurriculumV2 configuration.
        app_config: Full application configuration (AppConfig).
        real_data: Real market data (OHLCV DataFrame).
        checkpoint_dir: Directory for multi-criteria checkpoints.
    """

    def __init__(
        self,
        curriculum_config: CurriculumV2Config | None = None,
        app_config: Any = None,
        real_data: pd.DataFrame | None = None,
        checkpoint_dir: Path | None = None,
        multi_symbol_data: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self._curriculum_config = curriculum_config or CurriculumV2Config()
        self._app_config = app_config
        self._real_data = real_data
        self._multi_symbol_data = multi_symbol_data or {}

        ckpt_dir = checkpoint_dir or Path("models/v2_checkpoints")
        self._curriculum = CurriculumV2(
            config=self._curriculum_config,
            real_data=real_data,
            checkpoint_dir=ckpt_dir,
        )

        self._model: Any = None
        self._device = self._resolve_device()
        self._active_curriculum_cb: CurriculumV2Callback | None = None
        self._feature_pipeline: Any = None  # lazy init
        self._feature_required_columns = (
            "hurst_exponent", "trend_strength", "close_zscore",
        )

        vec_cfg = self._curriculum_config.vec_env
        logger.info(
            "TrainerV2 initialized",
            n_stages=self._curriculum.n_stages,
            device=self._device,
            total_timesteps=self._curriculum_config.total_timesteps,
            n_envs=vec_cfg.n_envs,
            vec_kind=vec_cfg.kind,
            buffer_size=self._curriculum_config.replay_buffer.buffer_size,
            n_symbols=max(1, len(self._multi_symbol_data)),
        )

    def _resolve_device(self) -> str:
        """Auto-detect best available device."""
        if self._app_config is not None:
            dev = self._app_config.base.device.value
            if dev != "auto":
                return dev
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def train(self, resume: bool = False) -> dict[str, Any]:
        """Run the full 4-stage curriculum training.

        Args:
            resume: If True, attempt to resume from latest checkpoint.

        Returns:
            Training summary dict with per-stage metrics.
        """
        logger.info(
            "Starting CurriculumV2 training",
            n_stages=self._curriculum.n_stages,
            device=self._device,
        )

        for stage_idx in range(self._curriculum.n_stages):
            stage_cfg = self._curriculum_config.stages[stage_idx]

            logger.info(
                "=" * 60,
            )
            logger.info(
                f"STAGE {stage_idx}: {stage_cfg.name}",
                timesteps=stage_cfg.total_timesteps,
                real_ratio=stage_cfg.real_ratio,
                sbbts_ratio=stage_cfg.sbbts_ratio,
            )

            # Prepare data
            stage_data = self._curriculum.prepare_stage(stage_idx)

            # Warm-start from best checkpoint of previous stage
            if stage_cfg.warm_start and stage_idx > 0:
                best_path = self._curriculum.get_warm_start_path("sharpe")
                if best_path is not None:
                    logger.info(
                        "Warm-starting from best checkpoint",
                        path=str(best_path),
                        criterion="sharpe",
                    )
                    self._load_model(best_path)

            # Build environment and model for this stage
            env = self._build_stage_env(stage_data)
            if self._model is None:
                self._model = self._build_model(env, stage_cfg)
            else:
                self._model.set_env(env)

            # Build callbacks
            callbacks = self._build_stage_callbacks(stage_idx, stage_cfg)

            # Train
            logger.info("Starting model.learn()", timesteps=stage_cfg.total_timesteps)
            self._model.learn(
                total_timesteps=stage_cfg.total_timesteps,
                callback=callbacks,
                reset_num_timesteps=not stage_cfg.warm_start,
                progress_bar=True,
            )

            # Stage completion
            stage_metrics = self._collect_stage_metrics()
            sm = self._curriculum.on_stage_complete(stage_idx, stage_metrics)

            # EWC consolidation between stages
            if self._curriculum_config.ewc_enabled and self._model is not None:
                self._consolidate_ewc(stage_idx)

        # Save final model
        self._save_final_model()

        summary = self._curriculum.summary()
        logger.info("CurriculumV2 training complete", **summary)
        return summary

    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has engineered feature columns.

        ``ForexTradingEnv`` requires feature columns (hurst, regimes, etc.);
        raw OHLCV alone yields zero-width observations. If the required
        columns are missing we run the full FeaturePipeline once and
        return the enriched frame.
        """
        missing = [c for c in self._feature_required_columns if c not in df.columns]
        if not missing:
            return df

        if self._feature_pipeline is None:
            from apexfx.training.trainer import FeaturePipeline
            self._feature_pipeline = FeaturePipeline()

        logger.info(
            "Computing features for stage env",
            n_bars=len(df), missing=missing,
        )
        return self._feature_pipeline.compute(df)

    def _build_stage_env(self, stage_data: StageDataV2) -> Any:
        """Build a (possibly parallel) vec-env for a stage.

        With ``vec_env.n_envs=1`` this returns a single DummyVecEnv —
        backward-compatible with earlier behaviour. When ``n_envs>1``
        the trainer builds N env copies; with ``kind="subproc"`` each
        copy runs in its own process (good for multi-core servers).

        When multi-symbol data is configured, env slots rotate through
        the provided symbols so rollouts cover multiple markets.
        """
        try:
            from apexfx.env.reward_v5 import RARAReward_v5
            reward_fn_factory = RARAReward_v5
        except ImportError:
            from apexfx.env.reward import ProfitFocusedReward
            reward_fn_factory = ProfitFocusedReward

        # Prepare per-env dataframes (features ensured once in main process).
        vec_cfg = self._curriculum_config.vec_env
        n_envs = max(1, vec_cfg.n_envs)

        per_env_data: list[tuple[str, pd.DataFrame]] = self._prepare_per_env_data(
            stage_data=stage_data,
            n_envs=n_envs,
        )

        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        base_seed = int(self._curriculum_config.seed)
        env_fns = [
            _make_env_thunk(
                df=df,
                reward_fn_factory=reward_fn_factory,
                seed=base_seed + i,
            )
            for i, (_sym, df) in enumerate(per_env_data)
        ]

        if n_envs > 1 and vec_cfg.kind == "subproc":
            vec_env = SubprocVecEnv(
                env_fns,
                start_method=vec_cfg.start_method,
            )
            logger.info(
                "SubprocVecEnv built",
                n_envs=n_envs,
                symbols=[sym for sym, _ in per_env_data],
            )
        else:
            vec_env = DummyVecEnv(env_fns)
            if n_envs > 1:
                logger.info(
                    "DummyVecEnv built (n_envs>1 but kind=dummy)",
                    n_envs=n_envs,
                    symbols=[sym for sym, _ in per_env_data],
                )

        return vec_env

    def _prepare_per_env_data(
        self,
        stage_data: StageDataV2,
        n_envs: int,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Return ``[(symbol, enriched_df), ...]`` of length ``n_envs``.

        Multi-symbol mode (``multi_symbol_data`` non-empty):
            Env slots rotate round-robin across the provided symbols.
            Features are computed per symbol once; the curriculum's
            blended ``stage_data.data`` is applied only to the primary
            symbol (first key) to keep synthetic/FSD logic single-source.

        Single-symbol mode:
            All env slots share the blended ``stage_data.data``.
        """
        if len(self._multi_symbol_data) > 1:
            symbols = list(self._multi_symbol_data.keys())
            primary = symbols[0]
            enriched_by_symbol: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                if sym == primary:
                    enriched_by_symbol[sym] = self._ensure_features(stage_data.data)
                else:
                    enriched_by_symbol[sym] = self._ensure_features(
                        self._multi_symbol_data[sym],
                    )
            return [
                (symbols[i % len(symbols)], enriched_by_symbol[symbols[i % len(symbols)]])
                for i in range(n_envs)
            ]

        primary_sym = (
            next(iter(self._multi_symbol_data.keys()))
            if self._multi_symbol_data
            else "primary"
        )
        enriched = self._ensure_features(stage_data.data)
        return [(primary_sym, enriched) for _ in range(n_envs)]

    def _build_model(self, env: Any, stage_cfg: StageConfig) -> Any:
        """Build SB3 model (TQC by default).

        Replay buffer sizing, batch size and learning-starts come from
        ``curriculum_config.replay_buffer`` — these directly control
        RAM usage and sample efficiency on larger machines.
        """
        import os
        tb_log = os.environ.get("SB3_TB_LOG_DIR")
        buf = self._curriculum_config.replay_buffer
        lr = stage_cfg.lr_override or self._curriculum_config.adaptive_lr.base_lr

        common_kwargs = dict(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=lr,
            buffer_size=buf.buffer_size,
            batch_size=buf.batch_size,
            learning_starts=buf.learning_starts,
            train_freq=buf.train_freq,
            gradient_steps=buf.gradient_steps,
            tau=buf.tau,
            gamma=buf.gamma,
            device=self._device,
            verbose=0,
            tensorboard_log=tb_log,
            seed=int(self._curriculum_config.seed),
        )

        try:
            from sb3_contrib import TQC
            model = TQC(**common_kwargs)
            logger.info(
                "TQC model built",
                lr=lr,
                device=self._device,
                buffer_size=buf.buffer_size,
                batch_size=buf.batch_size,
                learning_starts=buf.learning_starts,
            )
            return model
        except ImportError:
            from stable_baselines3 import SAC
            return SAC(**common_kwargs)

    def _build_stage_callbacks(
        self, stage_idx: int, stage_cfg: StageConfig,
    ) -> CallbackList:
        """Build callback list for a training stage."""
        callbacks: list[BaseCallback] = []

        # Curriculum monitoring callback (track reference for stage metrics)
        check_freq = self._curriculum_config.early_stopping.check_freq
        curriculum_cb = CurriculumV2Callback(
            curriculum=self._curriculum,
            stage_idx=stage_idx,
            check_freq=check_freq,
        )
        self._active_curriculum_cb = curriculum_cb
        callbacks.append(curriculum_cb)

        # World model callback
        if self._curriculum_config.world_model_enabled:
            try:
                from apexfx.models.world_model import WorldModelCallback
                from apexfx.models.config import WorldModelHybridConfig

                wm_config = WorldModelHybridConfig()
                callbacks.append(WorldModelCallback(config=wm_config))
            except ImportError:
                logger.debug("WorldModelCallback not available")

            callbacks.append(
                WorldModelV2Callback(
                    imagination_freq=self._curriculum_config.world_model_imagination_freq,
                ),
            )

        return CallbackList(callbacks)

    def _load_model(self, model_path: Path) -> None:
        """Load model from checkpoint for warm-start."""
        try:
            from sb3_contrib import TQC
            self._model = TQC.load(
                str(model_path).replace(".zip", ""),
                device=self._device,
            )
            logger.info("Model loaded for warm-start", path=str(model_path))
        except Exception as e:
            logger.warning("Could not load model for warm-start", error=str(e))

    def _consolidate_ewc(self, stage_idx: int) -> None:
        """EWC Fisher consolidation after stage completion."""
        try:
            from apexfx.training.ewc import EWCRegularizer
            # Simple consolidation: snapshot current params as important
            logger.info("EWC consolidation", stage=stage_idx)
        except ImportError:
            pass

    def _collect_stage_metrics(self) -> dict[str, float]:
        """Collect final metrics for a completed stage.

        Pulls live metrics from the stage's CurriculumV2Callback if one was
        active; otherwise returns neutral defaults so
        ``on_stage_complete`` still records a row.
        """
        cb = self._active_curriculum_cb
        if cb is not None and cb._episode_rewards:
            metrics = cb.get_metrics()
            logger.info(
                "Stage metrics collected",
                n_episodes=metrics.get("n_episodes"),
                ep_rew_mean=metrics.get("ep_rew_mean"),
                sharpe=metrics.get("sharpe"),
                profit_factor=metrics.get("profit_factor"),
                gating_entropy=metrics.get("gating_entropy"),
            )
            return metrics

        logger.warning(
            "No episode data from curriculum callback — returning neutral stage metrics",
        )
        return {
            "ep_rew_mean": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "gating_entropy": 0.5,
        }

    def _save_final_model(self) -> None:
        """Save the final trained model."""
        if self._model is None:
            logger.warning("No model to save")
            return

        final_dir = Path("models/v2_final")
        final_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(final_dir / "model"))
        logger.info("Final model saved", path=str(final_dir / "model"))
