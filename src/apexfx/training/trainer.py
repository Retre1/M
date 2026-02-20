"""Main training orchestrator integrating SB3 with HiveMind and curriculum learning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from apexfx.config.schema import AppConfig, RLAlgorithm
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.mtf_forex_env import MTFForexTradingEnv
from apexfx.env.reward import CalmarWeightedReward, DifferentialSharpeReward
from apexfx.env.wrappers import MonitorWrapper, NormalizeReward
from apexfx.features.pipeline import FeaturePipeline
from apexfx.models.ensemble.hive_mind import HiveMindExtractor, MTFHiveMindExtractor
from apexfx.training.callbacks import (
    EarlyStoppingCallback,
    MetricsCallback,
    TradingCheckpointCallback,
)
from apexfx.training.curriculum import CurriculumManager, MTFStageData
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Main training pipeline:
    1. Load config and data
    2. Run feature pipeline
    3. Create environments with curriculum data
    4. Train SB3 model with HiveMind feature extractor
    5. Save best checkpoints
    """

    def __init__(self, config: AppConfig, real_data: pd.DataFrame | None = None) -> None:
        self._config = config
        self._real_data = real_data
        self._feature_pipeline = FeaturePipeline()
        self._device = self._resolve_device()
        self._model: SAC | PPO | None = None

    def _resolve_device(self) -> str:
        dev = self._config.base.device.value
        if dev == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return dev

    @property
    def _mtf_enabled(self) -> bool:
        return self._config.model.mtf.enabled

    def train(self) -> None:
        """Run the full curriculum training pipeline."""
        logger.info("Starting training", device=self._device, mtf=self._mtf_enabled)

        curriculum = CurriculumManager(
            config=self._config.training,
            real_data=self._real_data,
            seed=self._config.base.seed,
            mtf_enabled=self._mtf_enabled,
        )

        for stage_data in curriculum.stages():
            logger.info(
                "Training stage",
                stage=stage_data.stage_idx,
                name=stage_data.stage.name,
                timesteps=stage_data.stage.total_timesteps,
            )

            if self._mtf_enabled and isinstance(stage_data, MTFStageData):
                self._train_mtf_stage(stage_data)
            else:
                self._train_single_tf_stage(stage_data)

            # Save stage checkpoint
            save_path = Path(self._config.base.paths.models_dir) / "checkpoints"
            save_path.mkdir(parents=True, exist_ok=True)
            self._model.save(str(save_path / f"stage_{stage_data.stage_idx}"))
            logger.info("Stage complete", stage=stage_data.stage_idx)

        # Save final model
        best_path = Path(self._config.base.paths.models_dir) / "best"
        best_path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(best_path / "final_model"))
        logger.info("Training complete")

    def _train_single_tf_stage(self, stage_data) -> None:
        """Train a single-timeframe stage (original behavior)."""
        # Compute features
        logger.info("Computing features", n_bars=len(stage_data.data))
        features = self._feature_pipeline.compute(stage_data.data)
        n_features = min(self._feature_pipeline.n_features, 30)
        logger.info("Features ready", n_features=n_features, n_bars=len(features))

        # Build environment
        logger.info("Building environment")
        env = self._build_env(features, n_features)

        # Build or update model
        if self._model is None or not stage_data.stage.warm_start:
            logger.info("Building model")
            self._model = self._build_model(env, n_features)
        else:
            self._model.set_env(env)
            logger.info("Warm-starting from previous stage")

        callbacks = self._build_callbacks(stage_data.stage_idx)

        logger.info("Starting model.learn()", timesteps=stage_data.stage.total_timesteps)
        self._model.learn(
            total_timesteps=stage_data.stage.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not stage_data.stage.warm_start,
            progress_bar=True,
        )

    def _train_mtf_stage(self, stage_data: MTFStageData) -> None:
        """Train a multi-timeframe stage."""
        # Compute features for each timeframe
        logger.info("Computing MTF features (D1, H1, M5)")

        d1_features = self._feature_pipeline.compute(stage_data.d1_data)
        logger.info("D1 features ready", n_bars=len(d1_features))

        h1_features = self._feature_pipeline.compute(stage_data.h1_data)
        logger.info("H1 features ready", n_bars=len(h1_features))

        m5_features = self._feature_pipeline.compute(stage_data.m5_data)
        logger.info("M5 features ready", n_bars=len(m5_features))

        n_features = min(self._feature_pipeline.n_features, 30)

        # Build MTF environment
        logger.info("Building MTF environment")
        env = self._build_mtf_env(d1_features, h1_features, m5_features, n_features)

        # Build or update model
        if self._model is None or not stage_data.stage.warm_start:
            logger.info("Building MTF model")
            self._model = self._build_mtf_model(env, n_features)
        else:
            self._model.set_env(env)
            logger.info("Warm-starting from previous stage")

        callbacks = self._build_callbacks(stage_data.stage_idx)

        logger.info("Starting model.learn()", timesteps=stage_data.stage.total_timesteps)
        self._model.learn(
            total_timesteps=stage_data.stage.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not stage_data.stage.warm_start,
            progress_bar=True,
        )

    def _build_env(self, data: pd.DataFrame, n_features: int) -> DummyVecEnv:
        """Create wrapped Gymnasium environment."""
        rl_cfg = self._config.model.rl
        risk_cfg = self._config.risk

        def make_env():
            env = ForexTradingEnv(
                data=data,
                initial_balance=100_000.0,
                max_position_pct=risk_cfg.position_sizing.max_position_pct,
                n_market_features=n_features,
                lookback=self._config.data.feature_window,
                reward_fn=CalmarWeightedReward(lambda_dd=2.0),
                max_drawdown_pct=0.15,
            )
            env = MonitorWrapper(env)
            env = NormalizeReward(env)
            return env

        return DummyVecEnv([make_env])

    def _build_model(self, env: DummyVecEnv, n_features: int) -> SAC | PPO:
        """Create SB3 model with HiveMind feature extractor."""
        rl_cfg = self._config.model.rl
        lookback = self._config.data.feature_window

        policy_kwargs = {
            "features_extractor_class": HiveMindExtractor,
            "features_extractor_kwargs": {
                "n_continuous_vars": n_features,
                "d_model": self._config.model.tft.d_model,
                "seq_len": lookback,
            },
            "net_arch": [256, 256],
            "optimizer_kwargs": {"weight_decay": 1e-4},
        }

        # Gradient clipping prevents explosion with deep TFT feature extractor
        policy_kwargs["optimizer_kwargs"]["eps"] = 1e-5  # tighter Adam epsilon

        tb_log_dir = str(Path(self._config.base.paths.logs_dir) / "tensorboard")

        if rl_cfg.algorithm == RLAlgorithm.SAC:
            return SAC(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=rl_cfg.learning_rate,
                buffer_size=rl_cfg.buffer_size,
                batch_size=rl_cfg.batch_size,
                ent_coef=rl_cfg.ent_coef,
                gamma=rl_cfg.gamma,
                tau=rl_cfg.tau,
                train_freq=rl_cfg.train_freq,
                gradient_steps=rl_cfg.gradient_steps,
                learning_starts=rl_cfg.learning_starts,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )
        else:
            return PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=rl_cfg.learning_rate,
                batch_size=rl_cfg.batch_size,
                gamma=rl_cfg.gamma,
                max_grad_norm=0.5,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )

    def _build_mtf_env(
        self,
        d1_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        m5_data: pd.DataFrame,
        n_features: int,
    ) -> DummyVecEnv:
        """Create MTF Gymnasium environment."""
        risk_cfg = self._config.risk
        mtf_cfg = self._config.model.mtf

        def make_env():
            env = MTFForexTradingEnv(
                h1_data=h1_data,
                d1_data=d1_data,
                m5_data=m5_data,
                initial_balance=100_000.0,
                max_position_pct=risk_cfg.position_sizing.max_position_pct,
                n_market_features=n_features,
                d1_lookback=mtf_cfg.lookback.d1,
                h1_lookback=mtf_cfg.lookback.h1,
                m5_lookback=mtf_cfg.lookback.m5,
                reward_fn=CalmarWeightedReward(lambda_dd=2.0),
                max_drawdown_pct=0.15,
            )
            env = MonitorWrapper(env)
            env = NormalizeReward(env)
            return env

        return DummyVecEnv([make_env])

    def _build_mtf_model(self, env: DummyVecEnv, n_features: int) -> SAC | PPO:
        """Create SB3 model with MTFHiveMindExtractor."""
        rl_cfg = self._config.model.rl
        mtf_cfg = self._config.model.mtf

        policy_kwargs = {
            "features_extractor_class": MTFHiveMindExtractor,
            "features_extractor_kwargs": {
                "n_continuous_vars": n_features,
                "d_model": self._config.model.tft.d_model,
                "d1_lookback": mtf_cfg.lookback.d1,
                "h1_lookback": mtf_cfg.lookback.h1,
                "m5_lookback": mtf_cfg.lookback.m5,
                "n_mtf_context": mtf_cfg.n_mtf_context,
            },
            "net_arch": [256, 256],
            "optimizer_kwargs": {"weight_decay": 1e-4, "eps": 1e-5},
        }

        tb_log_dir = str(Path(self._config.base.paths.logs_dir) / "tensorboard")

        if rl_cfg.algorithm == RLAlgorithm.SAC:
            return SAC(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=rl_cfg.learning_rate,
                buffer_size=rl_cfg.buffer_size,
                batch_size=rl_cfg.batch_size,
                ent_coef=rl_cfg.ent_coef,
                gamma=rl_cfg.gamma,
                tau=rl_cfg.tau,
                train_freq=rl_cfg.train_freq,
                gradient_steps=rl_cfg.gradient_steps,
                learning_starts=rl_cfg.learning_starts,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )
        else:
            return PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=rl_cfg.learning_rate,
                batch_size=rl_cfg.batch_size,
                gamma=rl_cfg.gamma,
                max_grad_norm=0.5,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )

    def _build_callbacks(self, stage_idx: int) -> CallbackList:
        """Build SB3 callback list."""
        save_dir = str(
            Path(self._config.base.paths.models_dir) / "checkpoints" / f"stage_{stage_idx}"
        )
        return CallbackList([
            MetricsCallback(),
            TradingCheckpointCallback(
                save_freq=self._config.training.checkpointing.save_freq,
                save_dir=save_dir,
                keep_best_n=self._config.training.checkpointing.keep_best_n,
            ),
            EarlyStoppingCallback(
                patience=self._config.training.early_stopping.patience,
                min_delta=self._config.training.early_stopping.min_delta,
            ),
        ])

    @property
    def model(self) -> SAC | PPO | None:
        return self._model
