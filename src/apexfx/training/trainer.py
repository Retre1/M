"""Main training orchestrator integrating SB3 with HiveMind and curriculum learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from apexfx.config.schema import AppConfig, RLAlgorithm
from apexfx.data.mtf_synthetic import resample_real_data
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.mtf_forex_env import MTFForexTradingEnv
from apexfx.env.reward import CalmarWeightedReward
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
from apexfx.utils.metrics import compute_all_metrics

if TYPE_CHECKING:
    import pandas as pd

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

        # Auto-backtest after training
        self._run_backtest(best_path)

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

    def _build_env(self, data: pd.DataFrame, n_features: int) -> VecEnv:
        """Create wrapped Gymnasium environment."""
        risk_cfg = self._config.risk
        n_envs = self._config.model.rl.n_envs

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

        if n_envs > 1:
            logger.info("Using SubprocVecEnv", n_envs=n_envs)
            return SubprocVecEnv([make_env for _ in range(n_envs)])
        return DummyVecEnv([make_env])

    def _build_model(self, env: VecEnv, n_features: int) -> SAC | PPO:
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
    ) -> VecEnv:
        """Create MTF Gymnasium environment."""
        risk_cfg = self._config.risk
        mtf_cfg = self._config.model.mtf
        n_envs = self._config.model.rl.n_envs

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

        if n_envs > 1:
            logger.info("Using SubprocVecEnv for MTF", n_envs=n_envs)
            return SubprocVecEnv([make_env for _ in range(n_envs)])
        return DummyVecEnv([make_env])

    def _build_mtf_model(self, env: VecEnv, n_features: int) -> SAC | PPO:
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

    def _run_backtest(self, save_dir: Path) -> None:
        """Run automatic backtest after training and save results."""
        if self._model is None:
            logger.warning("No model to backtest — skipping")
            return

        if self._real_data is None or self._real_data.empty:
            logger.warning("No real data available for backtest — skipping")
            return

        logger.info("=" * 60)
        logger.info("Running post-training backtest (out-of-sample: last 30%)")
        logger.info("=" * 60)

        try:
            # Compute features on full real data
            features = self._feature_pipeline.compute(self._real_data)
            n_features = min(self._feature_pipeline.n_features, 30)

            # Split: last 30% for out-of-sample test
            split_idx = int(len(features) * 0.7)
            test_data = features.iloc[split_idx:].reset_index(drop=True)
            logger.info("Test data", n_bars=len(test_data), split="last 30%")

            if len(test_data) < 50:
                logger.warning("Too few test bars for reliable backtest", n_bars=len(test_data))
                return

            # Build evaluation environment (no reward normalization, no drawdown kill)
            from apexfx.env.reward import DifferentialSharpeReward

            if self._mtf_enabled:
                d1_test, m5_test = resample_real_data(test_data)
                d1_features = self._feature_pipeline.compute(d1_test)
                m5_features = self._feature_pipeline.compute(m5_test)
                mtf_cfg = self._config.model.mtf

                eval_env = MTFForexTradingEnv(
                    h1_data=test_data,
                    d1_data=d1_features,
                    m5_data=m5_features,
                    initial_balance=100_000.0,
                    n_market_features=n_features,
                    d1_lookback=mtf_cfg.lookback.d1,
                    h1_lookback=mtf_cfg.lookback.h1,
                    m5_lookback=mtf_cfg.lookback.m5,
                    reward_fn=DifferentialSharpeReward(),
                    max_drawdown_pct=1.0,  # Don't terminate during eval
                )
            else:
                eval_env = ForexTradingEnv(
                    data=test_data,
                    initial_balance=100_000.0,
                    n_market_features=n_features,
                    lookback=self._config.data.feature_window,
                    reward_fn=DifferentialSharpeReward(),
                    max_drawdown_pct=1.0,  # Don't terminate during eval
                )

            # Run backtest loop
            obs, info = eval_env.reset()
            done = False
            returns: list[float] = []
            equity: list[float] = [100_000.0]
            trades: list[dict] = []
            prev_value = 100_000.0
            prev_position = 0.0
            step_count = 0

            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                current_value = info.get("portfolio_value", prev_value)
                position = info.get("position", 0.0)

                if prev_value > 0:
                    step_return = (current_value - prev_value) / prev_value
                    returns.append(step_return)

                # Track trade transitions
                if abs(position) > 0 and abs(prev_position) == 0:
                    trades.append({"step": step_count, "type": "open", "position": float(position)})
                elif abs(position) == 0 and abs(prev_position) > 0:
                    trades.append({
                        "step": step_count, "type": "close",
                        "pnl": float(current_value - prev_value),
                    })

                equity.append(current_value)
                prev_value = current_value
                prev_position = position
                step_count += 1

            # Compute metrics
            returns_arr = np.array(returns) if returns else np.array([0.0])
            metrics = compute_all_metrics(returns_arr)

            final_value = equity[-1]
            total_return_pct = (final_value - 100_000) / 100_000 * 100

            # Max drawdown from equity curve
            eq_arr = np.array(equity)
            peak = np.maximum.accumulate(eq_arr)
            dd_pct = ((peak - eq_arr) / peak * 100)
            max_dd_pct = float(np.max(dd_pct)) if len(dd_pct) > 0 else 0.0

            n_trades = len([t for t in trades if t["type"] == "open"])

            # Print results
            logger.info("=" * 60)
            logger.info("BACKTEST RESULTS (Out-of-Sample: last 30%)")
            logger.info("=" * 60)
            logger.info(f"  Test period:        {len(test_data)} bars")
            logger.info("  Initial balance:    $100,000.00")
            logger.info(f"  Final balance:      ${final_value:,.2f}")
            logger.info(f"  Total return:       {total_return_pct:+.2f}%")
            logger.info(f"  Max drawdown:       {max_dd_pct:.2f}%")
            logger.info(f"  Sharpe ratio:       {metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  Sortino ratio:      {metrics.get('sortino_ratio', 0):.4f}")
            logger.info(f"  Calmar ratio:       {metrics.get('calmar_ratio', 0):.4f}")
            logger.info(f"  Win rate:           {metrics.get('win_rate', 0):.2%}")
            logger.info(f"  Profit factor:      {metrics.get('profit_factor', 0):.4f}")
            logger.info(f"  Total trades:       {n_trades}")
            logger.info(f"  Steps:              {step_count}")
            logger.info("=" * 60)

            # Save results to JSON
            results = {
                "test_bars": len(test_data),
                "initial_balance": 100_000,
                "final_balance": float(final_value),
                "total_return_pct": float(total_return_pct),
                "max_drawdown_pct": float(max_dd_pct),
                "n_trades": n_trades,
                "n_steps": step_count,
                "mtf_enabled": self._mtf_enabled,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "equity_curve": [float(e) for e in equity],
                "trades": trades,
            }

            results_path = save_dir / "backtest_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Backtest results saved to {results_path}")

        except Exception as e:
            logger.error("Backtest failed (training succeeded though)", error=str(e))
            import traceback
            traceback.print_exc()

    @property
    def model(self) -> SAC | PPO | None:
        return self._model
