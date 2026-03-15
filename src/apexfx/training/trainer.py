"""Main training orchestrator integrating SB3 with HiveMind and curriculum learning.

Optimised for high-performance GPU training:
- SubprocVecEnv with N parallel environments (uses all CPU cores)
- CUDA optimizations (TF32, cuDNN benchmark, memory allocator)
- torch.compile() for graph-level speedup
- Larger batch/buffer sizes scaled for GPU VRAM
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sb3_contrib import TQC
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from apexfx.config.schema import AppConfig, RLAlgorithm
from apexfx.data.mtf_synthetic import resample_real_data
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.mtf_forex_env import MTFForexTradingEnv
from apexfx.env.reward import CalmarWeightedReward, DifferentialSharpeReward, LogReturnReward, TradingReward
from apexfx.env.wrappers import MonitorWrapper, NormalizeReward
from apexfx.features.pipeline import FeaturePipeline
from apexfx.features.selector import FeatureSelector
from apexfx.models.ensemble.hive_mind import HiveMindExtractor, MTFHiveMindExtractor
from apexfx.models.world_model import WorldModelCallback
from apexfx.training.adversarial import AdversarialObsWrapper, GradientPenaltyCallback
from apexfx.training.callbacks import (
    DiversityCallback,
    EarlyStoppingCallback,
    EWCCallback,
    MetricsCallback,
    TradingCheckpointCallback,
)
from apexfx.training.curriculum import CurriculumManager, MTFStageData
from apexfx.training.ewc import EWCRegularizer
from apexfx.training.per import PERCallback
from apexfx.training.hierarchical import ActionSmoothingWrapper, TemporalCommitmentWrapper
from apexfx.training.pretrain import TFTPretrainer
from apexfx.utils.gpu import (
    get_device_map,
    get_optimal_n_envs,
    log_gpu_memory,
    setup_cuda_optimizations,
    try_compile_model,
)
from apexfx.utils.logging import get_logger
from apexfx.utils.metrics import compute_all_metrics

logger = get_logger(__name__)


class Trainer:
    """
    Main training pipeline, optimised for multi-GPU production training.

    Key optimizations for 2× RTX 4090:
    - SubprocVecEnv: 16 parallel environments across 38 vCPUs
    - CUDA: TF32, cuDNN benchmark, expandable memory segments
    - torch.compile(): 15-30% graph-level speedup
    - Large batches (2048) saturate GPU Tensor Cores
    - 2M replay buffer fits in 128GB RAM
    """

    def __init__(self, config: AppConfig, real_data: pd.DataFrame | None = None) -> None:
        self._config = config
        self._real_data = real_data
        self._feature_pipeline = FeaturePipeline()
        self._feature_selector = FeatureSelector(top_n=15)
        self._device = self._resolve_device()
        self._model: TQC | SAC | PPO | None = None
        self._tft_pretrained: bool = False

        # --- GPU Optimizations ---
        perf_cfg = config.base.performance
        if self._device.startswith("cuda"):
            cuda_info = setup_cuda_optimizations(self._device)
            self._device_map = get_device_map(cuda_info.get("n_gpus", 1))
            log_gpu_memory("init: ")
        else:
            self._device_map = {"main": self._device, "auxiliary": self._device}

        # Determine number of parallel environments
        n_cpus = mp.cpu_count() or 4
        self._n_envs = perf_cfg.n_envs if perf_cfg.n_envs > 0 else get_optimal_n_envs(
            n_cpus, torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        logger.info(
            "Parallel environments configured",
            n_envs=self._n_envs,
            n_cpus=n_cpus,
            device=self._device,
        )

        # Online EWC regularizer (initialized once, accumulates Fisher across stages)
        ewc_cfg = config.training.ewc
        self._ewc_reg: EWCRegularizer | None = None
        if ewc_cfg.enabled:
            self._ewc_reg = EWCRegularizer(
                lambda_ewc=ewc_cfg.lambda_ewc,
                gamma_ewc=ewc_cfg.gamma_ewc,
            )

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

            # EWC: consolidate Fisher after each stage (before moving to next)
            if self._ewc_reg is not None and self._model is not None:
                ewc_cfg = self._config.training.ewc
                logger.info("Computing EWC Fisher consolidation", stage=stage_data.stage_idx)
                self._ewc_reg.consolidate(
                    self._model,
                    n_samples=ewc_cfg.fisher_n_samples,
                )

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

        # Auto walk-forward validation if enabled
        wf_cfg = self._config.training.walk_forward
        if wf_cfg.auto_validate and self._real_data is not None:
            try:
                wf_results = self.validate_walk_forward()
                p_val = wf_results.aggregate_metrics.get("mc_p_value", 1.0)
                if p_val > 0.05:
                    logger.warning(
                        "Walk-forward p-value above 0.05 — model may be overfit",
                        p_value=round(p_val, 4),
                    )
                else:
                    logger.info("Walk-forward validation passed", p_value=round(p_val, 4))
            except Exception as e:
                logger.error("Walk-forward validation failed", error=str(e))

    def validate_walk_forward(self):
        """Run walk-forward validation to detect overfitting.

        Returns:
            WalkForwardResults with per-fold and aggregate metrics.

        Raises:
            ValueError: If no real data is available.
        """
        from apexfx.training.walk_forward import WalkForwardValidator, WalkForwardResults

        if self._real_data is None:
            raise ValueError("Walk-forward validation requires real data")

        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 60)

        validator = WalkForwardValidator(self._config, self._real_data)
        results = validator.run()

        # Save results
        results_path = Path(self._config.base.paths.models_dir) / "walk_forward_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_wf_results(results, results_path)

        return results

    def _save_wf_results(self, results, results_path: Path) -> None:
        """Save walk-forward results to JSON."""
        import json

        data = {
            "n_folds": len(results.folds),
            "aggregate_metrics": {
                k: float(v) for k, v in results.aggregate_metrics.items()
            },
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_range": f"{f.train_start}-{f.train_end}",
                    "test_range": f"{f.test_start}-{f.test_end}",
                    "metrics": {k: float(v) for k, v in f.metrics.items()},
                }
                for f in results.folds
            ],
        }

        with open(results_path, "w") as fp:
            json.dump(data, fp, indent=2)
        logger.info("Walk-forward results saved", path=str(results_path))

    def _merge_intermarket(self, bars: pd.DataFrame, timeframe: str = "H1") -> pd.DataFrame:
        """Merge intermarket data (DXY, Gold, US10Y, SPX) into bars DataFrame.

        This enables IntermarketCorrExtractor to compute real correlation
        features instead of returning NaN.
        """
        intermarket_symbols = self._config.symbols.intermarket
        if not intermarket_symbols:
            return bars

        try:
            from apexfx.data.intermarket import IntermarketDataProvider
            provider = IntermarketDataProvider()  # No MT5 in training — uses fallback

            # Try loading cached intermarket data from Parquet
            from pathlib import Path
            data_dir = Path(self._config.base.paths.data_dir) / "processed"

            merged = bars.copy()
            for instrument in intermarket_symbols:
                parquet_path = data_dir / instrument / timeframe / "data.parquet"
                if parquet_path.exists():
                    idf = pd.read_parquet(parquet_path)
                    if "close" in idf.columns and "time" in idf.columns:
                        idf = idf[["time", "close"]].rename(
                            columns={"close": f"{instrument}_close"}
                        )
                        merged = merged.merge(idf, on="time", how="left")
                        logger.debug(
                            "Intermarket data merged for training",
                            instrument=instrument,
                            n_matched=merged[f"{instrument}_close"].notna().sum(),
                        )

            merged = merged.ffill()
            return merged

        except Exception as e:
            logger.debug("Intermarket merge skipped in training", error=str(e))
            return bars

    def _train_single_tf_stage(self, stage_data) -> None:
        """Train a single-timeframe stage (original behavior)."""
        # Merge intermarket data before feature computation
        bars_with_intermarket = self._merge_intermarket(stage_data.data, "H1")

        # Compute features
        logger.info("Computing features", n_bars=len(bars_with_intermarket))
        features = self._feature_pipeline.compute(bars_with_intermarket)

        # Feature selection: fit on first stage, reuse on subsequent
        if not self._feature_selector._is_fitted:
            logger.info("Running feature importance analysis")
            self._feature_selector.fit(features)
        selected = self._feature_selector.transform(features)
        n_features = len(self._feature_selector.selected_features)
        logger.info("Features ready (selected)", n_features=n_features, n_bars=len(selected))
        features = selected

        # Supervised pre-training of TFT (once, before first RL stage)
        self._pretrain_tft_if_needed(features, n_features)

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
        # Merge intermarket data for each timeframe
        d1_data = self._merge_intermarket(stage_data.d1_data, "D1")
        h1_data = self._merge_intermarket(stage_data.h1_data, "H1")
        m5_data = self._merge_intermarket(stage_data.m5_data, "M5")

        # Compute features for each timeframe
        logger.info("Computing MTF features (D1, H1, M5)")

        d1_features = self._feature_pipeline.compute(d1_data)
        logger.info("D1 features ready", n_bars=len(d1_features))

        h1_features = self._feature_pipeline.compute(h1_data)
        logger.info("H1 features ready", n_bars=len(h1_features))

        m5_features = self._feature_pipeline.compute(m5_data)
        logger.info("M5 features ready", n_bars=len(m5_features))

        # Feature selection: fit on H1 (primary timeframe), apply to all
        if not self._feature_selector._is_fitted:
            logger.info("Running feature importance analysis on H1 data")
            self._feature_selector.fit(h1_features)
        d1_features = self._feature_selector.transform(d1_features)
        h1_features = self._feature_selector.transform(h1_features)
        m5_features = self._feature_selector.transform(m5_features)
        n_features = len(self._feature_selector.selected_features)

        # Supervised pre-training of TFT on H1 data (once, before first RL stage)
        self._pretrain_tft_if_needed(h1_features, n_features)

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

    def _build_env(self, data: pd.DataFrame, n_features: int) -> DummyVecEnv | SubprocVecEnv:
        """Create vectorised environment with N parallel workers.

        With 38 vCPUs and n_envs=16, each worker runs on its own CPU core.
        SubprocVecEnv uses fork/spawn to parallelise env.step() calls,
        keeping the GPU busy while CPUs process environment logic.
        """
        risk_cfg = self._config.risk
        adv_cfg = self._config.training.adversarial
        tc_cfg = self._config.training.temporal_commitment
        aug_cfg = self._config.training.augmentation
        n_envs = self._n_envs

        def make_env_fn(rank: int):
            """Factory that captures rank for seed diversification."""
            def _init():
                env = ForexTradingEnv(
                    data=data,
                    initial_balance=100_000.0,
                    max_position_pct=risk_cfg.position_sizing.max_position_pct,
                    n_market_features=n_features,
                    lookback=self._config.data.feature_window,
                    reward_fn=TradingReward(loss_weight=2.0, reward_scale=1000.0),
                    max_drawdown_pct=0.15,
                )

                # Temporal commitment: prevent noisy flip-flopping
                if tc_cfg.enabled:
                    env = TemporalCommitmentWrapper(
                        env,
                        min_hold=tc_cfg.min_hold,
                        commitment_penalty=tc_cfg.commitment_penalty,
                    )

                # Action smoothing: EMA for smoother trading
                if tc_cfg.enabled and tc_cfg.action_smoothing_alpha < 1.0:
                    env = ActionSmoothingWrapper(
                        env, alpha=tc_cfg.action_smoothing_alpha,
                    )

                # Adversarial noise: robustness against distribution shift
                if adv_cfg.enabled:
                    env = AdversarialObsWrapper(
                        env,
                        noise_std=adv_cfg.noise_std,
                        noise_schedule=adv_cfg.noise_schedule,
                        decay_steps=adv_cfg.decay_steps,
                        warmup_steps=adv_cfg.warmup_steps,
                        adversarial_prob=adv_cfg.adversarial_prob,
                    )

                # Data augmentation: time warp, magnitude warp, window slice, mixup
                if aug_cfg.enabled:
                    from apexfx.training.augmentation import AugmentedObsWrapper
                    env = AugmentedObsWrapper(env, aug_cfg)

                env = MonitorWrapper(env)
                env = NormalizeReward(env)
                return env
            return _init

        env_fns = [make_env_fn(i) for i in range(n_envs)]

        if n_envs > 1:
            logger.info("Creating SubprocVecEnv", n_envs=n_envs)
            return SubprocVecEnv(env_fns, start_method="fork")
        else:
            return DummyVecEnv(env_fns)

    def _pretrain_tft_if_needed(self, features: pd.DataFrame, n_features: int) -> None:
        """Run supervised TFT pre-training once before RL begins.

        Loads cached weights from disk if available, otherwise trains from
        scratch and persists the result.
        """
        if self._tft_pretrained:
            return

        pretrain_path = (
            Path(self._config.base.paths.models_dir) / "pretrained" / "tft_pretrained.pt"
        )

        if pretrain_path.exists():
            logger.info("Loading cached pre-trained TFT weights from %s", pretrain_path)
            # We'll apply weights after model construction
            self._pretrain_cache_path = pretrain_path
            self._tft_pretrained = True
            return

        lookback = self._config.data.feature_window
        d_model = self._config.model.tft.d_model
        n_known_future = len(self._config.model.tft.known_future_inputs)

        # Create a temporary TFT for pre-training (same architecture as HiveMind's)
        from apexfx.models.tft.tft_model import TemporalFusionTransformer

        tft = TemporalFusionTransformer(
            n_continuous_vars=n_features,
            n_known_future_vars=n_known_future,
            d_model=d_model,
            n_heads=self._config.model.tft.n_heads,
            dropout=self._config.model.tft.dropout,
        )

        # Determine market feature columns
        market_cols = self._feature_selector.selected_features

        pretrainer = TFTPretrainer(
            tft=tft,
            device=self._device,
            lr=1e-3,
            epochs=30,
            batch_size=128,
            patience=5,
        )

        results = pretrainer.train(
            features=features,
            n_market_features=n_features,
            lookback=lookback,
            market_cols=market_cols,
        )

        logger.info(
            "TFT pre-training complete",
            val_loss=results["best_val_loss"],
            val_acc=results["final_val_acc"],
            epochs=results["epochs_trained"],
        )

        # Save pre-trained weights
        pretrainer.save_pretrained(pretrain_path)
        self._pretrain_cache_path = pretrain_path
        self._tft_pretrained = True

    def _apply_pretrained_tft(self, model: TQC | SAC | PPO) -> None:
        """Load pre-trained TFT weights into the SB3 model's feature extractor."""
        cache = getattr(self, "_pretrain_cache_path", None)
        if cache is None or not Path(cache).exists():
            return

        state = torch.load(cache, map_location="cpu", weights_only=True)
        extractor = model.policy.features_extractor

        # HiveMindExtractor.hive_mind.tft or MTFHiveMindExtractor.hive_mind.tft
        tft_module = extractor.hive_mind.tft
        try:
            tft_module.load_state_dict(state, strict=False)
            logger.info("Applied pre-trained TFT weights to SB3 model")
        except RuntimeError as e:
            logger.warning("Could not load pre-trained TFT weights (shape mismatch): %s", e)

    def _build_model(self, env, n_features: int) -> TQC | SAC | PPO:
        """Create SB3 model with HiveMind feature extractor.

        GPU optimizations applied:
        - Larger net_arch [512, 512, 256] for deeper actor/critic
        - torch.compile() on feature extractor for graph optimisation
        - AdamW with decoupled weight decay for better generalisation
        """
        rl_cfg = self._config.model.rl
        lookback = self._config.data.feature_window
        perf_cfg = self._config.base.performance

        # Cosine LR schedule: decays from initial LR to near-zero
        lr = rl_cfg.learning_rate
        if rl_cfg.use_cosine_lr:
            total_steps = sum(
                s.total_timesteps for s in self._config.training.curriculum.stages
            )
            lr = self._cosine_lr_schedule(rl_cfg.learning_rate, total_steps)

        policy_kwargs = {
            "features_extractor_class": HiveMindExtractor,
            "features_extractor_kwargs": {
                "n_continuous_vars": n_features,
                "d_model": self._config.model.tft.d_model,
                "seq_len": lookback,
            },
            "net_arch": [512, 512, 256],  # Wider actor/critic for GPU
            "optimizer_kwargs": {"weight_decay": 1e-4, "eps": 1e-5},
        }

        tb_log_dir = str(Path(self._config.base.paths.logs_dir) / "tensorboard")

        if rl_cfg.algorithm == RLAlgorithm.TQC:
            model = TQC(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                buffer_size=rl_cfg.buffer_size,
                batch_size=rl_cfg.batch_size,
                ent_coef=rl_cfg.ent_coef,
                gamma=rl_cfg.gamma,
                tau=rl_cfg.tau,
                train_freq=rl_cfg.train_freq,
                gradient_steps=rl_cfg.gradient_steps,
                learning_starts=rl_cfg.learning_starts,
                top_quantiles_to_drop_per_net=rl_cfg.top_quantiles_to_drop,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )
        elif rl_cfg.algorithm == RLAlgorithm.SAC:
            model = SAC(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
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
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                batch_size=rl_cfg.batch_size,
                gamma=rl_cfg.gamma,
                max_grad_norm=0.5,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )

        self._apply_pretrained_tft(model)

        # torch.compile() for graph-level speedup (PyTorch 2.x)
        if perf_cfg.torch_compile and self._device.startswith("cuda"):
            try:
                extractor = model.policy.features_extractor
                if extractor is not None:
                    compiled = try_compile_model(extractor, mode=perf_cfg.compile_mode)
                    model.policy.features_extractor = compiled
            except Exception as e:
                logger.warning("torch.compile skipped", error=str(e))

        # Log model size
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        logger.info(
            "Model built",
            total_params=f"{total_params:,}",
            trainable_params=f"{trainable_params:,}",
            vram_mb=round(total_params * 4 / 1e6, 1),
        )
        if self._device.startswith("cuda"):
            log_gpu_memory("after model build: ")

        return model

    @staticmethod
    def _cosine_lr_schedule(initial_lr: float, total_timesteps: int):
        """Cosine annealing LR schedule (returns callable for SB3)."""
        def schedule(progress_remaining: float) -> float:
            # progress_remaining goes from 1.0 → 0.0
            return initial_lr * (0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining))))
        return schedule

    def _build_mtf_env(
        self,
        d1_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        m5_data: pd.DataFrame,
        n_features: int,
    ) -> DummyVecEnv | SubprocVecEnv:
        """Create vectorised MTF environment with N parallel workers."""
        risk_cfg = self._config.risk
        mtf_cfg = self._config.model.mtf
        adv_cfg = self._config.training.adversarial
        tc_cfg = self._config.training.temporal_commitment
        aug_cfg = self._config.training.augmentation
        n_envs = self._n_envs

        def make_env_fn(rank: int):
            def _init():
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
                    reward_fn=TradingReward(loss_weight=2.0, reward_scale=1000.0),
                    max_drawdown_pct=0.15,
                )

                if tc_cfg.enabled:
                    env = TemporalCommitmentWrapper(
                        env,
                        min_hold=tc_cfg.min_hold,
                        commitment_penalty=tc_cfg.commitment_penalty,
                    )

                if tc_cfg.enabled and tc_cfg.action_smoothing_alpha < 1.0:
                    env = ActionSmoothingWrapper(
                        env, alpha=tc_cfg.action_smoothing_alpha,
                    )

                if adv_cfg.enabled:
                    env = AdversarialObsWrapper(
                        env,
                        noise_std=adv_cfg.noise_std,
                        noise_schedule=adv_cfg.noise_schedule,
                        decay_steps=adv_cfg.decay_steps,
                        warmup_steps=adv_cfg.warmup_steps,
                        adversarial_prob=adv_cfg.adversarial_prob,
                    )

                # Data augmentation: time warp, magnitude warp, window slice, mixup
                if aug_cfg.enabled:
                    from apexfx.training.augmentation import AugmentedObsWrapper
                    env = AugmentedObsWrapper(env, aug_cfg)

                env = MonitorWrapper(env)
                env = NormalizeReward(env)
                return env
            return _init

        env_fns = [make_env_fn(i) for i in range(n_envs)]

        if n_envs > 1:
            logger.info("Creating MTF SubprocVecEnv", n_envs=n_envs)
            return SubprocVecEnv(env_fns, start_method="fork")
        else:
            return DummyVecEnv(env_fns)

    def _build_mtf_model(self, env, n_features: int) -> TQC | SAC | PPO:
        """Create SB3 model with MTFHiveMindExtractor.

        GPU optimizations applied:
        - Larger net_arch [512, 512, 256] for deeper actor/critic
        - torch.compile() on feature extractor for graph optimisation
        - AdamW with decoupled weight decay for better generalisation
        """
        rl_cfg = self._config.model.rl
        mtf_cfg = self._config.model.mtf
        perf_cfg = self._config.base.performance

        lr = rl_cfg.learning_rate
        if rl_cfg.use_cosine_lr:
            total_steps = sum(
                s.total_timesteps for s in self._config.training.curriculum.stages
            )
            lr = self._cosine_lr_schedule(rl_cfg.learning_rate, total_steps)

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
            "net_arch": [512, 512, 256],  # Wider actor/critic for GPU
            "optimizer_kwargs": {"weight_decay": 1e-4, "eps": 1e-5},
        }

        tb_log_dir = str(Path(self._config.base.paths.logs_dir) / "tensorboard")

        if rl_cfg.algorithm == RLAlgorithm.TQC:
            model = TQC(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                buffer_size=rl_cfg.buffer_size,
                batch_size=rl_cfg.batch_size,
                ent_coef=rl_cfg.ent_coef,
                gamma=rl_cfg.gamma,
                tau=rl_cfg.tau,
                train_freq=rl_cfg.train_freq,
                gradient_steps=rl_cfg.gradient_steps,
                learning_starts=rl_cfg.learning_starts,
                top_quantiles_to_drop_per_net=rl_cfg.top_quantiles_to_drop,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )
        elif rl_cfg.algorithm == RLAlgorithm.SAC:
            model = SAC(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
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
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                batch_size=rl_cfg.batch_size,
                gamma=rl_cfg.gamma,
                max_grad_norm=0.5,
                tensorboard_log=tb_log_dir,
                device=self._device,
                verbose=1,
                seed=self._config.base.seed,
            )

        self._apply_pretrained_tft(model)

        # torch.compile() for graph-level speedup (PyTorch 2.x)
        if perf_cfg.torch_compile and self._device.startswith("cuda"):
            try:
                extractor = model.policy.features_extractor
                if extractor is not None:
                    compiled = try_compile_model(extractor, mode=perf_cfg.compile_mode)
                    model.policy.features_extractor = compiled
            except Exception as e:
                logger.warning("MTF torch.compile skipped", error=str(e))

        # Log model size
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        logger.info(
            "MTF Model built",
            total_params=f"{total_params:,}",
            trainable_params=f"{trainable_params:,}",
            vram_mb=round(total_params * 4 / 1e6, 1),
        )
        if self._device.startswith("cuda"):
            log_gpu_memory("after MTF model build: ")

        return model

    def _build_callbacks(self, stage_idx: int) -> CallbackList:
        """Build SB3 callback list (incl. diversity if enabled)."""
        save_dir = str(
            Path(self._config.base.paths.models_dir) / "checkpoints" / f"stage_{stage_idx}"
        )
        callbacks: list = [
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
        ]

        # EWC continual learning
        if self._ewc_reg is not None:
            ewc_cfg = self._config.training.ewc
            callbacks.append(
                EWCCallback(
                    ewc_regularizer=self._ewc_reg,
                    update_freq=ewc_cfg.update_freq,
                    lr=ewc_cfg.lr,
                )
            )
            logger.info(
                "EWC regularization enabled",
                lambda_ewc=ewc_cfg.lambda_ewc,
                gamma_ewc=ewc_cfg.gamma_ewc,
                update_freq=ewc_cfg.update_freq,
            )

        # Ensemble diversity regularization
        div_cfg = self._config.model.diversity
        if div_cfg.enabled:
            callbacks.append(
                DiversityCallback(
                    diversity_weight=div_cfg.diversity_weight,
                    entropy_weight=div_cfg.entropy_weight,
                    update_freq=div_cfg.update_freq,
                    batch_size=div_cfg.batch_size,
                    lr=div_cfg.lr,
                )
            )
            logger.info(
                "Diversity regularization enabled",
                weight=div_cfg.diversity_weight,
                entropy_weight=div_cfg.entropy_weight,
                freq=div_cfg.update_freq,
            )

        # Gradient Penalty for Lipschitz-smooth critic
        adv_cfg = self._config.training.adversarial
        if adv_cfg.enabled and adv_cfg.gradient_penalty_weight > 0:
            callbacks.append(
                GradientPenaltyCallback(
                    gp_weight=adv_cfg.gradient_penalty_weight,
                    update_freq=adv_cfg.gradient_penalty_freq,
                )
            )
            logger.info(
                "Gradient penalty enabled",
                weight=adv_cfg.gradient_penalty_weight,
                freq=adv_cfg.gradient_penalty_freq,
            )

        # World Model for model-based planning and curiosity
        wm_cfg = self._config.training.world_model
        if wm_cfg.enabled:
            # Determine features dim from model config
            d_model = self._config.model.tft.d_model
            d_features = d_model + 3 + 3 + 4  # encoded_state + agents + gating + position
            if self._mtf_enabled:
                d_features = d_model + 3 + 3 + 3 + 4 + self._config.model.mtf.n_mtf_context

            callbacks.append(
                WorldModelCallback(
                    d_features=d_features,
                    update_freq=wm_cfg.update_freq,
                    batch_size=wm_cfg.batch_size,
                    lr=wm_cfg.lr,
                    curiosity_weight=wm_cfg.curiosity_weight,
                    imagination_horizon=wm_cfg.imagination_horizon,
                )
            )
            logger.info(
                "World model enabled",
                d_features=d_features,
                n_ensemble=wm_cfg.n_ensemble,
                curiosity_weight=wm_cfg.curiosity_weight,
            )

        # Prioritized Experience Replay
        per_cfg = self._config.training.per
        if per_cfg.enabled:
            beta_frames = per_cfg.beta_annealing_steps
            if beta_frames <= 0:
                beta_frames = sum(
                    s.total_timesteps for s in self._config.training.curriculum.stages
                )
            callbacks.append(
                PERCallback(
                    alpha=per_cfg.alpha,
                    beta_start=per_cfg.beta_start,
                    beta_frames=beta_frames,
                    update_freq=per_cfg.update_freq,
                    batch_size=self._config.model.rl.batch_size,
                )
            )
            logger.info("PER enabled", alpha=per_cfg.alpha, beta_start=per_cfg.beta_start)

        return CallbackList(callbacks)

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

            # Apply feature selection (already fitted during training)
            if self._feature_selector._is_fitted:
                features = self._feature_selector.transform(features)
                n_features = len(self._feature_selector.selected_features)
            else:
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
                    trades.append({"step": step_count, "type": "close", "pnl": float(current_value - prev_value)})

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
            logger.info(f"  Initial balance:    $100,000.00")
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
    def model(self) -> TQC | SAC | PPO | None:
        return self._model
