"""Curriculum v2 — 4-stage progressive training orchestrator.

Manages the full training lifecycle:
    1. Data preparation (real + SBBTS blending per stage)
    2. Stage transitions with warm-start from best checkpoints
    3. Adaptive learning rate based on gating entropy
    4. Multi-metric early stopping
    5. Multi-criteria checkpoint saving
    6. EWC consolidation between stages
    7. Full TensorBoard + structlog monitoring

Usage::

    from apexfx.training.config import CurriculumV2Config
    from apexfx.training.curriculum_v2 import CurriculumV2

    curriculum = CurriculumV2(config=CurriculumV2Config(), real_data=df)
    stage_data = curriculum.prepare_stage(0)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from apexfx.training.config import (
    CurriculumV2Config,
    StageConfig,
)
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class StageDataV2:
    """Container for a curriculum stage's blended training data."""

    stage_cfg: StageConfig
    stage_idx: int
    data: pd.DataFrame                # blended real + synthetic
    real_data: pd.DataFrame            # pure real subset
    synthetic_data: pd.DataFrame | None = None
    fsd_features: np.ndarray | None = None  # (n_bars, 4) if FSD enabled
    metadata: dict = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Tracked metrics for a single stage."""

    stage_idx: int
    stage_name: str
    best_reward: float = float("-inf")
    best_sharpe: float = float("-inf")
    best_profit_factor: float = float("-inf")
    best_gating_entropy: float = 0.0
    jump_term_rmse: float = float("inf")
    final_lr: float = 0.0
    total_episodes: int = 0
    total_timesteps: int = 0
    wall_time_s: float = 0.0
    early_stopped: bool = False


# ── Checkpoint tracker ───────────────────────────────────────────────

class MultiCriteriaCheckpoint:
    """Tracks best checkpoints across multiple metrics.

    Maintains separate 'best' paths for each criterion:
        best_sharpe, best_entropy, best_reward, latest
    """

    def __init__(self, base_dir: Path, keep_n: int = 3) -> None:
        self._base_dir = base_dir
        self._keep_n = keep_n
        self._best: dict[str, tuple[float, Path]] = {
            "sharpe": (float("-inf"), Path()),
            "entropy": (float("-inf"), Path()),
            "reward": (float("-inf"), Path()),
        }
        base_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(
        self,
        model: Any,
        metrics: dict[str, float],
        stage_idx: int,
        step: int,
    ) -> list[str]:
        """Save checkpoint if any metric improved. Returns list of improved criteria."""
        improved = []
        tag = f"stage{stage_idx}_step{step}"

        checks = [
            ("sharpe", metrics.get("sharpe", float("-inf"))),
            ("entropy", metrics.get("gating_entropy", float("-inf"))),
            ("reward", metrics.get("ep_rew_mean", float("-inf"))),
        ]

        for criterion, value in checks:
            best_val, _ = self._best[criterion]
            if value > best_val:
                path = self._base_dir / f"best_{criterion}"
                path.mkdir(parents=True, exist_ok=True)
                model.save(str(path / "model"))
                self._best[criterion] = (value, path)
                improved.append(criterion)
                logger.info(
                    f"New best {criterion}",
                    value=round(value, 4),
                    previous=round(best_val, 4),
                    tag=tag,
                )

        # Always save latest
        latest_path = self._base_dir / "latest"
        latest_path.mkdir(parents=True, exist_ok=True)
        model.save(str(latest_path / "model"))

        # Save metadata
        meta = {
            "stage_idx": stage_idx,
            "step": step,
            "metrics": {k: round(v, 6) for k, v in metrics.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "improved": improved,
        }
        with open(self._base_dir / "latest" / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return improved

    def get_best_model_path(self, criterion: str = "sharpe") -> Path | None:
        """Get path to best model for a given criterion."""
        _, path = self._best.get(criterion, (float("-inf"), Path()))
        model_path = path / "model.zip"
        if model_path.exists():
            return model_path
        return None


# ── Adaptive LR controller ──────────────────────────────────────────

class AdaptiveLRController:
    """Entropy-based learning rate adjustment.

    Monitors gating entropy and adjusts the RL algorithm's LR:
    - Low entropy (collapse risk) → boost LR for exploration
    - High entropy (chaotic) → reduce LR for stability
    - Normal range → no change
    """

    def __init__(
        self,
        base_lr: float = 3e-4,
        entropy_low: float = 0.15,
        entropy_high: float = 0.90,
        boost_factor: float = 1.5,
        reduce_factor: float = 0.7,
        min_lr: float = 1e-5,
        max_lr: float = 1e-3,
    ) -> None:
        self._base_lr = base_lr
        self._current_lr = base_lr
        self._entropy_low = entropy_low
        self._entropy_high = entropy_high
        self._boost_factor = boost_factor
        self._reduce_factor = reduce_factor
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._history: list[tuple[int, float, float]] = []  # (step, entropy, lr)

    @property
    def current_lr(self) -> float:
        return self._current_lr

    def update(self, entropy: float, step: int) -> float:
        """Adjust LR based on current gating entropy.

        Args:
            entropy: Current gating entropy value.
            step: Current training step.

        Returns:
            New learning rate.
        """
        old_lr = self._current_lr

        if entropy < self._entropy_low:
            # Collapse risk — boost exploration
            self._current_lr = min(
                self._current_lr * self._boost_factor,
                self._max_lr,
            )
            if self._current_lr != old_lr:
                logger.info(
                    "LR boosted (low entropy)",
                    entropy=round(entropy, 4),
                    old_lr=old_lr,
                    new_lr=self._current_lr,
                    step=step,
                )
        elif entropy > self._entropy_high:
            # Too chaotic — reduce for stability
            self._current_lr = max(
                self._current_lr * self._reduce_factor,
                self._min_lr,
            )
            if self._current_lr != old_lr:
                logger.info(
                    "LR reduced (high entropy)",
                    entropy=round(entropy, 4),
                    old_lr=old_lr,
                    new_lr=self._current_lr,
                    step=step,
                )

        self._history.append((step, entropy, self._current_lr))
        return self._current_lr

    def reset(self, new_base_lr: float | None = None) -> None:
        """Reset LR for a new stage."""
        if new_base_lr is not None:
            self._base_lr = new_base_lr
        self._current_lr = self._base_lr


# ── Multi-metric early stopping ─────────────────────────────────────

class MultiMetricEarlyStopping:
    """Early stopping that monitors multiple metrics simultaneously.

    Only triggers if ALL tracked metrics fail to improve for `patience`
    consecutive checks. This prevents premature stopping when one metric
    plateaus but others continue improving.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta_reward: float = 0.01,
        min_delta_sharpe: float = 0.05,
        min_delta_profit_factor: float = 0.01,
    ) -> None:
        self._patience = patience
        self._deltas = {
            "ep_rew_mean": min_delta_reward,
            "sharpe": min_delta_sharpe,
            "profit_factor": min_delta_profit_factor,
        }
        self._best: dict[str, float] = {k: float("-inf") for k in self._deltas}
        self._no_improve_count: dict[str, int] = {k: 0 for k in self._deltas}

    def check(self, metrics: dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            metrics: Current metric values.

        Returns:
            True if ALL metrics exceeded patience without improvement.
        """
        for key, delta in self._deltas.items():
            val = metrics.get(key, float("-inf"))
            if val > self._best[key] + delta:
                self._best[key] = val
                self._no_improve_count[key] = 0
            else:
                self._no_improve_count[key] += 1

        # Stop only if ALL metrics stalled
        all_stalled = all(
            c >= self._patience for c in self._no_improve_count.values()
        )

        if all_stalled:
            logger.warning(
                "Early stopping triggered — all metrics stalled",
                counts=dict(self._no_improve_count),
                patience=self._patience,
            )

        return all_stalled

    def reset(self) -> None:
        """Reset for a new stage."""
        self._best = {k: float("-inf") for k in self._deltas}
        self._no_improve_count = {k: 0 for k in self._deltas}


# ── Data blender ─────────────────────────────────────────────────────

class DataBlender:
    """Blend real and SBBTS synthetic data according to stage ratios.

    Handles:
    - Real data filtering (quantile-based extreme bar removal)
    - SBBTS synthetic generation and mixing
    - DML jump injection into synthetic data
    - FSD feature computation
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    def blend(
        self,
        real_data: pd.DataFrame,
        stage_cfg: StageConfig,
    ) -> StageDataV2:
        """Prepare blended data for a curriculum stage.

        Args:
            real_data: Raw real market data (OHLCV).
            stage_cfg: Stage configuration with ratios and flags.

        Returns:
            StageDataV2 with blended data and metadata.
        """
        # 1. Filter real data if needed
        filtered_real = real_data.copy()
        if stage_cfg.filter_quantile is not None:
            filtered_real = self._filter_extreme_bars(
                filtered_real, stage_cfg.filter_quantile,
            )

        n_real_target = int(len(filtered_real) * stage_cfg.real_ratio)
        if n_real_target < len(filtered_real):
            # Subsample real data to match ratio
            idx = self._rng.choice(
                len(filtered_real), size=n_real_target, replace=False,
            )
            idx.sort()
            real_subset = filtered_real.iloc[idx].reset_index(drop=True)
        else:
            real_subset = filtered_real

        # 2. Generate synthetic data if needed
        synthetic_data = None
        if stage_cfg.sbbts_ratio > 0:
            n_synth = int(len(real_subset) * stage_cfg.sbbts_ratio / max(stage_cfg.real_ratio, 0.01))
            synthetic_data = self._generate_sbbts(
                real_data, n_synth, stage_cfg,
            )

        # 3. Blend
        if synthetic_data is not None and len(synthetic_data) > 0:
            blended = pd.concat(
                [real_subset, synthetic_data], ignore_index=True,
            ).sort_values("time" if "time" in real_subset.columns else real_subset.columns[0])
            blended = blended.reset_index(drop=True)
        else:
            blended = real_subset

        # 4. Compute FSD features if enabled
        fsd_features = None
        if stage_cfg.enable_fsd:
            fsd_features = self._compute_fsd(blended)

        # 5. Add noise if configured
        if stage_cfg.noise_std > 0:
            blended = self._add_noise(blended, stage_cfg.noise_std)

        metadata = {
            "n_real": len(real_subset),
            "n_synthetic": len(synthetic_data) if synthetic_data is not None else 0,
            "n_blended": len(blended),
            "real_ratio": stage_cfg.real_ratio,
            "sbbts_ratio": stage_cfg.sbbts_ratio,
            "filtered": stage_cfg.filter_quantile is not None,
            "dml_jumps": stage_cfg.enable_dml_jumps,
            "fsd_enabled": stage_cfg.enable_fsd,
        }

        logger.info(
            "Data blended for stage",
            stage=stage_cfg.name,
            **metadata,
        )

        return StageDataV2(
            stage_cfg=stage_cfg,
            stage_idx=0,  # set by caller
            data=blended,
            real_data=real_subset,
            synthetic_data=synthetic_data,
            fsd_features=fsd_features,
            metadata=metadata,
        )

    def _filter_extreme_bars(
        self, data: pd.DataFrame, quantile: float,
    ) -> pd.DataFrame:
        """Remove bars with extreme returns."""
        if "close" not in data.columns:
            return data
        returns = data["close"].pct_change().abs()
        threshold = returns.quantile(quantile)
        mask = returns <= threshold
        mask.iloc[0] = True
        filtered = data[mask].reset_index(drop=True)
        logger.info(
            "Filtered extreme bars",
            original=len(data),
            kept=len(filtered),
            quantile=quantile,
        )
        return filtered

    def _generate_sbbts(
        self,
        real_data: pd.DataFrame,
        n_steps: int,
        stage_cfg: StageConfig,
    ) -> pd.DataFrame:
        """Generate SBBTS synthetic data, optionally with DML jumps.

        Attempts to import SBBTSGenerator. Falls back to simple GBM
        if SBBTS dependencies are not available.
        """
        try:
            from apexfx.data.sbbts_generator import SBBTSGenerator
            generator = SBBTSGenerator()

            # Calibrate from real data (expects DataFrame with 'close' column)
            if "close" in real_data.columns:
                generator.calibrate(real_data)

            synthetic = generator.generate(real_data, n_bars=n_steps)

            if stage_cfg.enable_dml_jumps:
                synthetic = self._inject_dml_jumps(synthetic)

            return synthetic

        except ImportError:
            logger.warning(
                "SBBTSGenerator not available, using simple noise augmentation",
            )
            return self._fallback_synthetic(real_data, n_steps)

    def _inject_dml_jumps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Merton jump-diffusion events to synthetic data."""
        if "close" not in data.columns:
            return data

        augmented = data.copy()
        n = len(augmented)
        jump_intensity = 0.05  # 5% of bars have jumps
        jump_mean = -0.005
        jump_std = 0.02

        jump_mask = self._rng.random(n) < jump_intensity
        jumps = self._rng.normal(jump_mean, jump_std, n) * jump_mask

        augmented["close"] = augmented["close"] * (1 + jumps)
        if "high" in augmented.columns:
            augmented["high"] = augmented["high"] * (1 + np.maximum(jumps, 0))
        if "low" in augmented.columns:
            augmented["low"] = augmented["low"] * (1 + np.minimum(jumps, 0))

        n_jumps = int(jump_mask.sum())
        if n_jumps > 0:
            logger.info("DML jumps injected", n_jumps=n_jumps, n_bars=n)

        return augmented

    def _fallback_synthetic(
        self, real_data: pd.DataFrame, n_steps: int,
    ) -> pd.DataFrame:
        """Simple noise-augmented copy of real data as fallback."""
        if len(real_data) == 0:
            return pd.DataFrame()
        idx = self._rng.choice(len(real_data), size=min(n_steps, len(real_data)), replace=True)
        synthetic = real_data.iloc[idx].copy().reset_index(drop=True)
        numeric_cols = synthetic.select_dtypes(include=[np.number]).columns
        noise = self._rng.normal(0, 0.002, size=(len(synthetic), len(numeric_cols)))
        synthetic[numeric_cols] += noise
        return synthetic

    def _compute_fsd(self, data: pd.DataFrame) -> np.ndarray | None:
        """Compute FSD regime features if the data has enough numeric columns.

        FSDCalculator requires ≥3 numeric (non-OHLCV) feature columns to
        measure cross-sectional skewness dispersion. If the stage data has
        only raw bars, skip gracefully instead of crashing.
        """
        try:
            from apexfx.data.features import FSDCalculator
        except ImportError:
            logger.debug("FSDCalculator not available")
            return None

        # Count eligible numeric columns (exclude OHLCV)
        excluded = {"open", "high", "low", "close", "volume", "time"}
        numeric = data.select_dtypes(include=[np.number]).columns
        eligible = [c for c in numeric if c not in excluded]
        if len(eligible) < 3:
            logger.info(
                "Skipping FSD — insufficient feature columns",
                n_eligible=len(eligible),
            )
            return None

        try:
            fsd_df = FSDCalculator().compute(data)
            return fsd_df.values
        except Exception as e:
            logger.warning("FSD computation failed", error=str(e))
            return None

    def _add_noise(self, data: pd.DataFrame, noise_std: float) -> pd.DataFrame:
        """Add Gaussian noise to all numeric columns."""
        augmented = data.copy()
        numeric_cols = augmented.select_dtypes(include=[np.number]).columns
        noise = self._rng.normal(0, noise_std, size=(len(augmented), len(numeric_cols)))
        augmented[numeric_cols] += noise
        return augmented


# ── CurriculumV2 orchestrator ───────────────────────────────────────

class CurriculumV2:
    """4-stage progressive curriculum training orchestrator.

    Manages the complete training lifecycle with adaptive mechanisms,
    multi-criteria checkpointing, and full v2 component integration.

    Usage::

        curriculum = CurriculumV2(
            config=CurriculumV2Config(),
            real_data=real_df,
            checkpoint_dir=Path("models/v2_checkpoints"),
        )

        for stage_idx in range(curriculum.n_stages):
            stage_data = curriculum.prepare_stage(stage_idx)
            # ... train model ...
            curriculum.on_stage_complete(stage_idx, model, metrics)

    Attributes:
        config: Frozen curriculum configuration.
        blender: Data blending utility.
        lr_controller: Adaptive LR manager.
        early_stopper: Multi-metric early stopping.
        checkpoint_tracker: Multi-criteria checkpoint saver.
    """

    def __init__(
        self,
        config: CurriculumV2Config | None = None,
        real_data: pd.DataFrame | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.config = config or CurriculumV2Config()
        self._real_data = real_data
        self._stage_metrics: list[StageMetrics] = []

        # Set seed for reproducibility
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Sub-components
        self.blender = DataBlender(seed=self.config.seed)

        lr_cfg = self.config.adaptive_lr
        self.lr_controller = AdaptiveLRController(
            base_lr=lr_cfg.base_lr,
            entropy_low=lr_cfg.entropy_low,
            entropy_high=lr_cfg.entropy_high,
            boost_factor=lr_cfg.boost_factor,
            reduce_factor=lr_cfg.reduce_factor,
            min_lr=lr_cfg.min_lr,
            max_lr=lr_cfg.max_lr,
        )

        es_cfg = self.config.early_stopping
        self.early_stopper = MultiMetricEarlyStopping(
            patience=es_cfg.patience,
            min_delta_reward=es_cfg.min_delta_reward,
            min_delta_sharpe=es_cfg.min_delta_sharpe,
            min_delta_profit_factor=es_cfg.min_delta_profit_factor,
        )

        ckpt_dir = checkpoint_dir or Path("models/v2_checkpoints")
        self.checkpoint_tracker = MultiCriteriaCheckpoint(
            ckpt_dir,
            keep_n=self.config.checkpointing.keep_best_n,
        )

        logger.info(
            "CurriculumV2 initialized",
            n_stages=self.n_stages,
            total_timesteps=self.config.total_timesteps,
            seed=self.config.seed,
        )

    @property
    def n_stages(self) -> int:
        return self.config.n_stages

    def prepare_stage(self, stage_idx: int) -> StageDataV2:
        """Prepare blended data for a specific stage.

        Args:
            stage_idx: Stage index (0-3).

        Returns:
            StageDataV2 with blended real + synthetic data.

        Raises:
            ValueError: If no real data or invalid stage index.
        """
        if self._real_data is None or len(self._real_data) == 0:
            raise ValueError("Real data is required for CurriculumV2 training.")
        if stage_idx >= self.n_stages:
            raise ValueError(
                f"Stage {stage_idx} does not exist (max {self.n_stages - 1})."
            )

        stage_cfg = self.config.stages[stage_idx]
        logger.info(
            "Preparing stage",
            stage=stage_idx,
            name=stage_cfg.name,
            real_ratio=stage_cfg.real_ratio,
            sbbts_ratio=stage_cfg.sbbts_ratio,
            timesteps=stage_cfg.total_timesteps,
        )

        stage_data = self.blender.blend(self._real_data, stage_cfg)
        stage_data.stage_idx = stage_idx

        # Reset adaptive components for new stage
        lr_override = stage_cfg.lr_override
        self.lr_controller.reset(lr_override)
        self.early_stopper.reset()

        return stage_data

    def on_step(
        self,
        step: int,
        stage_idx: int,
        metrics: dict[str, float],
        model: Any,
    ) -> dict[str, Any]:
        """Called every N steps during training.

        Handles:
        - Adaptive LR adjustment
        - Multi-criteria checkpointing
        - Early stopping check

        Args:
            step: Current global step.
            stage_idx: Current stage index.
            metrics: Current metric values.
            model: SB3 model for checkpointing.

        Returns:
            Dict with 'should_stop', 'new_lr', 'improved_criteria'.
        """
        result: dict[str, Any] = {
            "should_stop": False,
            "new_lr": self.lr_controller.current_lr,
            "improved_criteria": [],
        }

        # Adaptive LR
        entropy = metrics.get("gating_entropy", 0.5)
        if step > self.config.adaptive_lr.warmup_steps:
            result["new_lr"] = self.lr_controller.update(entropy, step)

        # Multi-criteria checkpoint
        improved = self.checkpoint_tracker.maybe_save(
            model, metrics, stage_idx, step,
        )
        result["improved_criteria"] = improved

        # Early stopping
        if self.config.early_stopping.enabled:
            result["should_stop"] = self.early_stopper.check(metrics)

        return result

    def on_stage_complete(
        self,
        stage_idx: int,
        metrics: dict[str, float],
    ) -> StageMetrics:
        """Record stage completion metrics.

        Args:
            stage_idx: Completed stage index.
            metrics: Final metrics for the stage.

        Returns:
            StageMetrics summary.
        """
        stage_cfg = self.config.stages[stage_idx]
        sm = StageMetrics(
            stage_idx=stage_idx,
            stage_name=stage_cfg.name,
            best_reward=metrics.get("ep_rew_mean", float("-inf")),
            best_sharpe=metrics.get("sharpe", float("-inf")),
            best_profit_factor=metrics.get("profit_factor", float("-inf")),
            best_gating_entropy=metrics.get("gating_entropy", 0.0),
            jump_term_rmse=metrics.get("jump_term_rmse", float("inf")),
            final_lr=self.lr_controller.current_lr,
            total_timesteps=stage_cfg.total_timesteps,
        )
        self._stage_metrics.append(sm)

        logger.info(
            "Stage complete",
            stage=stage_idx,
            name=stage_cfg.name,
            best_sharpe=round(sm.best_sharpe, 4),
            best_reward=round(sm.best_reward, 4),
            gating_entropy=round(sm.best_gating_entropy, 4),
        )

        return sm

    def get_warm_start_path(self, criterion: str = "sharpe") -> Path | None:
        """Get best model path for warm-starting next stage."""
        return self.checkpoint_tracker.get_best_model_path(criterion)

    @property
    def stage_history(self) -> list[StageMetrics]:
        """All completed stage metrics."""
        return list(self._stage_metrics)

    def summary(self) -> dict[str, Any]:
        """Generate training summary across all stages."""
        return {
            "n_stages_completed": len(self._stage_metrics),
            "total_timesteps": sum(s.total_timesteps for s in self._stage_metrics),
            "stages": [
                {
                    "name": s.stage_name,
                    "best_sharpe": round(s.best_sharpe, 4),
                    "best_reward": round(s.best_reward, 4),
                    "gating_entropy": round(s.best_gating_entropy, 4),
                    "early_stopped": s.early_stopped,
                }
                for s in self._stage_metrics
            ],
        }
