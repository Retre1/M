"""Pydantic v2 frozen configs for Curriculum v2 training pipeline.

All configuration is immutable after construction. Override via YAML.

Stage progression:
    Stage 0 (real_warmup)      — 100% real, filtered, gentle learning
    Stage 1 (real_full)        — 70% real + 30% SBBTS synthetic
    Stage 2 (real_augmented)   — 50% real + 50% SBBTS + DML jumps + FSD
    Stage 3 (real_adversarial) — 30% real + 70% SBBTS + full adversarial
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AdaptiveLRConfig(BaseModel, frozen=True):
    """Entropy-based adaptive learning rate scheduling.

    When gating entropy drops below `entropy_low`, LR is boosted by
    `boost_factor` to encourage exploration. When entropy exceeds
    `entropy_high`, LR is reduced by `reduce_factor` for stability.
    """
    enabled: bool = True
    base_lr: float = 3e-4
    entropy_low: float = 0.15       # below this → boost LR
    entropy_high: float = 0.90      # above this → reduce LR
    boost_factor: float = 1.5       # LR *= boost when entropy low
    reduce_factor: float = 0.7      # LR *= reduce when entropy high
    min_lr: float = 1e-5
    max_lr: float = 1e-3
    adjustment_freq: int = 5000     # check every N steps
    warmup_steps: int = 10000       # no adjustment during warmup


class MultiMetricEarlyStopConfig(BaseModel, frozen=True):
    """Early stopping based on multiple metrics.

    Monitors ep_rew_mean, Sharpe, profit_factor, gating_entropy.
    Stops if ALL metrics fail to improve for `patience` evaluations.
    """
    enabled: bool = True
    patience: int = 15
    min_delta_reward: float = 0.01
    min_delta_sharpe: float = 0.05
    min_delta_profit_factor: float = 0.01
    check_freq: int = 10000


class CheckpointStrategyConfig(BaseModel, frozen=True):
    """Multi-criteria checkpoint saving strategy.

    Saves separate checkpoints for:
    - best_sharpe: highest Sharpe ratio
    - best_entropy: best gating entropy (diversity)
    - best_reward: highest mean episode reward
    - latest: most recent checkpoint
    """
    save_freq: int = 50000
    keep_best_n: int = 3
    save_best_sharpe: bool = True
    save_best_entropy: bool = True
    save_best_reward: bool = True
    save_latest: bool = True


class StageConfig(BaseModel, frozen=True):
    """Configuration for a single curriculum stage.

    Attributes:
        name: Human-readable stage identifier.
        total_timesteps: Training steps for this stage.
        real_ratio: Fraction of data from real market [0, 1].
        sbbts_ratio: Fraction from SBBTS synthetic generator.
        enable_dml_jumps: Activate DML jump-diffusion in synthetic data.
        enable_fsd: Activate FSD regime conditioning.
        enable_adversarial: Activate adversarial noise injection.
        filter_quantile: Remove extreme bars (None = no filter).
        lr_override: Per-stage LR override (None = use adaptive).
        warm_start: Load weights from previous stage.
    """
    name: str
    description: str = ""
    total_timesteps: int = 500_000
    real_ratio: float = 1.0
    sbbts_ratio: float = 0.0
    enable_dml_jumps: bool = False
    enable_fsd: bool = False
    enable_adversarial: bool = False
    filter_quantile: float | None = None
    lr_override: float | None = None
    warm_start: bool = True
    noise_std: float = 0.0
    reward_clip: float = 25.0


class CurriculumV2Config(BaseModel, frozen=True):
    """Master configuration for Curriculum v2 training pipeline.

    Defines the 4-stage progression with all hyperparameters.
    """
    seed: int = 42
    deterministic: bool = True

    stages: list[StageConfig] = Field(default_factory=lambda: [
        StageConfig(
            name="real_warmup",
            description="Gentle warm-up on filtered real data only",
            total_timesteps=500_000,
            real_ratio=1.0,
            sbbts_ratio=0.0,
            enable_dml_jumps=False,
            enable_fsd=False,
            enable_adversarial=False,
            filter_quantile=0.95,
            warm_start=False,
            lr_override=3e-4,
        ),
        StageConfig(
            name="real_full",
            description="Full real data + 30% SBBTS synthetic augmentation",
            total_timesteps=2_000_000,
            real_ratio=0.70,
            sbbts_ratio=0.30,
            enable_dml_jumps=False,
            enable_fsd=True,
            enable_adversarial=False,
            filter_quantile=None,
            warm_start=True,
        ),
        StageConfig(
            name="real_augmented",
            description="50/50 real+SBBTS with DML jumps and FSD conditioning",
            total_timesteps=1_500_000,
            real_ratio=0.50,
            sbbts_ratio=0.50,
            enable_dml_jumps=True,
            enable_fsd=True,
            enable_adversarial=False,
            filter_quantile=None,
            warm_start=True,
        ),
        StageConfig(
            name="real_adversarial",
            description="30% real + 70% SBBTS + full adversarial + jump-aware",
            total_timesteps=1_000_000,
            real_ratio=0.30,
            sbbts_ratio=0.70,
            enable_dml_jumps=True,
            enable_fsd=True,
            enable_adversarial=True,
            filter_quantile=None,
            warm_start=True,
            noise_std=0.01,
        ),
    ])

    adaptive_lr: AdaptiveLRConfig = Field(default_factory=AdaptiveLRConfig)
    early_stopping: MultiMetricEarlyStopConfig = Field(
        default_factory=MultiMetricEarlyStopConfig,
    )
    checkpointing: CheckpointStrategyConfig = Field(
        default_factory=CheckpointStrategyConfig,
    )

    # World model integration
    world_model_enabled: bool = True
    world_model_imagination_freq: int = 10  # imagination every N updates

    # Gating v2 integration
    gating_v2_enabled: bool = True
    gating_anticollapse_weight: float = 1.0  # scale for anticollapse loss

    # EWC between stages
    ewc_enabled: bool = True
    ewc_lambda: float = 5000.0
    ewc_gamma: float = 0.9
    ewc_fisher_samples: int = 2000

    @property
    def total_timesteps(self) -> int:
        """Total timesteps across all stages."""
        return sum(s.total_timesteps for s in self.stages)

    @property
    def n_stages(self) -> int:
        return len(self.stages)
