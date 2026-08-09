"""Pydantic v2 frozen configs for Curriculum v2 training pipeline.

All configuration is immutable after construction. Override via YAML.

Stage progression:
    Stage 0 (real_warmup)      — 100% real, filtered, gentle learning
    Stage 1 (real_full)        — 70% real + 30% SBBTS synthetic
    Stage 2 (real_augmented)   — 50% real + 50% SBBTS + DML jumps + FSD
    Stage 3 (real_adversarial) — 30% real + 70% SBBTS + full adversarial
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


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

    Monitors ep_rew_mean, Sharpe and profit_factor. Stops only if ALL of them
    fail to improve for `patience` evaluations.

    ``min_delta_profit_factor`` was set to -1.0 for Run 4, when profit_factor
    was computed from episode rewards and so was identically 0.0. A metric that
    never improves stalls immediately and, under the all-must-stall rule, stops
    holding training open — which is how runs 2 and 3 were cut at 18-27% of
    their stage budgets. A *negative* delta is the opposite failure: the
    comparison ``val > best + delta`` then passes almost always, the counter
    never stalls, and early stopping can never fire at all.

    Now that profit_factor is computed from realised trade returns it carries
    information again, so the threshold is a small positive number.
    """
    enabled: bool = True
    patience: int = 60
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


class VecEnvConfig(BaseModel, frozen=True):
    """Vectorized environment configuration.

    ``n_envs`` parallel copies of the env run rollouts concurrently.
    With ``kind="subproc"`` each env lives in its own Python process,
    which side-steps the GIL and scales nearly linearly with CPU cores.
    ``kind="dummy"`` keeps all envs in the main process — fine for
    debugging and for machines with a single core.
    """
    n_envs: int = 1
    kind: Literal["dummy", "subproc"] = "dummy"
    # If multi-symbol training is used and n_envs > len(symbols),
    # env slots wrap round-robin across symbols.
    start_method: Literal["spawn", "fork", "forkserver"] | None = None


class ReplayBufferConfig(BaseModel, frozen=True):
    """Off-policy replay buffer sizing.

    ``buffer_size`` directly controls RAM usage — each transition is
    roughly ``(obs_bytes + next_obs_bytes + action + reward + done)``.
    For a 3546-float32 Dict observation this is ~30 KB per sample, so
    1M transitions ≈ 30 GB, 5M ≈ 150 GB.

    ``learning_starts`` delays gradient updates until the buffer has
    enough random experience to start training usefully.
    """
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 10_000
    train_freq: int = 1           # gradient step every N rollout steps
    gradient_steps: int = 1
    tau: float = 0.005
    gamma: float = 0.99


class MultiSymbolConfig(BaseModel, frozen=True):
    """Multi-symbol training configuration.

    When ``symbols`` has more than one entry, the trainer loads each
    symbol's features independently and distributes them across
    vec-env slots. Each parallel env trains on a different pair,
    so the agent sees diverse market regimes per batch.
    """
    symbols: tuple[str, ...] = ("EURUSD",)
    timeframe: str = "H1"
    # When True the curriculum blending is applied per symbol; when
    # False only the first symbol is blended and other symbols feed
    # raw data into their env copies.
    blend_per_symbol: bool = True


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
    seed: int = 7
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
            total_timesteps=500_000,
            real_ratio=0.30,
            sbbts_ratio=0.70,
            enable_dml_jumps=True,
            enable_fsd=True,
            enable_adversarial=True,
            filter_quantile=None,
            warm_start=True,
            noise_std=0.003,
        ),
    ])

    adaptive_lr: AdaptiveLRConfig = Field(default_factory=AdaptiveLRConfig)
    early_stopping: MultiMetricEarlyStopConfig = Field(
        default_factory=MultiMetricEarlyStopConfig,
    )
    checkpointing: CheckpointStrategyConfig = Field(
        default_factory=CheckpointStrategyConfig,
    )
    vec_env: VecEnvConfig = Field(default_factory=VecEnvConfig)
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig)
    multi_symbol: MultiSymbolConfig = Field(default_factory=MultiSymbolConfig)

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


# ---------------------------------------------------------------------------
# YAML loading
#
# Run 6 was killed and restarted after it emerged that TrainerV2 built
# CurriculumV2Config() straight from the hardcoded defaults above and never
# read configs/training.yaml. Every knob tuned between runs 1-5 therefore may
# or may not have been in force, and the run logs cannot settle it.
#
# The YAML predates the v2 stage model and uses a different vocabulary: it says
# `data_source: real` and nests noise under `augmentation`, while StageConfig
# speaks in real_ratio / sbbts_ratio / enable_*. Only the overlapping keys can
# be mapped; the rest of each stage is inherited from the built-in stage of the
# same name.
#
# Anything the YAML sets that cannot be mapped is logged at WARNING rather than
# dropped in silence — a setting that looks applied but is not is the exact
# failure this loader exists to prevent.
# ---------------------------------------------------------------------------

# YAML stage key -> StageConfig field. Keys absent here are reported as ignored.
_STAGE_KEY_MAP: dict[str, str] = {
    "name": "name",
    "description": "description",
    "total_timesteps": "total_timesteps",
    "filter_quantile": "filter_quantile",
    "warm_start": "warm_start",
    "lr_override": "lr_override",
    "real_ratio": "real_ratio",
    "sbbts_ratio": "sbbts_ratio",
    "enable_dml_jumps": "enable_dml_jumps",
    "enable_fsd": "enable_fsd",
    "enable_adversarial": "enable_adversarial",
    "reward_clip": "reward_clip",
}

# YAML `<section>.<key>` -> CurriculumV2Config field.
_TOP_LEVEL_KEY_MAP: dict[str, str] = {
    "ewc.enabled": "ewc_enabled",
    "ewc.lambda_ewc": "ewc_lambda",
    "ewc.gamma_ewc": "ewc_gamma",
    "ewc.fisher_n_samples": "ewc_fisher_samples",
    "world_model.enabled": "world_model_enabled",
}


def _stage_from_yaml(
    raw: dict[str, Any],
    defaults_by_name: dict[str, StageConfig],
    ignored: list[str],
) -> StageConfig:
    """Build one StageConfig from a YAML stage entry."""
    name = raw.get("name")
    if not name:
        raise ValueError("curriculum.stages[] entry is missing a 'name'")

    # Start from the built-in stage of the same name so v2-only fields
    # (real_ratio, sbbts_ratio, enable_*) keep meaningful values.
    base = defaults_by_name.get(name, StageConfig(name=name))
    updates: dict[str, Any] = {}

    for key, value in raw.items():
        if key == "augmentation":
            if not isinstance(value, dict):
                ignored.append(f"curriculum.stages[{name}].augmentation")
                continue
            for aug_key, aug_value in value.items():
                if aug_key == "noise_std":
                    updates["noise_std"] = aug_value
                else:
                    # price_shift_std and friends have no StageConfig field.
                    ignored.append(f"curriculum.stages[{name}].augmentation.{aug_key}")
            continue

        field = _STAGE_KEY_MAP.get(key)
        if field is None:
            # `data_source` is the YAML's older way of saying real_ratio=1.0;
            # it carries no extra information, so report it like any other
            # unmapped key rather than guessing.
            ignored.append(f"curriculum.stages[{name}].{key}")
            continue
        updates[field] = value

    return base.model_copy(update=updates)


def load_curriculum_v2_config(
    config_dir: str | Path = "configs",
    *,
    strict: bool = False,
) -> CurriculumV2Config:
    """Build a :class:`CurriculumV2Config` from ``configs/training.yaml``.

    Args:
        config_dir: Directory holding ``training.yaml``.
        strict: Raise :class:`ValueError` when the YAML sets keys that cannot be
            mapped, instead of logging a warning. Useful in tests and for
            operators who want a run to abort rather than train with a setting
            that was silently discarded.

    Returns:
        The config with YAML values applied over the built-in defaults. Missing
        file or missing ``curriculum`` section yields the defaults unchanged.
    """
    # Imported here: apexfx.config.loader pulls in the app-wide schema, and
    # importing it at module scope would make this module cyclic.
    from apexfx.config.loader import load_yaml

    path = Path(config_dir) / "training.yaml"
    raw = load_yaml(path)
    if not raw:
        logger.warning("training.yaml not found — using built-in defaults", path=str(path))
        return CurriculumV2Config()

    defaults = CurriculumV2Config()
    defaults_by_name = {s.name: s for s in defaults.stages}
    ignored: list[str] = []
    updates: dict[str, Any] = {}

    curriculum = raw.get("curriculum") or {}
    raw_stages = curriculum.get("stages")
    if raw_stages:
        updates["stages"] = [
            _stage_from_yaml(entry, defaults_by_name, ignored) for entry in raw_stages
        ]
    for key in curriculum:
        if key != "stages":
            ignored.append(f"curriculum.{key}")

    for section, values in raw.items():
        if section == "curriculum" or not isinstance(values, dict):
            continue
        for key, value in values.items():
            field = _TOP_LEVEL_KEY_MAP.get(f"{section}.{key}")
            if field is not None:
                updates[field] = value

    # early_stopping / checkpointing are nested models with their own names.
    early = raw.get("early_stopping") or {}
    if early:
        es_updates: dict[str, Any] = {}
        for key, value in early.items():
            if key == "patience":
                es_updates["patience"] = value
            elif key == "min_delta":
                es_updates["min_delta_reward"] = value
            elif key == "enabled":
                es_updates["enabled"] = value
            else:
                ignored.append(f"early_stopping.{key}")
        if es_updates:
            updates["early_stopping"] = defaults.early_stopping.model_copy(update=es_updates)

    ckpt = raw.get("checkpointing") or {}
    if ckpt:
        ck_updates: dict[str, Any] = {}
        for key, value in ckpt.items():
            if key in {"save_freq", "keep_best_n"}:
                ck_updates[key] = value
            else:
                ignored.append(f"checkpointing.{key}")
        if ck_updates:
            updates["checkpointing"] = defaults.checkpointing.model_copy(update=ck_updates)

    config = defaults.model_copy(update=updates)

    if ignored:
        message = (
            "training.yaml sets keys that CurriculumV2Config cannot represent; "
            "they are NOT in effect"
        )
        if strict:
            raise ValueError(f"{message}: {sorted(ignored)}")
        logger.warning(message, keys=sorted(ignored))

    logger.info(
        "Loaded curriculum v2 config from YAML",
        path=str(path),
        n_stages=config.n_stages,
        total_timesteps=config.total_timesteps,
        ignored_keys=len(ignored),
    )
    return config
