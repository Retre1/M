"""Pydantic configuration models — single source of truth for all config types."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class PathsConfig(BaseModel):
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["json", "console"] = "json"
    file: str = "./logs/apexfx.log"
    rotation_mb: int = 50
    backup_count: int = 5


class BaseConfig(BaseModel):
    seed: int = 42
    device: DeviceType = DeviceType.AUTO
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# --- Symbols ---


class SessionHours(BaseModel):
    tokyo: tuple[int, int] = (0, 9)
    london: tuple[int, int] = (7, 16)
    new_york: tuple[int, int] = (12, 21)
    overlap_ln: tuple[int, int] = (12, 16)


class SymbolConfig(BaseModel):
    pip_value: float
    spread_limit_pips: float
    lot_step: float = 0.01
    contract_size: int = 100_000
    sessions: SessionHours = Field(default_factory=SessionHours)


class SymbolsConfig(BaseModel):
    symbols: dict[str, SymbolConfig]
    intermarket: list[str] = Field(default_factory=lambda: ["DXY", "XAUUSD", "US10Y"])


# --- Data ---


class StorageConfig(BaseModel):
    format: str = "parquet"
    compression: str = "snappy"
    partition_by: list[str] = Field(default_factory=lambda: ["symbol", "timeframe"])


class CollectionConfig(BaseModel):
    poll_interval_ms: int = 100
    flush_interval_s: int = 60
    max_ticks_per_flush: int = 50_000


class DataConfig(BaseModel):
    timeframes: list[str] = Field(default_factory=lambda: ["M1", "M5", "H1", "D1"])
    lookback_bars: int = 500
    feature_window: int = 100
    tick_buffer_size: int = 100_000
    bar_history_days: int = 750
    storage: StorageConfig = Field(default_factory=StorageConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)


# --- Model ---


class TFTConfig(BaseModel):
    d_model: int = 64
    n_heads: int = 4
    n_encoder_layers: int = 3
    n_decoder_layers: int = 1
    dropout: float = 0.1
    hidden_continuous_size: int = 16
    attention_head_size: int = 16
    known_future_inputs: list[str] = Field(
        default_factory=lambda: ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_id"]
    )


class AgentConfig(BaseModel):
    hidden_sizes: list[int] = Field(default_factory=lambda: [128, 128, 64])
    activation: str = "relu"
    dropout: float = 0.1


class ReversionAgentConfig(AgentConfig):
    z_score_threshold: float = 3.0


class GatingConfig(BaseModel):
    hidden_sizes: list[int] = Field(default_factory=lambda: [64, 32])
    n_agents: int = 3  # trend + reversion + breakout


class RLAlgorithm(str, Enum):
    SAC = "SAC"
    PPO = "PPO"


class RLConfig(BaseModel):
    algorithm: RLAlgorithm = RLAlgorithm.SAC
    gamma: float = 0.99
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    ent_coef: str | float = "auto"
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 10_000


class MTFLookbackConfig(BaseModel):
    d1: int = 5
    h1: int = 20
    m5: int = 20


class MTFFusionConfig(BaseModel):
    dropout: float = 0.1


class MTFConfig(BaseModel):
    enabled: bool = False
    lookback: MTFLookbackConfig = Field(default_factory=MTFLookbackConfig)
    fusion: MTFFusionConfig = Field(default_factory=MTFFusionConfig)
    n_mtf_context: int = 6


class ModelConfig(BaseModel):
    tft: TFTConfig = Field(default_factory=TFTConfig)
    trend_agent: AgentConfig = Field(default_factory=AgentConfig)
    reversion_agent: ReversionAgentConfig = Field(default_factory=ReversionAgentConfig)
    gating: GatingConfig = Field(default_factory=GatingConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    mtf: MTFConfig = Field(default_factory=MTFConfig)


# --- Training ---


class SyntheticParams(BaseModel):
    n_steps: int = 100_000
    mu: float = 0.0001
    sigma: float = 0.01
    noise_std: float = 0.0
    black_swan_intensity: float = 0.0
    black_swan_magnitude_std: float = 5.0


class CurriculumStage(BaseModel):
    name: str
    description: str
    total_timesteps: int
    data_source: Literal["synthetic", "mixed", "real"]
    synthetic_params: SyntheticParams = Field(default_factory=SyntheticParams)
    warm_start: bool = False


class WalkForwardConfig(BaseModel):
    train_window_days: int = 252
    test_window_days: int = 63
    step_size_days: int = 63
    n_folds: int | None = None
    retrain_each_fold: bool = True


class EvaluationConfig(BaseModel):
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    deterministic: bool = True


class CheckpointConfig(BaseModel):
    save_freq: int = 50_000
    keep_best_n: int = 3


class EarlyStoppingConfig(BaseModel):
    patience: int = 10
    min_delta: float = 0.01


class CurriculumConfig(BaseModel):
    stages: list[CurriculumStage] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


# --- Risk ---


class PositionSizingConfig(BaseModel):
    max_position_pct: float = 0.10
    kelly_fraction: float = 0.5
    min_trades_for_kelly: int = 30
    vol_lookback_bars: int = 20
    min_lot_size: float = 0.01


class CooldownConfig(BaseModel):
    after_n_losses: int = 3
    duration_minutes: int = 60
    tilt_drawdown_pct: float = 0.02
    tilt_window_minutes: int = 30


class HedgingConfig(BaseModel):
    enabled: bool = True
    hedge_ratio: float = 0.5
    hedge_instrument: str | None = None


class RiskConfig(BaseModel):
    daily_var_limit: float = 0.02
    max_drawdown_pct: float = 0.05
    var_confidence: float = 0.99
    var_lookback_days: int = 252
    var_method: Literal["historical", "parametric"] = "historical"
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)


# --- Execution ---


class SlippageConfig(BaseModel):
    max_slippage_pips: float = 1.0
    track_history: bool = True


class SessionFilterConfig(BaseModel):
    enabled: bool = True
    allowed_sessions: list[str] = Field(
        default_factory=lambda: ["london", "new_york", "overlap_ln"]
    )


class NewsBlackoutConfig(BaseModel):
    enabled: bool = True
    blackout_minutes_before: int = 15
    blackout_minutes_after: int = 10
    calendar_source: Literal["config", "external"] = "config"


class RetryConfig(BaseModel):
    max_retries: int = 3
    backoff_base_ms: int = 500
    backoff_max_ms: int = 5000


class ExecutionConfig(BaseModel):
    order_type: Literal["market", "limit"] = "limit"
    limit_offset_pips: float = 0.5
    limit_timeout_s: int = 30
    limit_fallback: Literal["market", "cancel"] = "market"
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    session_filter: SessionFilterConfig = Field(default_factory=SessionFilterConfig)
    news_blackout: NewsBlackoutConfig = Field(default_factory=NewsBlackoutConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)


# --- Dashboard ---


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False
    refresh_interval_s: int = 5
    pages: list[str] = Field(
        default_factory=lambda: ["overview", "signals", "risk", "training"]
    )
    theme: str = "darkly"


# --- Master Config ---


class AppConfig(BaseModel):
    """Master configuration combining all sections."""

    base: BaseConfig = Field(default_factory=BaseConfig)
    symbols: SymbolsConfig = Field(
        default_factory=lambda: SymbolsConfig(symbols={})
    )
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
