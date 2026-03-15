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


class PerformanceConfig(BaseModel):
    """Hardware-aware performance tuning."""
    n_envs: int = 1               # Parallel environments (auto-detect if 0)
    mixed_precision: bool = True   # FP16 AMP on CUDA
    torch_compile: bool = True     # torch.compile() graph optimisation
    compile_mode: str = "reduce-overhead"  # default | reduce-overhead | max-autotune
    cudnn_benchmark: bool = True
    tf32: bool = True             # Tensor Float 32 on Ampere+
    pin_memory: bool = True
    prefetch_factor: int = 2
    gradient_checkpointing: bool = False  # Trade speed for VRAM
    dataloader_workers: int = 4


class BaseConfig(BaseModel):
    seed: int = 42
    device: DeviceType = DeviceType.AUTO
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


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
    intermarket: list[str] = Field(default_factory=lambda: ["DXY", "XAUUSD", "US10Y", "SPX"])


# --- Data ---


class StorageConfig(BaseModel):
    format: str = "parquet"
    compression: str = "snappy"
    partition_by: list[str] = Field(default_factory=lambda: ["symbol", "timeframe"])


class CollectionConfig(BaseModel):
    poll_interval_ms: int = 100
    flush_interval_s: int = 60
    max_ticks_per_flush: int = 50_000


class CalendarConfig(BaseModel):
    """Economic calendar data configuration."""
    enabled: bool = True
    path: str = "data/economic_calendar.csv"
    currencies: list[str] = Field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])
    base_currency: str = "EUR"
    quote_currency: str = "USD"
    auto_fetch: bool = True
    fetch_interval_hours: int = 1
    source: str = "forex_factory"


class NewsConfig(BaseModel):
    """Real-time news configuration for fastest possible reaction.

    Priority chain: WebSocket (Finnhub, ~1s) → Fast RSS polling (30s) → Standard RSS (15m).
    """
    enabled: bool = True
    # --- Finnhub WebSocket (primary, fastest ~1s latency) ---
    finnhub_enabled: bool = True
    finnhub_api_key: str = ""  # env: FINNHUB_API_KEY
    finnhub_categories: list[str] = Field(
        default_factory=lambda: ["forex", "general", "merger"]
    )
    # --- Fast RSS polling (fallback, ~30s latency) ---
    rss_enabled: bool = True
    rss_poll_interval_s: int = 30
    rss_feeds: list[str] = Field(default_factory=lambda: [
        "https://www.forexlive.com/feed/news",
        "https://www.fxstreet.com/rss/news",
        "https://www.dailyfx.com/feeds/market-news",
    ])
    # --- Deduplication ---
    dedup_window_minutes: int = 60
    max_headlines_buffer: int = 50
    # --- Urgency detection (flash crash / breaking news) ---
    urgency_keywords: list[str] = Field(default_factory=lambda: [
        "breaking", "flash", "crash", "emergency", "surprise",
        "shock", "unexpected", "war", "attack", "default",
        "intervention", "halt", "circuit breaker",
    ])
    urgency_cooldown_s: int = 5


class DataConfig(BaseModel):
    timeframes: list[str] = Field(default_factory=lambda: ["M1", "M5", "H1", "D1"])
    lookback_bars: int = 500
    feature_window: int = 100
    tick_buffer_size: int = 100_000
    bar_history_days: int = 750
    storage: StorageConfig = Field(default_factory=StorageConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)


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
    TQC = "TQC"
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
    # TQC-specific
    top_quantiles_to_drop: int = 2
    n_quantiles: int = 25
    # Cosine LR schedule
    use_cosine_lr: bool = True


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


class UncertaintyConfig(BaseModel):
    """MC Dropout uncertainty estimation for position scaling."""
    enabled: bool = True
    n_samples: int = 10
    uncertainty_weight: float = 0.5
    min_position_scale: float = 0.1


class EnsembleDiversityConfig(BaseModel):
    """Ensemble diversity regularization to prevent agent collapse."""
    enabled: bool = True
    diversity_weight: float = 0.01
    entropy_weight: float = 0.1
    update_freq: int = 100
    batch_size: int = 256
    lr: float = 1e-4


class StructureConfig(BaseModel):
    """Structure break detection configuration."""
    swing_period: int = 5
    confluence_atr_mult: float = 1.0
    breakout_volume_mult: float = 1.5


class PositionManagementConfig(BaseModel):
    """Advanced position management configuration."""
    max_layers: int = 3
    breakeven_atr_mult: float = 1.5
    layer_size_decay: float = 0.5


class ModelConfig(BaseModel):
    tft: TFTConfig = Field(default_factory=TFTConfig)
    trend_agent: AgentConfig = Field(default_factory=AgentConfig)
    reversion_agent: ReversionAgentConfig = Field(default_factory=ReversionAgentConfig)
    gating: GatingConfig = Field(default_factory=GatingConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    mtf: MTFConfig = Field(default_factory=MTFConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    diversity: EnsembleDiversityConfig = Field(default_factory=EnsembleDiversityConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    position_management: PositionManagementConfig = Field(default_factory=PositionManagementConfig)


# --- Training ---


class SyntheticParams(BaseModel):
    """Legacy synthetic data parameters (kept for backward compatibility)."""
    n_steps: int = 100_000
    mu: float = 0.0001
    sigma: float = 0.01
    noise_std: float = 0.0
    black_swan_intensity: float = 0.0
    black_swan_magnitude_std: float = 5.0


class DataAugmentationConfig(BaseModel):
    """Noise augmentation applied to real data for robustness training."""
    noise_std: float = 0.002
    price_shift_std: float = 0.001


class CurriculumStage(BaseModel):
    name: str
    description: str
    total_timesteps: int
    data_source: Literal["real", "synthetic", "mixed"] = "real"
    synthetic_params: SyntheticParams = Field(default_factory=SyntheticParams)
    warm_start: bool = False
    # Real-data curriculum options
    filter_quantile: float | None = None
    augmentation: DataAugmentationConfig | None = None


class WalkForwardConfig(BaseModel):
    train_window_days: int = 252
    test_window_days: int = 63
    step_size_days: int = 63
    n_folds: int | None = None
    retrain_each_fold: bool = True
    auto_validate: bool = False  # Run walk-forward validation after training


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


class EWCConfig(BaseModel):
    """Elastic Weight Consolidation — prevents catastrophic forgetting between stages."""
    enabled: bool = True
    lambda_ewc: float = 5000.0
    gamma_ewc: float = 0.9
    fisher_n_samples: int = 2000
    update_freq: int = 10
    lr: float = 1e-4


class AdversarialConfig(BaseModel):
    """Adversarial training for robustness against distribution shift."""
    enabled: bool = True
    noise_std: float = 0.01
    noise_schedule: Literal["constant", "linear", "cosine"] = "cosine"
    decay_steps: int = 500_000
    warmup_steps: int = 1000
    adversarial_prob: float = 0.5
    gradient_penalty_weight: float = 0.1
    gradient_penalty_freq: int = 100


class WorldModelConfig(BaseModel):
    """Learned dynamics model for model-based planning and curiosity."""
    enabled: bool = True
    d_latent: int = 32
    d_hidden: int = 128
    n_ensemble: int = 5
    update_freq: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    curiosity_weight: float = 0.01
    imagination_horizon: int = 0


class TemporalCommitmentConfig(BaseModel):
    """Temporal commitment for trade plan persistence."""
    enabled: bool = True
    min_hold: int = 5
    commitment_penalty: float = 0.1
    action_smoothing_alpha: float = 0.3


class OnlineLearningConfig(BaseModel):
    """Adaptive online learning (retraining on recent data + live micro-updates)."""
    enabled: bool = False
    mode: Literal["fine_tune", "buffer_update"] = "fine_tune"
    retrain_window_days: int = 30
    retrain_steps: int = 10_000
    retrain_lr: float = 1e-5
    min_new_bars: int = 24     # Minimum bars before considering retrain
    validation_sharpe_min: float = 0.0  # Reject if new model is worse
    check_interval_hours: int = 1
    # Live micro-update fields
    mini_buffer_size: int = 500
    update_every_n_trades: int = 10
    gradient_steps: int = 3
    lr_scale: float = 0.1
    ewc_enabled: bool = True
    drift_detection_window: int = 50
    drift_threshold_sharpe: float = -0.5
    adaptation_gradient_steps: int = 10
    adaptation_lr_scale: float = 0.5
    max_rollback_checkpoints: int = 3


class ShadowTradingConfig(BaseModel):
    """A/B testing via shadow (paper) trading of candidate models."""
    enabled: bool = False
    evaluation_bars: int = 500
    promotion_sharpe_delta: float = 0.3  # Shadow must beat live by this delta
    gradual_rollout_bars: int = 200      # Bars for gradual 0→100% rollout


class PERConfig(BaseModel):
    """Prioritized Experience Replay configuration."""
    enabled: bool = True
    alpha: float = 0.6              # Prioritization exponent
    beta_start: float = 0.4         # Importance sampling weight start
    beta_end: float = 1.0           # IS weight end
    beta_annealing_steps: int = 0   # 0 = auto (sum of stage timesteps)
    epsilon: float = 1e-6           # Small constant for priority
    update_freq: int = 100          # Update priorities every N gradient steps


class AugmentationConfig(BaseModel):
    """Advanced time-series augmentation for training generalization."""
    enabled: bool = True
    time_warp_prob: float = 0.3
    time_warp_sigma: float = 0.2
    magnitude_warp_prob: float = 0.3
    magnitude_warp_sigma: float = 0.1
    window_slice_prob: float = 0.2
    window_slice_ratio: float = 0.9    # Keep 90% of points
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.2           # Beta distribution parameter


class TrainingConfig(BaseModel):
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    ewc: EWCConfig = Field(default_factory=EWCConfig)
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    temporal_commitment: TemporalCommitmentConfig = Field(default_factory=TemporalCommitmentConfig)
    online_learning: OnlineLearningConfig = Field(default_factory=OnlineLearningConfig)
    shadow_trading: ShadowTradingConfig = Field(default_factory=ShadowTradingConfig)
    per: PERConfig = Field(default_factory=PERConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


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


class StrategyFilterConfig(BaseModel):
    """Rule-based trade filter configuration."""
    enabled: bool = True
    news_blackout_threshold: float = 0.5
    time_to_event_threshold: float = 0.01
    min_fundamental_bias: float = 0.3
    require_structure_confirm: bool = True
    exit_on_conflict: bool = True
    reduce_scale_pre_news: float = 0.5
    pre_news_time_threshold: float = 0.1
    block_against_bias: bool = True
    min_bias_for_direction: float = 0.5


class PortfolioVaRConfig(BaseModel):
    """Portfolio-level VaR configuration for multi-asset risk."""
    multi_asset: bool = False
    daily_limit: float = 0.02  # Max portfolio VaR as fraction of equity (2%)
    correlation_lookback_days: int = 60
    use_cvar: bool = True  # Use CVaR (Expected Shortfall) instead of VaR
    default_symbol_vol: float = 0.01  # Default daily vol if no return data (1%)


class StressTestConfig(BaseModel):
    """Stress testing configuration."""
    enabled: bool = True
    run_on_startup: bool = True
    monte_carlo_sims: int = 10_000
    monte_carlo_horizon_days: int = 21
    survival_threshold_pct: float = 0.50  # Must survive with >50% equity


class RiskConfig(BaseModel):
    daily_var_limit: float = 0.02
    max_drawdown_pct: float = 0.05
    var_confidence: float = 0.99
    var_lookback_days: int = 252
    var_method: Literal["historical", "parametric"] = "historical"
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)
    strategy_filter: StrategyFilterConfig = Field(default_factory=StrategyFilterConfig)
    portfolio_var: PortfolioVaRConfig = Field(default_factory=PortfolioVaRConfig)
    stress_test: StressTestConfig = Field(default_factory=StressTestConfig)


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


class SmartExecutionConfig(BaseModel):
    """Smart order execution algorithm configuration."""
    algorithm: Literal["direct", "twap", "vwap", "is", "auto"] = "auto"
    urgency: float = 0.5  # 0=patient, 1=aggressive
    twap_threshold_lots: float = 0.5  # Orders < this use direct market
    vwap_threshold_lots: float = 2.0  # Orders < this use TWAP
    is_threshold_lots: float = 5.0    # Orders > vwap threshold use IS
    max_deviation_pct: float = 0.005  # Max price deviation before abort


class ExecutionConfig(BaseModel):
    order_type: Literal["market", "limit"] = "limit"
    limit_offset_pips: float = 0.5
    limit_timeout_s: int = 30
    limit_fallback: Literal["market", "cancel"] = "market"
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    session_filter: SessionFilterConfig = Field(default_factory=SessionFilterConfig)
    news_blackout: NewsBlackoutConfig = Field(default_factory=NewsBlackoutConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    smart_execution: SmartExecutionConfig = Field(default_factory=SmartExecutionConfig)


class PortfolioConfig(BaseModel):
    """Multi-symbol portfolio trading configuration."""
    enabled: bool = False
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    max_total_exposure: float = 0.40     # Max % of equity in all positions
    max_per_symbol: float = 0.25         # Max % of equity per symbol
    correlation_limit: float = 0.70      # Block trades with corr > this
    rebalance_freq_bars: int = 24


# --- Dashboard ---


class AlertConfig(BaseModel):
    """Alert/notification configuration for Telegram, Webhook, etc."""
    enabled: bool = False  # Off by default — requires setup
    min_level: str = "WARNING"  # INFO, WARNING, CRITICAL, EMERGENCY
    cooldown_s: float = 60.0
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""  # env: TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = ""    # env: TELEGRAM_CHAT_ID
    # Webhook (Slack, Discord, PagerDuty, etc.)
    webhook_enabled: bool = False
    webhook_url: str = ""         # env: ALERT_WEBHOOK_URL
    webhook_format: str = "json"  # json, slack, discord


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
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
