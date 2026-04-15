"""Data ingestion, processing, and synthetic generation.

Core modules:
    data_store          — Parquet storage layer (read/write bars, ticks, features)
    pipeline            — BiasFreePipeline (LIB/LAB correction, FSD, SBBTS orchestration)
    features            — FSDCalculator (Forex Skewness Dispersion regime detector)
    sbbts_generator     — SBBTSGenerator (Schrödinger–Bass synthetic trajectories)
    config              — Pydantic v2 configuration models for the data layer

Legacy / supporting:
    bar_aggregator      — OHLCV bar construction from ticks
    synthetic           — Legacy GBM/GARCH/regime-switching synthetic data
    mtf_synthetic       — Multi-timeframe resampling (H1 → D1/M5)
    mt5_client          — MetaTrader 5 live data connection
"""

__all__ = [
    # v2.0 bias-free pipeline
    "config",
    "pipeline",
    "features",
    "sbbts_generator",
    # Storage
    "data_store",
    # Legacy / supporting
    "bar_aggregator",
    "calendar_fetcher",
    "calendar_provider",
    "cot_fetcher",
    "intermarket",
    "mt5_client",
    "mtf_aligner",
    "mtf_synthetic",
    "news_fetcher",
    "realtime_news",
    "synthetic",
    "tick_collector",
    "volume_bar_aggregator",
]
