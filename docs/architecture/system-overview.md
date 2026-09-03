# System Architecture

> Формальное описание архитектуры для ТЗ (с таблицей соответствия реализации):
> [[tz-architecture|ТЗ — Архитектура программного комплекса]].

## Data Flow

```
MT5 / CSV / Parquet
       │
       ▼
  DataStore.read_bars()        ← src/apexfx/data/data_store.py
       │
       ▼
  FeaturePipeline.compute()    ← src/apexfx/features/pipeline.py
  (99 raw features)
       │
       ▼
  FeatureSelector              ← feature_selector.json (15 features)
       │
       ▼
  MTFForexTradingEnv           ← src/apexfx/env/mtf_forex_env.py
  (D1 + H1 + M5 obs)
       │
       ▼
  TFT Encoder → MARL Agents → Gating → Action
       │
       ▼
  ProfitFocusedReward          ← src/apexfx/env/reward.py
       │
       ▼
  TQC (sb3_contrib)            ← src/apexfx/training/trainer.py
```

## Directory Structure

```
src/apexfx/
├── config/         # Pydantic schemas, yaml loading
│   ├── schema.py   # All config dataclasses
│   └── registry.py # Config init + overlay
├── data/           # Data ingestion & processing
│   ├── data_store.py      # Read/write bars (parquet/csv)
│   ├── mtf_synthetic.py   # Resample H1 → D1/M5
│   ├── fsd_regime.py      # [v2] FSD regime detection
│   └── sbbts_generator.py # [v2] Synthetic data generation
├── env/            # Gymnasium environments
│   ├── forex_env.py       # Single-TF env
│   ├── mtf_forex_env.py   # Multi-TF env (D1+H1+M5)
│   ├── obs_builder.py     # Observation construction
│   ├── reward.py          # ProfitFocusedReward v4
│   └── reward_v5.py       # [v2] QuantumHybridReward
├── features/       # Feature engineering
│   └── pipeline.py        # 99 features from extractors
├── models/         # Neural network components
│   ├── tft/               # Temporal Fusion Transformer
│   ├── agents/            # Trend/Reversion/Breakout heads
│   ├── ensemble/          # Agent gating
│   ├── dml_network.py     # [v2] Differential ML
│   └── quantum_kernel.py  # [v2] QAOA/QAE/CVaR
├── training/       # Training loop & curriculum
│   ├── trainer.py         # Main trainer (curriculum stages)
│   ├── walk_forward.py    # Walk-forward validation
│   └── callbacks.py       # Checkpoint, metrics callbacks
├── risk/           # Position sizing, risk management
├── execution/      # Order execution
├── live/           # Live trading
└── utils/          # Logging, metrics, helpers
```

## Key Configs

| File | Purpose |
|------|---------|
| `configs/base.yaml` | Paths, logging, symbols |
| `configs/training.yaml` | Curriculum, EWC, adversarial, world model |
| `configs/gpu1.yaml` | RTX 4090 overlay + v2.0 module flags |

## Observation Space (MTF)

```
obs = flatten([
    d1_market_features,    # (15, 15) = 225
    h1_market_features,    # (40, 15) = 600
    m5_market_features,    # (30, 15) = 450
])
Total flat obs: 1275
```

Note: first 15 columns of market features (positional, from obs_builder.py).

#architecture #system
