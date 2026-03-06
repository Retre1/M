<p align="center">
  <h1 align="center">ApexFX Quantum</h1>
  <p align="center">
    <strong>Hedge-fund grade algorithmic Forex trading system</strong>
  </p>
  <p align="center">
    Temporal Fusion Transformer + Multi-Agent RL + Professional Trading Intelligence
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/pytorch-2.0+-red.svg" alt="PyTorch 2.0+">
    <img src="https://img.shields.io/badge/tests-144%20passed-brightgreen.svg" alt="Tests">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
</p>

---

## Overview

ApexFX Quantum is a professional-grade algorithmic trading system for the Forex market. It combines deep learning (Temporal Fusion Transformers), reinforcement learning (SAC/PPO with continuous action space), and an ensemble of specialized trading agents — all governed by institutional-level risk management and rule-based trade filters.

The system implements a complete trading pipeline: from raw tick data collection through feature engineering, model inference, risk evaluation, and order execution on MetaTrader 5.

### Key Principles

- **The model LEARNS the strategy** — fundamental and structural features teach the RL agent when and how to trade, rather than hardcoding rules
- **Non-negotiable safety layer** — rule-based filters and risk management cannot be overridden by the model
- **Institutional risk management** — VaR limits, drawdown monitoring, kill switches, daily loss guards, and weekend gap protection

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           DATA LAYER                     │
                        │  MT5 Ticks → Bar Aggregator → Parquet    │
                        │  Forex Factory → Calendar Provider       │
                        └──────────────────┬──────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────┐
                        │        FEATURE ENGINEERING               │
                        │                                          │
                        │  KDE Volume Profile    Hurst Exponent    │
                        │  Wavelet Decomposition Order Flow        │
                        │  Regime Detection      Spectral Analysis │
                        │  Fundamental (8 feat)  Structure (8 feat)│
                        └──────────────────┬──────────────────────┘
                                           │
          ┌────────────────────────────────▼────────────────────────────────┐
          │                     THE HIVE MIND                               │
          │                                                                 │
          │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
          │  │ Trend Agent  │  │Reversion Agent│  │  Breakout Agent      │  │
          │  │              │  │              │  │  + Structure Features │  │
          │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
          │         │                 │                      │              │
          │         └─────────────────┼──────────────────────┘              │
          │                           │                                     │
          │              ┌────────────▼────────────┐                       │
          │              │    Gating Network        │                       │
          │              │  + Fundamental Features  │                       │
          │              │    (regime-aware routing) │                       │
          │              └────────────┬────────────┘                       │
          │                           │                                     │
          │              ┌────────────▼────────────┐                       │
          │              │   TFT Encoder (d=64)    │                       │
          │              │   Multi-Head Attention   │                       │
          │              └────────────┬────────────┘                       │
          └───────────────────────────┼────────────────────────────────────┘
                                      │
          ┌───────────────────────────▼────────────────────────────────────┐
          │                   SAFETY LAYER                                  │
          │                                                                 │
          │  Strategy Filter ──► Risk Manager ──► Position Sizer            │
          │  (6 hard rules)      (10 checks)      (Kelly + Vol targeting)   │
          └───────────────────────────┬────────────────────────────────────┘
                                      │
          ┌───────────────────────────▼────────────────────────────────────┐
          │                   EXECUTION                                     │
          │                                                                 │
          │  Limit Orders → Slippage Model → Fill Tracking → MT5           │
          │  Break-even Stops → Position Layers (Pyramiding)                │
          └────────────────────────────────────────────────────────────────┘
```

---

## Features

### Multi-Agent Ensemble (The Hive Mind)

| Agent | Role | Input Features |
|-------|------|---------------|
| **Trend Agent** | Follows momentum and trends | Price features, regime |
| **Reversion Agent** | Mean-reversion with z-score gating | Statistical features |
| **Breakout Agent** | Structure break entries | Breakout + **Structure features (18d)** |
| **Gating Network** | Meta-controller routing | Regime + **Fundamental features (14d)** |

### Feature Engineering (32+ features)

| Extractor | Features | Description |
|-----------|----------|-------------|
| Volume Profile | KDE-based | Point of Control, Value Area High/Low |
| Hurst Exponent | R/S analysis | Trending vs mean-reverting detection |
| Wavelet | PyWavelets | Multi-scale decomposition of price |
| Regime | Hidden Markov | Trending / mean-reverting / volatile / flat |
| Spectral | FFT | Dominant cycles in price data |
| Order Flow | Tick analysis | Buy/sell imbalance, aggression |
| **Fundamental** | 8 features | News surprise, hawkish/dovish, bias, rate differential, conflicting signals |
| **Structure** | 8 features | Swing highs/lows, BOS (Break of Structure), confluence, retest |

### Professional Trading Intelligence

The system implements an institutional trading methodology:

**Fundamental Analysis:**
- Economic calendar integration (NFP, CPI, FOMC, ECB decisions)
- Normalized surprise scores per event type
- Hawkish/dovish sentiment scoring
- Rate differential tracking
- Conflicting signals detection

**Technical Confirmation:**
- Multi-timeframe analysis (D1 → H1 → M5)
- Swing high/low detection (fractal method)
- Structure Break (BOS) confirmation for entries
- Support/resistance confluence scoring
- Retest pattern detection

**Position Management:**
- Break-even stops (auto-lock at entry after N×ATR profit)
- Pyramiding (up to 3 layers with size decay)
- Weighted average entry price tracking

### Rule-Based Trade Filters

Non-negotiable rules enforced over the RL agent:

| Rule | Condition | Action |
|------|-----------|--------|
| News Blackout | High-impact event active | Block new entries |
| Conflicting Exit | Opposing fundamental signals | Force close position |
| Minimum Bias | \|fundamental_bias\| < 0.3 | Block entry |
| Structure Confirm | No BOS detected | Block entry |
| Direction Alignment | Action against fundamental bias | Block entry |
| Pre-News Scaling | Event approaching (~2.4h) | Reduce position 50% |

### Risk Management (10 checks)

```
Kill Switch → Weekend Guard → Daily Loss Guard → Strategy Filter →
Cooldown → Drawdown → Spread Check → VaR Limit → Vol Targeting →
Regime Scaling → Uncertainty Scaling → Position Sizing
```

- **Daily loss limit** (default 2%) — hard stop on daily losses
- **Kill switch** — emergency halt (auto + manual via file)
- **VaR scaling** — position sized to stay within VaR limits
- **Volatility targeting** — dynamic leverage to target portfolio vol
- **Weekend gap guard** — reduces/blocks trades before market close
- **Regime-adaptive risk** — different risk profiles per market regime

### Training Pipeline

- **SAC / TQC / PPO** with continuous action space
- **Curriculum learning** — 3-stage progression from simple to complex
- **Walk-forward validation** — rolling train/test windows
- **EWC** (Elastic Weight Consolidation) — prevents catastrophic forgetting
- **Adversarial training** — noise injection for robustness
- **World Model** — learned dynamics for curiosity-driven exploration
- **PER** (Prioritized Experience Replay) with SumTree

### Reward Function

Multi-component reward with professional trading incentives:

| Component | Description |
|-----------|-------------|
| Differential Sharpe | Risk-adjusted return metric |
| Asymmetric loss penalty | Losses penalized 2x vs gains |
| CVaR tail penalty | Penalizes extreme drawdowns |
| Hold winner bonus | Reward for holding profitable trades |
| Quick cut bonus | Reward for cutting losers fast |
| News trade penalty | Penalty for entries during news |
| Structure confirm bonus | Bonus for BOS-aligned entries |
| Churn penalty | Penalizes excessive position flipping |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/apexfx-quantum.git
cd apexfx-quantum

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with all dependencies
pip install -e ".[dev,dashboard]"

# Run tests to verify installation
pytest tests/ -v
```

### Download Economic Calendar

```bash
# Download 5 years of historical calendar data
python scripts/download_calendar.py --start 2020-01-01 --end 2024-12-31

# Update existing CSV with latest data
python scripts/download_calendar.py --update
```

### Training

```bash
# Train on synthetic data (no MT5 required)
python scripts/train.py --synthetic-only

# Train with curriculum learning
python scripts/train.py --config configs/training.yaml

# Hyperparameter optimization
python scripts/hyperopt_sac.py --n-trials 100
```

### Backtesting

```bash
# Quick backtest with default model
python scripts/quick_backtest.py --symbol EURUSD

# Full walk-forward backtest
python scripts/backtest.py --config configs/training.yaml
```

### Live Trading

```bash
# Start live trading (requires MetaTrader 5)
python scripts/live_trade.py --symbol EURUSD

# Launch monitoring dashboard
python scripts/launch_dashboard.py
```

---

## Project Structure

```
apexfx-quantum/
├── configs/                    # YAML configuration files
│   ├── base.yaml               #   Hardware, logging, paths
│   ├── data.yaml               #   Timeframes, storage, calendar
│   ├── model.yaml              #   TFT, agents, gating, MTF
│   ├── training.yaml           #   Curriculum, EWC, adversarial
│   ├── risk.yaml               #   VaR, drawdown, position sizing
│   ├── execution.yaml          #   Orders, slippage, strategy filter
│   ├── symbols.yaml            #   Pair configs (pip value, sessions)
│   └── dashboard.yaml          #   Monitoring UI settings
│
├── src/apexfx/
│   ├── config/                 # Pydantic schemas + YAML loader
│   │   └── schema.py           #   50+ config models
│   │
│   ├── data/                   # Data pipeline
│   │   ├── mt5_client.py       #   MetaTrader 5 connection
│   │   ├── data_store.py       #   Parquet storage (partitioned)
│   │   ├── tick_collector.py   #   Real-time tick collection
│   │   ├── bar_aggregator.py   #   Tick → OHLCV aggregation
│   │   ├── calendar_provider.py #  Economic calendar + surprise scores
│   │   ├── calendar_fetcher.py #   Forex Factory scraper (XML + HTML)
│   │   └── mtf_aligner.py      #   Multi-timeframe data alignment
│   │
│   ├── features/               # Feature engineering
│   │   ├── pipeline.py         #   Feature pipeline orchestrator
│   │   ├── fundamental.py      #   8 fundamental features
│   │   ├── structure.py        #   8 structure/level features
│   │   ├── regime.py           #   Market regime detection
│   │   ├── volume_profile.py   #   KDE volume profile
│   │   ├── hurst.py            #   Hurst exponent (R/S)
│   │   ├── spectral.py         #   FFT spectral analysis
│   │   └── ...                 #   7+ more extractors
│   │
│   ├── models/
│   │   ├── tft/                #   Temporal Fusion Transformer
│   │   ├── agents/             #   Trend / Reversion / Breakout
│   │   ├── ensemble/           #   HiveMind + GatingNetwork
│   │   │   ├── hive_mind.py    #     Multi-agent ensemble
│   │   │   └── gating.py       #     Regime-aware meta-controller
│   │   └── components/         #   GRN, Attention, custom layers
│   │
│   ├── env/                    # Trading environment
│   │   ├── forex_env.py        #   Gymnasium env (spread, slippage, stops)
│   │   ├── obs_builder.py      #   Observation space construction
│   │   ├── reward.py           #   Multi-component reward function
│   │   ├── trade_filter.py     #   StrategyFilter (6 hard rules)
│   │   └── wrappers.py         #   TradeFilterWrapper, NormalizeReward
│   │
│   ├── training/               # Training pipeline
│   │   ├── trainer.py          #   SB3 integration
│   │   ├── curriculum.py       #   3-stage curriculum learning
│   │   ├── ewc.py              #   Elastic Weight Consolidation
│   │   ├── adversarial.py      #   Adversarial noise training
│   │   ├── per.py              #   Prioritized Experience Replay
│   │   └── world_model.py      #   Learned dynamics model
│   │
│   ├── risk/                   # Risk management
│   │   ├── risk_manager.py     #   Central risk gate (10 checks)
│   │   ├── position_sizer.py   #   Kelly criterion + vol scaling
│   │   ├── var_calculator.py   #   Value-at-Risk (historical/parametric)
│   │   ├── drawdown_monitor.py #   Real-time drawdown tracking
│   │   ├── news_filter.py      #   News blackout enforcement
│   │   └── cooldown.py         #   Loss-streak cooldown
│   │
│   ├── execution/              # Order execution
│   │   └── executor.py         #   MT5 order management
│   │
│   ├── live/                   # Live trading
│   │   ├── trading_loop.py     #   Async event loop + calendar updates
│   │   ├── signal_generator.py #   Model inference wrapper
│   │   ├── state_manager.py    #   Portfolio state persistence
│   │   └── health_check.py     #   System health monitoring
│   │
│   └── dashboard/              # Monitoring UI
│       └── app.py              #   Plotly Dash application
│
├── scripts/                    # CLI entry points
│   ├── train.py                #   Training launcher
│   ├── live_trade.py           #   Live trading launcher
│   ├── backtest.py             #   Walk-forward backtesting
│   ├── download_calendar.py    #   Economic calendar downloader
│   ├── hyperopt_sac.py         #   Hyperparameter optimization
│   └── launch_dashboard.py     #   Dashboard launcher
│
├── tests/                      # Test suite (144 tests)
│   └── unit/
│       ├── test_env/           #   Reward function tests
│       ├── test_features/      #   Feature extractor tests
│       ├── test_risk/          #   Position sizer, VaR tests
│       ├── test_phase2.py      #   Gating, PER, MTF tests
│       ├── test_phase3.py      #   Fundamental, structure, HiveMind tests
│       └── test_phase3_5.py    #   Calendar fetcher, trade filter tests
│
└── pyproject.toml              # Package configuration
```

---

## Configuration

All configuration is managed through YAML files in `configs/` with Pydantic validation.

### Key Config Sections

```yaml
# configs/data.yaml
calendar:
  enabled: true
  auto_fetch: true              # Auto-download from Forex Factory
  source: forex_factory
  currencies: [USD, EUR, GBP, JPY]

# configs/execution.yaml
strategy_filter:
  enabled: true
  min_fundamental_bias: 0.3     # Minimum conviction for entry
  require_structure_confirm: true # Require BOS for entry
  exit_on_conflict: true        # Force close on conflicting signals
  block_against_bias: true      # Don't trade against fundamentals

# configs/risk.yaml (via schema)
risk:
  daily_var_limit: 0.02         # 2% daily VaR limit
  max_drawdown_pct: 0.05        # 5% max drawdown
  var_confidence: 0.99          # 99% VaR confidence
```

---

## Data Flow

```
Forex Factory XML ─► CalendarFetcher ─► CalendarProvider ─► FundamentalExtractor ─┐
                                                                                    │
MT5 Ticks ─► TickCollector ─► BarAggregator ─► FeaturePipeline ──────────────────┤
                                                    │                              │
                                          StructureExtractor ─────────────────────┤
                                          RegimeDetector ─────────────────────────┤
                                          VolumeProfile ──────────────────────────┤
                                          HurstExponent ──────────────────────────┘
                                                                                    │
                                                                              ObsBuilder
                                                                                    │
                                                                              HiveMind
                                                                            (3 agents)
                                                                                    │
                                                                          StrategyFilter
                                                                          (6 hard rules)
                                                                                    │
                                                                           RiskManager
                                                                           (10 checks)
                                                                                    │
                                                                            Executor
                                                                              (MT5)
```

---

## Requirements

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | >= 3.11 | Runtime |
| PyTorch | >= 2.0 | Deep learning |
| Gymnasium | >= 0.29 | RL environment |
| Stable-Baselines3 | >= 2.0 | RL algorithms |
| Pandas | >= 2.0 | Data processing |
| Pydantic | >= 2.0 | Config validation |
| Requests | >= 2.31 | Calendar API |
| BeautifulSoup4 | >= 4.12 | HTML parsing |
| MetaTrader5 | >= 5.0.45 | Live trading (optional, Windows) |

---

## Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific phase tests
pytest tests/unit/test_phase3.py -v      # Fundamental + Structure
pytest tests/unit/test_phase3_5.py -v    # Calendar + Trade Filters

# Run with coverage
pytest tests/ --cov=src/apexfx --cov-report=html
```

**Current: 144/144 tests passing**

---

## CLI Commands

```bash
# Training
apexfx-train                             # Start training
python scripts/hyperopt_sac.py           # Hyperparameter search

# Data
apexfx-collect                           # Collect live tick data
python scripts/download_calendar.py      # Download economic calendar
python scripts/download_mt5_data.py      # Download historical bars

# Trading
apexfx-live                              # Start live trading
apexfx-backtest                          # Run backtest

# Monitoring
apexfx-dashboard                         # Launch Dash UI
```

---

## License

MIT

---

<p align="center">
  <sub>Built for precision. Engineered for performance.</sub>
</p>
