# ApexFX Quantum

Hedge-fund grade algorithmic Forex trading system combining Temporal Fusion Transformers,
Reinforcement Learning with continuous action space, and ensemble of specialized agents.

## Architecture

- **Quantum Feature Engineering**: KDE Volume Profile, Hurst Exponent, Wavelet Decomposition, Order Flow Analysis
- **The Hive Mind**: TFT Encoder + Trend Agent + Reversion Agent + Meta-Controller Gating Network
- **RL Training**: SAC/PPO with Differential Sharpe Ratio reward, 3-stage curriculum learning
- **Hard Risk Management**: VaR limits, drawdown monitoring, cooldown system, Kelly position sizing
- **Execution Layer**: Liquidity guard, limit orders with fallback, fill quality tracking

## Quick Start

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install with dev and dashboard dependencies
pip install -e ".[dev,dashboard]"

# Train on synthetic data (no MT5 required)
python scripts/train.py --synthetic-only

# Run tests
pytest tests/ -v

# Launch dashboard
python scripts/launch_dashboard.py
```

## Project Structure

```
src/apexfx/
  config/       # Pydantic schemas + YAML loader
  data/         # MT5 client, Parquet storage, synthetic data generator
  features/     # 7 feature extractors + normalizer + pipeline
  models/       # TFT, RL agents, ensemble gating
  env/          # Gymnasium environment + reward functions
  training/     # SB3 integration, curriculum, walk-forward, hyperopt
  risk/         # VaR, drawdown, cooldown, position sizing
  execution/    # Liquidity guard, order management, fill tracking
  live/         # Async trading loop, signal generator, health checks
  dashboard/    # Plotly Dash monitoring UI
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- MetaTrader 5 (optional, Windows only — for live trading)

## License

MIT
