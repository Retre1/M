"""Backtesting engine for ApexFX Quantum."""

from apexfx.backtest.baselines import (
    BuyAndHold,
    DonchianBreakout,
    MACross,
    RandomStrategy,
    default_baselines,
)
from apexfx.backtest.comparison import (
    ComparisonResult,
    StrategyScore,
    compare_against_baselines,
)
from apexfx.backtest.engine import BacktestEngine
from apexfx.backtest.result import BacktestResult

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BuyAndHold",
    "ComparisonResult",
    "DonchianBreakout",
    "MACross",
    "RandomStrategy",
    "StrategyScore",
    "compare_against_baselines",
    "default_baselines",
]
