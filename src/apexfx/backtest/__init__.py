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
    FoldComparison,
    StrategyScore,
    compare_across_folds,
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
    "FoldComparison",
    "MACross",
    "RandomStrategy",
    "StrategyScore",
    "compare_across_folds",
    "compare_against_baselines",
    "default_baselines",
]
