"""Evaluation utilities: baselines, walk-forward reports, stress periods.

This module provides the missing infrastructure for *honest* validation —
the ability to measure whether the RL model has any real edge over trivial
strategies (Buy & Hold, MA crossover, Donchian breakout, Random).

Without baseline comparison, "Sharpe 0.8 OOS" is meaningless: it might mean
the market itself returned +20% (Buy & Hold would beat it) or -5% (the model
is genuinely good).  The contracts here exist to remove that ambiguity.

Usage::

    from apexfx.eval.baselines import BuyAndHoldBaseline, evaluate_on_data
    from apexfx.eval.walk_forward_report import compare_to_baselines

    # Standalone — evaluate a baseline on a price DataFrame
    bh = BuyAndHoldBaseline()
    metrics = evaluate_on_data(bh, price_df, initial_balance=100_000.0)

    # Inside walk-forward — compare model to all baselines per-fold
    table = compare_to_baselines(model_results, price_df_per_fold)
    print(table.to_markdown())
"""

from apexfx.eval.baselines import (
    BuyAndHoldBaseline,
    DonchianBaseline,
    MACrossBaseline,
    RandomBaseline,
    TradingBaseline,
    evaluate_on_data,
)
from apexfx.eval.walk_forward_report import (
    BaselineComparisonRow,
    compare_to_baselines,
    format_comparison_table,
)

__all__ = [
    "TradingBaseline",
    "BuyAndHoldBaseline",
    "MACrossBaseline",
    "DonchianBaseline",
    "RandomBaseline",
    "evaluate_on_data",
    "BaselineComparisonRow",
    "compare_to_baselines",
    "format_comparison_table",
]
