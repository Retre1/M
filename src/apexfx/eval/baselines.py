"""Trivial trading baselines for *honest* edge validation.

Why this module exists
----------------------
A model that produces ``Sharpe = 0.8`` on out-of-sample data tells you
nothing about whether the model has *edge*.  The same Sharpe would be
produced by simply buying and holding EURUSD during a trend.  To know
whether the RL agent learned anything beyond the market's natural drift,
you must measure four things on the *same* data:

1. **Buy & Hold** — take whatever direction is in the data.  Beats every
   trend system in a clean uptrend (see AUDIT_REPORT.md Part 4).
2. **MA crossover** — the simplest mechanical trend system in existence.
3. **Donchian breakout** — the simplest mechanical breakout system.
4. **Random** — sanity check that costs are realistic.  If random returns
   roughly zero (not strongly negative) your cost model is too generous.

If your RL model can't beat the *best* of these on a majority of walk-forward
folds, it has not demonstrated edge — period.  This module provides the four
baselines and a single helper, ``evaluate_on_data``, that runs each on a
price DataFrame using the same execution mechanics (spread cost on every
direction change) as the production env.

Design notes
------------
* Baselines are intentionally **price-only** — they do *not* use the rich
  feature observation that the env builds.  This is deliberate: a baseline's
  purpose is to ask "could a stupid trader have done this?"
* They consume the same OHLC bars as the trained model, with the same costs.
* They never look ahead: ``predict_action`` at bar ``i`` only sees prices
  up to and including bar ``i`` (the close).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineExecConfig:
    """Execution parameters shared between all baselines.

    The defaults mirror retail MT5 EURUSD reality (2.0 pip spread is the
    median across major retail brokers per audit research; 0.0001 pip value
    is the EURUSD pip).  Override per-call for backtesting other instruments.
    """

    initial_balance: float = 100_000.0
    transaction_cost_pips: float = 2.0  # retail-realistic, not 1.5 idealised
    pip_value: float = 0.0001
    position_pct: float = 1.0  # 1.0 = use full balance per trade


# ---------------------------------------------------------------------------
# Baseline protocol
# ---------------------------------------------------------------------------


class TradingBaseline(ABC):
    """Stateless-or-stateful trivial trader.

    A baseline maps an OHLC price history (up to and including the current
    bar) to a desired position in ``[-1, 1]``:

    * ``+1.0`` → fully long
    * ``0.0``  → flat
    * ``-1.0`` → fully short

    The ``evaluate_on_data`` runner translates these target positions into
    a portfolio equity curve, applying spread costs on every direction
    change (entry/exit/flip).
    """

    name: str = "Baseline"

    def reset(self) -> None:
        """Override if the baseline maintains internal state."""

    @abstractmethod
    def predict_action(
        self,
        close_history: np.ndarray,
        high_history: np.ndarray,
        low_history: np.ndarray,
    ) -> float:
        """Return target position in ``[-1, 1]`` given history up to ``t``.

        Histories are 1-D arrays where the last element is the current bar's
        close / high / low respectively.  No look-ahead allowed.
        """


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------


class BuyAndHoldBaseline(TradingBaseline):
    """Always-long. The toughest baseline in any uptrend market."""

    name = "B&H"

    def predict_action(
        self,
        close_history: np.ndarray,
        high_history: np.ndarray,
        low_history: np.ndarray,
    ) -> float:
        return 1.0


class MACrossBaseline(TradingBaseline):
    """Long when ``fast`` MA > ``slow`` MA, short otherwise.

    Defaults (20, 50) are the canonical "golden cross" parameters.  When
    the fast/slow line haven't been built up yet (``< slow`` bars seen),
    the baseline stays flat.
    """

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError(f"need 0 < fast < slow, got fast={fast}, slow={slow}")
        self.fast = fast
        self.slow = slow
        self.name = f"MA({fast},{slow})"

    def predict_action(
        self,
        close_history: np.ndarray,
        high_history: np.ndarray,
        low_history: np.ndarray,
    ) -> float:
        if len(close_history) < self.slow:
            return 0.0
        fast_ma = float(np.mean(close_history[-self.fast :]))
        slow_ma = float(np.mean(close_history[-self.slow :]))
        return 1.0 if fast_ma > slow_ma else -1.0


class DonchianBaseline(TradingBaseline):
    """Donchian channel breakout.

    Long when current close >= ``window``-bar high, short when it's <=
    the window's low, otherwise hold previous direction (bracket trader).

    A small tolerance (``tol_pct``) avoids whipsaws on equal-high closes.
    The ``long_only`` flag is useful for trending instruments where shorts
    bleed (per audit: Donchian long-only beat every other mechanical
    strategy on EURUSD H4 2024-2026).
    """

    def __init__(
        self,
        window: int = 20,
        tol_pct: float = 0.0005,
        long_only: bool = False,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = window
        self.tol_pct = tol_pct
        self.long_only = long_only
        self.name = f"Donchian({window})" + ("-LO" if long_only else "")
        self._last_action: float = 0.0

    def reset(self) -> None:
        self._last_action = 0.0

    def predict_action(
        self,
        close_history: np.ndarray,
        high_history: np.ndarray,
        low_history: np.ndarray,
    ) -> float:
        if len(close_history) < self.window:
            return 0.0
        ch_high = float(np.max(high_history[-self.window :]))
        ch_low = float(np.min(low_history[-self.window :]))
        c = float(close_history[-1])
        if c >= ch_high * (1.0 - self.tol_pct):
            self._last_action = 1.0
        elif c <= ch_low * (1.0 + self.tol_pct):
            self._last_action = 0.0 if self.long_only else -1.0
        # else: hold previous direction (bracket logic)
        return self._last_action


class RandomBaseline(TradingBaseline):
    """Sanity-check baseline: random ±1 / 0 actions.

    Should produce strongly negative returns under realistic costs — if it
    doesn't, your spread / slippage model is too lenient and inflates every
    other baseline (and the trained model) too.
    """

    def __init__(self, seed: int = 42, p_flat: float = 0.34) -> None:
        if not 0.0 <= p_flat <= 1.0:
            raise ValueError(f"p_flat must be in [0, 1], got {p_flat}")
        self.seed = seed
        self.p_flat = p_flat
        self.name = f"Random(seed={seed})"
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def predict_action(
        self,
        close_history: np.ndarray,
        high_history: np.ndarray,
        low_history: np.ndarray,
    ) -> float:
        u = self._rng.random()
        if u < self.p_flat:
            return 0.0
        # Split remaining mass evenly between +1 / -1
        return 1.0 if self._rng.random() < 0.5 else -1.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineEvalResult:
    """Result of running a single baseline on a price slice."""

    name: str
    n_bars: int
    n_trades: int
    final_balance: float
    total_return_pct: float
    metrics: dict[str, float] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def sharpe_ratio(self) -> float:
        return float(self.metrics.get("sharpe_ratio", 0.0))

    @property
    def max_drawdown(self) -> float:
        return float(self.metrics.get("max_drawdown", 0.0))

    @property
    def profit_factor(self) -> float:
        return float(self.metrics.get("profit_factor", 0.0))


def evaluate_on_data(
    baseline: TradingBaseline,
    data: pd.DataFrame,
    config: BaselineExecConfig | None = None,
    *,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    annualisation_periods: int = 252 * 6,  # H4 default
) -> BaselineEvalResult:
    """Run a baseline against an OHLC DataFrame and collect metrics.

    The execution model deliberately mirrors the production
    ``ForexTradingEnv`` cost model: a spread cost (in pips) is deducted on
    every direction change.  Open positions accrue PnL based on
    bar-to-bar close changes, scaled by ``position_pct`` of equity.

    Parameters
    ----------
    baseline : TradingBaseline
        The strategy to evaluate.  ``reset()`` is called before the loop.
    data : pd.DataFrame
        Must contain ``close_col`` and ideally ``high_col`` / ``low_col``;
        if high/low are missing, they default to ``close``.
    config : BaselineExecConfig | None
        Spread, balance, and sizing.  Defaults to retail-realistic values.
    annualisation_periods : int
        Bars per year for Sharpe annualisation.  Defaults to ``252 * 6``
        (H4 → 6 bars/day × 252 trading days).  Use ``252 * 24`` for H1.
    """
    if config is None:
        config = BaselineExecConfig()

    if close_col not in data.columns:
        raise ValueError(f"data missing required column '{close_col}'")
    close = data[close_col].to_numpy(dtype=np.float64)
    high = data[high_col].to_numpy(dtype=np.float64) if high_col in data.columns else close.copy()
    low = data[low_col].to_numpy(dtype=np.float64) if low_col in data.columns else close.copy()

    n = len(close)
    if n < 2:
        return BaselineEvalResult(
            name=baseline.name,
            n_bars=n,
            n_trades=0,
            final_balance=config.initial_balance,
            total_return_pct=0.0,
            metrics={},
            equity_curve=[config.initial_balance],
        )

    baseline.reset()

    spread_cost = config.transaction_cost_pips * config.pip_value  # in price units
    balance = config.initial_balance
    position = 0.0  # in {-1, 0, +1}
    equity_curve = [balance]
    returns: list[float] = []
    n_trades = 0

    for i in range(1, n):
        # Predict on history up to and including bar i-1 (no look-ahead)
        action = baseline.predict_action(
            close_history=close[: i],
            high_history=high[: i],
            low_history=low[: i],
        )
        # Snap to {-1, 0, +1}
        if action > 0.5:
            target = 1.0
        elif action < -0.5:
            target = -1.0
        else:
            target = 0.0

        # Apply position from previous step over (i-1 → i) close-to-close return
        prev_close = close[i - 1]
        cur_close = close[i]
        if prev_close > 0:
            bar_return = (cur_close - prev_close) / prev_close
        else:
            bar_return = 0.0

        # Position PnL on the bar
        pnl_pct = position * bar_return * config.position_pct

        # Direction change cost (spread × notional fraction of balance)
        if target != position:
            # Cost is fraction of price in spread terms
            cost_pct = spread_cost / prev_close if prev_close > 0 else 0.0
            pnl_pct -= cost_pct * config.position_pct
            if target != 0.0:
                n_trades += 1

        balance *= 1.0 + pnl_pct
        equity_curve.append(balance)
        returns.append(pnl_pct)
        position = target

    returns_arr = np.asarray(returns, dtype=np.float64)
    metrics = _compute_metrics(returns_arr, annualisation_periods)

    return BaselineEvalResult(
        name=baseline.name,
        n_bars=n,
        n_trades=n_trades,
        final_balance=float(balance),
        total_return_pct=float((balance - config.initial_balance) / config.initial_balance * 100.0),
        metrics=metrics,
        equity_curve=equity_curve,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_metrics(returns: np.ndarray, annualisation_periods: int) -> dict[str, float]:
    """Compute the metric set we care about for baseline comparison.

    Kept small and self-contained so this module has no hard dependency on
    ``apexfx.utils.metrics`` (which has its own annualisation conventions).
    """
    if len(returns) < 2:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "annualised_return": 0.0,
        }

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    sharpe = (mean_r / std_r * np.sqrt(annualisation_periods)) if std_r > 1e-12 else 0.0

    downside = returns[returns < 0]
    down_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else 0.0
    sortino = (mean_r / down_std * np.sqrt(annualisation_periods)) if down_std > 1e-12 else 0.0

    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / np.maximum(peak, 1e-12)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    n_pos = int(np.sum(returns > 0))
    win_rate = n_pos / len(returns)

    gains_sum = float(np.sum(returns[returns > 0]))
    losses_sum = float(np.sum(np.abs(returns[returns < 0])))
    profit_factor = (gains_sum / losses_sum) if losses_sum > 1e-12 else float("inf")

    total_return = float(equity[-1] - 1.0)
    ann_return = mean_r * annualisation_periods

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_dd,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "total_return": total_return,
        "annualised_return": float(ann_return),
    }
