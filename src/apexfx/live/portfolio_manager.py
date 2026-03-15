"""Multi-symbol portfolio manager — coordinates trading across multiple pairs.

Tracks aggregate exposure, per-symbol concentration, and cross-pair correlation
to prevent portfolio-level blow-up even when individual pair risk checks pass.

Usage
-----
>>> pm = PortfolioManager(config, ["EURUSD", "GBPUSD"], risk_manager)
>>> result = pm.check_new_trade("EURUSD", 1, 10000.0)
>>> if result.approved:
...     pm.update_position("EURUSD", 1, 0.1, 1.0850)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Known FX pair correlation groups (approximate static correlations — fallback only)
# Positive: move together; Negative: move inversely
CORRELATION_MAP: dict[tuple[str, str], float] = {
    ("EURUSD", "GBPUSD"): 0.85,
    ("EURUSD", "AUDUSD"): 0.70,
    ("EURUSD", "NZDUSD"): 0.65,
    ("EURUSD", "USDCHF"): -0.90,
    ("EURUSD", "USDJPY"): -0.30,
    ("GBPUSD", "AUDUSD"): 0.60,
    ("GBPUSD", "NZDUSD"): 0.55,
    ("GBPUSD", "USDCHF"): -0.80,
    ("AUDUSD", "NZDUSD"): 0.90,
    ("USDJPY", "USDCHF"): 0.60,
}

# Rolling window for dynamic correlation estimation
_DYNAMIC_CORRELATION_LOOKBACK = 60  # bars (60 H1 bars ≈ 2.5 trading days)
_MIN_BARS_FOR_DYNAMIC = 20  # Minimum bars before trusting dynamic estimate


class DynamicCorrelationTracker:
    """Rolling correlation tracker that replaces static estimates when enough data exists.

    Uses exponentially-weighted rolling returns to compute pairwise correlations
    and updates at configurable frequency. Falls back to static CORRELATION_MAP
    when insufficient data is available.
    """

    def __init__(
        self,
        lookback: int = _DYNAMIC_CORRELATION_LOOKBACK,
        min_bars: int = _MIN_BARS_FOR_DYNAMIC,
    ) -> None:
        self._lookback = lookback
        self._min_bars = min_bars
        self._returns: dict[str, list[float]] = {}
        self._dynamic_corr: dict[tuple[str, str], float] = {}

    def update_returns(self, symbol: str, bar_return: float) -> None:
        """Feed a new bar return for a symbol."""
        if symbol not in self._returns:
            self._returns[symbol] = []
        self._returns[symbol].append(bar_return)
        if len(self._returns[symbol]) > self._lookback * 2:
            self._returns[symbol] = self._returns[symbol][-self._lookback:]

    def recompute(self) -> None:
        """Recompute all pairwise correlations from return history."""
        symbols = list(self._returns.keys())
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                r_a = self._returns[sym_a]
                r_b = self._returns[sym_b]
                n = min(len(r_a), len(r_b))
                if n < self._min_bars:
                    continue
                arr_a = np.array(r_a[-n:])
                arr_b = np.array(r_b[-n:])
                corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                if not np.isnan(corr):
                    key = tuple(sorted([sym_a, sym_b]))
                    self._dynamic_corr[key] = corr

    def get_correlation(self, sym_a: str, sym_b: str) -> float:
        """Get correlation — prefers dynamic, falls back to static."""
        if sym_a == sym_b:
            return 1.0
        key = tuple(sorted([sym_a, sym_b]))
        # Prefer dynamic
        if key in self._dynamic_corr:
            return self._dynamic_corr[key]
        # Fall back to static
        if (sym_a, sym_b) in CORRELATION_MAP:
            return CORRELATION_MAP[(sym_a, sym_b)]
        if (sym_b, sym_a) in CORRELATION_MAP:
            return CORRELATION_MAP[(sym_b, sym_a)]
        return 0.30

    @property
    def has_dynamic_data(self) -> bool:
        return len(self._dynamic_corr) > 0

    @property
    def dynamic_pairs(self) -> dict[tuple[str, str], float]:
        return dict(self._dynamic_corr)


# Module-level singleton for shared correlation tracking
_correlation_tracker = DynamicCorrelationTracker()


def _get_correlation(sym_a: str, sym_b: str) -> float:
    """Look up correlation — uses dynamic if available, else static."""
    return _correlation_tracker.get_correlation(sym_a, sym_b)


def get_correlation_tracker() -> DynamicCorrelationTracker:
    """Get the module-level correlation tracker singleton."""
    return _correlation_tracker


@dataclass
class PositionInfo:
    """Snapshot of an open position."""
    symbol: str
    direction: int          # +1 long, -1 short
    volume: float           # lots
    entry_price: float
    notional: float         # direction * volume * contract_size * entry_price
    open_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioRiskResult:
    """Result of a portfolio-level trade check."""
    approved: bool
    scale_factor: float = 1.0   # May reduce size if partially approved
    reason: str = ""


@dataclass
class PortfolioMetrics:
    """Aggregate portfolio metrics."""
    total_exposure: float       # Sum of abs(notional) / equity
    n_positions: int
    per_symbol_exposure: dict[str, float]
    max_correlation: float      # Max pairwise correlation among open positions
    aggregate_equity: float


class PortfolioManager:
    """Coordinates multi-symbol trading with shared equity and risk limits.

    Parameters
    ----------
    max_total_exposure : float
        Maximum total exposure as fraction of equity (sum of |notional|/equity).
    max_per_symbol : float
        Maximum per-symbol exposure as fraction of equity.
    correlation_limit : float
        Block new trades if they have correlation > this with existing positions
        in the same direction.
    contract_size : int
        Standard FX contract size (default 100,000).
    """

    def __init__(
        self,
        max_total_exposure: float = 0.40,
        max_per_symbol: float = 0.25,
        correlation_limit: float = 0.70,
        contract_size: int = 100_000,
    ) -> None:
        self._max_total_exposure = max_total_exposure
        self._max_per_symbol = max_per_symbol
        self._correlation_limit = correlation_limit
        self._contract_size = contract_size
        self._positions: dict[str, PositionInfo] = {}
        self._equity: float = 100_000.0
        self._closed_pnl: float = 0.0

    def set_equity(self, equity: float) -> None:
        """Update the shared portfolio equity."""
        self._equity = equity

    def check_new_trade(
        self,
        symbol: str,
        direction: int,
        notional: float,
    ) -> PortfolioRiskResult:
        """Check if a new trade is allowed given portfolio constraints.

        Args:
            symbol: Trading pair (e.g. "EURUSD").
            direction: +1 for long, -1 for short.
            notional: Absolute notional value of proposed trade.

        Returns:
            PortfolioRiskResult with approval status and optional scale factor.
        """
        if self._equity <= 0:
            return PortfolioRiskResult(False, 0.0, "Zero equity")

        # 1. Total exposure check
        current_total = sum(abs(p.notional) for p in self._positions.values())
        new_total = current_total + notional
        total_exposure = new_total / self._equity

        if total_exposure > self._max_total_exposure:
            # Calculate how much we can allow
            room = max(0, self._max_total_exposure * self._equity - current_total)
            if room <= 0:
                return PortfolioRiskResult(
                    False, 0.0,
                    f"Total exposure {total_exposure:.1%} > limit {self._max_total_exposure:.1%}",
                )
            scale = room / notional
            return PortfolioRiskResult(
                True, scale,
                f"Total exposure scaled to {self._max_total_exposure:.1%}",
            )

        # 2. Per-symbol concentration check
        existing_symbol_notional = abs(self._positions[symbol].notional) if symbol in self._positions else 0.0
        symbol_exposure = (existing_symbol_notional + notional) / self._equity
        if symbol_exposure > self._max_per_symbol:
            room = max(0, self._max_per_symbol * self._equity - existing_symbol_notional)
            if room <= 0:
                return PortfolioRiskResult(
                    False, 0.0,
                    f"{symbol} exposure {symbol_exposure:.1%} > limit {self._max_per_symbol:.1%}",
                )
            scale = room / notional
            return PortfolioRiskResult(
                True, scale,
                f"{symbol} exposure scaled to {self._max_per_symbol:.1%}",
            )

        # 3. Correlation check: don't pile into correlated positions in same direction
        for existing_sym, existing_pos in self._positions.items():
            if existing_sym == symbol:
                continue
            corr = _get_correlation(symbol, existing_sym)
            same_direction = (direction == existing_pos.direction)
            # Correlated positions in same direction → concentrated risk
            # Anti-correlated positions in opposite direction → also concentrated
            effective_corr = corr if same_direction else -corr
            if effective_corr > self._correlation_limit:
                return PortfolioRiskResult(
                    False, 0.0,
                    f"Correlation {symbol}/{existing_sym} = {corr:.2f} "
                    f"(effective {effective_corr:.2f}) > limit {self._correlation_limit:.2f}",
                )

        return PortfolioRiskResult(True, 1.0, "Portfolio checks passed")

    def update_position(
        self,
        symbol: str,
        direction: int,
        volume: float,
        entry_price: float,
    ) -> None:
        """Record a new or updated position."""
        notional = abs(direction * volume * self._contract_size * entry_price)
        self._positions[symbol] = PositionInfo(
            symbol=symbol,
            direction=direction,
            volume=volume,
            entry_price=entry_price,
            notional=notional,
        )
        logger.info(
            "Portfolio position updated",
            symbol=symbol,
            direction=direction,
            volume=volume,
            notional=round(notional, 2),
        )

    def close_position(self, symbol: str, exit_price: float, pnl: float) -> None:
        """Remove a closed position and record P&L."""
        if symbol in self._positions:
            del self._positions[symbol]
            self._closed_pnl += pnl
            logger.info(
                "Portfolio position closed",
                symbol=symbol,
                pnl=round(pnl, 2),
                total_closed_pnl=round(self._closed_pnl, 2),
            )

    def get_open_positions(self) -> list[PositionInfo]:
        """Return list of all open positions."""
        return list(self._positions.values())

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Compute aggregate portfolio metrics."""
        per_symbol = {}
        total_notional = 0.0
        max_corr = 0.0

        for sym, pos in self._positions.items():
            per_symbol[sym] = abs(pos.notional) / self._equity if self._equity > 0 else 0.0
            total_notional += abs(pos.notional)

        # Max pairwise correlation
        symbols = list(self._positions.keys())
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                corr = abs(_get_correlation(sym_a, sym_b))
                max_corr = max(max_corr, corr)

        return PortfolioMetrics(
            total_exposure=total_notional / self._equity if self._equity > 0 else 0.0,
            n_positions=len(self._positions),
            per_symbol_exposure=per_symbol,
            max_correlation=max_corr,
            aggregate_equity=self._equity,
        )

    def get_aggregate_equity(self) -> float:
        """Return current portfolio equity including unrealized P&L."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self._equity + unrealized
