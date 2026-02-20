"""Correlation-based portfolio risk — prevents over-concentration in correlated positions.

If EURUSD and GBPUSD are both long, the effective exposure is much higher
than each position alone due to their ~0.85 correlation. This module
monitors cross-pair correlations and reduces position sizes accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionInfo:
    """Current position in a single symbol."""
    symbol: str
    direction: int  # +1 long, -1 short, 0 flat
    notional_value: float  # in account currency
    entry_price: float


@dataclass
class CorrelationRiskResult:
    """Result of portfolio correlation check."""
    approved: bool
    scale_factor: float  # 0.0-1.0
    effective_exposure: float  # portfolio-level
    concentration_ratio: float  # how concentrated is the portfolio
    correlated_pairs: list[tuple[str, str, float]]  # (sym1, sym2, corr)
    reason: str


class CorrelationRiskManager:
    """Monitors and limits correlated exposure across multiple positions.

    Maintains a rolling correlation matrix between traded pairs and
    limits the portfolio's effective exposure (correlation-adjusted).

    When opening a new position, checks if the combined exposure
    (considering existing positions' correlations) exceeds limits.
    """

    # Static correlation estimates (updated dynamically when data available)
    DEFAULT_CORRELATIONS = {
        ("EURUSD", "GBPUSD"): 0.85,
        ("EURUSD", "EURGBP"): -0.45,
        ("EURUSD", "USDJPY"): -0.30,
        ("GBPUSD", "USDJPY"): -0.25,
        ("EURUSD", "USDCHF"): -0.95,
        ("GBPUSD", "EURGBP"): -0.75,
        ("AUDUSD", "NZDUSD"): 0.90,
        ("EURUSD", "AUDUSD"): 0.50,
    }

    def __init__(
        self,
        max_effective_exposure_pct: float = 0.25,
        max_concentration: float = 0.70,
        correlation_lookback: int = 60,
    ) -> None:
        self._max_exposure = max_effective_exposure_pct
        self._max_concentration = max_concentration
        self._lookback = correlation_lookback

        self._positions: dict[str, PositionInfo] = {}
        self._returns_history: dict[str, list[float]] = {}
        self._dynamic_correlations: dict[tuple[str, str], float] = {}

    def update_position(self, position: PositionInfo) -> None:
        """Update or add a position."""
        if position.direction == 0:
            self._positions.pop(position.symbol, None)
        else:
            self._positions[position.symbol] = position

    def update_returns(self, symbol: str, daily_return: float) -> None:
        """Update return history for correlation computation."""
        if symbol not in self._returns_history:
            self._returns_history[symbol] = []
        self._returns_history[symbol].append(daily_return)
        if len(self._returns_history[symbol]) > self._lookback * 2:
            self._returns_history[symbol] = self._returns_history[symbol][-self._lookback:]

        # Recompute correlations when enough data
        self._recompute_correlations()

    def _recompute_correlations(self) -> None:
        """Recompute pairwise correlations from return history."""
        symbols = list(self._returns_history.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                r1 = self._returns_history[sym1]
                r2 = self._returns_history[sym2]
                min_len = min(len(r1), len(r2))
                if min_len < 20:
                    continue
                arr1 = np.array(r1[-min_len:])
                arr2 = np.array(r2[-min_len:])
                corr = np.corrcoef(arr1, arr2)[0, 1]
                if not np.isnan(corr):
                    key = tuple(sorted([sym1, sym2]))
                    self._dynamic_correlations[key] = float(corr)

    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        key = tuple(sorted([sym1, sym2]))
        # Prefer dynamic correlations
        if key in self._dynamic_correlations:
            return self._dynamic_correlations[key]
        # Fall back to static estimates
        if key in self.DEFAULT_CORRELATIONS:
            return self.DEFAULT_CORRELATIONS[key]
        # Unknown pair: assume moderate correlation
        return 0.3

    def check_new_position(
        self,
        new_symbol: str,
        new_direction: int,
        new_notional: float,
        portfolio_value: float,
    ) -> CorrelationRiskResult:
        """Check if a new position would create excessive correlated exposure.

        Args:
            new_symbol: Symbol to trade.
            new_direction: +1 long, -1 short.
            new_notional: Notional value of the new position.
            portfolio_value: Current portfolio value.

        Returns:
            CorrelationRiskResult with approval and scaling.
        """
        if portfolio_value <= 0:
            return CorrelationRiskResult(
                approved=False, scale_factor=0.0,
                effective_exposure=0.0, concentration_ratio=0.0,
                correlated_pairs=[], reason="Portfolio value <= 0",
            )

        # Build position vector including the new position
        all_positions = dict(self._positions)
        all_positions[new_symbol] = PositionInfo(
            symbol=new_symbol,
            direction=new_direction,
            notional_value=new_notional,
            entry_price=0.0,
        )

        symbols = list(all_positions.keys())
        n = len(symbols)

        if n <= 1:
            exposure = new_notional / portfolio_value
            return CorrelationRiskResult(
                approved=True, scale_factor=1.0,
                effective_exposure=exposure, concentration_ratio=1.0,
                correlated_pairs=[], reason="Single position",
            )

        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.get_correlation(symbols[i], symbols[j])
                # Adjust sign: if positions are in opposite directions, flip correlation
                dir_i = all_positions[symbols[i]].direction
                dir_j = all_positions[symbols[j]].direction
                effective_corr = c * dir_i * dir_j
                corr_matrix[i, j] = effective_corr
                corr_matrix[j, i] = effective_corr

        # Weights: fraction of total notional
        notionals = np.array([all_positions[s].notional_value for s in symbols])
        total_notional = np.sum(notionals)
        weights = notionals / (total_notional + 1e-10)

        # Portfolio variance: w'Cw (using correlation as covariance proxy)
        portfolio_var = float(weights @ corr_matrix @ weights)
        effective_exposure = total_notional / portfolio_value * np.sqrt(max(0, portfolio_var))

        # Concentration: Herfindahl index
        concentration = float(np.sum(weights ** 2))

        # Find highly correlated pairs
        correlated_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > 0.5:
                    correlated_pairs.append(
                        (symbols[i], symbols[j], float(corr_matrix[i, j]))
                    )

        # Determine if we need to scale down
        scale = 1.0
        reason = "Correlation check passed"

        if effective_exposure > self._max_exposure:
            scale = self._max_exposure / (effective_exposure + 1e-10)
            reason = f"Effective exposure {effective_exposure:.2%} > limit {self._max_exposure:.2%}"

        if concentration > self._max_concentration:
            conc_scale = self._max_concentration / concentration
            if conc_scale < scale:
                scale = conc_scale
                reason = f"Concentration {concentration:.2%} > limit {self._max_concentration:.2%}"

        scale = max(0.0, min(1.0, scale))
        approved = scale > 0.01

        if not approved:
            logger.warning(
                "Correlation risk: position blocked",
                symbol=new_symbol,
                effective_exposure=f"{effective_exposure:.2%}",
                concentration=f"{concentration:.2%}",
                correlated_pairs=correlated_pairs,
            )

        return CorrelationRiskResult(
            approved=approved,
            scale_factor=scale,
            effective_exposure=effective_exposure,
            concentration_ratio=concentration,
            correlated_pairs=correlated_pairs,
            reason=reason,
        )
