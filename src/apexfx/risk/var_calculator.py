"""Value at Risk (VaR) computation: Historical, Parametric, and CVaR."""

from __future__ import annotations

import numpy as np
from scipy import stats


class VaRCalculator:
    """
    Computes portfolio Value at Risk using multiple methods.
    Daily VaR limit (2% of portfolio) is the primary risk gate.
    """

    def __init__(
        self,
        confidence: float = 0.99,
        lookback_days: int = 252,
        method: str = "historical",
    ) -> None:
        self._confidence = confidence
        self._lookback = lookback_days
        self._method = method
        self._returns_history: list[float] = []

    def update(self, daily_return: float) -> None:
        """Add a new daily return to the history."""
        self._returns_history.append(daily_return)
        if len(self._returns_history) > self._lookback:
            self._returns_history = self._returns_history[-self._lookback :]

    @property
    def has_sufficient_data(self) -> bool:
        return len(self._returns_history) >= 30

    def compute_var(self, portfolio_value: float) -> float:
        """Compute VaR as a fraction of portfolio value."""
        if not self.has_sufficient_data:
            return 0.0  # No VaR estimate without data

        returns = np.array(self._returns_history)

        if self._method == "historical":
            return self._historical_var(returns, portfolio_value)
        elif self._method == "parametric":
            return self._parametric_var(returns, portfolio_value)
        else:
            return self._historical_var(returns, portfolio_value)

    def compute_cvar(self, portfolio_value: float) -> float:
        """Compute Conditional VaR (Expected Shortfall) — more conservative."""
        if not self.has_sufficient_data:
            return 0.0

        returns = np.array(self._returns_history)
        var_threshold = np.percentile(returns, (1 - self._confidence) * 100)
        tail_losses = returns[returns <= var_threshold]

        if len(tail_losses) == 0:
            return abs(var_threshold) * portfolio_value

        cvar = abs(np.mean(tail_losses))
        return cvar * portfolio_value

    def project_with_position(
        self,
        action: float,
        current_var: float,
        position_value: float,
        portfolio_value: float,
    ) -> float:
        """
        Project what VaR would be if we took the proposed action.
        Used to check if a trade would breach the VaR limit.
        """
        if not self.has_sufficient_data:
            # Conservative estimate: scale linearly with position size
            return abs(action) * position_value * 0.02

        returns = np.array(self._returns_history)
        vol = np.std(returns, ddof=1)

        # Project position VaR: larger position = higher VaR
        position_fraction = abs(action) * position_value / (portfolio_value + 1e-10)
        projected_var = position_fraction * vol * stats.norm.ppf(self._confidence)

        return abs(projected_var)

    def _historical_var(self, returns: np.ndarray, portfolio_value: float) -> float:
        """VaR from empirical return distribution."""
        percentile = (1 - self._confidence) * 100
        var_return = np.percentile(returns, percentile)
        return abs(var_return)

    def _parametric_var(self, returns: np.ndarray, portfolio_value: float) -> float:
        """VaR assuming normal distribution: μ - z_α × σ."""
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z_alpha = stats.norm.ppf(self._confidence)
        var_return = -(mu - z_alpha * sigma)
        return max(var_return, 0)
