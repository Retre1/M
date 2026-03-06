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

    # ------------------------------------------------------------------
    # Portfolio-level VaR (multi-asset)
    # ------------------------------------------------------------------

    def compute_portfolio_var(
        self,
        positions: dict[str, float],
        correlation_matrix: np.ndarray,
        individual_vars: dict[str, float],
    ) -> float:
        """Portfolio VaR using variance-covariance method with correlations.

        VaR_portfolio = sqrt(w' * Sigma * w) where Sigma is the VaR-covariance
        matrix constructed from individual VaRs and their correlations.

        Args:
            positions: {symbol: notional_value} for each asset.
            correlation_matrix: NxN correlation matrix (same order as positions keys).
            individual_vars: {symbol: var_fraction} individual asset VaRs.

        Returns:
            Portfolio VaR as a fraction of total portfolio value.
        """
        symbols = list(positions.keys())
        n = len(symbols)
        if n == 0:
            return 0.0
        if n == 1:
            sym = symbols[0]
            return individual_vars.get(sym, 0.0)

        total_value = sum(abs(v) for v in positions.values())
        if total_value < 1e-10:
            return 0.0

        # Weight vector (fraction of total exposure)
        weights = np.array([positions[s] / total_value for s in symbols])

        # VaR vector
        var_vec = np.array([individual_vars.get(s, 0.0) for s in symbols])

        # Construct VaR-covariance matrix: Sigma_ij = VaR_i * VaR_j * rho_ij
        corr = np.asarray(correlation_matrix)
        if corr.shape != (n, n):
            raise ValueError(
                f"correlation_matrix shape {corr.shape} != ({n}, {n})"
            )

        var_cov = np.outer(var_vec, var_vec) * corr

        # Portfolio VaR = sqrt(w' * Sigma * w)
        portfolio_var_sq = float(weights @ var_cov @ weights)
        return float(np.sqrt(max(portfolio_var_sq, 0.0)))

    def compute_component_var(
        self,
        positions: dict[str, float],
        correlation_matrix: np.ndarray,
        individual_vars: dict[str, float],
    ) -> dict[str, float]:
        """Component VaR: each asset's contribution to portfolio VaR.

        Component VaR_i = w_i * (Sigma * w)_i / VaR_portfolio
        Sum of component VaRs = portfolio VaR (additive decomposition).

        Returns:
            {symbol: component_var_fraction} for each asset.
        """
        symbols = list(positions.keys())
        n = len(symbols)
        if n == 0:
            return {}

        total_value = sum(abs(v) for v in positions.values())
        if total_value < 1e-10:
            return {s: 0.0 for s in symbols}

        weights = np.array([positions[s] / total_value for s in symbols])
        var_vec = np.array([individual_vars.get(s, 0.0) for s in symbols])
        corr = np.asarray(correlation_matrix)
        var_cov = np.outer(var_vec, var_vec) * corr

        portfolio_var = self.compute_portfolio_var(
            positions, correlation_matrix, individual_vars
        )
        if portfolio_var < 1e-10:
            return {s: 0.0 for s in symbols}

        # Component VaR_i = w_i * (Sigma @ w)_i / portfolio_var
        marginal = var_cov @ weights
        components = weights * marginal / portfolio_var

        return {s: float(components[i]) for i, s in enumerate(symbols)}

    def compute_marginal_var(
        self,
        new_position: str,
        new_value: float,
        existing_positions: dict[str, float],
        correlation_matrix: np.ndarray,
        individual_vars: dict[str, float],
    ) -> float:
        """Marginal VaR: how much portfolio VaR increases by adding a new position.

        Returns:
            Delta VaR (positive = increases risk, negative = diversification benefit).
        """
        # VaR without new position
        var_before = self.compute_portfolio_var(
            existing_positions, correlation_matrix, individual_vars
        )

        # VaR with new position
        combined = dict(existing_positions)
        combined[new_position] = combined.get(new_position, 0.0) + new_value

        # Expand correlation matrix if needed
        symbols_before = list(existing_positions.keys())
        symbols_after = list(combined.keys())

        if len(symbols_after) > len(symbols_before):
            # New asset not in existing matrix — assume zero correlation (conservative)
            n_old = len(symbols_before)
            n_new = len(symbols_after)
            new_corr = np.eye(n_new)
            new_corr[:n_old, :n_old] = correlation_matrix
            var_after = self.compute_portfolio_var(combined, new_corr, individual_vars)
        else:
            var_after = self.compute_portfolio_var(
                combined, correlation_matrix, individual_vars
            )

        return var_after - var_before
