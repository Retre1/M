"""Institutional-grade stress testing framework for portfolio risk analysis.

Provides deterministic scenario analysis, Monte Carlo simulation with fat-tailed
distributions, and reverse stress testing to identify minimum failure conditions.

This module is designed for production trading systems handling real capital.
All computations are conservative by default — when in doubt, the framework
assumes worse outcomes to protect the portfolio.

Components:
- StressScenario: defines a market shock event (historical or hypothetical)
- StressResult: outcome of a single scenario simulation
- MonteCarloResult: aggregate statistics from Monte Carlo path simulation
- ReverseStressResult: minimum shock required to breach a loss threshold
- StressTester: orchestrates all stress testing operations
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StressScenario:
    """Definition of a market stress event.

    Each scenario encodes the key dimensions of a market shock:
    price movement, spread widening, volatility expansion, event duration,
    and liquidity impairment.

    Attributes:
        name: Short human-readable label (e.g. "Flash Crash").
        description: Longer context about the historical or hypothetical event.
        price_shock_pct: Instantaneous price move as a decimal fraction
            (e.g. -0.05 for a 5% drop). Negative means adverse move.
        spread_multiplier: Factor by which typical bid-ask spread widens
            during the event (e.g. 10.0 means 10x normal spread).
        volatility_multiplier: Factor by which realized volatility increases
            relative to the pre-event baseline.
        duration_bars: Number of bars (time steps) the event persists before
            conditions begin normalizing.
        liquidity_factor: Fraction of normal market liquidity available during
            the event (0.0 = no liquidity, 1.0 = normal).
    """

    name: str
    description: str
    price_shock_pct: float
    spread_multiplier: float
    volatility_multiplier: float
    duration_bars: int
    liquidity_factor: float


@dataclass
class StressResult:
    """Outcome of running a single stress scenario against a portfolio.

    Attributes:
        scenario: The scenario that was simulated.
        pnl_impact: Absolute P&L change in account currency. Negative means loss.
        max_drawdown_pct: Peak-to-trough drawdown as a decimal fraction of
            portfolio value caused by this scenario.
        recovery_bars: Estimated number of bars required to recover from the
            drawdown under normal market conditions.
        var_breach: True if the scenario loss exceeds the configured VaR limit.
        margin_call: True if the scenario loss would trigger a margin call
            (remaining equity below maintenance margin requirement).
        survival: True if the portfolio retains more than 50% of its value
            after the scenario.
    """

    scenario: StressScenario
    pnl_impact: float
    max_drawdown_pct: float
    recovery_bars: int
    var_breach: bool
    margin_call: bool
    survival: bool


@dataclass
class MonteCarloResult:
    """Aggregate statistics from Monte Carlo path simulation.

    All VaR and CVaR figures are expressed as positive loss fractions
    (e.g. 0.05 means a 5% loss at that confidence level).

    Attributes:
        var_95: Value at Risk at the 95th percentile.
        var_99: Value at Risk at the 99th percentile.
        cvar_95: Conditional VaR (Expected Shortfall) at the 95th percentile.
        cvar_99: Conditional VaR (Expected Shortfall) at the 99th percentile.
        max_drawdown_distribution: List of maximum drawdown values across all
            simulated paths, one per path.
        median_return: Median terminal cumulative return across all paths.
        worst_path_return: Terminal return of the single worst-performing path.
        survival_rate: Fraction of paths that did not breach the maximum
            drawdown threshold (default 15%).
        n_simulations: Number of Monte Carlo paths that were simulated.
    """

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_distribution: list[float]
    median_return: float
    worst_path_return: float
    survival_rate: float
    n_simulations: int


@dataclass
class ReverseStressResult:
    """Result of a reverse stress test.

    Identifies the minimum market shock that causes a specified portfolio loss.

    Attributes:
        target_loss_pct: The target loss threshold that was tested against
            (e.g. 0.20 for a 20% portfolio loss).
        min_shock_pct: The minimum adverse price shock (as a positive decimal
            fraction) that causes the target loss.
        scenario_description: Human-readable description of the reverse
            stress scenario.
        probability_estimate: Estimated probability of such a shock occurring,
            derived from a Student-t fit to historical returns. None if no
            historical returns were provided.
    """

    target_loss_pct: float
    min_shock_pct: float
    scenario_description: str
    probability_estimate: float | None


# ---------------------------------------------------------------------------
# Stress tester
# ---------------------------------------------------------------------------


class StressTester:
    """Orchestrates stress testing across deterministic scenarios, Monte Carlo
    simulation, and reverse stress analysis.

    This is a stateless analysis tool — it does not modify any portfolio or
    risk manager state. It is safe to call from monitoring, reporting, or
    pre-trade analysis pipelines.

    Usage::

        tester = StressTester(var_limit=0.02, margin_requirement=0.01)

        # Run a single scenario
        result = tester.run_scenario(
            StressTester.PRESET_SCENARIOS["snb_shock"],
            portfolio_value=100_000.0,
        )

        # Run all historical presets
        results = tester.run_all_presets(portfolio_value=100_000.0)

        # Monte Carlo VaR/CVaR
        mc = tester.monte_carlo_stress(returns_array)

        # Reverse stress test
        rs = tester.reverse_stress_test(portfolio_value=100_000.0)
    """

    # Historical / hypothetical preset scenarios
    PRESET_SCENARIOS: dict[str, StressScenario] = {
        "flash_crash": StressScenario(
            "Flash Crash",
            "2010-style 5% instant drop",
            -0.05, 10.0, 5.0, 12, 0.1,
        ),
        "snb_shock": StressScenario(
            "SNB Shock",
            "2015 EURCHF floor removal",
            -0.20, 50.0, 20.0, 24, 0.01,
        ),
        "brexit": StressScenario(
            "Brexit Vote",
            "2016 GBPUSD crash",
            -0.10, 8.0, 8.0, 48, 0.3,
        ),
        "covid_march": StressScenario(
            "COVID March 2020",
            "Pandemic liquidity crisis",
            -0.08, 15.0, 12.0, 120, 0.2,
        ),
        "fed_surprise": StressScenario(
            "Fed Surprise",
            "Unexpected 50bp rate hike",
            -0.03, 5.0, 3.0, 24, 0.5,
        ),
        "gap_weekend": StressScenario(
            "Weekend Gap",
            "Geopolitical weekend event",
            -0.04, 20.0, 4.0, 6, 0.3,
        ),
    }

    def __init__(
        self,
        var_limit: float = 0.02,
        margin_requirement: float = 0.01,
    ) -> None:
        """Initialise the stress tester.

        Args:
            var_limit: Maximum acceptable Value at Risk as a fraction of
                portfolio value. Scenario losses exceeding this threshold
                are flagged as VaR breaches.
            margin_requirement: Maintenance margin as a fraction of portfolio
                value. If remaining equity falls below this after a scenario
                shock, a margin call is triggered.
        """
        if var_limit <= 0:
            raise ValueError(f"var_limit must be positive, got {var_limit}")
        if margin_requirement < 0:
            raise ValueError(
                f"margin_requirement must be non-negative, got {margin_requirement}"
            )

        self._var_limit = var_limit
        self._margin_req = margin_requirement
        self._historical_returns: np.ndarray | None = None

        logger.info(
            "StressTester initialised",
            var_limit=f"{var_limit:.4f}",
            margin_requirement=f"{margin_requirement:.4f}",
            n_preset_scenarios=len(self.PRESET_SCENARIOS),
        )

    # ------------------------------------------------------------------
    # Deterministic scenario analysis
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: StressScenario,
        portfolio_value: float,
        current_position_pct: float = 0.10,
        daily_var: float = 0.02,
    ) -> StressResult:
        """Simulate a single stress scenario against the portfolio.

        The simulation is instantaneous (single-step shock) — it does not
        model the path of recovery. Recovery bars are estimated heuristically.

        Args:
            scenario: The stress scenario to simulate.
            portfolio_value: Current total portfolio value in account currency.
            current_position_pct: Current net exposure as a fraction of
                portfolio value (e.g. 0.10 for 10% exposure).
            daily_var: Current daily Value at Risk as a decimal fraction.
                Used only for informational logging; VaR breach is computed
                from the scenario impact directly.

        Returns:
            StressResult with all impact metrics populated.

        Raises:
            ValueError: If portfolio_value is non-positive.
        """
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive, got {portfolio_value}"
            )

        # Core P&L impact: position notional * adverse price move
        pnl_impact = portfolio_value * current_position_pct * scenario.price_shock_pct

        # Maximum drawdown as a fraction of portfolio value
        max_drawdown_pct = abs(current_position_pct * scenario.price_shock_pct)

        # Rough recovery estimate: larger shocks over longer durations
        # take proportionally longer to recover from
        recovery_bars = int(
            abs(scenario.price_shock_pct) / 0.001 * scenario.duration_bars
        )

        # VaR breach: does the loss exceed the configured limit?
        loss_pct = abs(pnl_impact / portfolio_value)
        var_breach = loss_pct > self._var_limit

        # Margin call: remaining equity below maintenance requirement
        remaining_equity = portfolio_value + pnl_impact
        margin_call = abs(pnl_impact) > portfolio_value * (1 - self._margin_req)

        # Survival: portfolio retains more than 50% of its value
        survival = remaining_equity > portfolio_value * 0.5

        result = StressResult(
            scenario=scenario,
            pnl_impact=pnl_impact,
            max_drawdown_pct=max_drawdown_pct,
            recovery_bars=recovery_bars,
            var_breach=var_breach,
            margin_call=margin_call,
            survival=survival,
        )

        log_level = "error" if not survival else ("warning" if var_breach else "info")
        getattr(logger, log_level)(
            "Stress scenario result",
            scenario=scenario.name,
            pnl_impact=round(pnl_impact, 2),
            max_drawdown_pct=f"{max_drawdown_pct:.4f}",
            recovery_bars=recovery_bars,
            var_breach=var_breach,
            margin_call=margin_call,
            survival=survival,
        )

        return result

    def run_all_presets(
        self,
        portfolio_value: float,
        current_position_pct: float = 0.10,
    ) -> list[StressResult]:
        """Run all preset historical stress scenarios.

        Args:
            portfolio_value: Current total portfolio value in account currency.
            current_position_pct: Current net exposure as a fraction of
                portfolio value.

        Returns:
            List of StressResult objects, one per preset scenario, ordered
            by preset key name.
        """
        logger.info(
            "Running all preset stress scenarios",
            portfolio_value=round(portfolio_value, 2),
            position_pct=f"{current_position_pct:.2%}",
            n_scenarios=len(self.PRESET_SCENARIOS),
        )

        results: list[StressResult] = []
        for key in sorted(self.PRESET_SCENARIOS):
            scenario = self.PRESET_SCENARIOS[key]
            result = self.run_scenario(
                scenario=scenario,
                portfolio_value=portfolio_value,
                current_position_pct=current_position_pct,
            )
            results.append(result)

        # Summary logging
        n_breaches = sum(1 for r in results if r.var_breach)
        n_margin_calls = sum(1 for r in results if r.margin_call)
        n_failures = sum(1 for r in results if not r.survival)

        logger.info(
            "Preset stress test summary",
            n_scenarios=len(results),
            var_breaches=n_breaches,
            margin_calls=n_margin_calls,
            survival_failures=n_failures,
            worst_pnl=round(min(r.pnl_impact for r in results), 2),
        )

        return results

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo_stress(
        self,
        returns: np.ndarray,
        n_simulations: int = 10_000,
        horizon_days: int = 21,
        portfolio_value: float = 100_000.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation with fat-tailed return distribution.

        Fits a Student-t distribution to the provided historical returns to
        capture leptokurtosis (fat tails), then simulates forward paths.

        Args:
            returns: 1-D array of historical daily returns (as decimal
                fractions, e.g. 0.01 for a 1% gain).
            n_simulations: Number of independent paths to simulate.
            horizon_days: Number of trading days per simulated path.
            portfolio_value: Portfolio value for absolute P&L computation.
                Used only for logging; all returned metrics are in return
                (fraction) space.

        Returns:
            MonteCarloResult with VaR, CVaR, drawdown distribution, and
            survival statistics.

        Raises:
            ValueError: If returns array is too short (< 30 observations)
                or contains non-finite values.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()

        if len(returns) < 30:
            raise ValueError(
                f"Need at least 30 return observations, got {len(returns)}"
            )
        if not np.all(np.isfinite(returns)):
            raise ValueError("returns array contains non-finite values (NaN/Inf)")

        # Store for potential use in reverse stress testing
        self._historical_returns = returns

        # Fit Student-t distribution to capture fat tails
        df, loc, scale = stats.t.fit(returns)
        logger.info(
            "Fitted Student-t distribution to returns",
            df=round(df, 2),
            loc=f"{loc:.6f}",
            scale=f"{scale:.6f}",
            n_observations=len(returns),
        )

        # Generate simulation paths: (n_simulations, horizon_days)
        rng = np.random.default_rng(seed=42)
        simulated_returns = stats.t.rvs(
            df, loc=loc, scale=scale,
            size=(n_simulations, horizon_days),
            random_state=rng,
        )

        # Cumulative returns per path: product of (1 + r_t) across time
        cumulative_wealth = np.cumprod(1.0 + simulated_returns, axis=1)
        terminal_returns = cumulative_wealth[:, -1] - 1.0

        # Maximum drawdown per path
        max_drawdown_per_path = np.empty(n_simulations, dtype=np.float64)
        for i in range(n_simulations):
            wealth_path = cumulative_wealth[i]
            running_max = np.maximum.accumulate(wealth_path)
            drawdowns = (running_max - wealth_path) / running_max
            max_drawdown_per_path[i] = float(np.max(drawdowns))

        # VaR: loss at a given percentile (expressed as positive number)
        # We want the loss in the left tail, so we negate returns and take
        # the upper percentile of the loss distribution.
        losses = -terminal_returns
        var_95 = float(np.percentile(losses, 95))
        var_99 = float(np.percentile(losses, 99))

        # CVaR (Expected Shortfall): mean of losses beyond VaR
        cvar_95 = float(np.mean(losses[losses >= var_95]))
        cvar_99 = float(np.mean(losses[losses >= var_99]))

        median_return = float(np.median(terminal_returns))
        worst_path_return = float(np.min(terminal_returns))

        # Survival: paths that don't breach 15% max drawdown
        max_dd_threshold = 0.15
        survival_rate = float(
            np.mean(max_drawdown_per_path < max_dd_threshold)
        )

        result = MonteCarloResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_distribution=max_drawdown_per_path.tolist(),
            median_return=median_return,
            worst_path_return=worst_path_return,
            survival_rate=survival_rate,
            n_simulations=n_simulations,
        )

        logger.info(
            "Monte Carlo stress test complete",
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            var_95=f"{var_95:.4f}",
            var_99=f"{var_99:.4f}",
            cvar_95=f"{cvar_95:.4f}",
            cvar_99=f"{cvar_99:.4f}",
            median_return=f"{median_return:.4f}",
            worst_path_return=f"{worst_path_return:.4f}",
            survival_rate=f"{survival_rate:.2%}",
        )

        return result

    # ------------------------------------------------------------------
    # Reverse stress testing
    # ------------------------------------------------------------------

    def reverse_stress_test(
        self,
        portfolio_value: float,
        current_position_pct: float = 0.10,
        target_loss_pct: float = 0.20,
    ) -> ReverseStressResult:
        """Find the minimum price shock that causes a specified portfolio loss.

        Uses binary search (bisection) to find the smallest adverse price
        movement that, given the current position size, results in a loss
        equal to or greater than the target loss percentage.

        If historical returns have been provided (via a prior call to
        ``monte_carlo_stress``), the probability of such a shock is estimated
        from a Student-t CDF fit.

        Args:
            portfolio_value: Current total portfolio value in account currency.
            current_position_pct: Current net exposure as a fraction of
                portfolio value.
            target_loss_pct: Target loss as a positive decimal fraction
                (e.g. 0.20 for a 20% portfolio loss).

        Returns:
            ReverseStressResult with the minimum shock and probability estimate.

        Raises:
            ValueError: If portfolio_value is non-positive or target_loss_pct
                is not in (0, 1].
        """
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive, got {portfolio_value}"
            )
        if not (0 < target_loss_pct <= 1.0):
            raise ValueError(
                f"target_loss_pct must be in (0, 1], got {target_loss_pct}"
            )

        target_loss = portfolio_value * target_loss_pct

        # Binary search for minimum shock
        lo = 0.0
        hi = 1.0
        n_iterations = 100

        for _ in range(n_iterations):
            mid = (lo + hi) / 2.0
            # Shock is negative (adverse), loss magnitude from position:
            loss = portfolio_value * current_position_pct * mid
            if loss >= target_loss:
                hi = mid
            else:
                lo = mid

        min_shock_pct = hi

        # Probability estimate from historical return distribution
        probability_estimate: float | None = None
        if self._historical_returns is not None and len(self._historical_returns) >= 30:
            df, loc, scale = stats.t.fit(self._historical_returns)
            # Probability of a single-day move worse than -min_shock_pct
            probability_estimate = float(stats.t.cdf(-min_shock_pct, df, loc, scale))

        scenario_description = (
            f"Minimum {min_shock_pct:.2%} adverse price shock required to cause "
            f"{target_loss_pct:.2%} portfolio loss with "
            f"{current_position_pct:.2%} position exposure"
        )

        result = ReverseStressResult(
            target_loss_pct=target_loss_pct,
            min_shock_pct=min_shock_pct,
            scenario_description=scenario_description,
            probability_estimate=probability_estimate,
        )

        logger.info(
            "Reverse stress test complete",
            target_loss_pct=f"{target_loss_pct:.2%}",
            min_shock_pct=f"{min_shock_pct:.4%}",
            position_pct=f"{current_position_pct:.2%}",
            probability_estimate=(
                f"{probability_estimate:.6f}" if probability_estimate is not None
                else "unavailable"
            ),
        )

        return result
