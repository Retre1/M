"""Our metric math, checked against a reference implementation.

The project's rules require researching existing work before writing new code
(`.claude/rules/development-workflow.md`, step 0). That step was skipped when
`utils/metrics.py` was written, and running the comparison afterwards found
three divergences against empyrical:

* ``calmar_ratio`` annualised the return arithmetically (``mean * periods``)
  while ``annualized_return`` in the same module compounded it — two different
  "annual returns", and Calmar reading ~73% high.
* ``sortino_ratio`` divided the squared shortfalls by the count of negative
  observations instead of the total, which is a different quantity and read
  ~15% more favourable.
* ``max_drawdown`` returns a positive magnitude where empyrical returns a
  negative one. That is a convention, not an error, and is pinned here so the
  difference is deliberate rather than discovered again.

empyrical is a test-only dependency. Adopting it at runtime is a separate
decision; using it as an oracle costs nothing and would have caught all three.
"""

from __future__ import annotations

import numpy as np
import pytest

from apexfx.utils import metrics as ours

empyrical = pytest.importorskip(
    "empyrical", reason="reference implementation not installed",
)

PERIODS = 252
REL = 1e-9


@pytest.fixture(params=[0, 1, 2], ids=["mild_up", "choppy", "drawdown_heavy"])
def returns(request) -> np.ndarray:
    """Three return series with different shapes, not just one lucky seed."""
    rng = np.random.default_rng(request.param)
    if request.param == 0:
        return rng.normal(0.0004, 0.011, 900)
    if request.param == 1:
        return rng.normal(0.0, 0.02, 600)
    series = rng.normal(-0.0006, 0.015, 700)
    series[200:260] -= 0.01  # a sustained loss stretch
    return series


class TestAgreesWithReference:
    def test_sharpe(self, returns):
        assert ours.sharpe_ratio(returns, periods=PERIODS) == pytest.approx(
            empyrical.sharpe_ratio(returns, annualization=PERIODS), rel=REL,
        )

    def test_sortino(self, returns):
        assert ours.sortino_ratio(returns, periods=PERIODS) == pytest.approx(
            empyrical.sortino_ratio(returns, annualization=PERIODS), rel=REL,
        )

    def test_calmar(self, returns):
        assert ours.calmar_ratio(returns, periods=PERIODS) == pytest.approx(
            empyrical.calmar_ratio(returns, annualization=PERIODS), rel=REL,
        )

    def test_annualized_return(self, returns):
        assert ours.annualized_return(returns, PERIODS) == pytest.approx(
            empyrical.annual_return(returns, annualization=PERIODS), rel=REL,
        )

    def test_annualized_volatility(self, returns):
        assert ours.annualized_volatility(returns, PERIODS) == pytest.approx(
            empyrical.annual_volatility(returns, annualization=PERIODS), rel=REL,
        )


class TestDrawdownSignConvention:
    def test_magnitude_matches_but_sign_is_ours(self, returns):
        theirs = empyrical.max_drawdown(returns)
        assert ours.max_drawdown(returns) == pytest.approx(abs(theirs), rel=REL)

    def test_ours_is_never_negative(self, returns):
        assert ours.max_drawdown(returns) >= 0.0


class TestCalmarUsesGeometricReturn:
    """Pins the specific bug: Calmar must not reintroduce arithmetic scaling."""

    def test_calmar_equals_annualized_return_over_drawdown(self, returns):
        expected = ours.annualized_return(returns, PERIODS) / ours.max_drawdown(returns)
        assert ours.calmar_ratio(returns, periods=PERIODS) == pytest.approx(
            expected, rel=REL,
        )

    def test_arithmetic_scaling_would_disagree(self, returns):
        """Guard the guard: the two annualisations must actually differ here."""
        arithmetic = float(np.mean(returns) * PERIODS)
        geometric = ours.annualized_return(returns, PERIODS)
        assert arithmetic != pytest.approx(geometric, rel=1e-3)


class TestSortinoUsesTotalCount:
    def test_denominator_divides_by_all_observations(self, returns):
        """Downside deviation is over N, not over the count of losses."""
        shortfall = np.minimum(returns, 0.0)
        expected_dd = float(np.sqrt(np.mean(shortfall**2)))
        expected = float(np.mean(returns) / expected_dd * np.sqrt(PERIODS))
        assert ours.sortino_ratio(returns, periods=PERIODS) == pytest.approx(
            expected, rel=REL,
        )

    def test_dividing_by_the_loss_count_would_disagree(self, returns):
        losses = returns[returns < 0]
        naive_dd = float(np.std(losses, ddof=1))
        expected_dd = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        assert naive_dd != pytest.approx(expected_dd, rel=1e-3)
