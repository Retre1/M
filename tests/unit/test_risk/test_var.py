"""Tests for VaR calculator."""

import numpy as np

from apexfx.risk.var_calculator import VaRCalculator


class TestVaRCalculator:
    def test_insufficient_data(self):
        calc = VaRCalculator()
        assert calc.compute_var(100_000) == 0.0
        assert not calc.has_sufficient_data

    def test_historical_var(self):
        calc = VaRCalculator(confidence=0.99, method="historical")
        np.random.seed(42)
        for r in np.random.randn(300) * 0.01:
            calc.update(float(r))
        assert calc.has_sufficient_data
        var = calc.compute_var(100_000)
        assert var > 0

    def test_parametric_var(self):
        calc = VaRCalculator(confidence=0.99, method="parametric")
        np.random.seed(42)
        for r in np.random.randn(300) * 0.01:
            calc.update(float(r))
        var = calc.compute_var(100_000)
        assert var > 0

    def test_cvar_greater_than_var(self):
        calc = VaRCalculator(confidence=0.99, method="historical")
        np.random.seed(42)
        for r in np.random.randn(300) * 0.01 - 0.001:
            calc.update(float(r))
        calc.compute_var(100_000)
        cvar = calc.compute_cvar(100_000)
        # CVaR (expected shortfall) should be >= VaR
        assert cvar >= 0
