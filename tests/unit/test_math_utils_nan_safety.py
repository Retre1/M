"""NaN safety tests for volatility estimators in math_utils.

Regression tests for AUDIT_REPORT.md finding #3: Garman-Klass and Parkinson
estimators could produce NaN under sqrt for edge-case bar sequences,
silently poisoning the feature pipeline (observed in
``train_EURUSD_20260418_113833.log``).
"""

from __future__ import annotations

import numpy as np

from apexfx.utils.math_utils import garman_klass_volatility, parkinson_volatility


def _flat_bars(n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constant-price bars — an edge case that historically produced NaN."""
    price = 1.1000
    arr = np.full(n, price)
    return arr.copy(), arr.copy(), arr.copy(), arr.copy()


def _gapped_bars(n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bars where open-to-close move dominates high-low range — GK can go negative."""
    rng = np.random.RandomState(0)
    open_ = 1.1 + 0.0001 * rng.randn(n)
    close = open_ + 0.01  # huge jump vs tiny hl
    high = np.maximum(open_, close) + 1e-6
    low = np.minimum(open_, close) - 1e-6
    return open_, high, low, close


def _nan_input() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 30
    open_ = np.linspace(1.0, 1.1, n)
    high = open_ + 1e-4
    low = open_ - 1e-4
    close = open_.copy()
    # Inject a NaN bar
    high[10] = np.nan
    return open_, high, low, close


class TestGarmanKlass:
    def test_flat_bars_no_nan(self) -> None:
        open_, high, low, close = _flat_bars()
        out = garman_klass_volatility(open_, high, low, close, window=20)
        # Warm-up region is NaN by design; computed region must be finite
        computed = out[20:]
        assert np.all(np.isfinite(computed)), "flat bars should never produce NaN"
        assert np.all(computed >= 0.0), "volatility is non-negative"

    def test_gapped_bars_no_nan(self) -> None:
        """The exact pattern that triggered RuntimeWarning in production logs."""
        open_, high, low, close = _gapped_bars()
        out = garman_klass_volatility(open_, high, low, close, window=20)
        computed = out[20:]
        assert np.all(np.isfinite(computed))

    def test_nan_input_bar_does_not_poison_following_bars(self) -> None:
        open_, high, low, close = _nan_input()
        out = garman_klass_volatility(open_, high, low, close, window=5)
        # Window 11..14 spans the NaN bar — those rows are allowed to be NaN.
        # But bars well past the NaN must recover.
        assert np.isfinite(out[-1]), "downstream bars must recover from a single NaN input"


class TestParkinson:
    def test_flat_bars_no_nan(self) -> None:
        _, high, low, _ = _flat_bars()
        out = parkinson_volatility(high, low, window=10)
        computed = out[10:]
        assert np.all(np.isfinite(computed))
        assert np.all(computed >= 0.0)

    def test_result_non_negative(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        high = 1.1 + 0.001 * np.abs(rng.randn(n))
        low = 1.1 - 0.001 * np.abs(rng.randn(n))
        out = parkinson_volatility(high, low, window=20)
        computed = out[20:]
        assert np.all(np.isfinite(computed))
        assert np.all(computed >= 0.0)
