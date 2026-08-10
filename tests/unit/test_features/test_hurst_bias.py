"""The Hurst estimator must land where theory says, not just rank correctly.

`regime.py` classifies on absolute thresholds — ``h < 0.45`` is mean-reverting,
``h > 0.55`` is trending — and those labels reach the model as ``regime_label``
and ``hurst_regime``, which the pipeline treats as already-normalised. Unlike
every other feature, a bias here is *not* absorbed by the z-score.

The previous uncorrected R/S read 0.764 on white noise and 0.557 on a
genuinely anti-persistent series, so the mean-reverting class could never fire.
These tests fail if that bias returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from apexfx.features.hurst import HurstExtractor

WINDOW = 512
N = 4096
STEP = 256


@pytest.fixture(scope="module")
def series() -> dict[str, np.ndarray]:
    """White noise plus AR(1) processes on either side of it."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, N)
    persistent = np.zeros(N)
    anti = np.zeros(N)
    for i in range(1, N):
        persistent[i] = 0.7 * persistent[i - 1] + noise[i]
        anti[i] = -0.7 * anti[i - 1] + noise[i]
    return {"noise": noise, "persistent": persistent, "anti_persistent": anti}


def _mean_h(data: np.ndarray) -> float:
    extractor = HurstExtractor()
    return float(np.mean([
        extractor._compute_hurst(data[i - WINDOW:i]) for i in range(WINDOW, N, STEP)
    ]))


class TestEstimatorIsUnbiased:
    def test_white_noise_lands_near_one_half(self, series):
        """True H is 0.5. The uncorrected estimator gave 0.764."""
        assert _mean_h(series["noise"]) == pytest.approx(0.5, abs=0.1)

    def test_persistent_series_reads_above_the_trend_threshold(self, series):
        assert _mean_h(series["persistent"]) > 0.55

    def test_anti_persistent_series_reads_below_the_reversion_threshold(self, series):
        """The case the old estimator could not express — it returned 0.557."""
        assert _mean_h(series["anti_persistent"]) < 0.45

    def test_the_three_regimes_are_ordered(self, series):
        anti = _mean_h(series["anti_persistent"])
        noise = _mean_h(series["noise"])
        persistent = _mean_h(series["persistent"])
        assert anti < noise < persistent


class TestRegimeClassesAreReachable:
    """Every branch of regime.py must be attainable by some real input."""

    @staticmethod
    def _regime(h: float) -> int:
        extractor = HurstExtractor()
        if h > extractor._trend_threshold:
            return 2
        if h < extractor._reversion_threshold:
            return 0
        return 1

    def test_mean_reverting_class_can_fire(self, series):
        assert self._regime(_mean_h(series["anti_persistent"])) == 0

    def test_trending_class_can_fire(self, series):
        assert self._regime(_mean_h(series["persistent"])) == 2


class TestDegenerateWindows:
    def test_short_window_returns_neutral(self):
        extractor = HurstExtractor()
        assert extractor._compute_hurst(np.zeros(5)) == 0.5

    def test_constant_series_does_not_raise(self):
        extractor = HurstExtractor()
        value = extractor._compute_hurst(np.full(512, 0.001))
        assert np.isfinite(value)

    def test_result_stays_in_the_unit_interval(self, series):
        extractor = HurstExtractor()
        for data in series.values():
            for i in range(WINDOW, N, STEP * 4):
                assert 0.0 <= extractor._compute_hurst(data[i - WINDOW:i]) <= 1.0
