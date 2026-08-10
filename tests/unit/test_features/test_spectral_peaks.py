"""FFT features must report distinct cycles at their true amplitude.

Two defects the sine-wave check found:

* Taking the N largest *bins* returned the same peak several times, because a
  Hann window spreads each component across its neighbours. On a mixture of a
  period-20 and a period-8 component the result was 19.69 / 21.33 / 18.29 —
  one cycle three times, and the period-8 component missing entirely.
* Amplitudes read exactly half their true value: a Hann window has a coherent
  gain of 0.5, which was never divided back out.
"""

from __future__ import annotations

import numpy as np
import pytest

from apexfx.features.spectral import SpectralExtractor

N = 256
T = np.arange(N)


@pytest.fixture
def extractor() -> SpectralExtractor:
    return SpectralExtractor()


def _sine(period: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * T / period)


class TestPeriodDetection:
    @pytest.mark.parametrize("period", [8, 16, 32, 64])
    def test_exact_bin_periods_are_recovered(self, extractor, period):
        """Periods that land on an FFT bin must come back exactly."""
        periods, _ = extractor._compute_fft(_sine(period))
        assert periods[0] == pytest.approx(period, rel=1e-6)

    def test_off_bin_period_is_close(self, extractor):
        """256/20 is not an integer bin, so a small error is inherent."""
        periods, _ = extractor._compute_fft(_sine(20))
        assert periods[0] == pytest.approx(20, rel=0.02)


class TestAmplitudeScale:
    @pytest.mark.parametrize("amplitude", [0.5, 1.0, 2.5])
    def test_amplitude_is_not_halved_by_the_window(self, extractor, amplitude):
        _, amps = extractor._compute_fft(_sine(16, amplitude))
        assert amps[0] == pytest.approx(amplitude, rel=0.02)

    def test_half_amplitude_would_fail(self, extractor):
        """Guard the guard: the pre-fix value must be outside tolerance."""
        _, amps = extractor._compute_fft(_sine(16, 1.0))
        assert amps[0] != pytest.approx(0.5, rel=0.02)


class TestPeaksAreDistinct:
    def test_a_two_cycle_mixture_reports_both(self, extractor):
        """The case the old top-N-bins version could not express."""
        signal = _sine(20, 1.0) + _sine(8, 0.3)
        periods, amps = extractor._compute_fft(signal)

        assert periods[0] == pytest.approx(20, rel=0.02)
        assert amps[0] == pytest.approx(1.0, rel=0.05)
        assert periods[1] == pytest.approx(8, rel=0.02)
        assert amps[1] == pytest.approx(0.3, rel=0.05)

    def test_reported_periods_are_not_neighbours(self, extractor):
        """Adjacent bins around one peak are leakage, not separate cycles."""
        periods, _ = extractor._compute_fft(_sine(20, 1.0) + _sine(8, 0.3))
        top_two = sorted(periods[:2])
        assert top_two[1] / top_two[0] > 1.5, (
            f"periods {periods[:2]} sit on the same peak"
        )

    def test_three_cycles_are_all_found(self, extractor):
        signal = _sine(64, 1.0) + _sine(16, 0.6) + _sine(6, 0.4)
        periods, _ = extractor._compute_fft(signal)
        found = sorted(periods[:3])
        assert found[0] == pytest.approx(6, rel=0.05)
        assert found[1] == pytest.approx(16, rel=0.05)
        assert found[2] == pytest.approx(64, rel=0.05)


class TestDegenerateInput:
    def test_flat_signal_does_not_raise(self, extractor):
        periods, amps = extractor._compute_fft(np.zeros(N))
        assert len(periods) == len(amps)

    def test_pure_noise_returns_finite_values(self, extractor):
        rng = np.random.default_rng(0)
        periods, amps = extractor._compute_fft(rng.normal(0, 1, N))
        assert np.all(np.isfinite(periods))
        assert np.all(np.isfinite(amps))

    def test_short_signal_is_handled(self, extractor):
        periods, amps = extractor._compute_fft(np.sin(np.arange(8)))
        assert len(periods) == len(amps)
