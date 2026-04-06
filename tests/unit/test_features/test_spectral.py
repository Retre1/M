"""Tests for Spectral (FFT + Wavelet) extractor."""

import numpy as np
import pandas as pd
import pytest

from apexfx.features.spectral import SpectralExtractor


def _make_bars(prices: np.ndarray) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.0005),
        "low": prices - np.abs(np.random.randn(n) * 0.0005),
        "close": prices,
        "volume": np.random.lognormal(10, 1, n),
        "tick_count": np.random.randint(50, 500, n),
    })


class TestSpectralExtractor:
    def test_feature_names(self):
        ext = SpectralExtractor(top_n_cycles=3, wavelet_level=4)
        names = ext.feature_names
        assert "fft_period_1" in names
        assert "fft_amplitude_1" in names
        assert "fft_dominant_period" in names
        assert "wavelet_energy_d1" in names
        assert "wavelet_energy_approx" in names
        assert "wavelet_trend" in names

    def test_output_shape(self, sample_bars):
        ext = SpectralExtractor(fft_window=64)
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)
        assert set(ext.feature_names).issubset(set(result.columns))

    def test_dominant_period_detects_cycle(self):
        """A sinusoidal price series should detect the correct cycle period."""
        np.random.seed(42)
        n = 600
        period = 50
        t = np.arange(n, dtype=float)
        prices = 1.1 + 0.01 * np.sin(2 * np.pi * t / period) + np.random.randn(n) * 0.0001
        bars = _make_bars(prices)

        ext = SpectralExtractor(fft_window=256, top_n_cycles=3)
        result = ext.extract(bars)

        valid = result["fft_dominant_period"].dropna()
        assert len(valid) > 0
        # The dominant period should be close to 50
        median_period = valid.median()
        assert 30 < median_period < 80, f"Expected ~50, got {median_period:.1f}"

    def test_warmup_is_nan(self):
        np.random.seed(42)
        n = 300
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = _make_bars(prices)

        fft_window = 64
        ext = SpectralExtractor(fft_window=fft_window)
        result = ext.extract(bars)
        # First fft_window bars should be NaN for FFT features
        assert result["fft_dominant_period"].iloc[:fft_window].isna().all()

    def test_short_data(self):
        """Data shorter than FFT window should return all NaN."""
        np.random.seed(42)
        n = 10
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = _make_bars(prices)

        ext = SpectralExtractor(fft_window=256)
        result = ext.extract(bars)
        assert len(result) == n
        assert result["fft_dominant_period"].isna().all()
