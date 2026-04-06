"""Tests for Hurst exponent extractor."""

import numpy as np
import pandas as pd
import pytest

from apexfx.features.hurst import HurstExtractor


def _make_bars(prices: np.ndarray) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices,
        "high": prices + 0.001,
        "low": prices - 0.001,
        "close": prices,
        "volume": np.ones(n) * 1000,
        "tick_count": np.ones(n, dtype=int) * 100,
    })


class TestHurstExtractor:
    def test_feature_names(self):
        ext = HurstExtractor()
        assert ext.feature_names == ["hurst_exponent", "hurst_regime"]

    def test_trending_series(self):
        """A strongly trending series should have H > 0.5."""
        np.random.seed(42)
        n = 500
        trend = np.cumsum(np.ones(n) * 0.001 + np.random.randn(n) * 0.0001)
        bars = _make_bars(1.1 + trend)

        ext = HurstExtractor(window=100)
        result = ext.extract(bars)

        valid = result["hurst_exponent"].dropna()
        assert len(valid) > 0
        assert valid.mean() > 0.5, f"Trending series should have H > 0.5, got {valid.mean():.3f}"

    def test_random_walk(self):
        """A pure random walk should have H in reasonable range.

        Note: R/S analysis on small windows with max_lag=20 has positive bias,
        so random walks often show H ~ 0.6-0.8. We test that it's not extreme.
        """
        np.random.seed(123)
        n = 1000
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = _make_bars(prices)

        ext = HurstExtractor(window=200)
        result = ext.extract(bars)

        valid = result["hurst_exponent"].dropna()
        assert len(valid) > 0
        assert 0.3 < valid.mean() < 0.95, f"Random walk H out of range, got {valid.mean():.3f}"

    def test_output_shape(self, sample_bars):
        ext = HurstExtractor(window=100)
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)
        assert list(result.columns) == ["hurst_exponent", "hurst_regime"]

    def test_regime_classification(self):
        """Check regime values are 0, 1, or 2."""
        np.random.seed(42)
        n = 500
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = _make_bars(prices)

        ext = HurstExtractor(window=100)
        result = ext.extract(bars)

        regimes = result["hurst_regime"].dropna().unique()
        assert all(r in {0.0, 1.0, 2.0} for r in regimes)

    def test_warmup_period_is_nan(self):
        """First `window+1` bars should be NaN."""
        np.random.seed(42)
        n = 300
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = _make_bars(prices)

        window = 100
        ext = HurstExtractor(window=window)
        result = ext.extract(bars)

        # All bars up to window+1 should be NaN
        assert result["hurst_exponent"].iloc[:window + 1].isna().all()

    def test_values_in_range(self, sample_bars):
        """Hurst values should be clipped to [0, 1]."""
        ext = HurstExtractor(window=100)
        result = ext.extract(sample_bars)
        valid = result["hurst_exponent"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_short_data(self):
        """Should handle data shorter than window gracefully."""
        prices = np.array([1.1, 1.2, 1.3])
        bars = _make_bars(prices)
        ext = HurstExtractor(window=100)
        result = ext.extract(bars)
        assert len(result) == 3
        assert result["hurst_exponent"].isna().all()
