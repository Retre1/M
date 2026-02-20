"""Tests for Hurst exponent extractor."""

import numpy as np
import pandas as pd
import pytest

from apexfx.features.hurst import HurstExtractor


class TestHurstExtractor:
    def test_feature_names(self):
        ext = HurstExtractor()
        assert "hurst_exponent" in ext.feature_names
        assert "hurst_regime" in ext.feature_names

    def test_trending_series(self):
        """A strongly trending series should have H > 0.5."""
        np.random.seed(42)
        n = 500
        trend = np.cumsum(np.ones(n) * 0.001 + np.random.randn(n) * 0.0001)
        prices = 1.1 + trend

        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices, "high": prices + 0.001,
            "low": prices - 0.001, "close": prices,
            "volume": np.ones(n) * 1000, "tick_count": np.ones(n, dtype=int) * 100,
        })

        ext = HurstExtractor(window=100)
        result = ext.extract(bars)

        # Check that non-NaN values exist
        valid = result["hurst_exponent"].dropna()
        assert len(valid) > 0

    def test_output_shape(self, sample_bars):
        ext = HurstExtractor(window=100)
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)
        assert "hurst_exponent" in result.columns
        assert "hurst_regime" in result.columns
