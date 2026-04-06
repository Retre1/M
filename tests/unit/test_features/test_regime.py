"""Tests for Regime extractor."""

import numpy as np
import pandas as pd

from apexfx.features.regime import RegimeExtractor


class TestRegimeExtractor:
    def test_feature_names(self):
        ext = RegimeExtractor()
        names = ext.feature_names
        assert "realized_vol" in names
        assert "trend_strength" in names
        assert "regime_trending" in names
        assert "regime_mean_reverting" in names
        assert "regime_flat" in names
        assert "regime_label" in names

    def test_output_shape(self, sample_bars):
        ext = RegimeExtractor()
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)

    def test_regime_one_hot(self, sample_bars):
        """Exactly one regime flag should be 1 for each bar."""
        ext = RegimeExtractor()
        result = ext.extract(sample_bars)
        regime_sum = (
            result["regime_trending"]
            + result["regime_mean_reverting"]
            + result["regime_flat"]
        )
        assert (regime_sum == 1.0).all()

    def test_regime_label_consistency(self, sample_bars):
        ext = RegimeExtractor()
        result = ext.extract(sample_bars)

        trending_mask = result["regime_trending"] == 1.0
        reverting_mask = result["regime_mean_reverting"] == 1.0
        flat_mask = result["regime_flat"] == 1.0

        assert (result.loc[trending_mask, "regime_label"] == 2.0).all()
        assert (result.loc[reverting_mask, "regime_label"] == 0.0).all()
        assert (result.loc[flat_mask, "regime_label"] == 1.0).all()

    def test_adx_values_positive(self, sample_bars):
        ext = RegimeExtractor()
        result = ext.extract(sample_bars)
        valid_adx = result["trend_strength"].dropna()
        assert len(valid_adx) > 0
        assert (valid_adx >= 0).all()

    def test_uses_hurst_if_available(self):
        """If hurst_exponent column exists, regime should use it."""
        np.random.seed(42)
        n = 200
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices, "high": prices + 0.001,
            "low": prices - 0.001, "close": prices,
            "volume": np.ones(n) * 1000,
            "tick_count": np.ones(n, dtype=int) * 100,
            "hurst_exponent": np.full(n, 0.3),  # strongly mean-reverting
        })
        ext = RegimeExtractor(trend_threshold=25.0)
        result = ext.extract(bars)
        # With H=0.3 everywhere, should be mean-reverting
        assert (result["regime_mean_reverting"] == 1.0).all()
