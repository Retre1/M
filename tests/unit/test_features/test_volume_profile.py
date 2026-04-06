"""Tests for Volume Profile extractor."""

import numpy as np
import pandas as pd

from apexfx.features.volume_profile import VolumeProfileExtractor


class TestVolumeProfileExtractor:
    def test_feature_names(self):
        ext = VolumeProfileExtractor()
        assert ext.feature_names == [
            "hvn_distance", "lvn_distance", "volume_profile_skew",
            "poc_price", "poc_distance",
        ]

    def test_output_shape(self, sample_bars):
        ext = VolumeProfileExtractor(window=50)
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)
        for col in ext.feature_names:
            assert col in result.columns

    def test_warmup_is_nan(self, sample_bars):
        window = 50
        ext = VolumeProfileExtractor(window=window)
        result = ext.extract(sample_bars)
        assert result["poc_price"].iloc[:window].isna().all()
        assert result["poc_price"].iloc[window:].notna().any()

    def test_poc_is_valid_price(self, sample_bars):
        ext = VolumeProfileExtractor(window=50)
        result = ext.extract(sample_bars)
        valid_poc = result["poc_price"].dropna()
        assert len(valid_poc) > 0
        # POC should be within the price range
        price_min = sample_bars["low"].min()
        price_max = sample_bars["high"].max()
        assert (valid_poc >= price_min * 0.99).all()
        assert (valid_poc <= price_max * 1.01).all()

    def test_zero_volume(self):
        """Should handle zero volume gracefully."""
        np.random.seed(42)
        n = 200
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices, "high": prices + 0.001,
            "low": prices - 0.001, "close": prices,
            "volume": np.zeros(n),
            "tick_count": np.ones(n, dtype=int),
        })
        ext = VolumeProfileExtractor(window=50)
        result = ext.extract(bars)
        assert len(result) == n
