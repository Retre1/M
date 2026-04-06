"""Tests for Order Flow extractor."""

import numpy as np
import pandas as pd

from apexfx.features.order_flow import OrderFlowExtractor


class TestOrderFlowExtractor:
    def test_feature_names(self):
        ext = OrderFlowExtractor(delta_windows=[10, 50])
        names = ext.feature_names
        assert "delta" in names
        assert "delta_pct" in names
        assert "cumulative_delta" in names
        assert "delta_divergence" in names
        assert "delta_ma_10" in names
        assert "delta_pct_ma_50" in names

    def test_estimate_from_bars(self, sample_bars):
        ext = OrderFlowExtractor()
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)
        for col in ["delta", "delta_pct", "cumulative_delta", "delta_divergence"]:
            assert col in result.columns

    def test_delta_sign(self):
        """Bullish bars should have positive delta estimate."""
        np.random.seed(42)
        n = 100
        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": np.ones(n) * 1.0,
            "high": np.ones(n) * 1.05,
            "low": np.ones(n) * 0.99,
            "close": np.ones(n) * 1.04,  # close near high -> bullish
            "volume": np.ones(n) * 1000,
            "tick_count": np.ones(n, dtype=int) * 100,
        })
        ext = OrderFlowExtractor()
        result = ext.extract(bars)
        # Most deltas should be positive (close is near high)
        assert (result["delta"] > 0).mean() > 0.9

    def test_divergence_values(self, sample_bars):
        ext = OrderFlowExtractor()
        result = ext.extract(sample_bars)
        divs = result["delta_divergence"].unique()
        # Should only be -1, 0, or 1
        assert all(d in {-1.0, 0.0, 1.0} for d in divs)

    def test_cumulative_delta_is_cumsum(self, sample_bars):
        ext = OrderFlowExtractor()
        result = ext.extract(sample_bars)
        np.testing.assert_allclose(
            result["cumulative_delta"].values,
            np.cumsum(result["delta"].values),
            atol=1e-10,
        )
