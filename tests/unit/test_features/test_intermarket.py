"""Tests for Intermarket Correlation extractor."""

import numpy as np
import pandas as pd

from apexfx.features.intermarket_corr import IntermarketCorrExtractor


def _make_bars_with_intermarket(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    dxy = 100 + np.cumsum(np.random.randn(n) * 0.01)
    gold = 1900 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices, "high": prices + 0.001,
        "low": prices - 0.001, "close": prices,
        "volume": np.ones(n) * 1000,
        "tick_count": np.ones(n, dtype=int) * 100,
        "DXY_close": dxy,
        "XAUUSD_close": gold,
    })


class TestIntermarketCorrExtractor:
    def test_feature_names(self):
        ext = IntermarketCorrExtractor(instruments=["DXY"], windows=[20])
        assert ext.feature_names == ["corr_DXY_20", "rank_corr_DXY_20", "beta_DXY"]

    def test_output_shape(self):
        bars = _make_bars_with_intermarket()
        ext = IntermarketCorrExtractor(instruments=["DXY", "XAUUSD"], windows=[20, 60])
        result = ext.extract(bars)
        assert len(result) == len(bars)

    def test_missing_columns_graceful(self):
        """Should produce NaN features (not crash) when intermarket columns are missing."""
        np.random.seed(42)
        n = 100
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices, "high": prices + 0.001,
            "low": prices - 0.001, "close": prices,
            "volume": np.ones(n) * 1000,
            "tick_count": np.ones(n, dtype=int) * 100,
        })
        ext = IntermarketCorrExtractor()
        result = ext.extract(bars)
        # All intermarket features should be NaN (no intermarket data)
        for col in ext.feature_names:
            assert result[col].isna().all(), f"{col} should be all NaN"

    def test_correlation_range(self):
        """Correlations should be in [-1, 1]."""
        bars = _make_bars_with_intermarket(500)
        ext = IntermarketCorrExtractor(instruments=["DXY"], windows=[20])
        result = ext.extract(bars)
        corr = result["corr_DXY_20"].dropna()
        assert len(corr) > 0
        assert (corr >= -1.01).all() and (corr <= 1.01).all()

    def test_perfectly_correlated(self):
        """Two identical series should have correlation ~1."""
        np.random.seed(42)
        n = 200
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices, "high": prices + 0.001,
            "low": prices - 0.001, "close": prices,
            "volume": np.ones(n) * 1000,
            "tick_count": np.ones(n, dtype=int) * 100,
            "DXY_close": prices * 90,  # perfectly correlated (scaled)
        })
        ext = IntermarketCorrExtractor(instruments=["DXY"], windows=[20])
        result = ext.extract(bars)
        corr = result["corr_DXY_20"].dropna()
        assert corr.iloc[-1] > 0.95, f"Expected ~1.0, got {corr.iloc[-1]:.3f}"

    def test_all_nan_instrument(self):
        """Should handle all-NaN instrument data gracefully."""
        np.random.seed(42)
        n = 100
        prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
        bars = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
            "open": prices, "high": prices + 0.001,
            "low": prices - 0.001, "close": prices,
            "volume": np.ones(n) * 1000,
            "tick_count": np.ones(n, dtype=int) * 100,
            "DXY_close": np.full(n, np.nan),
        })
        ext = IntermarketCorrExtractor(instruments=["DXY"], windows=[20])
        result = ext.extract(bars)
        assert result["corr_DXY_20"].isna().all()
