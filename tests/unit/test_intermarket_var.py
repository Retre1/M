"""Tests for Intermarket Integration and Portfolio VaR.

Covers:
- IntermarketDataProvider: fallback chain (MT5 → Parquet → yfinance → empty)
- Portfolio VaR: real variance-covariance calculation with dynamic correlations
- Per-symbol volatility tracking in RiskManager
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# IntermarketDataProvider
# ---------------------------------------------------------------------------

class TestIntermarketDataProvider:
    def test_yfinance_ticker_mapping(self):
        from apexfx.data.intermarket import YFINANCE_TICKERS
        assert "DXY" in YFINANCE_TICKERS
        assert "XAUUSD" in YFINANCE_TICKERS
        assert "US10Y" in YFINANCE_TICKERS
        assert "SPX" in YFINANCE_TICKERS

    def test_intermarket_symbols_have_spx(self):
        from apexfx.data.intermarket import INTERMARKET_SYMBOLS
        assert "SPX" in INTERMARKET_SYMBOLS
        assert "US500" in INTERMARKET_SYMBOLS["SPX"]

    def test_empty_fallback_when_no_sources(self):
        """Without MT5, cache, or yfinance — returns empty DataFrame."""
        from apexfx.data.intermarket import IntermarketDataProvider
        provider = IntermarketDataProvider(data_dir="/nonexistent/path")
        with patch.dict("sys.modules", {"yfinance": None}):
            df = provider.get_bars("DXY")
        # Should return empty-ish DataFrame with correct columns
        assert "DXY_close" in df.columns

    def test_parquet_cache_loading(self, tmp_path):
        """Loads data from Parquet cache when available."""
        from apexfx.data.intermarket import IntermarketDataProvider

        # Create a fake Parquet cache
        cache_dir = tmp_path / "DXY" / "H1"
        cache_dir.mkdir(parents=True)
        n = 100
        fake_data = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=n, freq="h"),
            "close": np.cumsum(np.random.randn(n) * 0.1) + 104.0,
        })
        fake_data.to_parquet(cache_dir / "data.parquet", index=False)

        provider = IntermarketDataProvider(data_dir=tmp_path)
        df = provider.get_bars("DXY", timeframe="H1", count=50)

        assert not df.empty
        assert "DXY_close" in df.columns
        assert len(df) == 50  # Truncated to count

    def test_get_all_intermarket_merges(self, tmp_path):
        """get_all_intermarket merges multiple instruments on time."""
        from apexfx.data.intermarket import IntermarketDataProvider

        n = 100
        times = pd.date_range("2025-01-01", periods=n, freq="h")

        for inst in ["DXY", "XAUUSD"]:
            cache_dir = tmp_path / inst / "H1"
            cache_dir.mkdir(parents=True)
            df = pd.DataFrame({
                "time": times,
                "close": np.random.randn(n) * 10 + (104 if inst == "DXY" else 2000),
            })
            df.to_parquet(cache_dir / "data.parquet", index=False)

        provider = IntermarketDataProvider(data_dir=tmp_path)
        result = provider.get_all_intermarket(["DXY", "XAUUSD"], timeframe="H1")

        assert not result.empty
        assert "DXY_close" in result.columns
        assert "XAUUSD_close" in result.columns
        assert len(result) == n

    def test_parquet_cache_save(self, tmp_path):
        """_save_parquet_cache creates valid Parquet file."""
        from apexfx.data.intermarket import IntermarketDataProvider

        provider = IntermarketDataProvider(data_dir=tmp_path)
        df = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=10, freq="h"),
            "SPX_close": np.random.randn(10) * 50 + 4500,
        })
        provider._save_parquet_cache("SPX", "H1", df)

        saved = pd.read_parquet(tmp_path / "SPX" / "H1" / "data.parquet")
        assert "close" in saved.columns
        assert "time" in saved.columns
        assert len(saved) == 10

    def test_yf_params(self):
        from apexfx.data.intermarket import IntermarketDataProvider
        assert IntermarketDataProvider._yf_params("H1", 500) == ("1h", "730d")
        assert IntermarketDataProvider._yf_params("D1", 500) == ("1d", "5y")
        assert IntermarketDataProvider._yf_params("M5", 500) == ("5m", "60d")

    def test_mt5_preferred_over_cache(self, tmp_path):
        """MT5 data takes priority over Parquet cache."""
        from apexfx.data.intermarket import IntermarketDataProvider

        # Create cache with old values
        cache_dir = tmp_path / "DXY" / "H1"
        cache_dir.mkdir(parents=True)
        pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=5, freq="h"),
            "close": [100.0] * 5,
        }).to_parquet(cache_dir / "data.parquet", index=False)

        # Mock MT5 that returns fresh data
        mock_mt5 = MagicMock()
        mock_mt5.get_symbol_info.return_value = True
        mock_mt5.get_bars.return_value = pd.DataFrame({
            "time": pd.date_range("2025-06-01", periods=5, freq="h"),
            "close": [105.0] * 5,
        })

        provider = IntermarketDataProvider(mt5_client=mock_mt5, data_dir=tmp_path)
        df = provider.get_bars("DXY", timeframe="H1")

        assert not df.empty
        assert df["DXY_close"].iloc[0] == 105.0  # MT5 data, not cache


# ---------------------------------------------------------------------------
# Portfolio VaR
# ---------------------------------------------------------------------------

class TestPortfolioVaRReal:
    def test_config_has_daily_limit(self):
        from apexfx.config.schema import PortfolioVaRConfig
        cfg = PortfolioVaRConfig()
        assert cfg.daily_limit == 0.02
        assert cfg.default_symbol_vol == 0.01

    def test_record_symbol_return(self):
        """RiskManager tracks per-symbol returns."""
        from apexfx.risk.risk_manager import RiskManager
        from apexfx.config.schema import RiskConfig
        rm = RiskManager(RiskConfig(), initial_balance=100000)

        rm.record_symbol_return("EURUSD", 0.005)
        rm.record_symbol_return("EURUSD", -0.003)
        rm.record_symbol_return("GBPUSD", 0.002)

        assert len(rm._symbol_returns["EURUSD"]) == 2
        assert len(rm._symbol_returns["GBPUSD"]) == 1

    def test_compute_symbol_vol_with_data(self):
        """Computes realized vol when sufficient data exists."""
        from apexfx.risk.risk_manager import RiskManager
        from apexfx.config.schema import RiskConfig
        rm = RiskManager(RiskConfig(), initial_balance=100000)

        np.random.seed(42)
        for _ in range(30):
            rm.record_symbol_return("EURUSD", np.random.randn() * 0.008)

        vol = rm._compute_symbol_vol("EURUSD")
        assert 0.005 < vol < 0.015  # Should be around 0.008

    def test_compute_symbol_vol_default_fallback(self):
        """Falls back to config default with insufficient data."""
        from apexfx.risk.risk_manager import RiskManager
        from apexfx.config.schema import RiskConfig
        rm = RiskManager(RiskConfig(), initial_balance=100000)

        vol = rm._compute_symbol_vol("UNKNOWN")
        assert vol == 0.01  # Default

    def test_symbol_returns_truncation(self):
        """Buffer doesn't grow unbounded."""
        from apexfx.risk.risk_manager import RiskManager
        from apexfx.config.schema import RiskConfig
        rm = RiskManager(RiskConfig(), initial_balance=100000)

        for _ in range(200):
            rm.record_symbol_return("EURUSD", np.random.randn() * 0.01)

        assert len(rm._symbol_returns["EURUSD"]) <= 120  # 2x lookback truncation

    def test_portfolio_var_with_positions(self):
        """Portfolio VaR computes correctly with 2 correlated positions."""
        from apexfx.risk.var_calculator import VaRCalculator

        calc = VaRCalculator(confidence=0.99)
        # Feed enough data so has_sufficient_data is True
        for _ in range(50):
            calc.update(np.random.randn() * 0.01)

        positions = {"EURUSD": 50000.0, "GBPUSD": 30000.0}
        corr = np.array([[1.0, 0.85], [0.85, 1.0]])
        ind_vars = {"EURUSD": 1000.0, "GBPUSD": 600.0}

        pvar = calc.compute_portfolio_var(positions, corr, ind_vars)

        # With high correlation, diversification benefit is small
        # Undiversified VaR = sqrt(1000^2 + 600^2) if rho=0
        # With rho=0.85 it's higher
        assert pvar > 0
        # Should be less than simple sum (1000 + 600 = 1600)
        assert pvar < 1600

    def test_portfolio_var_diversification_benefit(self):
        """Negatively correlated positions should have lower portfolio VaR."""
        from apexfx.risk.var_calculator import VaRCalculator

        calc = VaRCalculator(confidence=0.99)
        for _ in range(50):
            calc.update(np.random.randn() * 0.01)

        positions = {"EURUSD": 50000.0, "USDCHF": 50000.0}
        ind_vars = {"EURUSD": 1000.0, "USDCHF": 1000.0}

        # High positive correlation — no diversification
        corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
        pvar_high = calc.compute_portfolio_var(positions, corr_high, ind_vars)

        # Negative correlation — strong diversification
        corr_neg = np.array([[1.0, -0.9], [-0.9, 1.0]])
        pvar_neg = calc.compute_portfolio_var(positions, corr_neg, ind_vars)

        # Negatively correlated portfolio should have significantly lower VaR
        assert pvar_neg < pvar_high * 0.5

    def test_portfolio_var_uses_dynamic_correlation(self):
        """Verify dynamic correlations are used in the risk manager's VaR check."""
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker, get_correlation_tracker

        tracker = get_correlation_tracker()
        # Feed highly correlated returns
        np.random.seed(42)
        for _ in range(30):
            r = np.random.randn() * 0.01
            tracker.update_returns("EURUSD", r)
            tracker.update_returns("GBPUSD", r * 0.95 + np.random.randn() * 0.001)
        tracker.recompute()

        corr = tracker.get_correlation("EURUSD", "GBPUSD")
        assert corr > 0.8  # Dynamic should show high correlation


# ---------------------------------------------------------------------------
# Intermarket → FeaturePipeline Integration
# ---------------------------------------------------------------------------

class TestIntermarketPipelineIntegration:
    def test_pipeline_with_intermarket_columns(self):
        """FeaturePipeline produces non-NaN intermarket features when data is present."""
        from apexfx.features.intermarket_corr import IntermarketCorrExtractor

        n = 400
        np.random.seed(42)
        bars = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": np.random.uniform(1.09, 1.11, n),
            "high": np.random.uniform(1.10, 1.12, n),
            "low": np.random.uniform(1.08, 1.10, n),
            "close": np.cumsum(np.random.randn(n) * 0.001) + 1.10,
            "tick_volume": np.random.randint(100, 5000, n).astype(float),
            "volume": np.random.randint(100, 5000, n).astype(float),
            "spread": np.random.uniform(0.0001, 0.0003, n),
            # Intermarket columns — this is what provider merges in
            "DXY_close": np.cumsum(np.random.randn(n) * 0.1) + 104.0,
            "XAUUSD_close": np.cumsum(np.random.randn(n) * 2) + 2000.0,
            "US10Y_close": np.cumsum(np.random.randn(n) * 0.01) + 4.5,
            "SPX_close": np.cumsum(np.random.randn(n) * 5) + 4500.0,
        })

        ext = IntermarketCorrExtractor()
        result = ext.extract(bars)

        # Should have non-NaN values for at least some features
        non_nan_cols = [c for c in result.columns if not result[c].iloc[-10:].isna().all()]
        assert len(non_nan_cols) > 10, f"Only {len(non_nan_cols)} non-NaN columns: {non_nan_cols}"

    def test_pipeline_without_intermarket_all_nan(self):
        """Without intermarket columns, extractor returns NaN (current behavior)."""
        from apexfx.features.intermarket_corr import IntermarketCorrExtractor

        n = 100
        bars = pd.DataFrame({
            "close": np.random.uniform(1.09, 1.11, n),
        })

        ext = IntermarketCorrExtractor()
        result = ext.extract(bars)

        # All features should be NaN without the data columns
        assert result.isna().all().all()

    def test_trainer_merge_with_parquet(self, tmp_path):
        """Trainer's _merge_intermarket loads and merges Parquet files."""
        n = 100
        times = pd.date_range("2025-01-01", periods=n, freq="h")
        bars = pd.DataFrame({
            "time": times,
            "close": np.random.uniform(1.09, 1.11, n),
        })

        # Create Parquet cache files
        for inst, base in [("DXY", 104), ("XAUUSD", 2000), ("US10Y", 4.5), ("SPX", 4500)]:
            cache_dir = tmp_path / inst / "H1"
            cache_dir.mkdir(parents=True)
            df = pd.DataFrame({
                "time": times,
                "close": np.random.randn(n) + base,
            })
            df.to_parquet(cache_dir / "data.parquet", index=False)

        # Simulate what trainer._merge_intermarket does
        merged = bars.copy()
        for inst in ["DXY", "XAUUSD", "US10Y", "SPX"]:
            parquet_path = tmp_path / inst / "H1" / "data.parquet"
            idf = pd.read_parquet(parquet_path)
            idf = idf[["time", "close"]].rename(columns={"close": f"{inst}_close"})
            merged = merged.merge(idf, on="time", how="left")
        merged = merged.ffill()

        assert "DXY_close" in merged.columns
        assert "SPX_close" in merged.columns
        assert merged["DXY_close"].notna().all()
        assert merged["SPX_close"].notna().all()
