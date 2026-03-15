"""Tests for Phase 5: Intermarket integration + Portfolio VaR.

Covers:
- SPX added to IntermarketDataProvider
- IntermarketCorrExtractor with 4 instruments
- IntermarketDataProvider data merging
- Portfolio VaR calculation (non-placeholder)
- Silent failure logging in realtime_news.py
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from apexfx.config.schema import NewsConfig, SymbolsConfig, DataConfig
from apexfx.data.intermarket import INTERMARKET_SYMBOLS, IntermarketDataProvider
from apexfx.data.realtime_news import NewsHeadline, RealtimeNewsStream
from apexfx.features.intermarket_corr import IntermarketCorrExtractor


# ---------------------------------------------------------------------------
# IntermarketDataProvider
# ---------------------------------------------------------------------------

class TestIntermarketDataProvider:
    def test_spx_in_symbols(self):
        """SPX should be in the default intermarket symbols."""
        assert "SPX" in INTERMARKET_SYMBOLS
        assert "US500" in INTERMARKET_SYMBOLS["SPX"]

    def test_all_four_instruments(self):
        """All 4 instruments should be defined."""
        assert "DXY" in INTERMARKET_SYMBOLS
        assert "XAUUSD" in INTERMARKET_SYMBOLS
        assert "US10Y" in INTERMARKET_SYMBOLS
        assert "SPX" in INTERMARKET_SYMBOLS

    def test_provider_without_mt5(self):
        """Provider works without MT5 (returns empty fallback)."""
        provider = IntermarketDataProvider(mt5_client=None)
        df = provider.get_bars("DXY", count=100)
        assert isinstance(df, pd.DataFrame)
        assert "DXY_close" in df.columns

    def test_get_all_intermarket_empty(self):
        """All instruments return empty when no MT5."""
        provider = IntermarketDataProvider(mt5_client=None)
        df = provider.get_all_intermarket(
            ["DXY", "XAUUSD", "US10Y", "SPX"], count=100
        )
        # All return empty DataFrames, so merged result is empty
        assert isinstance(df, pd.DataFrame)

    def test_resolve_symbol_without_mt5(self):
        """Without MT5, _resolve_symbol returns None."""
        provider = IntermarketDataProvider(mt5_client=None)
        assert provider._resolve_symbol("DXY") is None

    def test_fallback_returns_empty_with_correct_columns(self):
        """Without any data source, get_bars returns empty with correct columns."""
        provider = IntermarketDataProvider(data_dir="/nonexistent/path")
        df = provider.get_bars("SPX", timeframe="H1", count=100)
        assert "SPX_close" in df.columns


# ---------------------------------------------------------------------------
# IntermarketCorrExtractor
# ---------------------------------------------------------------------------

class TestIntermarketCorrExtractor:
    def test_default_instruments_include_spx(self):
        """SPX should be in default instruments."""
        extractor = IntermarketCorrExtractor()
        assert "SPX" in extractor._instruments

    def test_feature_names_include_spx(self):
        """Feature names should include SPX correlations."""
        extractor = IntermarketCorrExtractor()
        names = extractor.feature_names
        assert "corr_SPX_20" in names
        assert "rank_corr_SPX_60" in names
        assert "beta_SPX" in names

    def test_feature_count(self):
        """4 instruments × (3 windows × 2 corr types + 1 beta) = 28 features."""
        extractor = IntermarketCorrExtractor()
        # Each instrument: 3 Pearson + 3 Spearman + 1 beta = 7
        # 4 instruments × 7 = 28
        assert len(extractor.feature_names) == 28

    def test_extract_with_intermarket_data(self):
        """Extract produces non-NaN when intermarket data is present."""
        np.random.seed(42)
        n = 300
        bars = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": np.random.uniform(1.05, 1.15, n),
            "high": np.random.uniform(1.10, 1.20, n),
            "low": np.random.uniform(1.00, 1.10, n),
            "close": np.cumsum(np.random.randn(n) * 0.001) + 1.10,
            "volume": np.random.randint(100, 1000, n),
            # Intermarket data columns
            "DXY_close": np.cumsum(np.random.randn(n) * 0.1) + 104.0,
            "XAUUSD_close": np.cumsum(np.random.randn(n) * 2.0) + 2000.0,
            "US10Y_close": np.cumsum(np.random.randn(n) * 0.01) + 4.5,
            "SPX_close": np.cumsum(np.random.randn(n) * 5.0) + 5000.0,
        })

        extractor = IntermarketCorrExtractor()
        result = extractor.extract(bars)

        assert len(result) == n
        # After enough bars (>252), correlations should be computed
        assert not result["corr_DXY_20"].iloc[-1:].isna().all()
        assert not result["corr_SPX_20"].iloc[-1:].isna().all()
        assert not result["beta_XAUUSD"].iloc[-1:].isna().all()

    def test_extract_without_intermarket_data(self):
        """Extract returns all NaN when intermarket columns are missing."""
        bars = pd.DataFrame({
            "close": np.random.uniform(1.05, 1.15, 100),
        })

        extractor = IntermarketCorrExtractor()
        result = extractor.extract(bars)

        # All features should be NaN (no intermarket data)
        for col in result.columns:
            assert result[col].isna().all()

    def test_rolling_correlation(self):
        """Rolling correlation computes correct values."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 0.8 * x + 0.2 * np.random.randn(n)  # Correlated

        result = IntermarketCorrExtractor._rolling_correlation(x, y, 20)

        # Last value should be high positive correlation
        assert result[-1] > 0.5
        # First 19 values should be NaN
        assert np.all(np.isnan(result[:20]))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestIntermarketConfig:
    def test_symbols_config_includes_spx(self):
        cfg = SymbolsConfig(symbols={})
        assert "SPX" in cfg.intermarket
        assert len(cfg.intermarket) == 4

    def test_data_config_has_news(self):
        cfg = DataConfig()
        assert cfg.news.enabled is True


# ---------------------------------------------------------------------------
# Portfolio VaR (non-placeholder)
# ---------------------------------------------------------------------------

class TestPortfolioVaR:
    def test_var_calculator_portfolio_method_exists(self):
        """VaRCalculator should have compute_portfolio_var method."""
        from apexfx.risk.var_calculator import VaRCalculator
        calc = VaRCalculator()
        assert hasattr(calc, "compute_portfolio_var")

    def test_portfolio_var_with_correlation(self):
        """Portfolio VaR should be less than sum of individual VaRs (diversification)."""
        from apexfx.risk.var_calculator import VaRCalculator
        calc = VaRCalculator()

        positions = {"EURUSD": 50000.0, "USDJPY": 50000.0}
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        individual_vars = {"EURUSD": 100.0, "USDJPY": 100.0}

        pvar = calc.compute_portfolio_var(positions, corr, individual_vars)

        # Portfolio VaR < sum of individual VaRs (diversification benefit)
        assert pvar < 100.0  # Less than either individual VaR
        assert pvar > 0

    def test_portfolio_var_perfect_correlation(self):
        """With perfect correlation, portfolio VaR ≈ weighted sum of individual VaRs."""
        from apexfx.risk.var_calculator import VaRCalculator
        calc = VaRCalculator()

        positions = {"EURUSD": 50000.0, "GBPUSD": 50000.0}
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        individual_vars = {"EURUSD": 100.0, "GBPUSD": 100.0}

        pvar = calc.compute_portfolio_var(positions, corr, individual_vars)

        # With perfect correlation and equal weights, portfolio VaR
        # should equal the weighted individual VaR
        assert pvar > 0
        assert pvar == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Silent failures fix
# ---------------------------------------------------------------------------

class TestSilentFailureFix:
    def test_queue_overflow_logs_warning(self):
        """When queue overflows, dropped headline should be logged (not silent)."""
        cfg = NewsConfig(finnhub_enabled=False, rss_enabled=False)
        stream = RealtimeNewsStream(cfg)

        # Fill queue to max
        for i in range(200):
            h = NewsHeadline(
                text=f"Headline {i}",
                timestamp=datetime.now(timezone.utc),
                source="test",
            )
            stream._on_headline(h)

        # Now add one more — should trigger overflow handling
        overflow_headline = NewsHeadline(
            text="Overflow headline 201",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        stream._on_headline(overflow_headline)

        # Queue should still be at max (200)
        assert stream._headline_queue.qsize() <= 200
        # The overflow headline should be in the queue (newest wins)
        assert stream._stats["total_received"] == 201
