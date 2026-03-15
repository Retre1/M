"""Tests for Phase 7: Advanced Features.

Covers:
- Dynamic FX Correlations (DynamicCorrelationTracker)
- COT Extractor features
- Seasonal pattern features
- Central bank speech analysis
- COT Fetcher data parsing
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Dynamic Correlations
# ---------------------------------------------------------------------------

class TestDynamicCorrelationTracker:
    def test_static_fallback(self):
        """Without dynamic data, falls back to static CORRELATION_MAP."""
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker
        tracker = DynamicCorrelationTracker()
        corr = tracker.get_correlation("EURUSD", "GBPUSD")
        assert corr == pytest.approx(0.85)

    def test_self_correlation(self):
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker
        tracker = DynamicCorrelationTracker()
        assert tracker.get_correlation("EURUSD", "EURUSD") == 1.0

    def test_unknown_pair_default(self):
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker
        tracker = DynamicCorrelationTracker()
        corr = tracker.get_correlation("EURUSD", "ZARJPY")
        assert corr == 0.30

    def test_dynamic_overrides_static(self):
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker
        tracker = DynamicCorrelationTracker(lookback=60, min_bars=10)

        # Feed correlated returns
        np.random.seed(42)
        for _ in range(30):
            r = np.random.randn() * 0.01
            tracker.update_returns("EURUSD", r)
            tracker.update_returns("GBPUSD", r * 0.9 + np.random.randn() * 0.002)

        tracker.recompute()

        assert tracker.has_dynamic_data
        corr = tracker.get_correlation("EURUSD", "GBPUSD")
        # Should be high positive (since we fed correlated data)
        assert corr > 0.5

    def test_recompute_with_insufficient_data(self):
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker
        tracker = DynamicCorrelationTracker(min_bars=20)

        # Only 5 bars — not enough
        for _ in range(5):
            tracker.update_returns("EURUSD", np.random.randn() * 0.01)
            tracker.update_returns("GBPUSD", np.random.randn() * 0.01)

        tracker.recompute()
        assert not tracker.has_dynamic_data  # Not enough data

    def test_buffer_truncation(self):
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker
        tracker = DynamicCorrelationTracker(lookback=30)

        for _ in range(200):
            tracker.update_returns("EURUSD", np.random.randn() * 0.01)

        # Buffer should be truncated to lookback size
        assert len(tracker._returns["EURUSD"]) <= 60  # 2x lookback

    def test_module_level_get_correlation(self):
        """Module-level _get_correlation uses the singleton tracker."""
        from apexfx.live.portfolio_manager import _get_correlation
        corr = _get_correlation("EURUSD", "USDCHF")
        assert corr == pytest.approx(-0.90)

    def test_get_correlation_tracker(self):
        from apexfx.live.portfolio_manager import get_correlation_tracker
        tracker = get_correlation_tracker()
        assert tracker is not None


# ---------------------------------------------------------------------------
# COT Features
# ---------------------------------------------------------------------------

class TestCOTExtractor:
    def test_feature_names(self):
        from apexfx.features.cot import COTExtractor
        ext = COTExtractor()
        assert len(ext.feature_names) == 7
        assert "cot_spec_net_norm" in ext.feature_names
        assert "cot_divergence" in ext.feature_names

    def test_extract_without_cot_columns(self):
        """Returns all NaN when COT columns are missing."""
        from apexfx.features.cot import COTExtractor
        ext = COTExtractor()
        bars = pd.DataFrame({
            "close": np.random.uniform(1.05, 1.15, 100),
        })
        result = ext.extract(bars)
        assert result.shape == (100, 7)
        assert result["cot_spec_net_norm"].isna().all()

    def test_extract_with_cot_data(self):
        """Produces non-NaN values when COT columns present."""
        from apexfx.features.cot import COTExtractor
        ext = COTExtractor(z_score_lookback=10)

        n = 300
        np.random.seed(42)
        bars = pd.DataFrame({
            "close": np.cumsum(np.random.randn(n) * 0.001) + 1.10,
            "cot_spec_net": np.cumsum(np.random.randn(n) * 100) + 50000,
            "cot_comm_net": np.cumsum(np.random.randn(n) * 100) - 30000,
            "cot_open_interest": np.abs(np.random.randn(n) * 10000) + 200000,
        })

        result = ext.extract(bars)
        assert result.shape == (n, 7)
        # After lookback period, values should be computed
        assert not result["cot_spec_net_norm"].iloc[-1:].isna().all()
        assert not result["cot_spec_change"].iloc[-1:].isna().all()

    def test_spec_extreme_detection(self):
        """Detects extreme speculative positioning."""
        from apexfx.features.cot import COTExtractor
        ext = COTExtractor(z_score_lookback=5, extreme_threshold=1.5)

        n = 100
        # Create data with a spike in spec net at the end
        spec = np.zeros(n)
        spec[-5:] = 100000  # Extreme spike
        bars = pd.DataFrame({
            "close": np.ones(n) * 1.10,
            "cot_spec_net": spec,
            "cot_comm_net": np.zeros(n),
            "cot_open_interest": np.ones(n) * 200000,
        })

        result = ext.extract(bars)
        # The last values should show extreme
        assert result["cot_spec_extreme"].iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# Seasonal Patterns
# ---------------------------------------------------------------------------

class TestSeasonalExtractor:
    def test_feature_names(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()
        assert len(ext.feature_names) == 8
        assert "seasonal_month_vol" in ext.feature_names
        assert "seasonal_session" in ext.feature_names

    def test_extract_basic(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()

        bars = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=100, freq="h"),
            "close": np.random.uniform(1.05, 1.15, 100),
        })

        result = ext.extract(bars)
        assert result.shape == (100, 8)

        # January bars should have specific month_vol
        assert result["seasonal_month_vol"].iloc[0] > 0
        # No summer in January
        assert result["seasonal_summer"].iloc[0] == 0.0
        # Year-end for Jan 1 (within Jan 5 window)
        assert result["seasonal_year_end"].iloc[0] == 1.0

    def test_summer_detection(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()

        bars = pd.DataFrame({
            "time": pd.date_range("2025-07-15", periods=10, freq="h"),
            "close": np.ones(10),
        })
        result = ext.extract(bars)
        assert result["seasonal_summer"].iloc[0] == 1.0

    def test_quarter_end(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()

        # March 31 = quarter end
        bars = pd.DataFrame({
            "time": pd.to_datetime(["2025-03-31 12:00"]),
            "close": [1.10],
        })
        result = ext.extract(bars)
        assert result["seasonal_quarter_end"].iloc[0] == 1.0

    def test_session_detection(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()

        # Create bars at different hours
        bars = pd.DataFrame({
            "time": [
                datetime(2025, 1, 6, 3, 0),   # Asia (0-7 UTC)
                datetime(2025, 1, 6, 9, 0),   # London (7-12 UTC)
                datetime(2025, 1, 6, 14, 0),  # NY (12-17 UTC)
                datetime(2025, 1, 6, 19, 0),  # Late/overlap (17+)
            ],
            "close": [1.10, 1.11, 1.12, 1.13],
        })
        result = ext.extract(bars)
        sessions = result["seasonal_session"].values
        assert sessions[0] == pytest.approx(0 / 3.0)    # Asia
        assert sessions[1] == pytest.approx(1 / 3.0)    # London
        assert sessions[2] == pytest.approx(2 / 3.0)    # NY
        assert sessions[3] == pytest.approx(3 / 3.0)    # Late

    def test_without_time_column(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()
        bars = pd.DataFrame({"close": [1.10, 1.11]})
        result = ext.extract(bars)
        assert result.shape == (2, 8)
        # Should return zeros
        assert (result["seasonal_month_vol"] == 0.0).all()

    def test_dow_volatility(self):
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()

        # Monday vs Thursday
        bars = pd.DataFrame({
            "time": [
                datetime(2025, 1, 6, 12, 0),  # Monday
                datetime(2025, 1, 9, 12, 0),  # Thursday
            ],
            "close": [1.10, 1.11],
        })
        result = ext.extract(bars)
        # Thursday should have higher dow_vol than Monday
        assert result["seasonal_dow_vol"].iloc[1] > result["seasonal_dow_vol"].iloc[0]


# ---------------------------------------------------------------------------
# Central Bank Analysis
# ---------------------------------------------------------------------------

class TestCentralBankAnalyzer:
    def test_hawkish_detection(self):
        from apexfx.features.central_bank import CentralBankAnalyzer
        analyzer = CentralBankAnalyzer()

        stmt = analyzer.analyze_text(
            "The committee decided to raise rates by 25bps. "
            "Inflation remains above target and the labor market is strong. "
            "Further increases may be appropriate.",
            source="fomc",
        )
        assert stmt.hawkish_score > stmt.dovish_score
        assert stmt.net_score > 0

    def test_dovish_detection(self):
        from apexfx.features.central_bank import CentralBankAnalyzer
        analyzer = CentralBankAnalyzer()

        stmt = analyzer.analyze_text(
            "The committee decided to cut rates by 50bps. "
            "The economy shows signs of slowdown and weakness. "
            "Further easing may be needed to support growth.",
            source="ecb",
        )
        assert stmt.dovish_score > stmt.hawkish_score
        assert stmt.net_score < 0

    def test_neutral_text(self):
        from apexfx.features.central_bank import CentralBankAnalyzer
        analyzer = CentralBankAnalyzer()

        stmt = analyzer.analyze_text(
            "The weather forecast for tomorrow shows partly cloudy skies.",
            source="fomc",
        )
        assert stmt.net_score == 0.0

    def test_stance_momentum(self):
        from apexfx.features.central_bank import CentralBankAnalyzer
        analyzer = CentralBankAnalyzer()

        # First dovish
        analyzer.analyze_text("We will cut rates and ease policy.", "fomc")
        analyzer.analyze_text("Further easing is expected.", "fomc")
        # Then hawkish
        analyzer.analyze_text("We must raise rates to fight inflation.", "fomc")
        analyzer.analyze_text("Higher for longer rate policy.", "fomc")

        stance = analyzer.get_current_stance("fomc")
        assert stance["momentum"] > 0  # Shifted hawkish

    def test_get_stance_empty(self):
        from apexfx.features.central_bank import CentralBankAnalyzer
        analyzer = CentralBankAnalyzer()
        stance = analyzer.get_current_stance("fomc")
        assert stance["stance"] == 0.0
        assert stance["conviction"] == 0.0


class TestCentralBankExtractor:
    def test_feature_names(self):
        from apexfx.features.central_bank import CentralBankExtractor
        ext = CentralBankExtractor()
        assert len(ext.feature_names) == 6
        assert "cb_net_stance" in ext.feature_names

    def test_extract_without_data(self):
        from apexfx.features.central_bank import CentralBankExtractor
        ext = CentralBankExtractor()

        bars = pd.DataFrame({"close": np.random.uniform(1.05, 1.15, 50)})
        result = ext.extract(bars)
        assert result.shape == (50, 6)
        assert (result["cb_net_stance"] == 0.0).all()

    def test_extract_with_statements(self):
        from apexfx.features.central_bank import CentralBankExtractor
        ext = CentralBankExtractor(base_bank="ecb", quote_bank="fomc")

        # Feed ECB hawkish + FOMC dovish → EUR bullish
        ext.update_statement("ECB will raise rates aggressively.", "ecb")
        ext.update_statement("Fed considers cutting rates.", "fomc")

        bars = pd.DataFrame({"close": [1.10, 1.11, 1.12]})
        result = ext.extract(bars)

        # Net stance should be positive (ECB hawkish vs FOMC dovish = EUR bullish)
        assert result["cb_net_stance"].iloc[0] > 0


# ---------------------------------------------------------------------------
# COT Fetcher
# ---------------------------------------------------------------------------

class TestCOTFetcher:
    def test_parse_int(self):
        from apexfx.data.cot_fetcher import COTFetcher
        assert COTFetcher._parse_int("1,234") == 1234
        assert COTFetcher._parse_int("0") == 0
        assert COTFetcher._parse_int("") == 0
        assert COTFetcher._parse_int("  5678  ") == 5678

    def test_cot_currency_map(self):
        from apexfx.data.cot_fetcher import COT_CURRENCY_MAP
        assert "EUR" in COT_CURRENCY_MAP
        assert "GBP" in COT_CURRENCY_MAP
        assert "JPY" in COT_CURRENCY_MAP
        assert len(COT_CURRENCY_MAP) >= 7

    def test_to_dataframe_empty(self):
        from apexfx.data.cot_fetcher import COTData, COTFetcher
        fetcher = COTFetcher()
        data = COTData()
        df = fetcher.to_dataframe(data, "EUR")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_cot_record_fields(self):
        from apexfx.data.cot_fetcher import COTRecord
        record = COTRecord(
            report_date=datetime(2025, 1, 7, tzinfo=timezone.utc),
            currency="EUR",
            speculative_long=150000,
            speculative_short=100000,
            speculative_net=50000,
            commercial_long=200000,
            commercial_short=250000,
            commercial_net=-50000,
            open_interest=500000,
            spec_net_pct=0.10,
            commercial_net_pct=-0.10,
        )
        assert record.speculative_net == 50000
        assert record.spec_net_pct == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Portfolio Manager (regression)
# ---------------------------------------------------------------------------

class TestPortfolioManagerRegression:
    def test_check_new_trade_still_works(self):
        """Ensure portfolio manager still functions after dynamic correlation changes."""
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager(max_total_exposure=0.40, correlation_limit=0.70)
        pm.set_equity(100000)

        result = pm.check_new_trade("EURUSD", 1, 10000)
        assert result.approved

    def test_correlation_block_still_works(self):
        """Correlated positions in same direction should still be blocked."""
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager(correlation_limit=0.70)
        pm.set_equity(100000)

        pm.update_position("EURUSD", 1, 0.5, 1.10)
        # GBPUSD long should be blocked (0.85 > 0.70 with EURUSD long)
        result = pm.check_new_trade("GBPUSD", 1, 10000)
        assert not result.approved

    def test_opposite_direction_allowed(self):
        """EURUSD long + GBPUSD short should be fine (hedged)."""
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager(correlation_limit=0.70, max_total_exposure=0.80)
        pm.set_equity(200000)

        pm.update_position("EURUSD", 1, 0.1, 1.10)
        result = pm.check_new_trade("GBPUSD", -1, 10000)
        assert result.approved
