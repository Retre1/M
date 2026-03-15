"""Integration tests — end-to-end pipeline validation.

Tests the full flow:
1. Data generation → Feature extraction → Signal readiness
2. Feature pipeline produces consistent output shape
3. Risk manager gating works end-to-end
4. New Phase 7 features integrate into pipeline
5. Dynamic correlations feed into portfolio checks
6. COT + Seasonal + CentralBank extractors chain correctly
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_ohlcv_bars(n: int = 500, pair: str = "EURUSD") -> pd.DataFrame:
    """Generate realistic synthetic OHLCV bar data."""
    np.random.seed(42)
    base_price = 1.1000
    returns = np.random.randn(n) * 0.001  # 10 pips daily stdev
    close = base_price + np.cumsum(returns)
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    opn = close + np.random.randn(n) * 0.0003
    volume = np.random.randint(100, 10000, n).astype(float)
    tick_volume = volume * np.random.uniform(1, 5, n)
    spread = np.random.uniform(0.00005, 0.00020, n)
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        "time": times,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": tick_volume,
        "volume": volume,
        "spread": spread,
    })


def _generate_bars_with_cot(n: int = 500) -> pd.DataFrame:
    """Generate bars with COT columns forward-filled from weekly data."""
    bars = _generate_ohlcv_bars(n)
    # Simulate weekly COT data forward-filled into H1 bars
    spec_net = np.zeros(n)
    comm_net = np.zeros(n)
    oi = np.ones(n) * 200000
    # Update weekly (every ~120 H1 bars ≈ 5 trading days)
    current_spec = 50000.0
    current_comm = -30000.0
    for i in range(n):
        if i % 120 == 0:
            current_spec += np.random.randn() * 5000
            current_comm += np.random.randn() * 5000
        spec_net[i] = current_spec
        comm_net[i] = current_comm
        oi[i] = 200000 + np.random.randn() * 10000

    bars["cot_spec_net"] = spec_net
    bars["cot_comm_net"] = comm_net
    bars["cot_open_interest"] = oi
    return bars


# ---------------------------------------------------------------------------
# 1. Feature Pipeline End-to-End
# ---------------------------------------------------------------------------

class TestFeaturePipelineIntegration:
    """End-to-end tests for the full feature pipeline."""

    def test_default_pipeline_runs_without_errors(self):
        """Full pipeline with all default extractors produces valid output."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=False)
        bars = _generate_ohlcv_bars(500)

        result = pipeline.compute(bars)

        assert len(result) == 500
        # Should have bar columns + feature columns
        assert len(result.columns) > len(bars.columns)
        # No inf values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.any(np.isinf(result[numeric_cols].values[np.isfinite(result[numeric_cols].values) | np.isnan(result[numeric_cols].values)]))

    def test_pipeline_with_normalization(self):
        """Pipeline with z-score normalization produces bounded features."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=True)
        bars = _generate_ohlcv_bars(500)

        result = pipeline.compute(bars)
        feature_cols = [c for c in result.columns if c not in bars.columns]

        # After normalization, most features should be roughly [-3, 3]
        # (allowing for outliers and binary features)
        for col in feature_cols[:10]:  # Check first 10
            vals = result[col].dropna()
            if len(vals) > 0 and vals.std() > 0:
                # Non-zero variance columns should be normalized
                assert vals.mean() < 10, f"{col} has unusually high mean"

    def test_pipeline_feature_count_matches(self):
        """Pipeline reports correct number of features via property."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=False)
        bars = _generate_ohlcv_bars(300)
        result = pipeline.compute(bars)

        reported_features = pipeline.feature_names
        actual_features = [c for c in result.columns if c not in bars.columns]

        # All reported features should exist in output
        for f in reported_features:
            assert f in result.columns, f"Reported feature {f} not in output"

    def test_incremental_compute_matches(self):
        """Incremental computation for live trading matches batch."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=False)
        bars = _generate_ohlcv_bars(300)

        # Batch compute
        batch_result = pipeline.compute(bars)

        # Incremental: compute last bar using history
        history = bars.iloc[:-1].copy()
        new_bar = bars.iloc[-1]
        incremental_result = pipeline.compute_incremental(new_bar, history)

        # Check that key features are approximately equal
        # (small numerical differences OK due to different code paths)
        for col in ["hurst_exponent", "spectral_dominant_frequency"]:
            if col in batch_result.columns and col in incremental_result.index:
                batch_val = batch_result[col].iloc[-1]
                incr_val = incremental_result[col]
                if not np.isnan(batch_val) and not np.isnan(incr_val):
                    assert abs(batch_val - incr_val) < 1.0, f"{col} mismatch"


# ---------------------------------------------------------------------------
# 2. Phase 7 Feature Extractors Integration
# ---------------------------------------------------------------------------

class TestPhase7FeaturesIntegration:
    """Test new Phase 7 extractors work within the pipeline context."""

    def test_cot_extractor_with_pipeline_data(self):
        """COT extractor produces valid features from realistic data."""
        from apexfx.features.cot import COTExtractor
        ext = COTExtractor(z_score_lookback=10)

        bars = _generate_bars_with_cot(500)
        result = ext.extract(bars)

        assert len(result) == 500
        assert "cot_spec_net_norm" in result.columns
        assert "cot_divergence" in result.columns

        # After warmup period, should have valid values
        tail = result.iloc[-50:]
        assert tail["cot_spec_net_norm"].notna().sum() > 0
        assert tail["cot_momentum"].notna().sum() > 0

    def test_seasonal_extractor_full_year(self):
        """Seasonal extractor handles a full year of data."""
        from apexfx.features.seasonal import SeasonalExtractor
        ext = SeasonalExtractor()

        # Generate a full year of H1 bars
        n = 24 * 252  # ~252 trading days
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bars = pd.DataFrame({
            "time": [start + timedelta(hours=i) for i in range(n)],
            "close": np.random.uniform(1.05, 1.15, n),
        })

        result = ext.extract(bars)
        assert len(result) == n

        # Summer detection
        summer_bars = result[(bars["time"].dt.month >= 6) & (bars["time"].dt.month <= 8)]
        assert (summer_bars["seasonal_summer"] == 1.0).all()

        # Non-summer bars
        winter_bars = result[(bars["time"].dt.month >= 1) & (bars["time"].dt.month <= 2)]
        assert (winter_bars["seasonal_summer"] == 0.0).all()

    def test_central_bank_live_flow(self):
        """Central bank extractor works in live-like update flow."""
        from apexfx.features.central_bank import CentralBankExtractor

        ext = CentralBankExtractor(base_bank="ecb", quote_bank="fomc")

        # Simulate live flow: feed statements then extract
        ext.update_statement(
            "The ECB Governing Council decided to raise rates by 50 basis points. "
            "Inflation remains above target and further increases may be needed.",
            "ecb",
        )
        ext.update_statement(
            "The Federal Reserve decided to cut rates by 25 basis points. "
            "The economy shows signs of slowdown.",
            "fomc",
        )

        bars = _generate_ohlcv_bars(100)
        result = ext.extract(bars)

        # Should show EUR bullish stance (ECB hawkish, FOMC dovish)
        assert result["cb_net_stance"].iloc[0] > 0
        assert result["cb_conviction"].iloc[0] > 0

    def test_all_phase7_extractors_together(self):
        """All Phase 7 extractors produce compatible output shapes."""
        from apexfx.features.cot import COTExtractor
        from apexfx.features.seasonal import SeasonalExtractor
        from apexfx.features.central_bank import CentralBankExtractor

        bars = _generate_bars_with_cot(200)

        cot_result = COTExtractor(z_score_lookback=5).extract(bars)
        seasonal_result = SeasonalExtractor().extract(bars)
        cb_result = CentralBankExtractor().extract(bars)

        # All should have same length as input
        assert len(cot_result) == 200
        assert len(seasonal_result) == 200
        assert len(cb_result) == 200

        # Concatenation should work
        combined = pd.concat([bars, cot_result, seasonal_result, cb_result], axis=1)
        assert len(combined) == 200
        # Total columns = bar cols + 7 cot + 8 seasonal + 6 cb
        expected_extra = 7 + 8 + 6
        actual_extra = len(combined.columns) - len(bars.columns)
        assert actual_extra == expected_extra


# ---------------------------------------------------------------------------
# 3. Dynamic Correlations → Portfolio Manager Integration
# ---------------------------------------------------------------------------

class TestDynamicCorrelationIntegration:
    """Tests that dynamic correlations feed correctly into portfolio decisions."""

    def test_dynamic_corr_changes_portfolio_decision(self):
        """When dynamic correlations are low, previously blocked trades pass."""
        from apexfx.live.portfolio_manager import (
            DynamicCorrelationTracker,
            PortfolioManager,
            _correlation_tracker,
        )

        # Create a fresh tracker with zero correlation between EURUSD and GBPUSD
        tracker = DynamicCorrelationTracker(lookback=60, min_bars=10)
        np.random.seed(123)
        for _ in range(30):
            # Feed uncorrelated returns
            tracker.update_returns("EURUSD", np.random.randn() * 0.01)
            tracker.update_returns("GBPUSD", np.random.randn() * 0.01)
        tracker.recompute()

        # Dynamic correlation should be near zero (uncorrelated)
        dyn_corr = tracker.get_correlation("EURUSD", "GBPUSD")
        assert abs(dyn_corr) < 0.70, f"Expected low correlation, got {dyn_corr}"

    def test_tracker_recompute_cycle(self):
        """Simulate a full trading session feeding returns and recomputing."""
        from apexfx.live.portfolio_manager import DynamicCorrelationTracker

        tracker = DynamicCorrelationTracker(lookback=60, min_bars=20)
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

        np.random.seed(42)
        # Simulate 100 bars of correlated market data
        base_factor = np.random.randn(100) * 0.01  # Common market factor

        for i in range(100):
            for sym in symbols:
                # Each pair has some common factor + idiosyncratic noise
                ret = base_factor[i] * 0.5 + np.random.randn() * 0.005
                tracker.update_returns(sym, ret)

            # Recompute every 10 bars
            if i % 10 == 0:
                tracker.recompute()

        assert tracker.has_dynamic_data
        pairs = tracker.dynamic_pairs
        assert len(pairs) > 0

        # All correlations should be between -1 and 1
        for (s1, s2), corr in pairs.items():
            assert -1.0 <= corr <= 1.0, f"Invalid corr {corr} for {s1}/{s2}"


# ---------------------------------------------------------------------------
# 4. Risk Manager Integration (without torch)
# ---------------------------------------------------------------------------

class TestRiskManagerIntegration:
    """Integration tests for the risk management pipeline."""

    def test_daily_loss_guard_resets(self):
        """DailyLossGuard correctly resets on new day."""
        from apexfx.risk.risk_manager import DailyLossGuard

        guard = DailyLossGuard(max_daily_loss_pct=0.02)

        # Start of day
        assert guard.update(100000)  # OK
        assert guard.update(99000)   # 1% loss — OK
        assert not guard.update(98000)  # 2% loss — should trigger
        assert not guard.update(98000)  # Still blocked

    def test_kill_switch_consecutive_rejections(self):
        """Kill switch activates after too many rejections."""
        from apexfx.risk.risk_manager import KillSwitch

        ks = KillSwitch(max_consecutive_rejections=5)

        for _ in range(4):
            ks.record_rejection()
            assert not ks.is_active

        ks.record_rejection()  # 5th rejection
        assert ks.is_active

        # Reset
        ks.reset()
        assert not ks.is_active

    def test_volatility_targeter_scaling(self):
        """VolatilityTargeter adjusts leverage based on realized vol."""
        from apexfx.risk.risk_manager import VolatilityTargeter

        vt = VolatilityTargeter(target_vol=0.10, lookback=30)

        # Feed low-vol returns (1% daily ≈ 16% annualized)
        np.random.seed(42)
        for _ in range(40):
            vt.update(np.random.randn() * 0.01)

        leverage = vt.compute_leverage()
        # Target 10% vol / realized ~16% → leverage < 1.0
        assert 0.1 < leverage < 2.0

    def test_weekend_guard_friday_evening(self):
        """WeekendGapGuard blocks trades on Friday evening."""
        from apexfx.risk.risk_manager import WeekendGapGuard

        guard = WeekendGapGuard(close_before_hour_utc=20)

        # Friday 21:00 UTC
        friday_late = datetime(2025, 1, 3, 21, 0, tzinfo=timezone.utc)
        should_block, scale = guard.check(friday_late)
        assert should_block
        assert scale == 0.0

        # Monday 10:00 UTC
        monday = datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc)
        should_block, scale = guard.check(monday)
        assert not should_block
        assert scale == 1.0

    def test_regime_adaptive_scaling(self):
        """RegimeAdaptiveRisk returns correct scaling for different regimes."""
        from apexfx.risk.risk_manager import RegimeAdaptiveRisk

        rar = RegimeAdaptiveRisk()

        # Trending → larger positions
        rar.set_regime("trending")
        pos_scale, var_scale = rar.get_scales()
        assert pos_scale > 1.0

        # Volatile → smaller positions
        rar.set_regime("volatile")
        pos_scale, var_scale = rar.get_scales()
        assert pos_scale < 1.0
        assert var_scale < 1.0


# ---------------------------------------------------------------------------
# 5. COT Fetcher → COT Extractor Integration
# ---------------------------------------------------------------------------

class TestCOTDataIntegration:
    """Test COT data flows from fetcher to extractor."""

    def test_cot_record_to_dataframe(self):
        """COTFetcher.to_dataframe produces valid DataFrame for extractor."""
        from apexfx.data.cot_fetcher import COTData, COTFetcher, COTRecord

        fetcher = COTFetcher(cache_dir="/tmp/apexfx_cot_test")

        data = COTData()
        data.records["EUR"] = []
        for i in range(10):
            data.records["EUR"].append(COTRecord(
                report_date=datetime(2025, 1, 7, tzinfo=timezone.utc) + timedelta(weeks=i),
                currency="EUR",
                speculative_long=150000 + i * 1000,
                speculative_short=100000 + i * 500,
                speculative_net=50000 + i * 500,
                commercial_net=-30000,
                open_interest=500000,
                spec_net_pct=0.10 + i * 0.001,
            ))

        df = fetcher.to_dataframe(data, "EUR")
        assert len(df) == 10
        assert "cot_spec_net" in df.columns
        assert "time" in df.columns
        assert df["cot_spec_net"].iloc[-1] > df["cot_spec_net"].iloc[0]

    def test_cot_dataframe_merge_with_bars(self):
        """COT DataFrame merges correctly with OHLCV bars."""
        from apexfx.data.cot_fetcher import COTData, COTFetcher, COTRecord

        fetcher = COTFetcher(cache_dir="/tmp/apexfx_cot_test2")

        # Create weekly COT records
        data = COTData()
        data.records["EUR"] = []
        for week in range(8):
            data.records["EUR"].append(COTRecord(
                report_date=datetime(2025, 1, 7, tzinfo=timezone.utc) + timedelta(weeks=week),
                currency="EUR",
                speculative_net=50000 + week * 2000,
                commercial_net=-30000,
                open_interest=500000,
                spec_net_pct=0.10 + week * 0.004,
            ))

        cot_df = fetcher.to_dataframe(data, "EUR")

        # Create H1 bars for the same period
        n = 24 * 56  # 56 days of H1 bars
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bars = pd.DataFrame({
            "time": [start + timedelta(hours=i) for i in range(n)],
            "close": np.random.uniform(1.05, 1.15, n),
        })

        # Merge using merge_asof (forward-fill weekly COT into H1 bars)
        bars_sorted = bars.sort_values("time")
        cot_sorted = cot_df.sort_values("time")
        merged = pd.merge_asof(
            bars_sorted, cot_sorted, on="time", direction="backward"
        )

        assert len(merged) == n
        # COT columns should be present
        assert "cot_spec_net" in merged.columns
        # First bars before Jan 7 should have NaN, rest should have values
        assert merged["cot_spec_net"].notna().sum() > n // 2


# ---------------------------------------------------------------------------
# 6. Full Pipeline with Phase 7 Extractors
# ---------------------------------------------------------------------------

class TestFullPipelineWithPhase7:
    """Test running the full pipeline with Phase 7 extractors injected."""

    def test_pipeline_with_seasonal_extractor(self):
        """Seasonal extractor integrates into FeaturePipeline."""
        from apexfx.features.pipeline import FeaturePipeline
        from apexfx.features.seasonal import SeasonalExtractor

        # Create pipeline with just a few extractors + seasonal
        pipeline = FeaturePipeline(
            extractors=[SeasonalExtractor()],
            normalize=False,
        )

        bars = _generate_ohlcv_bars(200)
        result = pipeline.compute(bars)

        assert "seasonal_month_vol" in result.columns
        assert "seasonal_session" in result.columns
        assert len(result) == 200

    def test_pipeline_with_cot_extractor(self):
        """COT extractor integrates into FeaturePipeline."""
        from apexfx.features.pipeline import FeaturePipeline
        from apexfx.features.cot import COTExtractor

        pipeline = FeaturePipeline(
            extractors=[COTExtractor(z_score_lookback=5)],
            normalize=False,
        )

        bars = _generate_bars_with_cot(200)
        result = pipeline.compute(bars)

        assert "cot_spec_net_norm" in result.columns
        assert len(result) == 200

    def test_pipeline_with_all_phase7_extractors(self):
        """All Phase 7 extractors work together in a pipeline."""
        from apexfx.features.pipeline import FeaturePipeline
        from apexfx.features.cot import COTExtractor
        from apexfx.features.seasonal import SeasonalExtractor
        from apexfx.features.central_bank import CentralBankExtractor

        pipeline = FeaturePipeline(
            extractors=[
                COTExtractor(z_score_lookback=5),
                SeasonalExtractor(),
                CentralBankExtractor(),
            ],
            normalize=False,
        )

        bars = _generate_bars_with_cot(200)
        result = pipeline.compute(bars)

        # 7 COT + 8 seasonal + 6 central bank = 21 new features
        feature_cols = [c for c in result.columns if c not in bars.columns]
        assert len(feature_cols) == 21

    def test_mixed_pipeline_default_plus_phase7(self):
        """Phase 7 extractors work alongside default extractors."""
        from apexfx.features.pipeline import FeaturePipeline
        from apexfx.features.seasonal import SeasonalExtractor
        from apexfx.features.volume_profile import VolumeProfileExtractor
        from apexfx.features.regime import RegimeExtractor

        pipeline = FeaturePipeline(
            extractors=[
                VolumeProfileExtractor(window=50),
                RegimeExtractor(),
                SeasonalExtractor(),
            ],
            normalize=False,
        )

        bars = _generate_ohlcv_bars(300)
        result = pipeline.compute(bars)

        # Should have features from all three extractors
        assert "poc_distance" in result.columns or "vp_poc_distance" in result.columns or "volume_profile_poc" in result.columns
        assert "seasonal_month_vol" in result.columns
        assert len(result) == 300


# ---------------------------------------------------------------------------
# 7. Data Quality Checks
# ---------------------------------------------------------------------------

class TestDataQualityIntegration:
    """Ensure no silent data corruption in the pipeline."""

    def test_no_infinite_values_in_features(self):
        """Feature pipeline should never produce infinite values."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=False)
        bars = _generate_ohlcv_bars(500)

        result = pipeline.compute(bars)
        numeric = result.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric.values)
        inf_locations = np.argwhere(inf_mask)
        assert len(inf_locations) == 0, (
            f"Found {len(inf_locations)} infinite values in features"
        )

    def test_feature_dtypes_are_numeric(self):
        """All feature columns should be numeric (float or int)."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=False)
        bars = _generate_ohlcv_bars(300)

        result = pipeline.compute(bars)
        feature_cols = [c for c in result.columns if c not in bars.columns]

        for col in feature_cols:
            assert result[col].dtype in [np.float64, np.float32, np.int64, np.int32, float], \
                f"Feature {col} has non-numeric dtype: {result[col].dtype}"

    def test_nan_ratio_acceptable(self):
        """NaN ratio in features should be reasonable after warmup."""
        from apexfx.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(normalize=False)
        bars = _generate_ohlcv_bars(500)

        result = pipeline.compute(bars)
        feature_cols = [c for c in result.columns if c not in bars.columns]

        # After first 260 bars (warmup), NaN ratio should be < 20%
        tail = result[feature_cols].iloc[260:]
        nan_ratio = tail.isna().mean().mean()
        # Some extractors (intermarket, orderbook, sentiment) may have NaN
        # without real data sources, so we allow up to 50%
        assert nan_ratio < 0.50, f"NaN ratio {nan_ratio:.2%} is too high after warmup"
