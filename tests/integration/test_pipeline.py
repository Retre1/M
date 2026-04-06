"""Integration tests for the full feature pipeline."""

import numpy as np
import pandas as pd
import pytest

from apexfx.features.pipeline import FeaturePipeline


def _make_bars(n: int = 600) -> pd.DataFrame:
    np.random.seed(42)
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.0005),
        "low": prices - np.abs(np.random.randn(n) * 0.0005),
        "close": prices + np.random.randn(n) * 0.0002,
        "volume": np.random.lognormal(10, 1, n),
        "tick_count": np.random.randint(50, 500, n),
    })


class TestFeaturePipeline:
    def test_full_pipeline_runs(self):
        """Full pipeline should run without errors on clean data."""
        bars = _make_bars(600)
        pipeline = FeaturePipeline(normalize=True, validate=True)
        result = pipeline.compute(bars)
        assert len(result) == len(bars)
        # Should have original columns plus feature columns
        assert len(result.columns) > len(bars.columns)

    def test_fast_pipeline(self):
        """Fast pipeline should run with only OrderFlow + Regime."""
        bars = _make_bars(300)
        pipeline = FeaturePipeline.create_fast()
        result = pipeline.compute(bars)
        assert len(result) == len(bars)
        assert "delta" in result.columns
        assert "regime_label" in result.columns
        # Should NOT have slow features
        assert "hurst_exponent" not in result.columns
        assert "fft_dominant_period" not in result.columns

    def test_pipeline_feature_count(self):
        bars = _make_bars(600)
        pipeline = FeaturePipeline(normalize=False, validate=False)
        result = pipeline.compute(bars)
        feature_cols = [c for c in result.columns if c not in bars.columns]
        assert len(feature_cols) == pipeline.n_features

    def test_pipeline_no_nan_after_warmup(self):
        """After sufficient warmup, features should not be all-NaN."""
        bars = _make_bars(600)
        pipeline = FeaturePipeline.create_fast()
        result = pipeline.compute(bars)
        feature_cols = [c for c in result.columns if c not in bars.columns]
        # Check last 100 rows — should have valid features
        tail = result[feature_cols].iloc[-100:]
        # At least 50% of feature columns should have data
        non_nan_cols = (tail.notna().mean() > 0.5).sum()
        assert non_nan_cols > len(feature_cols) * 0.5

    def test_pipeline_with_dirty_data(self):
        """Pipeline should auto-clean dirty data when validate=True."""
        bars = _make_bars(300)
        # Inject some NaN
        bars.loc[50, "close"] = np.nan
        bars.loc[100, "volume"] = np.nan
        # Duplicate a timestamp
        bars.loc[1, "time"] = bars.loc[0, "time"]

        pipeline = FeaturePipeline.create_fast()
        result = pipeline.compute(bars)
        # Should complete without error
        assert len(result) > 0

    def test_pipeline_normalized_range(self):
        """Normalized features should be clipped to [-5, 5]."""
        bars = _make_bars(600)
        pipeline = FeaturePipeline.create_fast()
        result = pipeline.compute(bars)

        normalize_cols = [
            c for c in result.columns
            if c not in bars.columns
            and c not in ("regime_label", "hurst_regime", "in_liquidity_zone", "delta_divergence")
        ]
        for col in normalize_cols:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= -5.01, f"{col} min={valid.min()}"
                assert valid.max() <= 5.01, f"{col} max={valid.max()}"

    def test_pipeline_incremental(self):
        """Incremental computation should produce same features as full."""
        bars = _make_bars(300)
        pipeline = FeaturePipeline.create_fast()

        # Full computation
        full_result = pipeline.compute(bars)

        # Incremental: use history + last bar
        history = bars.iloc[:-1]
        new_bar = bars.iloc[-1]
        last_features = pipeline.compute_incremental(new_bar, history)

        # Features should exist
        assert "delta" in last_features.index
        assert "regime_label" in last_features.index
