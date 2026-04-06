"""Tests for BarDataValidator."""

import numpy as np
import pandas as pd
import pytest

from apexfx.data.validator import BarDataValidator, ValidationReport


def _make_good_bars(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.0005),
        "low": prices - np.abs(np.random.randn(n) * 0.0005),
        "close": prices + np.random.randn(n) * 0.0002,
        "volume": np.random.lognormal(10, 1, n),
    })


class TestBarDataValidator:
    def test_good_data_passes(self):
        bars = _make_good_bars()
        v = BarDataValidator()
        report = v.validate(bars)
        assert report.is_valid

    def test_missing_columns(self):
        bars = pd.DataFrame({"time": [1, 2, 3], "close": [1.1, 1.2, 1.3]})
        v = BarDataValidator()
        report = v.validate(bars)
        assert not report.is_valid
        assert any(i.check == "required_columns" for i in report.issues)

    def test_too_few_bars(self):
        bars = _make_good_bars(10)
        v = BarDataValidator(min_bars=100)
        report = v.validate(bars)
        assert not report.is_valid
        assert any(i.check == "min_bars" for i in report.issues)

    def test_duplicate_timestamps(self):
        bars = _make_good_bars(200)
        # Duplicate the first timestamp
        bars.loc[1, "time"] = bars.loc[0, "time"]
        v = BarDataValidator()
        report = v.validate(bars)
        assert any(i.check == "duplicates" for i in report.issues)

    def test_nan_detection(self):
        bars = _make_good_bars(200)
        # Inject 50% NaN in close
        bars.loc[:99, "close"] = np.nan
        v = BarDataValidator(max_nan_pct=0.10)
        report = v.validate(bars)
        assert not report.is_valid
        assert any(i.check == "nan_values" and i.severity == "error" for i in report.issues)

    def test_time_gap_warning(self):
        bars = _make_good_bars(200)
        # Create a large gap
        bars.loc[100, "time"] = bars.loc[99, "time"] + pd.Timedelta(days=10)
        for i in range(101, 200):
            bars.loc[i, "time"] = bars.loc[i - 1, "time"] + pd.Timedelta(hours=1)
        v = BarDataValidator(max_gap_factor=5.0)
        report = v.validate(bars)
        assert any(i.check == "time_gaps" for i in report.issues)

    def test_ohlc_consistency(self):
        bars = _make_good_bars(200)
        # high < close is invalid
        bars.loc[50, "high"] = bars.loc[50, "close"] - 0.01
        v = BarDataValidator()
        report = v.validate(bars)
        assert any(i.check == "ohlc_consistency" for i in report.issues)

    def test_negative_price(self):
        bars = _make_good_bars(200)
        bars.loc[100, "close"] = -1.0
        v = BarDataValidator()
        report = v.validate(bars)
        assert not report.is_valid
        assert any(i.check == "price_sanity" for i in report.issues)

    def test_negative_volume(self):
        bars = _make_good_bars(200)
        bars.loc[50, "volume"] = -100
        v = BarDataValidator()
        report = v.validate(bars)
        assert not report.is_valid
        assert any(i.check == "volume_sanity" and i.severity == "error" for i in report.issues)

    def test_clean_removes_duplicates(self):
        bars = _make_good_bars(200)
        bars.loc[1, "time"] = bars.loc[0, "time"]
        v = BarDataValidator()
        cleaned = v.clean(bars)
        assert cleaned["time"].is_unique

    def test_clean_sorts_by_time(self):
        bars = _make_good_bars(200)
        shuffled = bars.sample(frac=1, random_state=42)
        v = BarDataValidator()
        cleaned = v.clean(shuffled)
        times = cleaned["time"].values
        assert (times[1:] >= times[:-1]).all()

    def test_clean_fixes_ohlc(self):
        bars = _make_good_bars(200)
        bars.loc[50, "high"] = bars.loc[50, "low"] - 0.01
        v = BarDataValidator()
        cleaned = v.clean(bars)
        assert (cleaned["high"] >= cleaned["low"]).all()

    def test_clean_ffill_nan(self):
        bars = _make_good_bars(200)
        bars.loc[100, "close"] = np.nan
        v = BarDataValidator()
        cleaned = v.clean(bars)
        assert cleaned["close"].notna().all()

    def test_report_summary(self):
        bars = _make_good_bars()
        v = BarDataValidator()
        report = v.validate(bars)
        summary = report.summary()
        assert "500 bars" in summary
