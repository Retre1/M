"""Tests for volume-based bar aggregation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from apexfx.data.bar_aggregator import FinalizedBar
from apexfx.data.volume_bar_aggregator import (
    AdaptiveVolumeThreshold,
    VolumeBarAggregator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(seconds_offset: float, bid: float, ask: float, volume: float, base: datetime | None = None):
    """Create a tick tuple for convenience."""
    base = base or datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return base + timedelta(seconds=seconds_offset), bid, ask, volume


def _tick_df(ticks: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame from (time, bid, ask, volume) tuples."""
    return pd.DataFrame(ticks, columns=["time", "bid", "ask", "volume"])


# ---------------------------------------------------------------------------
# VolumeBarAggregator — basic threshold
# ---------------------------------------------------------------------------

class TestVolumeBarAggregatorBasic:
    def test_single_bar_finalization(self):
        """Bar finalizes when cumulative volume >= threshold."""
        agg = VolumeBarAggregator(volume_threshold=100.0, min_bar_duration_sec=0.0)

        # Feed ticks totalling 100 volume
        result = None
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        for i in range(10):
            result = agg.process_tick(
                time=base + timedelta(seconds=i),
                bid=1.1000, ask=1.1002, volume=10.0,
            )
        assert result is not None
        assert isinstance(result, FinalizedBar)
        assert result.volume == 100.0
        assert result.tick_count == 10
        assert agg.bars_generated == 1

    def test_no_finalization_below_threshold(self):
        """No bar produced when volume stays below threshold."""
        agg = VolumeBarAggregator(volume_threshold=100.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            result = agg.process_tick(
                time=base + timedelta(seconds=i),
                bid=1.1000, ask=1.1002, volume=10.0,
            )
            assert result is None
        assert agg.bars_generated == 0

    def test_exact_threshold_match(self):
        """Bar finalizes at exactly the threshold volume."""
        agg = VolumeBarAggregator(volume_threshold=50.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = agg.process_tick(time=base, bid=1.10, ask=1.10, volume=50.0)
        assert result is not None
        assert result.volume == 50.0

    def test_timeframe_label(self):
        """FinalizedBar.timeframe is set to V<threshold>."""
        agg = VolumeBarAggregator(volume_threshold=500.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        bar = agg.process_tick(time=base, bid=1.10, ask=1.10, volume=500.0)
        assert bar is not None
        assert bar.timeframe == "V500"


# ---------------------------------------------------------------------------
# OHLCV correctness
# ---------------------------------------------------------------------------

class TestOHLCV:
    def test_ohlcv_values(self):
        """Verify open/high/low/close/volume from known tick sequence."""
        agg = VolumeBarAggregator(volume_threshold=30.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # mid prices: 1.10, 1.12, 1.08
        agg.process_tick(time=base, bid=1.10, ask=1.10, volume=10.0)
        agg.process_tick(time=base + timedelta(seconds=1), bid=1.12, ask=1.12, volume=10.0)
        bar = agg.process_tick(time=base + timedelta(seconds=2), bid=1.08, ask=1.08, volume=10.0)

        assert bar is not None
        assert bar.open == pytest.approx(1.10)
        assert bar.high == pytest.approx(1.12)
        assert bar.low == pytest.approx(1.08)
        assert bar.close == pytest.approx(1.08)
        assert bar.volume == pytest.approx(30.0)
        assert bar.tick_count == 3

    def test_mid_price_calculation(self):
        """OHLC uses mid-price (bid+ask)/2."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        bar = agg.process_tick(time=base, bid=1.0998, ask=1.1002, volume=10.0)
        assert bar is not None
        assert bar.open == pytest.approx(1.1000)

    def test_vwap(self):
        """VWAP is volume-weighted average price."""
        agg = VolumeBarAggregator(volume_threshold=30.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        # price 1.10 * vol 10 + price 1.12 * vol 10 + price 1.08 * vol 10 = 32.90
        # vwap = 32.90 / 30 = 1.09666...
        agg.process_tick(time=base, bid=1.10, ask=1.10, volume=10.0)
        agg.process_tick(time=base + timedelta(seconds=1), bid=1.12, ask=1.12, volume=10.0)
        bar = agg.process_tick(time=base + timedelta(seconds=2), bid=1.08, ask=1.08, volume=10.0)
        assert bar is not None
        expected_vwap = (1.10 * 10 + 1.12 * 10 + 1.08 * 10) / 30.0
        assert bar.vwap == pytest.approx(expected_vwap)


# ---------------------------------------------------------------------------
# Min bar duration
# ---------------------------------------------------------------------------

class TestMinBarDuration:
    def test_min_duration_prevents_micro_bars(self):
        """Bar is NOT finalized if elapsed time < min_bar_duration_sec."""
        agg = VolumeBarAggregator(volume_threshold=100.0, min_bar_duration_sec=5.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Single large trade — volume met instantly
        result = agg.process_tick(time=base, bid=1.10, ask=1.10, volume=200.0)
        assert result is None, "Should not finalize because min duration not met"

    def test_finalization_after_min_duration(self):
        """Bar finalizes once volume threshold is met AND min duration has elapsed."""
        agg = VolumeBarAggregator(volume_threshold=100.0, min_bar_duration_sec=5.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Large trade at t=0
        agg.process_tick(time=base, bid=1.10, ask=1.10, volume=200.0)

        # Small tick at t=6s — triggers finalization (volume already met, now time met)
        bar = agg.process_tick(
            time=base + timedelta(seconds=6), bid=1.10, ask=1.10, volume=1.0,
        )
        assert bar is not None
        assert bar.volume == pytest.approx(201.0)


# ---------------------------------------------------------------------------
# Batch process_ticks
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def test_process_ticks_dataframe(self):
        """Batch processing produces correct bars from a DataFrame."""
        agg = VolumeBarAggregator(volume_threshold=50.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        ticks = _tick_df([
            (base + timedelta(seconds=i), 1.10, 1.10, 10.0) for i in range(12)
        ])
        bars = agg.process_ticks(ticks)

        # 12 ticks * 10 vol = 120 total -> should get 2 bars (50 + 50), leftover 20
        assert len(bars) == 2
        assert bars[0].volume == pytest.approx(50.0)
        assert bars[1].volume == pytest.approx(50.0)
        assert agg.bars_generated == 2

    def test_process_ticks_empty_dataframe(self):
        """Empty DataFrame yields no bars."""
        agg = VolumeBarAggregator(volume_threshold=50.0, min_bar_duration_sec=0.0)
        bars = agg.process_ticks(pd.DataFrame(columns=["time", "bid", "ask", "volume"]))
        assert bars == []


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_callback_fires_on_finalization(self):
        """Registered callback is called with each finalized bar."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        received = []
        agg.on_bar(lambda bar: received.append(bar))

        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        agg.process_tick(time=base, bid=1.10, ask=1.10, volume=10.0)

        assert len(received) == 1
        assert isinstance(received[0], FinalizedBar)

    def test_callback_error_does_not_propagate(self):
        """A failing callback does not prevent bar return."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        agg.on_bar(lambda bar: (_ for _ in ()).throw(ValueError("boom")))

        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        bar = agg.process_tick(time=base, bid=1.10, ask=1.10, volume=10.0)
        assert bar is not None  # bar still returned despite callback error

    def test_multiple_callbacks(self):
        """Multiple callbacks all receive the bar."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        r1, r2 = [], []
        agg.on_bar(lambda bar: r1.append(bar))
        agg.on_bar(lambda bar: r2.append(bar))

        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        agg.process_tick(time=base, bid=1.10, ask=1.10, volume=10.0)

        assert len(r1) == 1
        assert len(r2) == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        """Reset discards partial bar and resets counter."""
        agg = VolumeBarAggregator(volume_threshold=100.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Build up partial state
        agg.process_tick(time=base, bid=1.10, ask=1.10, volume=50.0)
        assert agg.bars_generated == 0

        # Generate one bar
        agg.process_tick(time=base + timedelta(seconds=1), bid=1.10, ask=1.10, volume=50.0)
        assert agg.bars_generated == 1

        agg.reset()
        assert agg.bars_generated == 0

        # After reset, need full threshold again
        result = agg.process_tick(
            time=base + timedelta(seconds=2), bid=1.10, ask=1.10, volume=50.0,
        )
        assert result is None  # only 50, threshold is 100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_volume_ticks(self):
        """Zero-volume ticks update price but don't contribute to volume."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Zero-volume tick
        result = agg.process_tick(time=base, bid=1.12, ask=1.12, volume=0.0)
        assert result is None

        # Now send volume
        bar = agg.process_tick(
            time=base + timedelta(seconds=1), bid=1.10, ask=1.10, volume=10.0,
        )
        assert bar is not None
        assert bar.high == pytest.approx(1.12)  # zero-vol tick's price is tracked
        assert bar.volume == pytest.approx(10.0)

    def test_negative_volume_clamped(self):
        """Negative volume is clamped to zero."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        result = agg.process_tick(time=base, bid=1.10, ask=1.10, volume=-5.0)
        assert result is None  # treated as 0 volume

    def test_single_tick_bar(self):
        """A single tick can finalize a bar if volume >= threshold."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        bar = agg.process_tick(time=base, bid=1.10, ask=1.10, volume=10.0)
        assert bar is not None
        assert bar.open == bar.close == bar.high == bar.low

    def test_invalid_mid_price_skipped(self):
        """Tick with zero mid-price is ignored."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = agg.process_tick(time=base, bid=0.0, ask=0.0, volume=10.0)
        assert result is None
        assert agg.bars_generated == 0

    def test_naive_datetime_gets_utc(self):
        """Naive datetimes are converted to UTC."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        naive_time = datetime(2025, 1, 1, 12, 0, 0)  # no tzinfo
        bar = agg.process_tick(time=naive_time, bid=1.10, ask=1.10, volume=10.0)
        assert bar is not None
        assert bar.time.tzinfo is not None

    def test_consecutive_bars(self):
        """Multiple bars can be produced in sequence."""
        agg = VolumeBarAggregator(volume_threshold=10.0, min_bar_duration_sec=0.0)
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        bars = []
        for i in range(30):
            bar = agg.process_tick(
                time=base + timedelta(seconds=i),
                bid=1.10, ask=1.10, volume=10.0,
            )
            if bar is not None:
                bars.append(bar)

        assert len(bars) == 30
        assert agg.bars_generated == 30

    def test_invalid_threshold_raises(self):
        """Zero or negative threshold raises ValueError."""
        with pytest.raises(ValueError):
            VolumeBarAggregator(volume_threshold=0.0)
        with pytest.raises(ValueError):
            VolumeBarAggregator(volume_threshold=-10.0)


# ---------------------------------------------------------------------------
# AdaptiveVolumeThreshold
# ---------------------------------------------------------------------------

class TestAdaptiveVolumeThreshold:
    def _make_bars(self, n: int, interval_sec: float) -> list[FinalizedBar]:
        """Create n bars spaced interval_sec apart."""
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        return [
            FinalizedBar(
                timeframe="V1000",
                time=base + timedelta(seconds=i * interval_sec),
                open=1.10, high=1.10, low=1.10, close=1.10,
                volume=1000.0, tick_count=10, vwap=1.10,
            )
            for i in range(n)
        ]

    def test_initial_threshold(self):
        """Initial threshold is the midpoint of min/max."""
        avt = AdaptiveVolumeThreshold(
            min_threshold=100.0, max_threshold=10000.0,
        )
        assert avt.get_threshold() == pytest.approx(5050.0)

    def test_threshold_increases_when_bars_too_fast(self):
        """Threshold increases when bars are produced faster than target."""
        avt = AdaptiveVolumeThreshold(
            target_bars_per_hour=12,  # 1 every 5 minutes
            min_threshold=100.0,
            max_threshold=100_000.0,
        )
        initial = avt.get_threshold()

        # Feed bars at 1/minute = 60/hour, way above target of 12/hour
        for bar in self._make_bars(20, interval_sec=60.0):
            avt.update(bar)

        assert avt.get_threshold() > initial

    def test_threshold_decreases_when_bars_too_slow(self):
        """Threshold decreases when bars are produced slower than target."""
        avt = AdaptiveVolumeThreshold(
            target_bars_per_hour=60,  # 1 per minute
            min_threshold=100.0,
            max_threshold=100_000.0,
        )
        initial = avt.get_threshold()

        # Feed bars at 1/hour = way below target of 60/hour
        for bar in self._make_bars(10, interval_sec=3600.0):
            avt.update(bar)

        assert avt.get_threshold() < initial

    def test_threshold_respects_min_bound(self):
        """Threshold never drops below min_threshold."""
        avt = AdaptiveVolumeThreshold(
            target_bars_per_hour=10000,
            min_threshold=500.0,
            max_threshold=100_000.0,
        )
        # Feed very slow bars
        for bar in self._make_bars(20, interval_sec=7200.0):
            avt.update(bar)

        assert avt.get_threshold() >= 500.0

    def test_threshold_respects_max_bound(self):
        """Threshold never exceeds max_threshold."""
        avt = AdaptiveVolumeThreshold(
            target_bars_per_hour=1,
            min_threshold=100.0,
            max_threshold=5000.0,
        )
        # Feed very fast bars
        for bar in self._make_bars(50, interval_sec=1.0):
            avt.update(bar)

        assert avt.get_threshold() <= 5000.0

    def test_single_bar_no_crash(self):
        """A single bar update does not crash (needs at least 2 for rate)."""
        avt = AdaptiveVolumeThreshold()
        bar = self._make_bars(1, interval_sec=0.0)[0]
        avt.update(bar)  # Should not raise
        # Threshold unchanged with only 1 sample
        assert avt.get_threshold() > 0

    def test_invalid_target_raises(self):
        """Zero or negative target_bars_per_hour raises ValueError."""
        with pytest.raises(ValueError):
            AdaptiveVolumeThreshold(target_bars_per_hour=0)
        with pytest.raises(ValueError):
            AdaptiveVolumeThreshold(target_bars_per_hour=-5)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestVolumeBarConfig:
    def test_default_config(self):
        """VolumeBarConfig has sensible defaults."""
        from apexfx.config.schema import VolumeBarConfig
        cfg = VolumeBarConfig()
        assert cfg.enabled is False
        assert cfg.threshold == 1000.0
        assert cfg.adaptive is True
        assert cfg.target_bars_per_hour == 12
        assert cfg.min_bar_duration_sec == 1.0

    def test_data_config_includes_volume_bars(self):
        """DataConfig has volume_bars field."""
        from apexfx.config.schema import DataConfig
        dc = DataConfig()
        assert hasattr(dc, "volume_bars")
        assert dc.volume_bars.enabled is False
