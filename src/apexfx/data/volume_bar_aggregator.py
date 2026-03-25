"""Volume-based bar aggregation — samples when cumulative volume hits a threshold.

Removes noise from low-liquidity periods (e.g., Asian session for EUR/USD)
by producing bars at a constant *information* rate rather than constant *time* rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from apexfx.data.bar_aggregator import FinalizedBar, PartialBar
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class VolumeBarAggregator:
    """Aggregates ticks into OHLCV bars based on cumulative volume thresholds.

    Unlike time-based bars, volume bars finalize when accumulated volume reaches
    ``volume_threshold``, producing more bars during active markets and fewer
    during quiet periods.

    Parameters
    ----------
    volume_threshold:
        Cumulative tick volume required to finalize a bar.
    min_bar_duration_sec:
        Minimum elapsed seconds between bar open and close.  Prevents
        micro-bars from single large trades.
    """

    def __init__(
        self,
        volume_threshold: float,
        min_bar_duration_sec: float = 1.0,
    ) -> None:
        if volume_threshold <= 0:
            raise ValueError("volume_threshold must be positive")

        self._volume_threshold = volume_threshold
        self._min_bar_duration_sec = min_bar_duration_sec
        self._timeframe_label = f"V{int(volume_threshold)}"

        self._partial: PartialBar | None = None
        self._bars_generated: int = 0
        self._on_bar_callbacks: list = []

    # --- public API --------------------------------------------------------

    @property
    def bars_generated(self) -> int:
        """Number of finalized bars produced since creation / last reset."""
        return self._bars_generated

    def on_bar(self, callback) -> None:
        """Register callback invoked with each :class:`FinalizedBar`."""
        self._on_bar_callbacks.append(callback)

    def reset(self) -> None:
        """Discard partial bar state and reset counter."""
        self._partial = None
        self._bars_generated = 0

    def process_tick(
        self,
        time: datetime,
        bid: float,
        ask: float,
        volume: float,
    ) -> FinalizedBar | None:
        """Ingest a single tick, returning a bar if the threshold was reached.

        Mid-price ``(bid + ask) / 2`` is used for OHLC computation.
        Ticks with ``volume <= 0`` are treated as zero-volume updates (price
        still updates the partial bar but won't push volume over threshold).
        """
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None

        # Clamp negative volumes to zero — still update price
        tick_vol = max(volume, 0.0)

        if time.tzinfo is None:
            time = time.replace(tzinfo=UTC)

        partial = self._partial
        if partial is None:
            partial = PartialBar(
                timeframe=self._timeframe_label,
                open_time=time,
                open=mid,
                high=mid,
                low=mid,
                close=mid,
                volume=tick_vol,
                tick_count=1,
                vwap_numerator=mid * tick_vol,
            )
            self._partial = partial
        else:
            partial.high = max(partial.high, mid)
            partial.low = min(partial.low, mid)
            partial.close = mid
            partial.volume += tick_vol
            partial.tick_count += 1
            partial.vwap_numerator += mid * tick_vol

        # Check finalization criteria
        if partial.volume >= self._volume_threshold:
            elapsed = (time - partial.open_time).total_seconds()
            if elapsed < self._min_bar_duration_sec:
                # Not enough time — keep accumulating
                return None
            return self._finalize(time)

        return None

    def process_ticks(self, ticks: pd.DataFrame) -> list[FinalizedBar]:
        """Batch-process a DataFrame of ticks.

        Expected columns: ``time``, ``bid``, ``ask``, ``volume``.
        """
        finalized: list[FinalizedBar] = []
        for _, row in ticks.iterrows():
            tick_time = row["time"]
            if hasattr(tick_time, "to_pydatetime"):
                tick_time = tick_time.to_pydatetime()
            if tick_time.tzinfo is None:
                tick_time = tick_time.replace(tzinfo=UTC)

            bar = self.process_tick(
                time=tick_time,
                bid=row.get("bid", 0.0),
                ask=row.get("ask", 0.0),
                volume=row.get("volume", 0.0),
            )
            if bar is not None:
                finalized.append(bar)

        return finalized

    # --- internals ---------------------------------------------------------

    def _finalize(self, close_time: datetime) -> FinalizedBar:
        """Convert the current partial bar into a finalized bar."""
        p = self._partial
        assert p is not None

        bar = FinalizedBar(
            timeframe=self._timeframe_label,
            time=p.open_time,
            open=p.open,
            high=p.high,
            low=p.low,
            close=p.close,
            volume=p.volume,
            tick_count=p.tick_count,
            vwap=p.vwap,
        )

        self._partial = None
        self._bars_generated += 1

        for cb in self._on_bar_callbacks:
            try:
                cb(bar)
            except Exception as e:
                logger.error("Volume bar callback error", error=str(e))

        return bar


@dataclass
class _BarRecord:
    """Lightweight timestamp record for adaptive threshold tracking."""
    time: datetime


class AdaptiveVolumeThreshold:
    """Dynamically adjusts the volume threshold to target a desired bar frequency.

    Uses exponential smoothing of the observed bars-per-hour rate and adjusts
    the threshold proportionally.  If bars are produced too fast, the threshold
    increases; if too slow, it decreases.

    Parameters
    ----------
    target_bars_per_hour:
        Desired average number of bars per hour.
    adjustment_window:
        Number of recent bars used to estimate the current rate.
    min_threshold:
        Lower bound for the volume threshold.
    max_threshold:
        Upper bound for the volume threshold.
    """

    def __init__(
        self,
        target_bars_per_hour: int = 12,
        adjustment_window: int = 60,
        min_threshold: float = 100.0,
        max_threshold: float = 100_000.0,
    ) -> None:
        if target_bars_per_hour <= 0:
            raise ValueError("target_bars_per_hour must be positive")

        self._target_bph = target_bars_per_hour
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._smoothing_alpha = 0.1  # EMA smoothing factor

        self._bar_times: deque[datetime] = deque(maxlen=adjustment_window)
        self._current_threshold = (min_threshold + max_threshold) / 2.0
        self._smoothed_bph: float | None = None

    @property
    def current_threshold(self) -> float:
        return self._current_threshold

    def update(self, bar: FinalizedBar) -> None:
        """Record a bar and recalculate the adaptive threshold."""
        bar_time = bar.time
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=UTC)

        self._bar_times.append(bar_time)

        if len(self._bar_times) < 2:
            return

        # Compute observed bars per hour from the window
        span_seconds = (
            self._bar_times[-1] - self._bar_times[0]
        ).total_seconds()
        if span_seconds <= 0:
            return

        observed_bph = (len(self._bar_times) - 1) / (span_seconds / 3600.0)

        # Exponential smoothing
        if self._smoothed_bph is None:
            self._smoothed_bph = observed_bph
        else:
            self._smoothed_bph = (
                self._smoothing_alpha * observed_bph
                + (1 - self._smoothing_alpha) * self._smoothed_bph
            )

        # Adjust threshold: if producing too many bars, increase threshold
        if self._smoothed_bph > 0:
            ratio = self._smoothed_bph / self._target_bph
            self._current_threshold = max(
                self._min_threshold,
                min(self._max_threshold, self._current_threshold * ratio),
            )

    def get_threshold(self) -> float:
        """Return the current adaptive volume threshold."""
        return self._current_threshold
