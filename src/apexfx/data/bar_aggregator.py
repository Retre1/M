"""Aggregate raw ticks into OHLCV bars at multiple timeframes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


@dataclass
class PartialBar:
    """Accumulator for an in-progress bar."""
    timeframe: str
    open_time: datetime
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0
    vwap_numerator: float = 0.0

    @property
    def vwap(self) -> float:
        return self.vwap_numerator / self.volume if self.volume > 0 else self.close


@dataclass
class FinalizedBar:
    """A completed OHLCV bar."""
    timeframe: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    vwap: float


class BarAggregator:
    """Aggregates ticks into OHLCV bars at multiple timeframes simultaneously."""

    def __init__(self, timeframes: list[str] | None = None) -> None:
        self._timeframes = timeframes or ["M1", "M5", "H1", "D1"]
        self._partial_bars: dict[str, PartialBar | None] = {tf: None for tf in self._timeframes}
        self._on_bar_callbacks: list = []

    def on_bar(self, callback) -> None:
        """Register callback for finalized bars: callback(FinalizedBar)."""
        self._on_bar_callbacks.append(callback)

    def process_ticks(self, ticks: pd.DataFrame) -> list[FinalizedBar]:
        """Process a batch of ticks, returning any finalized bars."""
        finalized: list[FinalizedBar] = []

        for _, tick in ticks.iterrows():
            tick_time = tick["time"]
            if hasattr(tick_time, "to_pydatetime"):
                tick_time = tick_time.to_pydatetime()
            if tick_time.tzinfo is None:
                tick_time = tick_time.replace(tzinfo=timezone.utc)

            mid_price = (tick.get("bid", 0) + tick.get("ask", 0)) / 2
            if mid_price <= 0:
                mid_price = tick.get("last", 0)
            if mid_price <= 0:
                continue

            tick_volume = tick.get("volume", 1.0)
            if tick_volume <= 0:
                tick_volume = 1.0

            for tf in self._timeframes:
                bars = self._process_single_tick(tf, tick_time, mid_price, tick_volume)
                finalized.extend(bars)

        for bar in finalized:
            for cb in self._on_bar_callbacks:
                try:
                    cb(bar)
                except Exception as e:
                    logger.error("Bar callback error", error=str(e))

        return finalized

    def _process_single_tick(
        self, tf: str, tick_time: datetime, price: float, volume: float
    ) -> list[FinalizedBar]:
        """Process one tick for one timeframe."""
        finalized = []
        bar_open_time = self._get_bar_open_time(tick_time, tf)
        partial = self._partial_bars[tf]

        if partial is not None and partial.open_time != bar_open_time:
            # Bar boundary crossed — finalize the old bar
            finalized.append(FinalizedBar(
                timeframe=tf,
                time=partial.open_time,
                open=partial.open,
                high=partial.high,
                low=partial.low,
                close=partial.close,
                volume=partial.volume,
                tick_count=partial.tick_count,
                vwap=partial.vwap,
            ))
            partial = None

        if partial is None:
            partial = PartialBar(
                timeframe=tf,
                open_time=bar_open_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                tick_count=1,
                vwap_numerator=price * volume,
            )
        else:
            partial.high = max(partial.high, price)
            partial.low = min(partial.low, price)
            partial.close = price
            partial.volume += volume
            partial.tick_count += 1
            partial.vwap_numerator += price * volume

        self._partial_bars[tf] = partial
        return finalized

    @staticmethod
    def _get_bar_open_time(tick_time: datetime, tf: str) -> datetime:
        """Compute the opening time of the bar that contains tick_time."""
        minutes = TIMEFRAME_MINUTES[tf]

        if minutes >= 1440:  # Daily
            return tick_time.replace(hour=0, minute=0, second=0, microsecond=0)

        total_minutes = tick_time.hour * 60 + tick_time.minute
        bar_start_minute = (total_minutes // minutes) * minutes
        hour = bar_start_minute // 60
        minute = bar_start_minute % 60

        return tick_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def bars_to_dataframe(self, bars: list[FinalizedBar]) -> pd.DataFrame:
        """Convert finalized bars to a DataFrame."""
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "time": b.time,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "tick_count": b.tick_count,
                "vwap": b.vwap,
            }
            for b in bars
        ])
