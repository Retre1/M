"""Async tick collector with ring buffer and periodic flush to storage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from apexfx.data.data_store import DataStore
from apexfx.data.mt5_client import MT5Client
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class TickCollector:
    """Polls MT5 for ticks and buffers them in a ring buffer."""

    def __init__(
        self,
        mt5_client: MT5Client,
        data_store: DataStore,
        symbol: str,
        buffer_size: int = 100_000,
        poll_interval_ms: int = 100,
        flush_interval_s: int = 60,
    ) -> None:
        self._mt5 = mt5_client
        self._store = data_store
        self._symbol = symbol
        self._poll_interval = poll_interval_ms / 1000.0
        self._flush_interval = flush_interval_s

        # Ring buffer for ticks
        self._buffer_size = buffer_size
        self._buffer: np.ndarray = np.zeros(
            buffer_size,
            dtype=[
                ("time", "datetime64[us]"),
                ("bid", "f8"),
                ("ask", "f8"),
                ("last", "f8"),
                ("volume", "f8"),
                ("flags", "i4"),
            ],
        )
        self._write_idx = 0
        self._count = 0
        self._last_tick_time: datetime | None = None
        self._running = False

        # Callbacks for downstream consumers
        self._on_tick_callbacks: list = []

    def on_tick(self, callback) -> None:
        """Register a callback for new tick events."""
        self._on_tick_callbacks.append(callback)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._count

    def get_recent_ticks(self, n: int = 1000) -> pd.DataFrame:
        """Get the most recent n ticks from the buffer."""
        actual_count = min(n, self._count)
        if actual_count == 0:
            return pd.DataFrame(columns=["time", "bid", "ask", "last", "volume", "flags"])

        if self._count <= self._buffer_size:
            start = max(0, self._write_idx - actual_count)
            data = self._buffer[start : self._write_idx]
        else:
            end = self._write_idx
            start = end - actual_count
            if start >= 0:
                data = self._buffer[start:end]
            else:
                data = np.concatenate([
                    self._buffer[start % self._buffer_size :],
                    self._buffer[:end],
                ])

        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    async def start(self) -> None:
        """Start the tick collection loop."""
        self._running = True
        logger.info("Tick collector started", symbol=self._symbol)

        flush_task = asyncio.create_task(self._flush_loop())
        try:
            while self._running:
                await self._poll_ticks()
                await asyncio.sleep(self._poll_interval)
        finally:
            flush_task.cancel()
            await self._flush_to_storage()
            logger.info("Tick collector stopped", symbol=self._symbol)

    def stop(self) -> None:
        """Signal the collector to stop."""
        self._running = False

    async def _poll_ticks(self) -> None:
        """Poll MT5 for new ticks."""
        try:
            from_dt = self._last_tick_time or datetime.now(timezone.utc)
            ticks_df = self._mt5.get_ticks(self._symbol, from_dt, count=1000)

            if ticks_df.empty:
                return

            # Filter only new ticks
            if self._last_tick_time is not None:
                ticks_df = ticks_df[ticks_df["time"] > self._last_tick_time]

            if ticks_df.empty:
                return

            self._last_tick_time = ticks_df["time"].iloc[-1]

            # Write to ring buffer
            for _, row in ticks_df.iterrows():
                idx = self._write_idx % self._buffer_size
                self._buffer[idx] = (
                    row["time"].to_datetime64(),
                    row.get("bid", 0.0),
                    row.get("ask", 0.0),
                    row.get("last", 0.0),
                    row.get("volume", 0.0),
                    row.get("flags", 0),
                )
                self._write_idx += 1
                self._count += 1

            # Notify callbacks
            for cb in self._on_tick_callbacks:
                try:
                    cb(ticks_df)
                except Exception as e:
                    logger.error("Tick callback error", error=str(e))

        except ConnectionError:
            logger.warning("MT5 connection lost during tick poll")
        except Exception as e:
            logger.error("Tick poll error", error=str(e))

    async def _flush_loop(self) -> None:
        """Periodically flush buffer to persistent storage."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush_to_storage()

    async def _flush_to_storage(self) -> None:
        """Flush current buffer contents to parquet storage."""
        recent = self.get_recent_ticks(n=self._count)
        if not recent.empty:
            self._store.append_ticks(self._symbol, recent)
            logger.debug("Flushed ticks", symbol=self._symbol, count=len(recent))
