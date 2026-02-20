"""Multi-Timeframe Data Aligner — synchronizes D1/H1/M5 bars by timestamp.

Uses searchsorted for O(log n) lookups: given an H1 bar timestamp, finds the
corresponding D1 and M5 slices for the lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MTFSlice:
    """Aligned multi-timeframe data slices for a single H1 step."""

    d1: pd.DataFrame  # (d1_lookback, features) — daily context
    h1: pd.DataFrame  # (h1_lookback, features) — hourly primary
    m5: pd.DataFrame  # (m5_lookback, features) — 5-min detail


class MTFDataAligner:
    """Aligns D1, H1, and M5 DataFrames by UTC timestamp.

    All DataFrames must have a ``time`` column with UTC-aware timestamps
    (or naive UTC timestamps). The aligner uses ``np.searchsorted`` on the
    time column for efficient lookup.

    Typical usage::

        aligner = MTFDataAligner(d1_df, h1_df, m5_df)
        slc = aligner.get_slice(h1_idx=42)
        # slc.d1 -> last 5 D1 bars up to the H1 bar's time
        # slc.h1 -> last 20 H1 bars ending at h1_idx
        # slc.m5 -> last 20 M5 bars up to the H1 bar's time
    """

    def __init__(
        self,
        d1_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        m5_data: pd.DataFrame,
        d1_lookback: int = 5,
        h1_lookback: int = 20,
        m5_lookback: int = 20,
    ) -> None:
        self._d1 = d1_data.reset_index(drop=True)
        self._h1 = h1_data.reset_index(drop=True)
        self._m5 = m5_data.reset_index(drop=True)

        self._d1_lookback = d1_lookback
        self._h1_lookback = h1_lookback
        self._m5_lookback = m5_lookback

        # Pre-convert time columns to int64 (nanoseconds) for fast searchsorted
        self._d1_times = self._to_timestamps(self._d1)
        self._h1_times = self._to_timestamps(self._h1)
        self._m5_times = self._to_timestamps(self._m5)

        logger.debug(
            "MTFDataAligner initialized",
            d1_bars=len(self._d1),
            h1_bars=len(self._h1),
            m5_bars=len(self._m5),
        )

    @staticmethod
    def _to_timestamps(df: pd.DataFrame) -> np.ndarray:
        """Convert the 'time' column to int64 nanoseconds for searchsorted."""
        if "time" not in df.columns:
            # Fallback: use index as sequential time
            return np.arange(len(df), dtype=np.int64)
        times = pd.to_datetime(df["time"], utc=True)
        return times.astype(np.int64).values

    def get_slice(self, h1_idx: int) -> MTFSlice:
        """Get aligned MTF slices for the given H1 bar index.

        Parameters
        ----------
        h1_idx : int
            Index into the H1 DataFrame (the primary trading timeframe).

        Returns
        -------
        MTFSlice
            Contains D1, H1, M5 DataFrames for the lookback windows
            aligned to the H1 bar's timestamp.
        """
        # H1 slice — straightforward window
        h1_start = max(0, h1_idx - self._h1_lookback + 1)
        h1_slice = self._h1.iloc[h1_start: h1_idx + 1]

        # Get the H1 bar's timestamp for cross-timeframe alignment
        h1_time = self._h1_times[h1_idx]

        # D1 slice — find the latest D1 bar at or before the H1 timestamp
        d1_end_idx = int(np.searchsorted(self._d1_times, h1_time, side="right"))
        d1_start_idx = max(0, d1_end_idx - self._d1_lookback)
        d1_slice = self._d1.iloc[d1_start_idx: d1_end_idx]

        # M5 slice — find the latest M5 bars at or before the H1 timestamp
        m5_end_idx = int(np.searchsorted(self._m5_times, h1_time, side="right"))
        m5_start_idx = max(0, m5_end_idx - self._m5_lookback)
        m5_slice = self._m5.iloc[m5_start_idx: m5_end_idx]

        return MTFSlice(d1=d1_slice, h1=h1_slice, m5=m5_slice)

    @property
    def n_h1_bars(self) -> int:
        """Number of H1 bars available."""
        return len(self._h1)

    @property
    def min_h1_idx(self) -> int:
        """Minimum H1 index with enough lookback for all timeframes."""
        # Need at least h1_lookback H1 bars
        min_h1 = self._h1_lookback

        # Also need enough time to have d1_lookback D1 bars
        # A D1 bar covers ~24 H1 bars, so we need at least d1_lookback * 24 H1 bars
        min_for_d1 = self._d1_lookback * 24

        # And enough M5 bars (1 H1 = 12 M5, we need m5_lookback M5 bars)
        # Usually satisfied if we have enough H1 bars
        min_for_m5 = max(1, self._m5_lookback // 12)

        return max(min_h1, min_for_d1, min_for_m5)


def align_timeframes(
    d1_data: pd.DataFrame,
    h1_data: pd.DataFrame,
    m5_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure all three DataFrames cover the same time range.

    Trims each DataFrame to the overlapping time window so that
    the aligner can always find corresponding bars.

    Returns
    -------
    tuple
        (d1_trimmed, h1_trimmed, m5_trimmed)
    """
    d1_times = pd.to_datetime(d1_data["time"], utc=True)
    h1_times = pd.to_datetime(h1_data["time"], utc=True)
    m5_times = pd.to_datetime(m5_data["time"], utc=True)

    # Find overlapping range
    start = max(d1_times.min(), h1_times.min(), m5_times.min())
    end = min(d1_times.max(), h1_times.max(), m5_times.max())

    d1_trimmed = d1_data[(d1_times >= start) & (d1_times <= end)].reset_index(drop=True)
    h1_trimmed = h1_data[(h1_times >= start) & (h1_times <= end)].reset_index(drop=True)
    m5_trimmed = m5_data[(m5_times >= start) & (m5_times <= end)].reset_index(drop=True)

    logger.info(
        "Timeframes aligned",
        start=str(start),
        end=str(end),
        d1_bars=len(d1_trimmed),
        h1_bars=len(h1_trimmed),
        m5_bars=len(m5_trimmed),
    )

    return d1_trimmed, h1_trimmed, m5_trimmed
