"""Parquet-based columnar storage layer for tick and bar data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class DataStore:
    """Parquet storage partitioned by symbol/timeframe/date."""

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _ticks_path(self, symbol: str, date_str: str) -> Path:
        return self._data_dir / "raw" / "ticks" / symbol / f"{date_str}.parquet"

    def _bars_path(self, symbol: str, timeframe: str, date_str: str) -> Path:
        return self._data_dir / "raw" / "bars" / symbol / timeframe / f"{date_str}.parquet"

    def _features_path(self, symbol: str, date_str: str) -> Path:
        return self._data_dir / "processed" / "features" / symbol / f"{date_str}.parquet"

    def append_ticks(self, symbol: str, df: pd.DataFrame) -> None:
        """Append tick data, partitioned by date."""
        if df.empty:
            return

        for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
            path = self._ticks_path(symbol, str(date_str))
            self._append_parquet(path, group)

        logger.debug("Ticks stored", symbol=symbol, count=len(df))

    def append_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Append OHLCV bar data, partitioned by date."""
        if df.empty:
            return

        for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
            path = self._bars_path(symbol, timeframe, str(date_str))
            self._append_parquet(path, group)

        logger.debug("Bars stored", symbol=symbol, timeframe=timeframe, count=len(df))

    def read_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Read bar data for a symbol/timeframe within a date range."""
        bars_dir = self._data_dir / "raw" / "bars" / symbol / timeframe
        if not bars_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(bars_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        if start:
            start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
            parquet_files = [f for f in parquet_files if f.stem >= start_str]
        if end:
            end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
            parquet_files = [f for f in parquet_files if f.stem <= end_str]

        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        result = pd.concat(dfs, ignore_index=True)

        if start:
            result = result[result["time"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            result = result[result["time"] <= pd.Timestamp(end, tz="UTC")]

        return result.sort_values("time").reset_index(drop=True)

    def read_ticks(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Read tick data for a symbol within a date range."""
        ticks_dir = self._data_dir / "raw" / "ticks" / symbol
        if not ticks_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(ticks_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        if start:
            start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
            parquet_files = [f for f in parquet_files if f.stem >= start_str]
        if end:
            end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
            parquet_files = [f for f in parquet_files if f.stem <= end_str]

        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)

    def save_features(self, symbol: str, df: pd.DataFrame) -> None:
        """Save computed feature matrices."""
        if df.empty:
            return
        for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
            path = self._features_path(symbol, str(date_str))
            self._write_parquet(path, group)

    def read_features(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Read pre-computed feature matrices."""
        feat_dir = self._data_dir / "processed" / "features" / symbol
        if not feat_dir.exists():
            return pd.DataFrame()

        parquet_files = sorted(feat_dir.glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        if start:
            start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
            parquet_files = [f for f in parquet_files if f.stem >= start_str]
        if end:
            end_str = pd.Timestamp(end).strftime("%Y-%m-%d")
            parquet_files = [f for f in parquet_files if f.stem <= end_str]

        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)

    @staticmethod
    def _append_parquet(path: Path, df: pd.DataFrame) -> None:
        """Append data to a parquet file, creating if necessary."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression="snappy")

    @staticmethod
    def _write_parquet(path: Path, df: pd.DataFrame) -> None:
        """Write data to a parquet file (overwrite)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
