"""Fetch correlated instruments (DXY, Gold, US Treasuries, SPX) for intermarket analysis.

Data source priority:
1. MT5 (live trading)
2. Cached Parquet files (training / offline)
3. yfinance download (auto-bootstrap)
4. Empty fallback (last resort — logged as warning)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Default intermarket instruments and their MT5 symbol names
INTERMARKET_SYMBOLS: dict[str, list[str]] = {
    "DXY": ["DXY", "USDX", "DX"],
    "XAUUSD": ["XAUUSD", "GOLD"],
    "US10Y": ["US10Y", "TNX", "US10YR"],
    "SPX": ["US500", "SP500", "SPX500", "US500.cash", "SPX", "S&P500"],
}

# yfinance ticker mapping for each intermarket instrument
YFINANCE_TICKERS: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "XAUUSD": "GC=F",
    "US10Y": "^TNX",
    "SPX": "^GSPC",
}


class IntermarketDataProvider:
    """Provides aligned intermarket data from MT5, cached files, or yfinance."""

    def __init__(
        self,
        mt5_client=None,
        data_dir: str | Path | None = None,
    ) -> None:
        self._mt5 = mt5_client
        self._symbol_cache: dict[str, str] = {}
        self._data_dir = Path(data_dir) if data_dir else self._default_data_dir()

    @staticmethod
    def _default_data_dir() -> Path:
        """Default data directory from config or fallback."""
        try:
            from apexfx.config.schema import AppConfig
            cfg = AppConfig()
            return Path(cfg.base.paths.data_dir) / "processed"
        except Exception:
            return Path("data") / "processed"

    def _resolve_symbol(self, instrument: str) -> str | None:
        """Find the actual MT5 symbol name for an instrument."""
        if instrument in self._symbol_cache:
            return self._symbol_cache[instrument]

        candidates = INTERMARKET_SYMBOLS.get(instrument, [instrument])
        if self._mt5 is None:
            return None

        for candidate in candidates:
            try:
                info = self._mt5.get_symbol_info(candidate)
                if info:
                    self._symbol_cache[instrument] = candidate
                    return candidate
            except (ValueError, ConnectionError):
                continue

        logger.warning("Intermarket symbol not found on MT5", instrument=instrument)
        return None

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def get_bars(
        self,
        instrument: str,
        timeframe: str = "H1",
        from_dt: datetime | None = None,
        count: int = 500,
    ) -> pd.DataFrame:
        """Get bars for an intermarket instrument.

        Tries in order: MT5 → Parquet cache → yfinance → empty fallback.
        """
        # 1. Try MT5
        if self._mt5 is not None:
            df = self._try_mt5(instrument, timeframe, from_dt, count)
            if df is not None and not df.empty:
                return df

        # 2. Try cached Parquet
        df = self._try_parquet_cache(instrument, timeframe)
        if df is not None and not df.empty:
            if count and len(df) > count:
                df = df.tail(count).reset_index(drop=True)
            return df

        # 3. Try yfinance download + cache
        df = self._try_yfinance(instrument, timeframe, count)
        if df is not None and not df.empty:
            return df

        # 4. Empty fallback
        logger.warning(
            "No intermarket data available — features will be NaN",
            instrument=instrument,
        )
        return pd.DataFrame(columns=["time", f"{instrument}_close"])

    def get_all_intermarket(
        self,
        instruments: list[str],
        timeframe: str = "H1",
        from_dt: datetime | None = None,
        count: int = 500,
    ) -> pd.DataFrame:
        """Get aligned bars for all intermarket instruments."""
        dfs: list[pd.DataFrame] = []

        for instrument in instruments:
            df = self.get_bars(instrument, timeframe, from_dt, count)
            if not df.empty:
                dfs.append(df)
                logger.debug(
                    "Intermarket data loaded",
                    instrument=instrument,
                    rows=len(df),
                )

        if not dfs:
            logger.warning("No intermarket data loaded for any instrument")
            return pd.DataFrame()

        # Merge on time column
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on="time", how="outer")

        result = result.sort_values("time").reset_index(drop=True)
        result = result.ffill()  # Forward-fill gaps
        result = result.bfill()  # Back-fill leading NaNs

        logger.info(
            "Intermarket data merged",
            instruments=len(dfs),
            rows=len(result),
            columns=list(result.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _try_mt5(
        self,
        instrument: str,
        timeframe: str,
        from_dt: datetime | None,
        count: int,
    ) -> pd.DataFrame | None:
        """Try loading from MT5."""
        symbol = self._resolve_symbol(instrument)
        if symbol is None:
            return None
        try:
            df = self._mt5.get_bars(symbol, timeframe, from_dt, count)
            if not df.empty:
                df = df.rename(columns={"close": f"{instrument}_close"})
                return df[["time", f"{instrument}_close"]]
        except Exception as e:
            logger.debug("MT5 intermarket fetch failed", instrument=instrument, error=str(e))
        return None

    def _try_parquet_cache(
        self,
        instrument: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Try loading from cached Parquet file."""
        parquet_path = self._data_dir / instrument / timeframe / "data.parquet"
        if not parquet_path.exists():
            return None

        try:
            df = pd.read_parquet(parquet_path)
            if "close" in df.columns and "time" in df.columns:
                df = df[["time", "close"]].rename(
                    columns={"close": f"{instrument}_close"}
                )
                logger.debug(
                    "Loaded intermarket from Parquet cache",
                    instrument=instrument,
                    rows=len(df),
                    path=str(parquet_path),
                )
                return df
        except Exception as e:
            logger.debug("Parquet cache read failed", instrument=instrument, error=str(e))
        return None

    def _try_yfinance(
        self,
        instrument: str,
        timeframe: str,
        count: int,
    ) -> pd.DataFrame | None:
        """Try downloading from yfinance and cache the result."""
        ticker = YFINANCE_TICKERS.get(instrument)
        if ticker is None:
            return None

        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance not installed — cannot download intermarket data")
            return None

        try:
            # Map timeframe to yfinance interval + period
            interval, period = self._yf_params(timeframe, count)

            logger.info(
                "Downloading intermarket data from yfinance",
                instrument=instrument,
                ticker=ticker,
                interval=interval,
                period=period,
            )

            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                logger.warning("yfinance returned empty data", ticker=ticker)
                return None

            # Normalize to our format
            df = pd.DataFrame({
                "time": data.index.tz_localize(timezone.utc) if data.index.tz is None else data.index.tz_convert(timezone.utc),
                f"{instrument}_close": data["Close"].values.flatten(),
            })
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

            # Cache to Parquet for next time
            self._save_parquet_cache(instrument, timeframe, df)

            logger.info(
                "Downloaded intermarket data",
                instrument=instrument,
                rows=len(df),
            )
            return df

        except Exception as e:
            logger.warning(
                "yfinance download failed",
                instrument=instrument,
                ticker=ticker,
                error=str(e),
            )
            return None

    def _save_parquet_cache(
        self,
        instrument: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """Save downloaded data to Parquet cache."""
        try:
            cache_dir = self._data_dir / instrument / timeframe
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "data.parquet"

            # Save in canonical format (time, close)
            col = f"{instrument}_close"
            save_df = df[["time", col]].rename(columns={col: "close"})
            save_df.to_parquet(cache_path, index=False)

            logger.debug(
                "Cached intermarket data to Parquet",
                instrument=instrument,
                path=str(cache_path),
                rows=len(save_df),
            )
        except Exception as e:
            logger.debug("Failed to cache Parquet", error=str(e))

    @staticmethod
    def _yf_params(timeframe: str, count: int) -> tuple[str, str]:
        """Convert our timeframe to yfinance interval + period."""
        tf_map = {
            "M1": ("1m", "7d"),
            "M5": ("5m", "60d"),
            "M15": ("15m", "60d"),
            "M30": ("30m", "60d"),
            "H1": ("1h", "730d"),
            "H4": ("1h", "730d"),  # yf doesn't have 4h, use 1h
            "D1": ("1d", "5y"),
        }
        return tf_map.get(timeframe, ("1h", "730d"))

    # ------------------------------------------------------------------
    # Bootstrap utility
    # ------------------------------------------------------------------

    def bootstrap_all(
        self,
        instruments: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, int]:
        """Download and cache all intermarket data. Returns {instrument: n_rows}.

        Call this once before training to populate the Parquet cache.

        Usage:
            provider = IntermarketDataProvider()
            stats = provider.bootstrap_all()
            print(stats)  # {'DXY': 5000, 'XAUUSD': 5000, 'US10Y': 5000, 'SPX': 5000}
        """
        instruments = instruments or list(INTERMARKET_SYMBOLS.keys())
        timeframes = timeframes or ["H1", "D1"]

        stats: dict[str, int] = {}
        for instrument in instruments:
            total = 0
            for tf in timeframes:
                df = self._try_yfinance(instrument, tf, count=5000)
                if df is not None and not df.empty:
                    total += len(df)
            stats[instrument] = total
            logger.info(
                "Bootstrap complete",
                instrument=instrument,
                total_rows=total,
            )

        return stats
