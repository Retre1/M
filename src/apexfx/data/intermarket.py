"""Fetch correlated instruments (DXY, Gold, US Treasuries) for intermarket analysis."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from apexfx.data.mt5_client import MT5Client
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Default intermarket instruments and their MT5/OANDA symbol name candidates.
# The provider tries each name in order and uses the first one that resolves.
INTERMARKET_SYMBOLS: dict[str, list[str]] = {
    "DXY": ["DXY", "USDX", "DX"],
    "XAUUSD": ["XAUUSD", "GOLD"],
    "US10Y": ["US10Y", "TNX", "US10YR"],
    # S&P 500 — different brokers use different symbol names
    "SP500": ["SP500", "US500", "SPX500", "SPXUSD", "SP500m", ".SPX", "SPX"],
}


class IntermarketDataProvider:
    """Provides aligned intermarket data from MT5 or fallback sources."""

    def __init__(self, mt5_client: MT5Client | None = None) -> None:
        self._mt5 = mt5_client
        self._symbol_cache: dict[str, str] = {}

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

    def get_bars(
        self,
        instrument: str,
        timeframe: str = "H1",
        from_dt: datetime | None = None,
        count: int = 500,
    ) -> pd.DataFrame:
        """Get bars for an intermarket instrument."""
        if self._mt5 is None:
            return self._fallback_data(instrument, timeframe, count)

        symbol = self._resolve_symbol(instrument)
        if symbol is None:
            return self._fallback_data(instrument, timeframe, count)

        try:
            df = self._mt5.get_bars(symbol, timeframe, from_dt, count)
            if not df.empty:
                df = df.rename(columns={"close": f"{instrument}_close"})
                return df[["time", f"{instrument}_close"]]
        except Exception as e:
            logger.warning("Failed to get intermarket data", instrument=instrument, error=str(e))

        return self._fallback_data(instrument, timeframe, count)

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

        if not dfs:
            return pd.DataFrame()

        # Merge on time column
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on="time", how="outer")

        result = result.sort_values("time").reset_index(drop=True)
        result = result.ffill()  # Forward-fill gaps
        return result

    @staticmethod
    def _fallback_data(instrument: str, timeframe: str, count: int) -> pd.DataFrame:
        """Return empty DataFrame as fallback when data is unavailable."""
        logger.debug("Using empty fallback for intermarket", instrument=instrument)
        return pd.DataFrame(columns=["time", f"{instrument}_close"])
