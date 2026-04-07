"""Download free historical Forex data (no MT5 required).

Sources:
  1. yfinance  — Yahoo Finance (reliable, H1/D1, ~2 years history)
  2. dukascopy — Dukascopy tick data (free, unlimited history, slower)

Usage:
    # Quick start — download EURUSD H1 data from Yahoo Finance:
    python scripts/download_forex_data.py

    # Download multiple pairs:
    python scripts/download_forex_data.py --symbols EURUSD GBPUSD USDJPY

    # Download 3 years of data:
    python scripts/download_forex_data.py --days 1095

    # Use Dukascopy source (more data, slower):
    python scripts/download_forex_data.py --source dukascopy --days 365
"""

from __future__ import annotations

import argparse
import io
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lzma
import numpy as np
import pandas as pd
import requests

from apexfx.data.data_store import DataStore
from apexfx.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Yahoo Finance symbol mapping
YAHOO_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}

# Dukascopy instrument mapping
DUKASCOPY_SYMBOLS = {
    "EURUSD": ("EURUSD", 5),  # (instrument, pip decimals)
    "GBPUSD": ("GBPUSD", 5),
    "USDJPY": ("USDJPY", 3),
    "USDCHF": ("USDCHF", 5),
    "AUDUSD": ("AUDUSD", 5),
    "USDCAD": ("USDCAD", 5),
    "NZDUSD": ("NZDUSD", 5),
    "EURGBP": ("EURGBP", 5),
    "EURJPY": ("EURJPY", 3),
    "GBPJPY": ("GBPJPY", 3),
}


def download_yfinance(symbol: str, days: int = 730) -> pd.DataFrame:
    """Download H1 data from Yahoo Finance.

    Yahoo provides free forex data with H1 granularity for up to ~2 years.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance")

    yahoo_sym = YAHOO_SYMBOLS.get(symbol, f"{symbol}=X")
    logger.info("Downloading from Yahoo Finance", symbol=yahoo_sym, days=days)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    ticker = yf.Ticker(yahoo_sym)
    # Yahoo allows H1 data for max 730 days
    interval = "1h" if days <= 730 else "1d"
    df = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        logger.error("No data returned from Yahoo Finance", symbol=symbol)
        return pd.DataFrame()

    # Normalize to ApexFX format
    df = df.reset_index()
    col_map = {"Datetime": "time", "Date": "time", "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=col_map)

    # Ensure UTC timezone
    if df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    # Keep only needed columns
    cols = ["time", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]

    # Forex volume from Yahoo is often 0; fill with synthetic
    if df["volume"].sum() == 0:
        rng = np.random.default_rng(42)
        df["volume"] = rng.lognormal(10, 1.0, len(df))

    # Drop NaN rows
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    logger.info("Yahoo Finance data ready", bars=len(df),
                start=str(df["time"].iloc[0])[:10],
                end=str(df["time"].iloc[-1])[:10])
    return df


def download_dukascopy_hour(symbol: str, year: int, month: int, day: int, hour: int) -> pd.DataFrame:
    """Download one hour of tick data from Dukascopy and aggregate to ticks."""
    instrument, decimals = DUKASCOPY_SYMBOLS.get(symbol, (symbol, 5))
    point = 10 ** (-decimals)

    # Dukascopy URL format (months are 0-indexed)
    url = (
        f"https://datafeed.dukascopy.com/datafeed/{instrument}/"
        f"{year:04d}/{month - 1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    )

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200 or len(resp.content) == 0:
            return pd.DataFrame()

        # Decompress LZMA
        try:
            data = lzma.decompress(resp.content)
        except lzma.LZMAError:
            return pd.DataFrame()

        if len(data) == 0:
            return pd.DataFrame()

        # Parse binary tick data: each tick = 20 bytes
        # Format: uint32 ms_offset, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol
        n_ticks = len(data) // 20
        if n_ticks == 0:
            return pd.DataFrame()

        base_time = datetime(year, month, day, hour, tzinfo=timezone.utc)
        ticks = []
        for i in range(n_ticks):
            offset = i * 20
            ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
                ">IIIff", data[offset:offset + 20]
            )
            tick_time = base_time + timedelta(milliseconds=ms)
            ask = ask_raw * point
            bid = bid_raw * point
            ticks.append({
                "time": tick_time,
                "bid": bid,
                "ask": ask,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
            })

        return pd.DataFrame(ticks)

    except requests.RequestException:
        return pd.DataFrame()


def aggregate_ticks_to_h1(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tick data to H1 OHLCV bars."""
    if ticks_df.empty:
        return pd.DataFrame()

    ticks_df = ticks_df.sort_values("time")
    ticks_df["mid"] = (ticks_df["bid"] + ticks_df["ask"]) / 2
    ticks_df["spread"] = ticks_df["ask"] - ticks_df["bid"]
    ticks_df["volume"] = ticks_df["bid_volume"] + ticks_df["ask_volume"]

    # Resample to H1
    ticks_df = ticks_df.set_index("time")
    bars = ticks_df["mid"].resample("1h").ohlc()
    bars.columns = ["open", "high", "low", "close"]
    bars["volume"] = ticks_df["volume"].resample("1h").sum()
    bars["spread"] = ticks_df["spread"].resample("1h").mean()
    bars["tick_count"] = ticks_df["mid"].resample("1h").count()

    bars = bars.dropna(subset=["open"]).reset_index()
    bars = bars.rename(columns={"time": "time"})

    # Ensure UTC
    if bars["time"].dt.tz is None:
        bars["time"] = pd.to_datetime(bars["time"], utc=True)

    return bars


def download_dukascopy(symbol: str, days: int = 365) -> pd.DataFrame:
    """Download historical tick data from Dukascopy and aggregate to H1.

    Dukascopy provides free tick data with unlimited history.
    Downloads hour-by-hour, then aggregates to H1 bars.
    """
    if symbol not in DUKASCOPY_SYMBOLS:
        logger.error("Symbol not supported for Dukascopy", symbol=symbol,
                     supported=list(DUKASCOPY_SYMBOLS.keys()))
        return pd.DataFrame()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    logger.info("Downloading from Dukascopy", symbol=symbol, days=days,
                start=str(start)[:10], end=str(end)[:10])

    all_ticks = []
    current = start.replace(minute=0, second=0, microsecond=0)
    total_hours = int((end - current).total_seconds() / 3600)
    downloaded = 0
    errors = 0

    while current < end:
        # Skip weekends (Sat=5, Sun=6)
        if current.weekday() in (5, 6):
            current += timedelta(hours=1)
            continue

        ticks = download_dukascopy_hour(
            symbol, current.year, current.month, current.day, current.hour
        )

        if not ticks.empty:
            all_ticks.append(ticks)
            downloaded += len(ticks)
        else:
            errors += 1

        current += timedelta(hours=1)

        # Progress every 24 hours of data
        hours_done = int((current - start).total_seconds() / 3600)
        if hours_done % 168 == 0:  # weekly progress
            pct = hours_done / max(total_hours, 1) * 100
            logger.info(f"Dukascopy progress: {pct:.0f}%",
                        ticks=downloaded, hours=hours_done)

        # Rate limiting: ~2 requests per second
        time.sleep(0.5)

    if not all_ticks:
        logger.error("No tick data downloaded", symbol=symbol)
        return pd.DataFrame()

    # Combine all ticks
    ticks_df = pd.concat(all_ticks, ignore_index=True)
    logger.info("Ticks downloaded", total=len(ticks_df), errors=errors)

    # Aggregate to H1
    bars = aggregate_ticks_to_h1(ticks_df)
    logger.info("Dukascopy data ready", bars=len(bars),
                start=str(bars["time"].iloc[0])[:10] if not bars.empty else "N/A",
                end=str(bars["time"].iloc[-1])[:10] if not bars.empty else "N/A")

    return bars


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download free Forex data (no MT5 required)"
    )
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"],
                        help="Currency pairs to download")
    parser.add_argument("--days", type=int, default=730,
                        help="Days of history (Yahoo max ~730 for H1)")
    parser.add_argument("--source", choices=["yahoo", "dukascopy"], default="yahoo",
                        help="Data source")
    parser.add_argument("--timeframe", default="H1",
                        help="Target timeframe (H1 default)")
    parser.add_argument("--data-dir", default="data",
                        help="Data directory")
    args = parser.parse_args()

    setup_logging(level="INFO", fmt="console")

    store = DataStore(args.data_dir)

    for symbol in args.symbols:
        logger.info(f"{'=' * 50}")
        logger.info(f"Downloading {symbol}", source=args.source, days=args.days)

        if args.source == "yahoo":
            df = download_yfinance(symbol, days=args.days)
        elif args.source == "dukascopy":
            df = download_dukascopy(symbol, days=args.days)
        else:
            logger.error("Unknown source", source=args.source)
            continue

        if df.empty:
            logger.error("No data downloaded", symbol=symbol)
            continue

        # Validate
        required = ["time", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("Missing columns", missing=missing)
            continue

        # Store
        store.append_bars(symbol, args.timeframe, df)

        logger.info(
            f"Saved {symbol}",
            bars=len(df),
            timeframe=args.timeframe,
            period=f"{str(df['time'].iloc[0])[:10]} → {str(df['time'].iloc[-1])[:10]}",
        )

    # Verify stored data
    logger.info(f"{'=' * 50}")
    logger.info("Verifying stored data...")
    for symbol in args.symbols:
        loaded = store.read_bars(symbol, args.timeframe)
        if loaded.empty:
            logger.error("Verification FAILED", symbol=symbol)
        else:
            logger.info(
                f"OK: {symbol}",
                bars=len(loaded),
                start=str(loaded["time"].iloc[0])[:10],
                end=str(loaded["time"].iloc[-1])[:10],
            )


if __name__ == "__main__":
    main()
