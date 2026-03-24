"""
Download Forex historical data WITHOUT MetaTrader 5.
Uses yfinance as primary source, with CSV import as fallback.

Usage:
    # Install yfinance first:
    pip install yfinance

    # Download EURUSD H1 data (2 years):
    python scripts/download_data.py --symbol EURUSD --years 2

    # Download multiple pairs:
    python scripts/download_data.py --symbol EURUSD GBPUSD USDJPY --years 2

    # Import your own CSV file:
    python scripts/download_data.py --from-csv path/to/data.csv --symbol EURUSD --timeframe H1
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── yfinance symbol mapping ────────────────────────────────────────
YFINANCE_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "XAUUSD": "GC=F",        # Gold futures (proxy)
    "DXY": "DX-Y.NYB",       # Dollar index
    "US10Y": "^TNX",          # 10-year yield
}

# yfinance interval mapping
YF_INTERVALS = {
    "M1": "1m",     # max 7 days history
    "M5": "5m",     # max 60 days
    "M15": "15m",   # max 60 days
    "M30": "30m",   # max 60 days
    "H1": "1h",     # max 730 days
    "H4": "4h",     # not supported in yfinance — use 1h
    "D1": "1d",     # unlimited
}


def download_yfinance(
    symbol: str,
    timeframe: str,
    years: float,
    data_dir: Path,
) -> pd.DataFrame:
    """Download data from yfinance and save as Parquet."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    yf_symbol = YFINANCE_MAP.get(symbol)
    if not yf_symbol:
        print(f"ERROR: Unknown symbol '{symbol}'. Available: {list(YFINANCE_MAP.keys())}")
        sys.exit(1)

    yf_interval = YF_INTERVALS.get(timeframe)
    if not yf_interval:
        print(f"ERROR: Unsupported timeframe '{timeframe}'. Available: {list(YF_INTERVALS.keys())}")
        sys.exit(1)

    # yfinance limits for intraday data
    end = datetime.now()
    if timeframe == "M1":
        max_days = 7
        start = end - timedelta(days=min(int(years * 365), max_days))
        print("WARNING: M1 data limited to 7 days in yfinance")
    elif timeframe in ("M5", "M15", "M30"):
        max_days = 60
        start = end - timedelta(days=min(int(years * 365), max_days))
        print(f"WARNING: {timeframe} data limited to 60 days in yfinance")
    elif timeframe == "H1":
        max_days = 730
        start = end - timedelta(days=min(int(years * 365), max_days))
    else:
        start = end - timedelta(days=int(years * 365))

    print(f"Downloading {symbol} ({yf_symbol}) {timeframe} from {start.date()} to {end.date()}...")

    df = yf.download(
        yf_symbol,
        start=start,
        end=end,
        interval=yf_interval,
        progress=True,
    )

    if df.empty:
        print(f"ERROR: No data received for {symbol}")
        return pd.DataFrame()

    # Normalize columns
    df = df.reset_index()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Rename columns to our format
    rename_map = {
        "Datetime": "time",
        "Date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Keep only needed columns
    keep_cols = ["time", "open", "high", "low", "close", "volume"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Ensure time is UTC
    if not hasattr(df["time"].dtype, "tz") or df["time"].dt.tz is None:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    # If volume is missing or zero, generate synthetic volume
    if "volume" not in df.columns or df["volume"].sum() == 0:
        import numpy as np
        np.random.seed(42)
        # Session-based synthetic volume
        hours = df["time"].dt.hour
        base_vol = 1000
        session_mult = pd.Series(1.0, index=df.index)
        session_mult = session_mult.where(~hours.between(7, 16), 2.5)   # London
        session_mult = session_mult.where(~hours.between(12, 21), 3.0)  # NY
        session_mult = session_mult.where(~hours.between(12, 16), 4.0)  # Overlap
        df["volume"] = (base_vol * session_mult * np.random.lognormal(0, 0.3, len(df))).astype(int)
        print("NOTE: Volume data is synthetic (Forex has no centralized volume)")

    # Remove NaN rows
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Save as Parquet partitioned by date
    output_dir = data_dir / "raw" / "bars" / symbol / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save in daily parquet files (matches DataStore.read_bars format)
    n_saved = 0
    for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        path = output_dir / f"{date_str}.parquet"
        group.to_parquet(path, index=False, compression="snappy")
        n_saved += len(group)

    print(f"OK: Saved {n_saved} bars to {output_dir}/")
    print(f"    Date range: {df['time'].min()} — {df['time'].max()}")
    return df


def import_csv(
    csv_path: str,
    symbol: str,
    timeframe: str,
    data_dir: Path,
) -> pd.DataFrame:
    """Import CSV file and convert to project Parquet format."""
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print(f"Importing {csv_path}...")
    df = pd.read_csv(csv_path)

    # Try to detect column names
    col_lower = {c.lower().strip(): c for c in df.columns}

    rename = {}
    for target, candidates in {
        "time": ["time", "datetime", "date", "timestamp", "date_time", "gmt time"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v", "tick_volume", "tickvol"],
    }.items():
        for cand in candidates:
            if cand in col_lower:
                rename[col_lower[cand]] = target
                break

    df = df.rename(columns=rename)

    # Validate required columns
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"       Found: {list(df.columns)}")
        print("       Expected: time, open, high, low, close, volume")
        sys.exit(1)

    # Parse time
    df["time"] = pd.to_datetime(df["time"], utc=True, infer_datetime_format=True)

    if "volume" not in df.columns:
        df["volume"] = 0

    # Keep needed columns
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").reset_index(drop=True)

    # Save
    output_dir = data_dir / "raw" / "bars" / symbol / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        path = output_dir / f"{date_str}.parquet"
        group.to_parquet(path, index=False, compression="snappy")
        n_saved += len(group)

    print(f"OK: Imported {n_saved} bars to {output_dir}/")
    print(f"    Date range: {df['time'].min()} — {df['time'].max()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Forex data for ApexFX Quantum (no MT5 needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download EURUSD H1 data (2 years):
  python scripts/download_data.py --symbol EURUSD --years 2

  # Download multiple pairs + D1 for intermarket:
  python scripts/download_data.py --symbol EURUSD GBPUSD DXY XAUUSD --timeframe H1 D1 --years 2

  # Import your own CSV:
  python scripts/download_data.py --from-csv data.csv --symbol EURUSD --timeframe H1

Free data sources (manual download):
  Dukascopy : dukascopy.com/swiss/english/marketwatch/historical/
  HistData  : histdata.com/download-free-forex-data/
  TrueFX    : truefx.com/market-data/
""",
    )
    parser.add_argument(
        "--symbol", nargs="+", default=["EURUSD"],
        help="Symbols to download (default: EURUSD)",
    )
    parser.add_argument(
        "--timeframe", nargs="+", default=["H1"],
        help="Timeframes to download (default: H1)",
    )
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Years of history (default: 2)",
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Data directory (default: ./data)",
    )
    parser.add_argument(
        "--from-csv", default=None,
        help="Import from CSV file instead of downloading",
    )
    parser.add_argument(
        "--mtf", action="store_true",
        help="Download all MTF timeframes (D1 + H1 + M5) for each symbol",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.from_csv:
        # CSV import mode
        if len(args.symbol) != 1:
            print("ERROR: --from-csv requires exactly one --symbol")
            sys.exit(1)
        if len(args.timeframe) != 1:
            print("ERROR: --from-csv requires exactly one --timeframe")
            sys.exit(1)
        import_csv(args.from_csv, args.symbol[0], args.timeframe[0], data_dir)
    else:
        # If --mtf flag is set, override timeframes to D1 + H1 + M5
        timeframes = args.timeframe
        if args.mtf:
            timeframes = ["D1", "H1", "M5"]
            print("MTF mode: downloading D1 + H1 + M5 for each symbol")

        # Download mode
        total_bars = 0
        for symbol in args.symbol:
            for tf in timeframes:
                print(f"\n{'='*60}")
                print(f"  {symbol} / {tf}")
                print(f"{'='*60}")
                df = download_yfinance(symbol, tf, args.years, data_dir)
                total_bars += len(df)

        print(f"\n{'='*60}")
        print(f"  DONE: {total_bars} total bars downloaded")
        print(f"  Data stored in: {data_dir / 'raw' / 'bars'}")
        print(f"{'='*60}")
        print("\nNext step — train the model:")
        print(f"  python scripts/train.py --symbol {args.symbol[0]}")


if __name__ == "__main__":
    main()
