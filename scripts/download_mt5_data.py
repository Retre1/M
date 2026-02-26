"""Download historical data from MetaTrader 5.

Requires:
- Windows with MetaTrader 5 installed and logged in
- pip install MetaTrader5

Usage:
    # Download EURUSD MTF data (D1 + H1 + M5):
    python scripts/download_mt5_data.py --symbol EURUSD --mtf --bars 10000

    # Download single timeframe:
    python scripts/download_mt5_data.py --symbol EURUSD --timeframe H1 --bars 5000

Environment variables (optional):
    MT5_PATH     - path to terminal64.exe
    MT5_LOGIN    - account number
    MT5_PASSWORD - account password
    MT5_SERVER   - server name
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from apexfx.data.mt5_client import MT5Client


def download_mt5(
    client: MT5Client,
    symbol: str,
    timeframe: str,
    bars: int,
    data_dir: Path,
) -> pd.DataFrame:
    """Download bars from MT5 and save as Parquet."""
    print(f"Downloading {symbol} {timeframe} ({bars} bars) from MT5...")

    df = client.get_bars(symbol, timeframe, count=bars)

    if df.empty:
        print(f"ERROR: No data received for {symbol} {timeframe}")
        return pd.DataFrame()

    # Rename tick_volume to volume if needed
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})

    # Keep standard columns
    keep_cols = ["time", "open", "high", "low", "close", "volume"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Add tick_count if not present
    if "tick_count" not in df.columns:
        df["tick_count"] = 0

    # Remove NaN
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Save as Parquet partitioned by date
    output_dir = data_dir / "raw" / "bars" / symbol / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        path = output_dir / f"{date_str}.parquet"
        group.to_parquet(path, index=False, compression="snappy")
        n_saved += len(group)

    print(f"  OK: {n_saved} bars -> {output_dir}/")
    print(f"  Range: {df['time'].min()} — {df['time'].max()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Forex data from MetaTrader 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MTF data (D1 + H1 + M5):
  python scripts/download_mt5_data.py --symbol EURUSD --mtf --bars 10000

  # Download H1 only:
  python scripts/download_mt5_data.py --symbol EURUSD --timeframe H1 --bars 5000

  # Multiple symbols:
  python scripts/download_mt5_data.py --symbol EURUSD GBPUSD --mtf --bars 5000

  # With MT5 credentials:
  set MT5_LOGIN=12345678
  set MT5_PASSWORD=mypassword
  set MT5_SERVER=ICMarkets-Demo
  python scripts/download_mt5_data.py --symbol EURUSD --mtf
""",
    )
    parser.add_argument("--symbol", nargs="+", default=["EURUSD"])
    parser.add_argument("--timeframe", nargs="+", default=["H1"])
    parser.add_argument("--bars", type=int, default=10000,
                        help="Number of bars to download per timeframe (default: 10000)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--mtf", action="store_true",
                        help="Download D1 + H1 + M5 for each symbol")
    parser.add_argument("--mt5-path", default=None, help="Path to terminal64.exe")
    parser.add_argument("--login", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Connect to MT5
    client = MT5Client(
        path=args.mt5_path,
        login=args.login,
        password=args.password,
        server=args.server,
    )

    try:
        client.connect()
    except ConnectionError as e:
        print(f"ERROR: {e}")
        print("\nMake sure MetaTrader 5 is running and logged in.")
        print("You can also set env variables: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER")
        sys.exit(1)

    # Determine timeframes
    timeframes = args.timeframe
    if args.mtf:
        timeframes = ["D1", "H1", "M5"]
        print("MTF mode: downloading D1 + H1 + M5")

    # Adjust bar counts per timeframe
    bar_counts = {}
    for tf in timeframes:
        if tf == "D1":
            bar_counts[tf] = min(args.bars, 5000)  # ~20 years of D1
        elif tf == "M5":
            bar_counts[tf] = min(args.bars * 12, 100000)  # 12x more M5 bars
        else:
            bar_counts[tf] = args.bars

    # Download
    total_bars = 0
    try:
        for symbol in args.symbol:
            for tf in timeframes:
                print(f"\n{'='*60}")
                print(f"  {symbol} / {tf}")
                print(f"{'='*60}")
                df = download_mt5(client, symbol, tf, bar_counts[tf], data_dir)
                total_bars += len(df)
    finally:
        client.disconnect()

    print(f"\n{'='*60}")
    print(f"  DONE: {total_bars} total bars downloaded")
    print(f"  Data stored in: {data_dir / 'raw' / 'bars'}")
    print(f"{'='*60}")
    print("\nNext step — train the model:")
    print(f"  python scripts/train.py --symbol {args.symbol[0]}")


if __name__ == "__main__":
    main()
