"""
ApexFX Quantum — Download historical data via yfinance (no MT5 required).
Works on Linux/macOS.

Downloads:
    - Forex pairs: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, XAUUSD
    - Intermarket: DXY, SPX, US10Y
    - Timeframes: H1 (daily bars), D1 (daily bars)
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/raw/bars")

# yfinance tickers mapping
TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "XAUUSD": "GC=F",
    "DXY":    "DX-Y.NYB",
    "SPX":    "^GSPC",
    "US10Y":  "^TNX",
}

TIMEFRAMES = {
    "H1": {"interval": "1h",  "period": "730d"},
    "D1": {"interval": "1d",  "period": "max"},
}


def save_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    out_dir = DATA_DIR / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "all.csv"
    df.to_csv(csv_path, index=False)

    saved = 0
    for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        path = out_dir / f"{date_str}.parquet"
        group.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        saved += 1

    return csv_path


def download(symbol: str, ticker: str, tf_name: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    if data is None or data.empty:
        return None

    data = data.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() if col[1] == "" else col[0].lower() for col in data.columns]
    else:
        data.columns = [c.lower() for c in data.columns]

    col_map = {}
    for col in data.columns:
        if col in ("date", "datetime"):
            col_map[col] = "time"

    data = data.rename(columns=col_map)

    if "time" not in data.columns:
        return None

    keep = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in data.columns]
    data = data[keep].copy()

    if "volume" in data.columns:
        data = data.rename(columns={"volume": "tick_volume"})
    else:
        data["tick_volume"] = 0

    data["spread"] = 0
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data = data.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return data


def main():
    print("=" * 60)
    print("  ApexFX Quantum — Data Downloader (Linux/yfinance)")
    print("=" * 60)
    print()

    stats = {}

    for symbol, ticker in TICKERS.items():
        print(f"\n[{symbol}] ({ticker})")
        for tf_name, cfg in TIMEFRAMES.items():
            df = download(symbol, ticker, tf_name, cfg["interval"], cfg["period"])
            if df is not None and len(df) > 10:
                csv_path = save_bars(symbol, tf_name, df)
                date_from = df["time"].iloc[0].strftime("%Y-%m-%d")
                date_to = df["time"].iloc[-1].strftime("%Y-%m-%d")
                print(f"  {tf_name}: {len(df):>8,} bars  [{date_from} → {date_to}]  saved: {csv_path}")
                stats[f"{symbol}_{tf_name}"] = len(df)
            else:
                print(f"  {tf_name}: NO DATA")
                stats[f"{symbol}_{tf_name}"] = 0

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n{'Symbol':<12} {'TF':<6} {'Bars':>10}")
    print("-" * 30)

    total_bars = 0
    for key, count in sorted(stats.items()):
        parts = key.rsplit("_", 1)
        sym, tf = parts[0], parts[1]
        status = f"{count:>10,}" if count > 0 else "    FAILED"
        print(f"{sym:<12} {tf:<6} {status}")
        total_bars += count

    print("-" * 30)
    print(f"{'TOTAL':<19} {total_bars:>10,} bars")
    print()

    data_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    print(f"Disk usage: {data_size / 1024 / 1024:.1f} MB")
    print(f"Location:   {DATA_DIR.resolve()}")

    missing = [k for k, v in stats.items() if v == 0]
    if missing:
        print("\nWARNING: No data for:")
        for m in missing:
            print(f"  - {m}")

    print("\nDone! Data ready for training and backtesting.")


if __name__ == "__main__":
    main()
