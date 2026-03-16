"""
ApexFX Quantum — Download all historical data from MT5 + yfinance.

Usage:
    1. Open MetaTrader 5, login to your account
    2. Run: python download_data.py

Downloads:
    - Forex pairs: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, XAUUSD
    - Timeframes: M5 (150k bars), H1 (18k bars), D1 (1000 bars)
    - Intermarket: DXY, SPX, US10Y via yfinance
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Check MT5 available ──────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed.")
    print("Run:  pip install MetaTrader5")
    sys.exit(1)

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "XAUUSD"]

TIMEFRAMES = {
    "M5":  {"mt5_tf": mt5.TIMEFRAME_M5,  "count": 150_000},  # ~2 years
    "H1":  {"mt5_tf": mt5.TIMEFRAME_H1,  "count": 18_000},   # ~2.5 years
    "D1":  {"mt5_tf": mt5.TIMEFRAME_D1,  "count": 1_000},    # ~4 years
}

# Intermarket symbols — try multiple names per instrument
INTERMARKET = {
    "DXY":   ["DXY", "USDX", "DX", "DXY.f", "USDX.f"],
    "SPX":   ["US500", "SPX500", "US500.cash", "SP500", "SPXm", "US500m"],
    "US10Y": ["US10Y", "TNX", "US10Y.f", "USTBOND"],
}

DATA_DIR = Path("data/raw/bars")

# ── Helpers ──────────────────────────────────────────────────────────

def save_bars(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    """Save bars to parquet, partitioned by date."""
    out_dir = DATA_DIR / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also save as single CSV for easy inspection
    csv_path = out_dir / "all.csv"
    df.to_csv(csv_path, index=False)

    # Save partitioned parquet
    saved = 0
    for date_str, group in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        path = out_dir / f"{date_str}.parquet"
        group.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        saved += 1

    return csv_path


def download_mt5(symbol: str, tf_name: str, mt5_tf: int, count: int) -> pd.DataFrame | None:
    """Download bars from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None or len(rates) == 0:
        print(f"    WARNING: No data for {symbol} {tf_name}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def try_intermarket_mt5(names: list[str], tf_name: str, mt5_tf: int, count: int):
    """Try multiple symbol names for intermarket instruments."""
    for name in names:
        info = mt5.symbol_info(name)
        if info is not None:
            # Enable symbol in Market Watch
            mt5.symbol_select(name, True)
            time.sleep(0.5)
            df = download_mt5(name, tf_name, mt5_tf, count)
            if df is not None and len(df) > 100:
                return name, df
    return None, None


def download_yfinance(instrument: str, tf_name: str) -> pd.DataFrame | None:
    """Fallback: download from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("    yfinance not installed. Run: pip install yfinance")
        return None

    ticker_map = {
        "DXY": "DX-Y.NYB",
        "SPX": "^GSPC",
        "US10Y": "^TNX",
        "XAUUSD": "GC=F",
    }

    ticker = ticker_map.get(instrument)
    if not ticker:
        return None

    interval_map = {"M5": "5m", "H1": "1h", "D1": "1d"}
    interval = interval_map.get(tf_name, "1h")

    # yfinance limits: 5m = 60 days, 1h = 730 days, 1d = unlimited
    period_map = {"M5": "60d", "H1": "730d", "D1": "max"}
    period = period_map.get(tf_name, "730d")

    print(f"    Downloading {instrument} ({ticker}) from yfinance [{interval}]...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        return None

    data = data.reset_index()
    # Normalize column names
    col_map = {}
    for col in data.columns:
        name = col[0].lower() if isinstance(col, tuple) else col.lower()
        if name in ("date", "datetime"):
            col_map[col] = "time"
        elif name == "open":
            col_map[col] = "open"
        elif name == "high":
            col_map[col] = "high"
        elif name == "low":
            col_map[col] = "low"
        elif name == "close":
            col_map[col] = "close"
        elif name == "volume":
            col_map[col] = "tick_volume"

    data = data.rename(columns=col_map)
    keep = [c for c in ["time", "open", "high", "low", "close", "tick_volume"] if c in data.columns]
    data = data[keep]

    if "tick_volume" not in data.columns:
        data["tick_volume"] = 0
    data["spread"] = 0

    data["time"] = pd.to_datetime(data["time"], utc=True)
    data = data.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return data


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ApexFX Quantum — Data Downloader")
    print("=" * 60)
    print()

    # Initialize MT5 with explicit credentials
    init_ok = mt5.initialize(
        login=591316579,
        server="FxPro-MT5 Demo",
        password="asdjJLAsnd21#",
    )
    if not init_ok:
        print(f"ERROR: MT5 init failed: {mt5.last_error()}")
        print("Make sure MetaTrader 5 is running!")
        sys.exit(1)

    info = mt5.terminal_info()
    account = mt5.account_info()
    print(f"MT5 connected: {info.name}")
    print(f"Account: {account.login} ({account.server})")
    print(f"Balance: {account.balance} {account.currency}")
    print()

    stats = {}

    # ── 1. Download Forex pairs ──────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: Forex Pairs")
    print("=" * 60)

    for symbol in PAIRS:
        print(f"\n[{symbol}]")

        # Enable in Market Watch
        if not mt5.symbol_select(symbol, True):
            print(f"  WARNING: Cannot select {symbol}, skipping...")
            continue

        time.sleep(0.3)

        for tf_name, tf_cfg in TIMEFRAMES.items():
            df = download_mt5(symbol, tf_name, tf_cfg["mt5_tf"], tf_cfg["count"])
            if df is not None:
                csv_path = save_bars(symbol, tf_name, df)
                date_from = df["time"].iloc[0].strftime("%Y-%m-%d")
                date_to = df["time"].iloc[-1].strftime("%Y-%m-%d")
                print(f"  {tf_name}: {len(df):>8,} bars  [{date_from} → {date_to}]  saved: {csv_path}")
                stats[f"{symbol}_{tf_name}"] = len(df)
            else:
                stats[f"{symbol}_{tf_name}"] = 0

    # ── 2. Download Intermarket ──────────────────────────────────────
    print()
    print("=" * 60)
    print("  STEP 2: Intermarket (DXY, SPX, US10Y)")
    print("=" * 60)

    for instrument, names in INTERMARKET.items():
        print(f"\n[{instrument}]")

        for tf_name, tf_cfg in TIMEFRAMES.items():
            # Try MT5 first
            found_name, df = try_intermarket_mt5(names, tf_name, tf_cfg["mt5_tf"], tf_cfg["count"])

            if df is not None:
                csv_path = save_bars(instrument, tf_name, df)
                date_from = df["time"].iloc[0].strftime("%Y-%m-%d")
                date_to = df["time"].iloc[-1].strftime("%Y-%m-%d")
                print(f"  {tf_name}: {len(df):>8,} bars  [{date_from} → {date_to}]  (MT5: {found_name})")
                stats[f"{instrument}_{tf_name}"] = len(df)
            else:
                # Fallback to yfinance
                df = download_yfinance(instrument, tf_name)
                if df is not None:
                    csv_path = save_bars(instrument, tf_name, df)
                    date_from = df["time"].iloc[0].strftime("%Y-%m-%d")
                    date_to = df["time"].iloc[-1].strftime("%Y-%m-%d")
                    print(f"  {tf_name}: {len(df):>8,} bars  [{date_from} → {date_to}]  (yfinance)")
                    stats[f"{instrument}_{tf_name}"] = len(df)
                else:
                    print(f"  {tf_name}: NO DATA (MT5 + yfinance failed)")
                    stats[f"{instrument}_{tf_name}"] = 0

    # ── 3. Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Symbol':<12} {'TF':<6} {'Bars':>10}")
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

    # Disk usage
    data_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    print(f"Disk usage: {data_size / 1024 / 1024:.1f} MB")
    print(f"Location:   {DATA_DIR.resolve()}")
    print()

    # Warnings
    missing = [k for k, v in stats.items() if v == 0]
    if missing:
        print("WARNING: Missing data for:")
        for m in missing:
            print(f"  - {m}")
        print()

    print("Done! Data ready for training and backtesting.")

    mt5.shutdown()


if __name__ == "__main__":
    main()
