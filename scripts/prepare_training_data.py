#!/usr/bin/env python3
"""Скачать и подготовить ВСЕ данные для обучения ApexFX.

Источник: Yahoo Finance (бесплатно, без авторизации).
Скачивает EURUSD + DXY + XAUUSD + US10Y на D1/H1, мержит intermarket
колонки в основной DataFrame и валидирует через BarDataValidator.

Использование:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --years 5
    python scripts/prepare_training_data.py --skip-download  # только мерж

Результат:
    data/raw/bars/{SYMBOL}/{TF}/*.parquet   — сырые бары
    data/prepared/EURUSD_H1_train.parquet   — готовый DataFrame для обучения
    data/prepared/EURUSD_D1_train.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from apexfx.data.data_store import DataStore
from apexfx.data.validator import BarDataValidator

# ───────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────

# Yahoo Finance тикеры
TICKERS = {
    "EURUSD": "EURUSD=X",
    "DXY":    "DX-Y.NYB",
    "XAUUSD": "GC=F",
    "US10Y":  "^TNX",
    "SP500":  "^GSPC",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

# Таймфреймы для каждого инструмента
DOWNLOAD_PLAN = {
    "EURUSD": ["D1", "H1"],
    "DXY":    ["D1", "H1"],
    "XAUUSD": ["D1", "H1"],
    "US10Y":  ["D1"],
    "SP500":  ["D1"],
}

YAHOO_INTERVAL = {"D1": "1d", "H1": "1h", "M5": "5m"}
YAHOO_MAX_DAYS = {"1d": 10 * 365, "1h": 730, "5m": 60}

# Intermarket инструменты для мержа в EURUSD
INTERMARKET = ["DXY", "XAUUSD", "US10Y", "SP500"]


# ───────────────────────────────────────────────────────────────────
# Download
# ───────────────────────────────────────────────────────────────────

def download_yahoo(
    symbol: str,
    ticker: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance."""
    import yfinance as yf

    max_days = YAHOO_MAX_DAYS.get(interval, 730)
    actual_start = max(start, end - timedelta(days=max_days))

    print(f"  Скачиваю {symbol} {interval} ({actual_start.date()} → {end.date()})...", end=" ")

    # For intraday intervals, Yahoo requires `period` instead of start/end
    # for some tickers (especially Forex). Try period-based first, fallback to dates.
    if interval in ("1h", "5m"):
        period_map = {"1h": "2y", "5m": "60d"}
        period = period_map.get(interval, "2y")
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
            if df.empty:
                # Fallback to yf.download
                df = yf.download(
                    ticker, period=period, interval=interval,
                    auto_adjust=True, prepost=False, progress=False,
                )
        except Exception:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=True, prepost=False, progress=False,
            )
    else:
        df = yf.download(
            ticker,
            start=actual_start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            prepost=False,
            progress=False,
        )

    if df.empty:
        print("0 баров")
        return pd.DataFrame()

    # yf.download can return MultiIndex columns for single ticker — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Normalize columns
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    rename = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl == "open": rename[col] = "open"
        elif cl == "high": rename[col] = "high"
        elif cl == "low": rename[col] = "low"
        elif cl in ("close", "adj close"): rename[col] = "close"
        elif cl == "volume": rename[col] = "volume"
    df = df.rename(columns=rename)

    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[needed].copy()

    if "close" not in df.columns:
        print("нет close")
        return pd.DataFrame()

    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0

    df["time"] = df.index
    df = df.reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_values("time").reset_index(drop=True)

    print(f"{len(df)} баров")
    return df


def download_all(store: DataStore, years: int) -> dict[str, dict[str, int]]:
    """Download all instruments and save to DataStore."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365)
    stats: dict[str, dict[str, int]] = {}

    for symbol, timeframes in DOWNLOAD_PLAN.items():
        ticker = TICKERS[symbol]
        stats[symbol] = {}

        for tf in timeframes:
            interval = YAHOO_INTERVAL[tf]
            df = download_yahoo(symbol, ticker, interval, start, end)
            if df.empty:
                stats[symbol][tf] = 0
                continue

            store.append_bars(symbol, tf, df)
            stats[symbol][tf] = len(df)

    return stats


# ───────────────────────────────────────────────────────────────────
# Merge intermarket data into main EURUSD DataFrame
# ───────────────────────────────────────────────────────────────────

def merge_intermarket(
    main_df: pd.DataFrame,
    store: DataStore,
    timeframe: str,
) -> pd.DataFrame:
    """Merge intermarket close prices into main DataFrame as {INST}_close columns.

    Uses asof merge to align daily intermarket data with hourly main data
    (intermarket value = latest available value at or before main bar time).
    """
    result = main_df.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result = result.sort_values("time").reset_index(drop=True)

    for inst in INTERMARKET:
        # Try same timeframe first, fallback to D1
        inst_df = store.read_bars(inst, timeframe)
        if inst_df.empty and timeframe != "D1":
            inst_df = store.read_bars(inst, "D1")

        if inst_df.empty:
            print(f"    [!] {inst} — нет данных, колонка {inst}_close будет NaN")
            result[f"{inst}_close"] = np.nan
            continue

        inst_df["time"] = pd.to_datetime(inst_df["time"], utc=True)
        inst_df = inst_df.sort_values("time").reset_index(drop=True)
        inst_df = inst_df[["time", "close"]].rename(columns={"close": f"{inst}_close"})

        # asof merge: for each main bar, find the latest intermarket value
        result = pd.merge_asof(
            result,
            inst_df,
            on="time",
            direction="backward",
            tolerance=pd.Timedelta(days=7),  # max 7 days gap (weekends + holidays)
        )

        valid_pct = result[f"{inst}_close"].notna().mean()
        print(f"    {inst}_close merged: {valid_pct:.0%} заполнено")

    return result


# ───────────────────────────────────────────────────────────────────
# Validate and prepare
# ───────────────────────────────────────────────────────────────────

def prepare_dataset(
    store: DataStore,
    symbol: str,
    timeframe: str,
    output_dir: Path,
) -> pd.DataFrame | None:
    """Load raw bars, merge intermarket, validate, and save prepared dataset."""
    print(f"\n  Подготовка {symbol} {timeframe}...")

    df = store.read_bars(symbol, timeframe)
    if df.empty:
        print(f"    [!] Нет данных для {symbol} {timeframe}")
        return None

    print(f"    Сырых баров: {len(df)} ({df['time'].min()} → {df['time'].max()})")

    # Merge intermarket
    print("    Мержу intermarket данные:")
    df = merge_intermarket(df, store, timeframe)

    # Validate
    validator = BarDataValidator(min_bars=100)
    report = validator.validate(df)
    if report.issues:
        for issue in report.issues:
            level = issue.severity.upper()
            print(f"    [{level}] {issue.check}: {issue.message}")

    # Clean
    df = validator.clean(df)
    print(f"    После очистки: {len(df)} баров")

    # Add tick_count placeholder (needed by some extractors)
    if "tick_count" not in df.columns:
        df["tick_count"] = 100

    # Save prepared dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_{timeframe}_train.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    Сохранено: {out_path} ({size_mb:.1f} MB)")

    return df


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка данных для обучения ApexFX")
    parser.add_argument("--years", type=int, default=10, help="Лет истории (default: 10)")
    parser.add_argument("--data-dir", default="data", help="Директория данных")
    parser.add_argument("--skip-download", action="store_true", help="Пропустить загрузку")
    parser.add_argument("--symbol", default="EURUSD", help="Основной символ")
    args = parser.parse_args()

    data_dir = _ROOT / args.data_dir
    store = DataStore(data_dir)
    output_dir = data_dir / "prepared"

    print("=" * 60)
    print("  ApexFX — Подготовка данных для обучения")
    print("=" * 60)

    # ─── 1. Download ───
    if not args.skip_download:
        print(f"\n  Период: {args.years} лет")
        print(f"  Источник: Yahoo Finance (бесплатно)\n")
        stats = download_all(store, args.years)

        print("\n  Итог загрузки:")
        total = 0
        for sym, tf_stats in stats.items():
            for tf, n in tf_stats.items():
                if n > 0:
                    print(f"    {sym:<8} {tf:<4} {n:>8,} баров")
                    total += n
        print(f"    {'─' * 30}")
        print(f"    {'ВСЕГО':<13} {total:>8,} баров")
    else:
        print("\n  Загрузка пропущена (--skip-download)")

    # ─── 2. Prepare datasets ───
    print("\n" + "=" * 60)
    print("  Подготовка датасетов для обучения")
    print("=" * 60)

    for tf in ["D1", "H1"]:
        df = prepare_dataset(store, args.symbol, tf, output_dir)
        if df is not None:
            # Quick stats
            log_ret = np.log(df["close"].values[1:] / df["close"].values[:-1])
            print(f"    Статистика:")
            print(f"      Mean return:     {log_ret.mean():.6f}")
            print(f"      Std return:      {log_ret.std():.6f}")
            print(f"      Ann. volatility: {log_ret.std() * np.sqrt(252):.2%}")
            print(f"      Min price:       {df['close'].min():.5f}")
            print(f"      Max price:       {df['close'].max():.5f}")

            # Check intermarket coverage
            for inst in INTERMARKET:
                col = f"{inst}_close"
                if col in df.columns:
                    pct = df[col].notna().mean()
                    print(f"      {col}: {pct:.0%} filled")

    # ─── 3. Summary ───
    print("\n" + "=" * 60)
    print("  Готово!")
    print("=" * 60)
    print(f"\n  Подготовленные датасеты:")
    for f in sorted(output_dir.glob("*.parquet")):
        size = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size:.1f} MB)")

    print(f"\n  Следующий шаг:")
    print(f"    python scripts/train.py --symbol {args.symbol}")
    print()


if __name__ == "__main__":
    main()
