"""Загрузка ВСЕХ данных нужных для торгового бота.

Источники данных
----------------
1. OANDA REST API  — Forex бары (EURUSD, GBPUSD, USDJPY)
   Требует: OANDA_API_TOKEN, OANDA_ACCOUNT_ID
   Регистрация бесплатная: https://www.oanda.com/register/#/sign-up/demo

2. Yahoo Finance   — Intermarket (DXY, Gold, US10Y, SP500)
   Не требует авторизации. Работает всегда.

Использование
-------------
    # Всё сразу (750 дней):
    python scripts/fetch_all_data.py

    # Только Yahoo Finance (без OANDA):
    python scripts/fetch_all_data.py --skip-oanda

    # Другой период:
    python scripts/fetch_all_data.py --days 500

    # Только определённые символы:
    python scripts/fetch_all_data.py --symbols EURUSD --timeframes H1 D1

Структура сохранённых данных
----------------------------
    data/raw/bars/EURUSD/H1/2024-01-15.parquet   ← Forex (OANDA)
    data/raw/bars/DXY/D1/2024-01-15.parquet       ← Intermarket (Yahoo)
    data/raw/bars/XAUUSD/D1/2024-01-15.parquet
    data/raw/bars/US10Y/D1/2024-01-15.parquet
    data/raw/bars/SP500/D1/2024-01-15.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Добавляем src/ в путь если запускаем напрямую
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from apexfx.data.data_store import DataStore

# ---------------------------------------------------------------------------
# Yahoo Finance тикеры для intermarket инструментов
# ---------------------------------------------------------------------------
YAHOO_TICKERS: dict[str, str] = {
    "DXY":    "DX-Y.NYB",   # US Dollar Index
    "XAUUSD": "GC=F",        # Gold Futures (XAU/USD proxy)
    "US10Y":  "^TNX",        # US 10-Year Treasury Yield
    "SP500":  "^GSPC",       # S&P 500 Index
}

# Yahoo Finance интервалы
YAHOO_INTERVALS: dict[str, str] = {
    "M5": "5m",
    "H1": "1h",
    "D1": "1d",
}

# Максимальная история Yahoo Finance по интервалам (в днях)
YAHOO_MAX_DAYS: dict[str, int] = {
    "5m": 60,    # 5-минутки — только 60 дней
    "1h": 730,   # часовики — до 2 лет
    "1d": 9999,  # дневки — практически без ограничений
}

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _print_step(symbol: str, timeframe: str, source: str, n_bars: int,
                date_from: str, date_to: str) -> None:
    status = f"[OK] {n_bars:>6} bars" if n_bars > 0 else "[SKIP]  0 bars"
    print(f"  {status}  {symbol:<8} {timeframe:<4}  {source:<7}  {date_from} → {date_to}")


def _print_error(symbol: str, timeframe: str, error: str) -> None:
    print(f"  [ERR]          {symbol:<8} {timeframe:<4}  ERROR: {error[:60]}")


# ---------------------------------------------------------------------------
# OANDA загрузчик
# ---------------------------------------------------------------------------

def _check_oanda_credentials() -> tuple[str, str, str]:
    """Проверить наличие OANDA переменных окружения."""
    token = os.environ.get("OANDA_API_TOKEN", "")
    account = os.environ.get("OANDA_ACCOUNT_ID", "")
    env = os.environ.get("OANDA_ENVIRONMENT", "practice")
    return token, account, env


def _fetch_oanda(
    symbols: list[str],
    timeframes: list[str],
    days: int,
    store: DataStore,
    api_token: str | None,
    account_id: str | None,
    environment: str | None,
) -> dict[str, int]:
    """Загрузить Forex бары через OANDA API. Возвращает словарь symbol→bars."""
    from apexfx.data.oanda_client import OANDAClient

    client = OANDAClient(
        api_token=api_token,
        account_id=account_id,
        environment=environment,
    )
    client.connect()

    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=days)
    totals: dict[str, int] = {}

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                bars = client.download_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    from_dt=from_dt,
                    to_dt=to_dt,
                )
                if bars.empty:
                    _print_step(symbol, timeframe, "OANDA", 0, "-", "-")
                    continue

                store.append_bars(symbol, timeframe, bars)
                n = len(bars)
                totals[f"{symbol}/{timeframe}"] = n
                _print_step(
                    symbol, timeframe, "OANDA", n,
                    str(bars["time"].min().date()),
                    str(bars["time"].max().date()),
                )
            except Exception as e:
                _print_error(symbol, timeframe, str(e))

    client.disconnect()
    return totals


# ---------------------------------------------------------------------------
# Yahoo Finance загрузчик
# ---------------------------------------------------------------------------

def _fetch_yahoo(
    instruments: list[str],
    timeframes: list[str],
    days: int,
    store: DataStore,
) -> dict[str, int]:
    """Загрузить intermarket данные через Yahoo Finance (бесплатно)."""
    try:
        import yfinance as yf
    except ImportError:
        print("\n  [!] yfinance не установлен. Установите: pip install yfinance")
        print("      Intermarket данные не будут загружены.")
        return {}

    totals: dict[str, int] = {}
    to_dt = datetime.now(timezone.utc)

    for instrument in instruments:
        ticker_symbol = YAHOO_TICKERS.get(instrument, instrument)

        for timeframe in timeframes:
            yahoo_interval = YAHOO_INTERVALS.get(timeframe)
            if yahoo_interval is None:
                continue  # этот таймфрейм не поддерживается Yahoo

            # Учитываем ограничения Yahoo по истории
            max_days = YAHOO_MAX_DAYS.get(yahoo_interval, 730)
            actual_days = min(days, max_days)
            from_dt = to_dt - timedelta(days=actual_days)

            try:
                ticker = yf.Ticker(ticker_symbol)
                df_raw = ticker.history(
                    start=from_dt.strftime("%Y-%m-%d"),
                    end=to_dt.strftime("%Y-%m-%d"),
                    interval=yahoo_interval,
                    auto_adjust=True,
                    prepost=False,
                )

                if df_raw.empty:
                    _print_step(instrument, timeframe, "Yahoo", 0, "-", "-")
                    continue

                # Привести к стандартному формату DataStore
                df = _normalize_yahoo(df_raw, instrument)
                if df.empty:
                    _print_step(instrument, timeframe, "Yahoo", 0, "-", "-")
                    continue

                store.append_bars(instrument, timeframe, df)
                n = len(df)
                totals[f"{instrument}/{timeframe}"] = n
                _print_step(
                    instrument, timeframe, "Yahoo", n,
                    str(df["time"].min().date()),
                    str(df["time"].max().date()),
                )

            except Exception as e:
                _print_error(instrument, timeframe, str(e))

    return totals


def _normalize_yahoo(df_raw: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Привести DataFrame от yfinance к формату DataStore (time, open, high, low, close, volume)."""
    df = df_raw.copy()

    # Индекс — это datetime
    df.index = pd.to_datetime(df.index, utc=True)

    # Убираем timezone-aware → UTC naive (как в OANDA клиенте)
    df.index = df.index.tz_convert("UTC")

    # Стандартные имена столбцов
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "open":
            rename_map[col] = "open"
        elif col_lower == "high":
            rename_map[col] = "high"
        elif col_lower == "low":
            rename_map[col] = "low"
        elif col_lower in ("close", "adj close"):
            rename_map[col] = "close"
        elif col_lower in ("volume",):
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    # Оставляем только нужные столбцы
    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[needed].copy()

    if "close" not in df.columns:
        return pd.DataFrame()

    # Заполняем отсутствующие OHLC через close
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

    return df


# ---------------------------------------------------------------------------
# Проверка результатов
# ---------------------------------------------------------------------------

def _print_summary(oanda_totals: dict, yahoo_totals: dict, data_dir: str) -> None:
    """Вывести итоговую таблицу."""
    _print_header("ИТОГ")

    all_totals = {**oanda_totals, **yahoo_totals}
    total_bars = sum(all_totals.values())

    if all_totals:
        print(f"  Загружено инструментов: {len(all_totals)}")
        print(f"  Всего баров:            {total_bars:,}")
        print(f"  Данные сохранены в:     {data_dir}/raw/bars/")
    else:
        print("  Данные не были загружены.")

    print()
    if oanda_totals:
        print("  Forex (OANDA):")
        for key, n in sorted(oanda_totals.items()):
            symbol, tf = key.split("/")
            print(f"    {symbol:<8} {tf:<4}  {n:>6,} баров")

    if yahoo_totals:
        print("\n  Intermarket (Yahoo Finance):")
        for key, n in sorted(yahoo_totals.items()):
            symbol, tf = key.split("/")
            print(f"    {symbol:<8} {tf:<4}  {n:>6,} баров")

    print()
    if total_bars > 0:
        print("  Следующий шаг:")
        print("    python scripts/train.py --symbol EURUSD")
    else:
        print("  Данные не загружены. Проверьте OANDA credentials и yfinance.")


def _print_oanda_setup_instructions() -> None:
    print("""
  OANDA API не настроен. Forex данные будут пропущены.

  Чтобы настроить OANDA (бесплатная practice account):
  ─────────────────────────────────────────────────────
  1. Зарегистрируйтесь: https://www.oanda.com/register/#/sign-up/demo
  2. Войдите → My Services → Manage API Access → Create Token
  3. Установите переменные окружения:

       export OANDA_API_TOKEN=ваш_токен
       export OANDA_ACCOUNT_ID=ваш_account_id
       export OANDA_ENVIRONMENT=practice

  4. Запустите снова: python scripts/fetch_all_data.py
  ─────────────────────────────────────────────────────
  Пока запускаем только Yahoo Finance (intermarket данные)...
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Загрузить все данные для торгового бота",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days", type=int, default=750,
        help="Количество дней истории (default: 750)",
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["EURUSD", "GBPUSD", "USDJPY"],
        help="Forex символы для OANDA (default: EURUSD GBPUSD USDJPY)",
    )
    parser.add_argument(
        "--timeframes", nargs="+",
        default=["M5", "H1", "D1"],
        help="Таймфреймы для Forex (default: M5 H1 D1)",
    )
    parser.add_argument(
        "--intermarket", nargs="+",
        default=["DXY", "XAUUSD", "US10Y", "SP500"],
        help="Intermarket инструменты для Yahoo Finance",
    )
    parser.add_argument(
        "--intermarket-timeframes", nargs="+",
        default=["H1", "D1"],
        help="Таймфреймы для intermarket данных (default: H1 D1)",
    )
    parser.add_argument(
        "--skip-oanda", action="store_true",
        help="Пропустить OANDA, загрузить только Yahoo Finance",
    )
    parser.add_argument(
        "--skip-yahoo", action="store_true",
        help="Пропустить Yahoo Finance, загрузить только OANDA",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Директория для сохранения данных (default: data)",
    )
    parser.add_argument(
        "--api-token", default=None,
        help="OANDA API token (по умолчанию: OANDA_API_TOKEN env)",
    )
    parser.add_argument(
        "--account-id", default=None,
        help="OANDA Account ID (по умолчанию: OANDA_ACCOUNT_ID env)",
    )
    parser.add_argument(
        "--environment", default=None, choices=["practice", "live"],
        help="OANDA environment (default: practice)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _print_header("ApexFX — Загрузка данных")
    print(f"  Период:      {args.days} дней")
    print(f"  Forex:       {' '.join(args.symbols)} × {' '.join(args.timeframes)}")
    print(f"  Intermarket: {' '.join(args.intermarket)} × {' '.join(args.intermarket_timeframes)}")
    print(f"  Данные → {args.data_dir}/")

    store = DataStore(args.data_dir)
    oanda_totals: dict[str, int] = {}
    yahoo_totals: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 1. OANDA — Forex бары
    # ------------------------------------------------------------------
    if not args.skip_oanda:
        _print_header("1/2  Forex бары (OANDA API)")

        token = args.api_token or os.environ.get("OANDA_API_TOKEN", "")
        account = args.account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        environment = args.environment or os.environ.get("OANDA_ENVIRONMENT", "practice")

        if not token or not account:
            _print_oanda_setup_instructions()
        else:
            print(f"  Account: {account}  Environment: {environment}\n")
            try:
                oanda_totals = _fetch_oanda(
                    symbols=args.symbols,
                    timeframes=args.timeframes,
                    days=args.days,
                    store=store,
                    api_token=token,
                    account_id=account,
                    environment=environment,
                )
            except Exception as e:
                print(f"\n  [!] OANDA ошибка: {e}")
                print("      Проверьте токен и account_id.")
    else:
        print("\n  [SKIP] OANDA пропущен (--skip-oanda)")

    # ------------------------------------------------------------------
    # 2. Yahoo Finance — Intermarket
    # ------------------------------------------------------------------
    if not args.skip_yahoo:
        _print_header("2/2  Intermarket данные (Yahoo Finance)")
        print("  Источник: бесплатный, авторизация не нужна\n")
        yahoo_totals = _fetch_yahoo(
            instruments=args.intermarket,
            timeframes=args.intermarket_timeframes,
            days=args.days,
            store=store,
        )
    else:
        print("\n  [SKIP] Yahoo Finance пропущен (--skip-yahoo)")

    # ------------------------------------------------------------------
    # Итог
    # ------------------------------------------------------------------
    _print_summary(oanda_totals, yahoo_totals, args.data_dir)


if __name__ == "__main__":
    main()
