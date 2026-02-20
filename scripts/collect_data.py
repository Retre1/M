"""Script to collect historical data from MT5 and store in Parquet."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.data.mt5_client import MT5Client
from apexfx.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical Forex data from MT5")
    parser.add_argument("--config-dir", default="configs", help="Path to config directory")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to collect")
    parser.add_argument("--days", type=int, default=750, help="Days of history to collect")
    parser.add_argument("--timeframes", nargs="+", default=["M1", "M5", "H1", "D1"])
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(
        level=config.base.logging.level,
        fmt=config.base.logging.format,
    )
    logger = get_logger(__name__)

    mt5 = MT5Client()
    store = DataStore(config.base.paths.data_dir)

    try:
        mt5.connect()
        logger.info("Starting data collection", symbol=args.symbol, days=args.days)

        from_dt = datetime.now(timezone.utc) - timedelta(days=args.days)

        for tf in args.timeframes:
            logger.info("Collecting bars", timeframe=tf)
            bars = mt5.get_bars(args.symbol, tf, from_dt=from_dt, count=100_000)
            if not bars.empty:
                store.append_bars(args.symbol, tf, bars)
                logger.info("Stored bars", timeframe=tf, count=len(bars))
            else:
                logger.warning("No bars received", timeframe=tf)

        logger.info("Data collection complete")

    except Exception as e:
        logger.error("Data collection failed", error=str(e))
        raise
    finally:
        mt5.disconnect()


if __name__ == "__main__":
    main()
