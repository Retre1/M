"""Download historical OHLCV data from OANDA REST API.

Cross-platform alternative to collect_data.py (which requires MT5 on Windows).
Works on Linux, macOS, and Windows without any broker terminal installation.

Setup
-----
    pip install "apexfx-quantum[oanda]"
    export OANDA_API_TOKEN=your-token
    export OANDA_ACCOUNT_ID=your-account-id
    export OANDA_ENVIRONMENT=practice  # or "live"

Usage
-----
    # Download 750 days of EURUSD H1 bars
    python scripts/collect_data_oanda.py --symbol EURUSD --days 750

    # Download multiple symbols and timeframes
    python scripts/collect_data_oanda.py \\
        --symbols EURUSD GBPUSD USDJPY \\
        --timeframes M5 H1 D1 \\
        --days 750
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.data.oanda_client import OANDAClient
from apexfx.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical data from OANDA v20 REST API"
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--symbols", nargs="+", default=["EURUSD"],
        help="Symbols to download (default: EURUSD)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=["M5", "H1", "D1"],
        help="Timeframes to download (default: M5 H1 D1)"
    )
    parser.add_argument(
        "--days", type=int, default=750,
        help="Number of calendar days to download (default: 750)"
    )
    parser.add_argument(
        "--api-token", default=None,
        help="OANDA API token (default: OANDA_API_TOKEN env var)"
    )
    parser.add_argument(
        "--account-id", default=None,
        help="OANDA account ID (default: OANDA_ACCOUNT_ID env var)"
    )
    parser.add_argument(
        "--environment", default=None,
        choices=["practice", "live"],
        help="OANDA environment (default: OANDA_ENVIRONMENT env var, fallback: practice)"
    )
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(
        level=config.base.logging.level,
        fmt=config.base.logging.format,
        log_file=config.base.logging.file,
    )
    logger = get_logger(__name__)
    logger.info("Starting OANDA data collection", symbols=args.symbols, timeframes=args.timeframes)

    client = OANDAClient(
        api_token=args.api_token,
        account_id=args.account_id,
        environment=args.environment,
    )
    client.connect()

    store = DataStore(config.base.paths.data_dir)
    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=args.days)

    total_bars = 0
    for symbol in args.symbols:
        for timeframe in args.timeframes:
            logger.info(
                "Downloading",
                symbol=symbol,
                timeframe=timeframe,
                from_dt=str(from_dt.date()),
                to_dt=str(to_dt.date()),
            )
            try:
                bars = client.download_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    from_dt=from_dt,
                    to_dt=to_dt,
                )
                if bars.empty:
                    logger.warning("No data returned", symbol=symbol, timeframe=timeframe)
                    continue

                store.append_bars(symbol, timeframe, bars)
                total_bars += len(bars)
                logger.info(
                    "Stored",
                    symbol=symbol,
                    timeframe=timeframe,
                    n_bars=len(bars),
                    from_bar=str(bars["time"].min()),
                    to_bar=str(bars["time"].max()),
                )
            except Exception as e:
                logger.error(
                    "Download failed",
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(e),
                )

    client.disconnect()
    logger.info("Data collection complete", total_bars=total_bars)


if __name__ == "__main__":
    main()
