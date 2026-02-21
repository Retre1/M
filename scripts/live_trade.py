"""Start the live trading loop."""

from __future__ import annotations

import argparse
import asyncio

from apexfx.config.registry import init_config
from apexfx.live.trading_loop import LiveTradingLoop
from apexfx.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Start ApexFX Quantum live trading")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--model-path", default=None, help="Path to trained model")
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(
        level=config.base.logging.level,
        fmt=config.base.logging.format,
        log_file=config.base.logging.file,
    )
    logger = get_logger(__name__)

    loop = LiveTradingLoop(
        config=config,
        symbol=args.symbol,
        model_path=args.model_path,
    )

    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
