"""Run the paper trading simulation.

Paper trading is the mandatory pre-live validation step.  It replays
historical data through the exact same feature → signal → risk → execution
pipeline used in live trading, but without submitting real orders.

Modes
-----
Standard paper trade (recent unseen data):
    python scripts/paper_trade.py --symbol EURUSD --days 90

Final OOS evaluation (unlocks the sacred 20% held-out set):
    python scripts/paper_trade.py --symbol EURUSD --oos-eval

    WARNING: --oos-eval is IRREVERSIBLE.  Only run it once the model is
    fully frozen and you are ready to make the go/no-go decision for live.

Go-live gate (exits with code 1 if results are below threshold):
    python scripts/paper_trade.py --symbol EURUSD --oos-eval --gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.data.oos_guard import OOSGuard
from apexfx.features.pipeline import FeaturePipeline
from apexfx.live.paper_trading_loop import PaperTradingLoop
from apexfx.utils.logging import get_logger, setup_logging

# Minimum performance gates for live deployment
_MIN_SHARPE = 1.0
_MAX_DRAWDOWN_PCT = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="ApexFX paper trading / OOS evaluation")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to trained model (default: models/best/final_model)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of recent days to replay (ignored when --oos-eval is set)",
    )
    parser.add_argument(
        "--oos-eval",
        action="store_true",
        help=(
            "IRREVERSIBLE: unlock the sacred OOS test set and run the final "
            "pre-live evaluation.  Only use this once the model is fully frozen."
        ),
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Exit with code 1 if paper trading results fail the go-live gate "
            f"(Sharpe < {_MIN_SHARPE} or max DD > {_MAX_DRAWDOWN_PCT}%%). "
            "Useful for CI/CD pipelines."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save JSON report to this path (default: models/best/paper_trade_report.json)",
    )
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(
        level=config.base.logging.level,
        fmt=config.base.logging.format,
        log_file=config.base.logging.file,
    )
    logger = get_logger(__name__)

    model_path = args.model_path or str(
        Path(config.base.paths.models_dir) / "best" / "final_model"
    )

    store = DataStore(config.base.paths.data_dir)
    pipeline = FeaturePipeline()

    # --- Load and process data ---
    raw_data = store.read_bars(args.symbol, args.timeframe)
    if raw_data.empty:
        logger.error("No data found for %s %s — cannot run paper trade", args.symbol, args.timeframe)
        sys.exit(1)

    full_features = pipeline.compute(raw_data)
    logger.info("Features computed", n_bars=len(full_features))

    # --- Determine which data to replay ---
    if args.oos_eval:
        logger.warning(
            "OOS EVALUATION MODE — unlocking the sacred test set. "
            "This action is logged and should only happen once."
        )
        guard = OOSGuard(
            data=full_features,
            oos_fraction=config.data.oos_fraction,
            data_dir=config.base.paths.data_dir,
        )
        _, _ = guard.split()  # establish the guard boundary
        replay_data = guard.unlock_oos(reason=f"paper_trade.py --oos-eval symbol={args.symbol}")

        output_path = args.output or str(
            Path(config.base.paths.models_dir) / "best" / f"oos_paper_trade_{args.symbol}.json"
        )
    else:
        # Standard paper trade on recent data (never touching OOS)
        bars_per_day = 24  # H1
        n_bars = args.days * bars_per_day
        replay_data = full_features.iloc[-n_bars:].reset_index(drop=True)

        output_path = args.output or str(
            Path(config.base.paths.models_dir) / "best" / f"paper_trade_{args.symbol}.json"
        )
        logger.info(
            "Standard paper trade (not OOS)",
            symbol=args.symbol,
            days=args.days,
            n_bars=len(replay_data),
        )

    if replay_data.empty or len(replay_data) < config.data.feature_window + 10:
        logger.error("Insufficient bars for paper trading", n_bars=len(replay_data))
        sys.exit(1)

    # --- Run paper trading ---
    loop = PaperTradingLoop(
        config=config,
        model_path=model_path,
        symbol=args.symbol,
        data=replay_data,
    )
    report = loop.run()

    # --- Save report ---
    report.save(output_path)

    # --- Go-live gate ---
    if args.gate:
        sharpe = report.metrics.get("sharpe_ratio", 0.0)
        max_dd = report.max_drawdown_pct
        passed = sharpe >= _MIN_SHARPE and max_dd <= _MAX_DRAWDOWN_PCT

        if passed:
            logger.info(
                "GO-LIVE GATE: PASSED",
                sharpe=round(sharpe, 4),
                max_dd=round(max_dd, 2),
                min_sharpe=_MIN_SHARPE,
                max_dd_limit=_MAX_DRAWDOWN_PCT,
            )
        else:
            logger.error(
                "GO-LIVE GATE: FAILED — do NOT deploy to live trading",
                sharpe=round(sharpe, 4),
                max_dd=round(max_dd, 2),
                min_sharpe=_MIN_SHARPE,
                max_dd_limit=_MAX_DRAWDOWN_PCT,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
