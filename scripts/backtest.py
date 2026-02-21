"""Run walk-forward backtest on historical data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.features.pipeline import FeaturePipeline
from apexfx.training.walk_forward import WalkForwardValidator
from apexfx.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--output", default="backtest_results.json")
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(level=config.base.logging.level, fmt="console")
    logger = get_logger(__name__)

    # Load data
    store = DataStore(config.base.paths.data_dir)
    data = store.read_bars(args.symbol, args.timeframe)

    if data.empty:
        logger.error("No data available for backtest")
        return

    # Compute features
    pipeline = FeaturePipeline()
    features = pipeline.compute(data)
    logger.info("Data ready", n_bars=len(features))

    # Run walk-forward validation
    validator = WalkForwardValidator(config, features)
    results = validator.run()

    # Print results
    logger.info("=" * 60)
    logger.info("WALK-FORWARD BACKTEST RESULTS")
    logger.info("=" * 60)

    for key, value in results.aggregate_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    for fold in results.folds:
        logger.info(
            f"  Fold {fold.fold_idx}: "
            f"Sharpe={fold.metrics.get('sharpe_ratio', 0):.3f}, "
            f"Return={fold.metrics.get('total_return', 0):.4f}, "
            f"MaxDD={fold.metrics.get('max_drawdown', 0):.4f}"
        )

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(
            {
                "aggregate": {k: float(v) for k, v in results.aggregate_metrics.items()},
                "folds": [
                    {
                        "fold_idx": f.fold_idx,
                        "metrics": {k: float(v) for k, v in f.metrics.items()},
                    }
                    for f in results.folds
                ],
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
