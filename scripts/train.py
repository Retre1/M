"""Launch the training pipeline."""

from __future__ import annotations

import argparse

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.features.pipeline import FeaturePipeline
from apexfx.training.trainer import Trainer
from apexfx.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ApexFX Quantum model")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Use only synthetic data (no MT5 needed)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint. "
                             "Useful for free-tier platforms (Colab, Kaggle) "
                             "with session time limits.")
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(
        level=config.base.logging.level,
        fmt=config.base.logging.format,
        log_file=config.base.logging.file,
    )
    logger = get_logger(__name__)

    if args.resume:
        logger.info("ApexFX Quantum training — RESUME mode")
    else:
        logger.info("Starting ApexFX Quantum training")

    # Load real data if available
    real_data = None
    if not args.synthetic_only:
        store = DataStore(config.base.paths.data_dir)
        real_data = store.read_bars(args.symbol, args.timeframe)
        if real_data.empty:
            logger.warning("No real data found, training with synthetic data only")
            real_data = None
        else:
            # Compute features on real data
            pipeline = FeaturePipeline()
            real_data = pipeline.compute(real_data)
            logger.info("Real data loaded and features computed", n_bars=len(real_data))

    # Run training
    trainer = Trainer(config, real_data=real_data)
    trainer.train(resume=args.resume)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
