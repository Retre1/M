"""Launch the training pipeline.

OOS Safety
----------
When real data is available, the last ``oos_fraction`` (default 20%) is
automatically locked in an :class:`~apexfx.data.oos_guard.OOSGuard`.
The trainer **never sees this data**. Only the training pool (first 80%)
is passed to the curriculum and walk-forward stages.

To run a final evaluation on the OOS set after training is complete, use::

    python scripts/paper_trade.py --oos-eval --symbol EURUSD
"""

from __future__ import annotations

import argparse

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.data.oos_guard import OOSGuard
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
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(
        level=config.base.logging.level,
        fmt=config.base.logging.format,
        log_file=config.base.logging.file,
    )
    logger = get_logger(__name__)
    logger.info("Starting ApexFX Quantum training")

    # Load real data if available
    train_data = None
    if not args.synthetic_only:
        store = DataStore(config.base.paths.data_dir)
        raw_data = store.read_bars(args.symbol, args.timeframe)

        if raw_data.empty:
            logger.warning("No real data found, training with synthetic data only")
        else:
            # Compute features before the OOS split so the guard operates on
            # feature-enriched bars and the split index is consistent.
            pipeline = FeaturePipeline()
            full_features = pipeline.compute(raw_data)
            logger.info("Features computed on full dataset", n_bars=len(full_features))

            # --- OOS Guard: lock the last oos_fraction of data in a safe ---
            guard = OOSGuard(
                data=full_features,
                oos_fraction=config.data.oos_fraction,
                data_dir=config.base.paths.data_dir,
            )
            train_data, _ = guard.split()  # OOS is intentionally withheld
            logger.info(
                "OOS guard active — training pool only",
                train_bars=len(train_data),
                oos_bars=guard.n_oos_bars,
                oos_fraction=config.data.oos_fraction,
            )

    # Run training on the training pool (never the OOS set)
    trainer = Trainer(config, real_data=train_data)
    trainer.train()

    logger.info("Training complete — OOS set remains untouched")
    logger.info(
        "Run 'python scripts/paper_trade.py --oos-eval --symbol %s' "
        "for the final out-of-sample evaluation",
        args.symbol,
    )


if __name__ == "__main__":
    main()
