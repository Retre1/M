#!/usr/bin/env python3
"""Standalone TFT supervised pre-training.

Trains the TFT encoder on direction classification and volatility
regression before RL training begins.

Usage:
    python scripts/pretrain_tft.py [--epochs 30] [--lr 1e-3] [--synthetic-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apexfx.config.loader import load_config
from apexfx.data.synthetic import SyntheticDataGenerator
from apexfx.features.pipeline import FeaturePipeline
from apexfx.features.selector import FeatureSelector
from apexfx.models.tft.tft_model import TemporalFusionTransformer
from apexfx.training.pretrain import TFTPretrainer
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="TFT supervised pre-training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_config()

    # Load data
    raw_dir = Path(config.base.paths.data_dir) / "raw" / "bars"
    parquet_files = list(raw_dir.rglob("*.parquet")) if not args.synthetic_only else []

    if parquet_files:
        import pandas as pd
        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        data = pd.concat(dfs, ignore_index=True)
        logger.info("Loaded %d bars from real data", len(data))
    else:
        logger.info("Generating synthetic data")
        gen = SyntheticDataGenerator(seed=config.base.seed)
        data = gen.generate_gbm(n_steps=10_000, mu=0.0001, sigma=0.01)

    # Feature pipeline + selection
    pipeline = FeaturePipeline()
    features = pipeline.compute(data)

    selector = FeatureSelector(top_n=args.top_n)
    selected = selector.fit_transform(features)
    n_features = len(selector.selected_features)
    market_cols = selector.selected_features

    # Create TFT
    lookback = config.data.feature_window
    tft = TemporalFusionTransformer(
        n_continuous_vars=n_features,
        n_known_future_vars=len(config.model.tft.known_future_inputs),
        d_model=config.model.tft.d_model,
        n_heads=config.model.tft.n_heads,
        dropout=config.model.tft.dropout,
    )

    # Pre-train
    pretrainer = TFTPretrainer(
        tft=tft,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    results = pretrainer.train(
        features=selected,
        n_market_features=n_features,
        lookback=lookback,
        market_cols=market_cols,
    )

    # Save
    save_path = Path(config.base.paths.models_dir) / "pretrained" / "tft_pretrained.pt"
    pretrainer.save_pretrained(save_path)

    print("\n" + "=" * 50)
    print("TFT PRE-TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Epochs trained:  {results['epochs_trained']}")
    print(f"  Best val loss:   {results['best_val_loss']:.6f}")
    print(f"  Val direction:   {results['final_val_acc']:.4f}")
    print(f"  Saved to:        {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
