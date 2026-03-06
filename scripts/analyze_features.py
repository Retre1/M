#!/usr/bin/env python3
"""Standalone feature importance analysis.

Runs the full feature pipeline on available data and ranks features
by GradientBoosting importance for predicting next-bar direction.

Usage:
    python scripts/analyze_features.py [--top-n 15] [--synthetic-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apexfx.config.loader import load_config
from apexfx.data.synthetic import SyntheticDataGenerator
from apexfx.features.pipeline import FeaturePipeline
from apexfx.features.selector import FeatureSelector
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def load_data(config, synthetic_only: bool = False) -> pd.DataFrame:
    """Load real data from parquet store, or generate synthetic."""
    if not synthetic_only:
        raw_dir = Path(config.base.paths.data_dir) / "raw" / "bars"
        parquet_files = list(raw_dir.rglob("*.parquet"))
        if parquet_files:
            logger.info("Loading real data from %d parquet files", len(parquet_files))
            dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
            data = pd.concat(dfs, ignore_index=True)
            logger.info("Loaded %d bars", len(data))
            return data

    logger.info("Generating synthetic data for analysis")
    gen = SyntheticDataGenerator(seed=config.base.seed)
    data = gen.generate_gbm(n_steps=10_000, mu=0.0001, sigma=0.01)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--top-n", type=int, default=15, help="Number of features to select")
    parser.add_argument("--synthetic-only", action="store_true", help="Use synthetic data only")
    args = parser.parse_args()

    config = load_config()
    data = load_data(config, synthetic_only=args.synthetic_only)

    # Run feature pipeline
    logger.info("Computing features...")
    pipeline = FeaturePipeline()
    features = pipeline.compute(data)
    logger.info("Total features computed: %d", pipeline.n_features)
    logger.info("Total bars: %d", len(features))

    # Run feature importance
    selector = FeatureSelector(top_n=args.top_n)
    selector.fit(features)

    # Print detailed results
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RANKING")
    print("=" * 70)

    ranked = sorted(selector.importance_scores.items(), key=lambda kv: kv[1], reverse=True)

    print(f"\n{'Rank':<6}{'Feature':<40}{'Importance':<12}{'Selected'}")
    print("-" * 70)

    selected_set = set(selector.selected_features)
    for rank, (name, score) in enumerate(ranked, 1):
        marker = " *" if name in selected_set else ""
        print(f"{rank:<6}{name:<40}{score:<12.6f}{marker}")

    print("\n" + "=" * 70)
    print(f"Selected {args.top_n} features (marked with *):")
    for i, name in enumerate(selector.selected_features, 1):
        print(f"  {i:2d}. {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
