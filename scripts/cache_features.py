#!/usr/bin/env python3
"""Pre-compute and cache features for faster training/hyperopt iterations.

Avoids re-running HurstExtractor + SpectralExtractor (~20 min) every time.

Usage:
    python scripts/cache_features.py
    python scripts/cache_features.py --top-n 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from apexfx.config.loader import load_config
from apexfx.features.pipeline import FeaturePipeline
from apexfx.features.selector import FeatureSelector
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache computed features")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    config = load_config()

    # Load all raw data
    raw_dir = Path(config.base.paths.data_dir) / "raw" / "bars"
    parquet_files = sorted(raw_dir.rglob("*.parquet"))

    if not parquet_files:
        logger.error("No parquet files found in %s", raw_dir)
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in parquet_files]
    data = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d bars from %d files", len(data), len(parquet_files))

    # Compute features
    pipeline = FeaturePipeline()
    features = pipeline.compute(data)
    logger.info("Features computed: %d columns", len(features.columns))

    # Feature selection
    selector = FeatureSelector(top_n=args.top_n)
    selected = selector.fit_transform(features)
    logger.info("Selected %d features", len(selector.selected_features))

    # Save cache
    cache_dir = Path(config.base.paths.data_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_path = cache_dir / "features_all.parquet"
    selected_path = cache_dir / "features_selected.parquet"
    meta_path = cache_dir / "feature_meta.json"

    features.to_parquet(features_path)
    selected.to_parquet(selected_path)

    import json
    meta = {
        "n_bars": len(data),
        "n_files": len(parquet_files),
        "n_features_total": len(features.columns),
        "n_features_selected": len(selector.selected_features),
        "selected_features": selector.selected_features,
        "top_n": args.top_n,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFeature cache saved:")
    print(f"  All features:      {features_path} ({features_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Selected features: {selected_path} ({selected_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Metadata:          {meta_path}")
    print(f"\nTop {args.top_n} features: {selector.selected_features}")


if __name__ == "__main__":
    main()
