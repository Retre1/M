"""Pre-compute engineered features and cache to parquet.

The full FeaturePipeline takes ~2 minutes per 12k bars because of the
O(n²) HurstExtractor and SpectralExtractor. On a GPU server we want to
pay this cost once (per symbol/timeframe), cache to parquet, and let
``train_v2.py`` load the enriched frame instantly.

Usage::

    python scripts/cache_features_v2.py --symbol EURUSD --timeframe H1
    python scripts/cache_features_v2.py --symbols EURUSD GBPUSD USDJPY AUDUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apexfx.data.bar_loader import load_bars
from apexfx.training.trainer import FeaturePipeline
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def cache_one(symbol: str, timeframe: str, data_root: Path, cache_root: Path) -> Path:
    """Compute features for one symbol/timeframe and write to cache."""
    df = load_bars(symbol, timeframe, data_root)
    pipeline = FeaturePipeline()
    logger.info("Computing features", symbol=symbol, timeframe=timeframe,
                n_bars=len(df))
    enriched = pipeline.compute(df)
    out_dir = cache_root / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    enriched.to_parquet(out_path, index=False)
    logger.info("Features cached", path=str(out_path),
                n_rows=len(enriched), n_cols=len(enriched.columns))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None,
                        help="Single symbol (shortcut for --symbols)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Multiple symbols to cache")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/bars"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/features"))
    args = parser.parse_args()

    symbols = args.symbols or ([args.symbol] if args.symbol else ["EURUSD"])
    for sym in symbols:
        cache_one(sym, args.timeframe, args.data_root, args.cache_root)


if __name__ == "__main__":
    main()
