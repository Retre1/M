"""Load and consolidate partitioned daily OHLCV parquet bars.

Reads ``data/raw/bars/<symbol>/<tf>/*.parquet`` files, concatenates,
deduplicates by time, and returns a clean chronologically-sorted frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def load_features_cache(
    symbol: str,
    timeframe: str = "H1",
    cache_root: Path | str = "data/cache/features",
) -> pd.DataFrame | None:
    """Load pre-computed features from cache if available; return None otherwise."""
    path = Path(cache_root) / symbol / timeframe / "features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    logger.info(
        "Loaded features cache",
        symbol=symbol, timeframe=timeframe,
        path=str(path), n_rows=len(df), n_cols=len(df.columns),
    )
    return df


def load_bars(
    symbol: str,
    timeframe: str = "H1",
    data_root: Path | str = "data/raw/bars",
) -> pd.DataFrame:
    """Load all daily parquet bar files for a symbol/timeframe.

    Args:
        symbol: Instrument code (e.g. ``"EURUSD"``).
        timeframe: Timeframe folder name (e.g. ``"H1"``).
        data_root: Root directory holding ``<symbol>/<tf>/*.parquet`` files.

    Returns:
        DataFrame sorted by ``time`` with columns
        ``[time, open, high, low, close, volume]``.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    root = Path(data_root) / symbol / timeframe
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {root}. "
            f"Expected daily files like YYYY-MM-DD.parquet.",
        )

    frames = [pd.read_parquet(f) for f in files]
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    logger.info(
        "Loaded bars",
        symbol=symbol,
        timeframe=timeframe,
        n_files=len(files),
        n_bars=len(df),
        start=str(df["time"].min()),
        end=str(df["time"].max()),
    )
    return df


def load_multi_symbol_features(
    symbols: tuple[str, ...] | list[str],
    timeframe: str = "H1",
    cache_root: Path | str = "data/cache/features",
    data_root: Path | str = "data/raw/bars",
    fallback_to_raw: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load cached features for each symbol, fall back to raw bars if missing.

    Returns a dict keyed by symbol. Symbols with neither cached features
    nor raw bars are skipped with a warning (not raised) so partial
    multi-symbol runs remain possible.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_features_cache(symbol, timeframe, cache_root)
        if df is None and fallback_to_raw:
            try:
                df = load_bars(symbol, timeframe, data_root)
                logger.warning(
                    "Multi-symbol: features cache missing, loaded raw bars",
                    symbol=symbol,
                )
            except FileNotFoundError as e:
                logger.warning("Multi-symbol: symbol skipped", symbol=symbol,
                               reason=str(e))
                continue
        if df is None:
            logger.warning("Multi-symbol: symbol skipped (no data)", symbol=symbol)
            continue
        result[symbol] = df

    if not result:
        raise FileNotFoundError(
            f"No data found for any of {list(symbols)} in either "
            f"{cache_root} or {data_root}.",
        )

    logger.info(
        "Multi-symbol data loaded",
        symbols=list(result.keys()),
        n_symbols=len(result),
        total_bars=sum(len(d) for d in result.values()),
    )
    return result
