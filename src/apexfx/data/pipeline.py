"""Bias-free data pipeline orchestrator.

Implements the complete data preparation workflow that eliminates both
Look-Ahead Bias (LAB) and Look-Back Bias (LIB) as described in
"The Corporate Bond Factor Replication Crisis" (Paper 2), adapted
for forex time-series.

Pipeline stages (in order):
1. **Gap procedure** (LIB fix) — ensures signal at close(t) is paired
   with return from open(t+1) → close(t+1).
2. **Ex-ante filtering** (LAB fix) — winsorisation and normalisation
   use only expanding-window historical statistics.
3. **FSD computation** — Forex Skewness Dispersion regime features.
4. **SBBTS augmentation** — optional synthetic data generation.

All stages are independently configurable and can be disabled.

Usage::

    from apexfx.data.config import BiasFreePipelineConfig
    from apexfx.data.pipeline import BiasFreePipeline

    cfg = BiasFreePipelineConfig()
    pipe = BiasFreePipeline(cfg)
    result = pipe.run(raw_bars, pair="EURUSD")
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from apexfx.data.config import BiasFreePipelineConfig
from apexfx.data.features import FSDCalculator
from apexfx.data.sbbts_generator import SBBTSGenerator
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class BiasFreePipeline:
    """Orchestrate bias-free data preparation for training and evaluation.

    The pipeline is stateless: each call to :meth:`run` operates on the
    provided DataFrame without mutating any internal state.  This makes
    it safe for use across multiple pairs and curriculum stages.

    Args:
        config: Pipeline configuration.  Defaults to all-enabled with
            conservative parameters.

    Example::

        pipe = BiasFreePipeline(BiasFreePipelineConfig(
            fsd=FSDConfig(enabled=True, window=252),
            sbbts=SBBTSConfig(enabled=True, ratio=0.3),
        ))
        clean = pipe.run(raw_bars, pair="EURUSD")
    """

    def __init__(self, config: BiasFreePipelineConfig | None = None) -> None:
        self._cfg = config or BiasFreePipelineConfig()
        self._fsd = FSDCalculator(self._cfg.fsd) if self._cfg.fsd.enabled else None
        self._sbbts = SBBTSGenerator(self._cfg.sbbts) if self._cfg.sbbts.enabled else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        bars: pd.DataFrame,
        pair: str = "EURUSD",
        features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Execute the full bias-free pipeline on a single pair.

        Args:
            bars: Raw OHLCV DataFrame.  Must contain at least
                ``open``, ``high``, ``low``, ``close`` columns.
            pair: Symbol name (for SBBTS per-pair config lookup).
            features: Pre-computed feature columns to attach.
                If provided, these are used *instead of* the raw bar
                columns for FSD computation.

        Returns:
            Processed DataFrame with:
            - Gap-corrected forward returns (``forward_return_gap``).
            - Ex-ante filtered feature columns.
            - FSD regime columns (if enabled).
            - ``synthetic`` flag column (False for all rows unless
              SBBTS augmentation is active).
        """
        logger.info(
            "BiasFreePipeline.run",
            pair=pair,
            n_bars=len(bars),
            gap=self._cfg.gap_procedure.enabled,
            exante=self._cfg.ex_ante_filter.enabled,
            fsd=self._cfg.fsd.enabled,
            sbbts=self._cfg.sbbts.enabled,
        )

        df = bars.copy()

        # Stage 1: Gap procedure (LIB correction)
        if self._cfg.gap_procedure.enabled:
            df = self._apply_gap_procedure(df)

        # Stage 2: Attach features (if provided separately)
        if features is not None:
            # Merge feature columns that don't already exist in df
            new_cols = [c for c in features.columns if c not in df.columns]
            if new_cols:
                df = pd.concat([df, features[new_cols]], axis=1)

        # Stage 3: Ex-ante filtering (LAB correction)
        if self._cfg.ex_ante_filter.enabled:
            df = self._apply_ex_ante_filter(df)

        # Stage 4: FSD regime features
        if self._fsd is not None:
            df = self._apply_fsd(df)

        # Stage 5: SBBTS synthetic augmentation
        if self._sbbts is not None and self._cfg.sbbts.ratio > 0:
            df = self._apply_sbbts(df, pair=pair)
        else:
            df["synthetic"] = False

        logger.info(
            "BiasFreePipeline complete",
            pair=pair,
            n_rows=len(df),
            n_cols=len(df.columns),
        )
        return df

    def run_multi_pair(
        self,
        pair_data: dict[str, pd.DataFrame],
        pair_features: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Execute the pipeline for multiple pairs, optionally in parallel.

        If ``config.n_jobs > 1``, pairs are processed in parallel using
        a process pool.  Cross-pair correlation is preserved in the SBBTS
        stage via :meth:`SBBTSGenerator.generate_multi_pair`.

        Args:
            pair_data: Dict mapping symbol → raw OHLCV DataFrame.
            pair_features: Optional dict mapping symbol → features DataFrame.

        Returns:
            Dict mapping symbol → processed DataFrame.
        """
        n_jobs = self._cfg.n_jobs
        pairs = list(pair_data.keys())

        logger.info(
            "BiasFreePipeline.run_multi_pair",
            pairs=pairs,
            n_jobs=n_jobs,
        )

        if n_jobs <= 1 or len(pairs) <= 1:
            # Sequential
            results = {}
            for pair in pairs:
                features = pair_features.get(pair) if pair_features else None
                results[pair] = self.run(pair_data[pair], pair=pair, features=features)
            return results

        # Parallel
        results: dict[str, pd.DataFrame] = {}
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(pairs))) as pool:
            futures = {}
            for pair in pairs:
                features = pair_features.get(pair) if pair_features else None
                future = pool.submit(self.run, pair_data[pair], pair, features)
                futures[future] = pair

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    results[pair] = future.result()
                except Exception:
                    logger.exception("Pipeline failed for pair", pair=pair)
                    raise

        return results

    def save_parquet(
        self,
        df: pd.DataFrame,
        path: str,
        pair: str = "EURUSD",
    ) -> None:
        """Save processed data to parquet with metadata.

        Args:
            df: Processed DataFrame from :meth:`run`.
            path: Output file path.
            pair: Symbol name stored as parquet metadata.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        metadata = {
            b"pair": pair.encode(),
            b"pipeline_version": b"2.0",
            b"gap_procedure": str(self._cfg.gap_procedure.enabled).encode(),
            b"ex_ante_filter": str(self._cfg.ex_ante_filter.enabled).encode(),
            b"fsd_enabled": str(self._cfg.fsd.enabled).encode(),
            b"sbbts_ratio": str(self._cfg.sbbts.ratio).encode(),
        }
        table = table.replace_schema_metadata({
            **table.schema.metadata, **metadata,
        })
        pq.write_table(
            table, path, compression=self._cfg.output_compression,
        )
        logger.info("Saved parquet", path=path, n_rows=len(df))

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _apply_gap_procedure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply gap procedure to eliminate look-back bias.

        Signal at close(t) is paired with return from open(t+1) → close(t+1).
        This prevents using the close(t) → close(t+1) return which overlaps
        with the signal computation window.

        The resulting ``forward_return_gap`` column is the *unbiased* return
        that should be used for any predictive evaluation.
        """
        cfg = self._cfg.gap_procedure
        lag = cfg.signal_lag

        if "open" not in df.columns or "close" not in df.columns:
            logger.warning("Gap procedure skipped: missing open/close columns")
            return df

        # Forward return: (close[t+lag] - open[t+lag]) / open[t+lag]
        future_open = df["open"].shift(-lag)
        future_close = df["close"].shift(-lag)
        gap_return = (future_close - future_open) / future_open

        df[cfg.return_col] = gap_return

        # Drop last `lag` rows where forward return is NaN
        n_before = len(df)
        df = df.iloc[:-lag].copy() if lag > 0 else df
        logger.debug(
            "Gap procedure applied",
            return_col=cfg.return_col,
            lag=lag,
            rows_dropped=n_before - len(df),
        )
        return df

    def _apply_ex_ante_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply expanding-window winsorisation and clipping (no look-ahead).

        For each numeric feature column, at each bar *t*:
        - Quantile boundaries are computed from bars [0 .. t-1].
        - The value at bar *t* is clipped to those boundaries.
        - Z-score is computed using expanding mean/std from [0 .. t-1].

        This is computationally expensive (O(T²) per column in the naïve
        case), so we use a vectorised rolling approach with
        ``expanding().quantile()`` and ``.shift(1)`` for strict causality.
        """
        cfg = self._cfg.ex_ante_filter
        low_q, high_q = cfg.winsor_quantiles
        min_hist = cfg.min_history

        # Identify feature columns to filter (exclude OHLCV, time, metadata)
        exclude = {
            "time", "datetime", "date",
            "open", "high", "low", "close", "volume",
            "tick_count", "tick_volume", "spread", "real_volume",
            "synthetic", "regime_label", "hurst_regime",
            self._cfg.gap_procedure.return_col,
        }
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if not feature_cols:
            logger.debug("Ex-ante filter: no feature columns to filter")
            return df

        logger.debug(
            "Ex-ante filter",
            n_cols=len(feature_cols),
            winsor=cfg.winsor_quantiles,
        )

        result = df.copy()

        for col in feature_cols:
            series = result[col].astype(np.float64)

            # Expanding quantiles shifted by 1 (strictly ex-ante)
            expanding = series.expanding(min_periods=min_hist)
            low_bound = expanding.quantile(low_q).shift(1)
            high_bound = expanding.quantile(high_q).shift(1)

            # Winsorise: clip to ex-ante quantile boundaries
            clipped = series.clip(lower=low_bound, upper=high_bound)

            # Z-score with expanding mean/std (shifted)
            exp_mean = series.expanding(min_periods=min_hist).mean().shift(1)
            exp_std = series.expanding(min_periods=min_hist).std().shift(1)
            exp_std = exp_std.clip(lower=1e-8)

            zscore = (clipped - exp_mean) / exp_std
            zscore = zscore.clip(-cfg.clip_zscore, cfg.clip_zscore)

            # Fill NaN warm-up period with 0
            result[col] = zscore.fillna(0.0)

        return result

    def _apply_fsd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append FSD regime features."""
        try:
            return self._fsd.compute(df)
        except ValueError as e:
            logger.warning("FSD computation skipped", reason=str(e))
            # Add zero columns so downstream code doesn't break
            for col in self._cfg.fsd.output_cols:
                df[col] = 0.0
            return df

    def _apply_sbbts(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Generate synthetic data and merge with real data."""
        # Extract only the OHLCV subset for generation
        ohlcv_cols = ["time", "open", "high", "low", "close", "volume", "tick_count"]
        available = [c for c in ohlcv_cols if c in df.columns]
        real_ohlcv = df[available].copy()

        augmented = self._sbbts.generate_augmented_dataset(
            real_ohlcv, pair=pair,
        )

        # The real rows in `augmented` are plain OHLCV — we need to re-attach
        # the full feature columns from `df`.  Synthetic rows get NaN features
        # (they'll need to go through the feature pipeline separately).
        real_mask = ~augmented["synthetic"]
        n_real = int(real_mask.sum())

        if n_real == len(df):
            # Real portion matches — merge features back
            result = df.copy()
            result["synthetic"] = False

            synth_rows = augmented[augmented["synthetic"]].copy()
            # Synthetic rows get NaN for feature columns
            for col in df.columns:
                if col not in synth_rows.columns:
                    synth_rows[col] = np.nan

            result = pd.concat([result, synth_rows], ignore_index=True)
            result = result.sort_values("time").reset_index(drop=True)
        else:
            # Fallback: just return augmented OHLCV
            result = augmented

        return result
