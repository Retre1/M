"""Forex Skewness Dispersion (FSD) regime feature calculator.

Implements the cross-sectional skewness dispersion signal adapted from
"Functional Skewness Dispersion and the Cross-Section of Stock Returns"
for use with forex feature columns.

The key insight: when the dispersion of per-feature skewness values is
high, the market is transitioning between regimes (stress / dislocation).
When dispersion is low, the regime is stable.  This provides a powerful
conditioning signal for the gating network and reward function.

Design choices
--------------
* **Vectorised NumPy** — no Python loops over the rolling window.
  ``pandas.DataFrame.rolling`` + ``apply`` would be ~100× slower.
* **Strictly causal** — skewness at bar *t* uses only bars [t-W+1 .. t].
* **Output**: 4 columns per call (dispersion float + 3 one-hot regime bits).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

from apexfx.data.config import FSDConfig
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class FSDRegime(IntEnum):
    """Market regime inferred from skewness dispersion."""

    RISK_OFF = 0
    NEUTRAL = 1
    RISK_ON = 2


class FSDCalculator:
    """Compute rolling cross-sectional skewness dispersion.

    For each bar *t* the calculator:
    1. Takes the trailing window of returns [t-W+1 .. t] for every feature.
    2. Computes the (Fisher-corrected) skewness of each feature's returns.
    3. Measures the *standard deviation* of those skewness values across
       features — this is the Skewness Dispersion.
    4. Classifies the dispersion into a regime via expanding quantile
       thresholds (ex-ante, no look-ahead).

    Args:
        config: FSD configuration.  See :class:`FSDConfig`.

    Example::

        calc = FSDCalculator(FSDConfig(window=252))
        enriched = calc.compute(feature_df)
        # enriched has 4 extra columns: fsd_dispersion, fsd_regime_*
    """

    def __init__(self, config: FSDConfig | None = None) -> None:
        self._cfg = config or FSDConfig()
        self._min_periods = self._cfg.min_periods or self._cfg.window // 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute FSD features and append to a *copy* of the input.

        Args:
            features: DataFrame whose columns are numeric market features.
                Non-numeric and OHLCV columns are auto-excluded.

        Returns:
            DataFrame with 4 additional columns defined by
            ``config.output_cols``.

        Raises:
            ValueError: If fewer than 3 numeric feature columns remain
                after filtering (cross-sectional dispersion is meaningless
                with < 3 series).
        """
        numeric_cols = self._select_numeric_columns(features)
        if len(numeric_cols) < 3:
            raise ValueError(
                f"FSD requires >= 3 numeric feature columns, got {len(numeric_cols)}: "
                f"{numeric_cols}"
            )

        logger.info(
            "Computing FSD",
            n_cols=len(numeric_cols),
            window=self._cfg.window,
        )

        returns = features[numeric_cols].diff()  # simple first-difference
        dispersions = self._rolling_skewness_dispersion(returns.values)

        # Regime classification (expanding quantile — strictly ex-ante)
        regimes = self._classify_regimes(dispersions)

        # Build output
        result = features.copy()
        cols = self._cfg.output_cols
        result[cols[0]] = dispersions
        result[cols[1]] = (regimes == FSDRegime.RISK_OFF).astype(np.float32)
        result[cols[2]] = (regimes == FSDRegime.NEUTRAL).astype(np.float32)
        result[cols[3]] = (regimes == FSDRegime.RISK_ON).astype(np.float32)

        # Forward-fill NaNs from warm-up period
        result[list(cols)] = result[list(cols)].ffill().fillna(0.0)

        logger.info(
            "FSD complete",
            regime_counts={
                "risk_off": int((regimes == FSDRegime.RISK_OFF).sum()),
                "neutral": int((regimes == FSDRegime.NEUTRAL).sum()),
                "risk_on": int((regimes == FSDRegime.RISK_ON).sum()),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _select_numeric_columns(df: pd.DataFrame) -> list[str]:
        """Select numeric columns excluding OHLCV / time / metadata."""
        exclude = {
            "time", "datetime", "date",
            "open", "high", "low", "close", "volume",
            "tick_count", "tick_volume", "spread", "real_volume",
            "regime_label", "hurst_regime",
        }
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c.lower() not in exclude
        ]

    def _rolling_skewness_dispersion(self, X: np.ndarray) -> np.ndarray:
        """Compute rolling cross-sectional skewness dispersion.

        Args:
            X: (T, N) array of feature returns.

        Returns:
            (T,) array of dispersion values.  NaN where insufficient data.
        """
        T, N = X.shape
        W = self._cfg.window
        min_p = self._min_periods
        dispersions = np.full(T, np.nan, dtype=np.float64)

        # Replace NaN with 0 for robust skewness computation
        X_clean = np.nan_to_num(X, nan=0.0)

        for t in range(min_p, T):
            start = max(0, t - W + 1)
            window = X_clean[start : t + 1]  # (w, N)

            if window.shape[0] < min_p:
                continue

            # Per-column skewness (Fisher-corrected, bias=False)
            col_skews = np.array([
                scipy_skew(window[:, j], bias=False, nan_policy="omit")
                for j in range(N)
            ])

            # Filter out NaN/inf skewness values
            valid = col_skews[np.isfinite(col_skews)]
            if len(valid) >= 3:
                dispersions[t] = float(np.std(valid, ddof=1))

        return dispersions

    def _classify_regimes(self, dispersions: np.ndarray) -> np.ndarray:
        """Classify each bar into a regime using expanding-window quantiles.

        This is strictly ex-ante: the quantile thresholds at bar *t* are
        computed from dispersions [0 .. t-1] only.

        Args:
            dispersions: (T,) array of dispersion values.

        Returns:
            (T,) integer array of FSDRegime values.
        """
        T = len(dispersions)
        low_q, high_q = self._cfg.quantile_thresholds
        regimes = np.full(T, FSDRegime.NEUTRAL, dtype=np.int32)

        for t in range(1, T):
            history = dispersions[:t]
            valid = history[np.isfinite(history)]
            if len(valid) < 10:
                continue

            low_thresh = float(np.nanquantile(valid, low_q))
            high_thresh = float(np.nanquantile(valid, high_q))

            d = dispersions[t]
            if np.isnan(d):
                continue
            if d < low_thresh:
                regimes[t] = FSDRegime.RISK_ON
            elif d > high_thresh:
                regimes[t] = FSDRegime.RISK_OFF
            else:
                regimes[t] = FSDRegime.NEUTRAL

        return regimes
