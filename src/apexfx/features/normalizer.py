"""Feature normalization with rolling statistics (no look-ahead bias)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RunningStats:
    """Exponentially weighted running mean and variance."""
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    var: np.ndarray = field(default_factory=lambda: np.array([]))
    count: int = 0
    alpha: float = 0.01  # EMA decay rate


class FeatureNormalizer:
    """
    Normalizes features using rolling statistics.
    Maintains running stats for live inference without look-ahead bias.
    """

    def __init__(self, method: str = "zscore", window: int = 252) -> None:
        self._method = method
        self._window = window
        self._stats: RunningStats | None = None

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling normalization on a full DataFrame (training mode)."""
        if self._method == "zscore":
            return self._rolling_zscore(features)
        elif self._method == "rank":
            return self._rolling_rank(features)
        elif self._method == "minmax":
            return self._rolling_minmax(features)
        return features

    def transform_online(self, features: np.ndarray) -> np.ndarray:
        """Normalize a single observation using running statistics (live mode)."""
        if self._stats is None:
            n_features = features.shape[-1]
            self._stats = RunningStats(
                mean=np.zeros(n_features),
                var=np.ones(n_features),
                count=0,
            )

        stats = self._stats
        alpha = stats.alpha

        # Update running statistics
        stats.count += 1
        delta = features - stats.mean
        stats.mean = stats.mean + alpha * delta
        stats.var = (1 - alpha) * stats.var + alpha * delta * (features - stats.mean)

        # Normalize
        std = np.sqrt(np.maximum(stats.var, 1e-8))
        normalized = (features - stats.mean) / std

        return np.clip(normalized, -5.0, 5.0)

    def get_state(self) -> dict:
        """Get normalizer state for serialization."""
        if self._stats is None:
            return {}
        return {
            "mean": self._stats.mean,
            "var": self._stats.var,
            "count": self._stats.count,
            "alpha": self._stats.alpha,
        }

    def load_state(self, state: dict) -> None:
        """Load normalizer state from serialized form."""
        if not state:
            return
        self._stats = RunningStats(
            mean=state["mean"],
            var=state["var"],
            count=state["count"],
            alpha=state.get("alpha", 0.01),
        )

    def _rolling_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling z-score normalization (strictly causal -- no look-ahead).

        Uses ``.shift(1)`` so that each bar is normalised using statistics
        computed from *previous* bars only.  The current bar's value is never
        included in its own mean/std calculation.

        During the warm-up period (fewer than ``self._window`` prior bars)
        an expanding window is used automatically by ``min_periods=1``.
        """
        result = df.copy()
        for col in df.columns:
            series = pd.Series(df[col].values.astype(float))
            rolling_mean = series.rolling(self._window, min_periods=1).mean().shift(1)
            rolling_std = series.rolling(self._window, min_periods=1).std().shift(1)
            rolling_std = rolling_std.fillna(0).values
            rolling_std = np.maximum(rolling_std, 1e-8)
            rolling_mean = rolling_mean.fillna(0).values
            result[col] = np.clip(
                (series.values - rolling_mean) / rolling_std, -5.0, 5.0
            )
        return result

    def _rolling_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling rank normalization: values mapped to [0, 1] within window."""
        result = df.copy()
        for col in df.columns:
            values = df[col].values.astype(float)
            n = len(values)
            ranked = np.full(n, np.nan)
            for i in range(self._window, n):
                window = values[i - self._window : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) > 1:
                    rank = np.searchsorted(np.sort(valid), values[i]) / (len(valid) - 1)
                    ranked[i] = rank
                else:
                    ranked[i] = 0.5
            result[col] = ranked
        return result

    def _rolling_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling min-max normalization to [0, 1]."""
        result = df.copy()
        for col in df.columns:
            values = df[col].values.astype(float)
            rolling_min = pd.Series(values).rolling(self._window, min_periods=1).min().values
            rolling_max = pd.Series(values).rolling(self._window, min_periods=1).max().values
            range_ = rolling_max - rolling_min
            range_ = np.maximum(range_, 1e-8)
            result[col] = (values - rolling_min) / range_
        return result
