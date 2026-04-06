"""DBSCAN clustering for support/resistance liquidity zone detection.

Optimized: pre-allocated NumPy arrays, no iloc writes.
DBSCAN is inherently per-window, but we minimize Python overhead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.math_utils import atr


class ClusteringExtractor(BaseFeatureExtractor):
    """
    Applies DBSCAN to historical price levels weighted by volume
    to identify support/resistance liquidity zones.
    Parameters eps and min_samples are ATR-adaptive.
    """

    def __init__(
        self,
        window: int = 200,
        atr_period: int = 14,
        eps_atr_multiplier: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        self._window = window
        self._atr_period = atr_period
        self._eps_multiplier = eps_atr_multiplier
        self._min_samples = min_samples

    @property
    def feature_names(self) -> list[str]:
        return [
            "nearest_support_distance",
            "nearest_resistance_distance",
            "in_liquidity_zone",
            "n_clusters",
            "cluster_density",
        ]

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        n = len(bars)
        data = np.full((n, 5), np.nan)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        volume = bars["volume"].values.astype(np.float64)

        atr_values = atr(high, low, close, self._atr_period)

        col_support = 0
        col_resistance = 1
        col_in_zone = 2
        col_n_clusters = 3
        col_density = 4

        for i in range(self._window, n):
            current_price = close[i]
            current_atr = atr_values[i]
            if np.isnan(current_atr) or current_atr <= 0:
                continue

            eps = current_atr * self._eps_multiplier
            s = i - self._window

            w_close = close[s:i]
            w_high = high[s:i]
            w_low = low[s:i]
            w_volume = volume[s:i]

            # Build volume-weighted price samples
            price_levels = np.concatenate([w_high, w_low, w_close])
            vol_weights = np.concatenate([w_volume, w_volume, w_volume])

            total_w = vol_weights.sum()
            if total_w > 0:
                weights_norm = np.maximum((vol_weights / total_w * 100).astype(int), 1)
                price_samples = np.repeat(price_levels, weights_norm)
            else:
                price_samples = price_levels

            if len(price_samples) < self._min_samples:
                continue

            # DBSCAN
            labels = DBSCAN(eps=eps, min_samples=self._min_samples).fit_predict(
                price_samples.reshape(-1, 1)
            )

            unique_labels = set(labels)
            unique_labels.discard(-1)
            n_clusters = len(unique_labels)
            data[i, col_n_clusters] = n_clusters

            if n_clusters == 0:
                continue

            # Compute cluster centers and sizes via vectorized groupby
            label_arr = np.array(list(unique_labels))
            cluster_centers = np.empty(n_clusters)
            cluster_sizes = np.empty(n_clusters)
            for j, lbl in enumerate(label_arr):
                mask = labels == lbl
                cluster_centers[j] = price_samples[mask].mean()
                cluster_sizes[j] = mask.sum()

            # Nearest support (below current price)
            supports = cluster_centers[cluster_centers < current_price]
            if len(supports) > 0:
                nearest_support = supports.max()
                data[i, col_support] = (current_price - nearest_support) / current_price
            else:
                data[i, col_support] = 0.0

            # Nearest resistance (above current price)
            resistances = cluster_centers[cluster_centers > current_price]
            if len(resistances) > 0:
                nearest_resistance = resistances.min()
                data[i, col_resistance] = (nearest_resistance - current_price) / current_price
            else:
                data[i, col_resistance] = 0.0

            # In liquidity zone?
            min_dist = np.abs(cluster_centers - current_price).min()
            data[i, col_in_zone] = 1.0 if min_dist < eps else 0.0

            # Cluster density
            data[i, col_density] = cluster_sizes.mean() / len(price_samples)

        return pd.DataFrame(data, index=bars.index, columns=self.feature_names)
