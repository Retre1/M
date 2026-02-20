"""DBSCAN clustering for support/resistance liquidity zone detection."""

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
        result = pd.DataFrame(index=bars.index)
        for col in self.feature_names:
            result[col] = np.nan

        close = bars["close"].values
        high = bars["high"].values
        low = bars["low"].values
        volume = bars["volume"].values

        # Compute ATR for adaptive eps
        atr_values = atr(high, low, close, self._atr_period)

        for i in range(self._window, n):
            current_price = close[i]
            current_atr = atr_values[i]
            if np.isnan(current_atr) or current_atr <= 0:
                continue

            # Adaptive DBSCAN eps
            eps = current_atr * self._eps_multiplier

            # Create price points weighted by volume
            w_close = close[i - self._window : i]
            w_high = high[i - self._window : i]
            w_low = low[i - self._window : i]
            w_volume = volume[i - self._window : i]

            # Use high/low/close as key price levels, weighted by volume
            price_levels = np.concatenate([w_high, w_low, w_close])
            vol_weights = np.concatenate([w_volume, w_volume, w_volume])

            # Repeat price levels by volume weight (normalized)
            if vol_weights.sum() > 0:
                weights_norm = (vol_weights / vol_weights.sum() * 100).astype(int)
                weights_norm = np.maximum(weights_norm, 1)
                price_samples = np.repeat(price_levels, weights_norm)
            else:
                price_samples = price_levels

            if len(price_samples) < self._min_samples:
                continue

            # Run DBSCAN
            X = price_samples.reshape(-1, 1)
            clustering = DBSCAN(eps=eps, min_samples=self._min_samples).fit(X)

            labels = clustering.labels_
            unique_labels = set(labels) - {-1}
            n_clusters = len(unique_labels)

            result.iloc[i, result.columns.get_loc("n_clusters")] = n_clusters

            if n_clusters == 0:
                continue

            # Compute cluster centers
            cluster_centers = []
            cluster_sizes = []
            for label in unique_labels:
                mask = labels == label
                center = X[mask].mean()
                size = mask.sum()
                cluster_centers.append(center)
                cluster_sizes.append(size)

            cluster_centers = np.array(cluster_centers)
            cluster_sizes = np.array(cluster_sizes)

            # Find nearest support (cluster center below current price)
            supports = cluster_centers[cluster_centers < current_price]
            if len(supports) > 0:
                nearest_support = supports[np.argmax(supports)]  # Closest support below
                result.iloc[i, result.columns.get_loc("nearest_support_distance")] = (
                    (current_price - nearest_support) / current_price
                )
            else:
                result.iloc[i, result.columns.get_loc("nearest_support_distance")] = 0.0

            # Find nearest resistance (cluster center above current price)
            resistances = cluster_centers[cluster_centers > current_price]
            if len(resistances) > 0:
                nearest_resistance = resistances[np.argmin(resistances)]  # Closest resistance above
                result.iloc[i, result.columns.get_loc("nearest_resistance_distance")] = (
                    (nearest_resistance - current_price) / current_price
                )
            else:
                result.iloc[i, result.columns.get_loc("nearest_resistance_distance")] = 0.0

            # Is current price inside a cluster?
            min_dist = np.min(np.abs(cluster_centers - current_price))
            result.iloc[i, result.columns.get_loc("in_liquidity_zone")] = (
                1.0 if min_dist < eps else 0.0
            )

            # Cluster density: weighted average cluster size
            result.iloc[i, result.columns.get_loc("cluster_density")] = (
                cluster_sizes.mean() / len(price_samples)
            )

        return result
