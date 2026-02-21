"""KDE-based Volume Profile for identifying High Volume Nodes (HVN) and Low Volume Nodes (LVN)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from apexfx.features import BaseFeatureExtractor


class VolumeProfileExtractor(BaseFeatureExtractor):
    """
    Uses Kernel Density Estimation over price-volume data to identify
    key price levels where liquidity is concentrated.
    """

    def __init__(self, window: int = 100, n_price_bins: int = 200) -> None:
        self._window = window
        self._n_bins = n_price_bins

    @property
    def feature_names(self) -> list[str]:
        return [
            "hvn_distance",
            "lvn_distance",
            "volume_profile_skew",
            "poc_price",
            "poc_distance",
        ]

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        n = len(bars)
        result = pd.DataFrame(index=bars.index)
        for col in self.feature_names:
            result[col] = np.nan

        close = bars["close"].values
        volume = bars["volume"].values
        high = bars["high"].values
        low = bars["low"].values

        for i in range(self._window, n):
            window_close = close[i - self._window : i]
            window_volume = volume[i - self._window : i]
            window_high = high[i - self._window : i]
            window_low = low[i - self._window : i]

            current_price = close[i]

            hvn, lvn, poc, skew = self._compute_profile(
                window_close, window_volume, window_high, window_low, current_price
            )

            result.iloc[i, result.columns.get_loc("hvn_distance")] = (
                (current_price - hvn) / current_price if hvn > 0 else 0
            )
            result.iloc[i, result.columns.get_loc("lvn_distance")] = (
                (current_price - lvn) / current_price if lvn > 0 else 0
            )
            result.iloc[i, result.columns.get_loc("volume_profile_skew")] = skew
            result.iloc[i, result.columns.get_loc("poc_price")] = poc
            result.iloc[i, result.columns.get_loc("poc_distance")] = (
                (current_price - poc) / current_price if poc > 0 else 0
            )

        return result

    def _compute_profile(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        current_price: float,
    ) -> tuple[float, float, float, float]:
        """Compute KDE volume profile and return (nearest_hvn, nearest_lvn, poc, skew)."""
        # Create price samples weighted by volume
        price_range = np.linspace(low.min(), high.max(), self._n_bins)

        # Weight each bar's contribution by its volume
        total_vol = volume.sum()
        if total_vol <= 0:
            return 0.0, 0.0, current_price, 0.0

        weights = volume / total_vol
        # Create weighted samples: repeat each price proportional to its volume
        samples = np.repeat(close, np.maximum(1, (weights * 1000).astype(int)))

        if len(samples) < 3:
            return 0.0, 0.0, current_price, 0.0

        try:
            kde = gaussian_kde(samples, bw_method="silverman")
            density = kde(price_range)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 0.0, current_price, 0.0

        # Point of Control (POC): price level with highest density
        poc_idx = np.argmax(density)
        poc = price_range[poc_idx]

        # Find HVN: nearest price level with density > 70th percentile
        threshold_high = np.percentile(density, 70)
        threshold_low = np.percentile(density, 30)

        hvn_mask = density >= threshold_high
        lvn_mask = density <= threshold_low

        hvn_prices = price_range[hvn_mask]
        lvn_prices = price_range[lvn_mask]

        # Nearest HVN to current price
        if len(hvn_prices) > 0:
            hvn_distances = np.abs(hvn_prices - current_price)
            nearest_hvn = hvn_prices[np.argmin(hvn_distances)]
        else:
            nearest_hvn = poc

        # Nearest LVN to current price
        if len(lvn_prices) > 0:
            lvn_distances = np.abs(lvn_prices - current_price)
            nearest_lvn = lvn_prices[np.argmin(lvn_distances)]
        else:
            nearest_lvn = current_price

        # Volume profile skew: density above vs below current price
        above_mask = price_range >= current_price
        below_mask = price_range < current_price
        density_above = density[above_mask].sum() if above_mask.any() else 0
        density_below = density[below_mask].sum() if below_mask.any() else 0
        total_density = density_above + density_below
        skew = (density_above - density_below) / total_density if total_density > 0 else 0

        return nearest_hvn, nearest_lvn, poc, skew
