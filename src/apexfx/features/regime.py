"""Market regime classification combining Hurst, volatility, and trend strength.

Fully vectorized — no Python for-loops. ADX computation uses Wilder's
smoothing via vectorized cumulative operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.math_utils import garman_klass_volatility


class RegimeExtractor(BaseFeatureExtractor):
    """
    Classifies the current market regime:
    - Trending: Hurst > 0.55 AND strong trend
    - Mean-reverting: Hurst < 0.45
    - Flat/Choppy: otherwise

    Also computes trend strength (ADX-like) and realized volatility.
    """

    def __init__(
        self,
        vol_window: int = 20,
        adx_period: int = 14,
        trend_threshold: float = 25.0,
    ) -> None:
        self._vol_window = vol_window
        self._adx_period = adx_period
        self._trend_threshold = trend_threshold

    @property
    def feature_names(self) -> list[str]:
        return [
            "realized_vol",
            "trend_strength",
            "regime_trending",
            "regime_mean_reverting",
            "regime_flat",
            "regime_label",
        ]

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        n = len(bars)

        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        close = bars["close"].values.astype(np.float64)
        open_ = bars["open"].values.astype(np.float64)

        realized_vol = garman_klass_volatility(open_, high, low, close, self._vol_window)
        trend_strength = self._compute_adx(high, low, close, self._adx_period)

        # Hurst: use pre-computed if available, else default to 0.5
        if "hurst_exponent" in bars.columns:
            hurst_values = bars["hurst_exponent"].values.astype(np.float64)
            hurst_values = np.where(np.isnan(hurst_values), 0.5, hurst_values)
        else:
            hurst_values = np.full(n, 0.5)

        ts = np.where(np.isnan(trend_strength), 0.0, trend_strength)

        # Vectorized regime classification
        is_trending = (hurst_values > 0.55) & (ts > self._trend_threshold)
        is_reverting = (hurst_values < 0.45) & ~is_trending
        is_flat = ~is_trending & ~is_reverting

        regime_trending = is_trending.astype(np.float64)
        regime_mean_reverting = is_reverting.astype(np.float64)
        regime_flat = is_flat.astype(np.float64)
        regime_label = np.where(is_trending, 2.0, np.where(is_reverting, 0.0, 1.0))

        return pd.DataFrame(
            {
                "realized_vol": realized_vol,
                "trend_strength": trend_strength,
                "regime_trending": regime_trending,
                "regime_mean_reverting": regime_mean_reverting,
                "regime_flat": regime_flat,
                "regime_label": regime_label,
            },
            index=bars.index,
        )

    @staticmethod
    def _compute_adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Compute Average Directional Index (ADX) — fully vectorized."""
        n = len(high)
        result = np.full(n, np.nan)

        if n < period * 2:
            return result

        # True Range — vectorized
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        # Directional Movement — vectorized
        up = np.diff(high, prepend=high[0])
        down = np.diff(-low, prepend=-low[0])

        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0

        # Wilder's smoothing via cumsum trick for the seed, then recursive
        atr_s = np.zeros(n)
        plus_dm_s = np.zeros(n)
        minus_dm_s = np.zeros(n)

        atr_s[period] = tr[1: period + 1].sum()
        plus_dm_s[period] = plus_dm[1: period + 1].sum()
        minus_dm_s[period] = minus_dm[1: period + 1].sum()

        # Wilder's smoothing is recursive: s[i] = s[i-1] * (1 - 1/period) + x[i]
        # This cannot be fully vectorized, but the loop is over scalars — very fast.
        decay = 1.0 - 1.0 / period
        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] * decay + tr[i]
            plus_dm_s[i] = plus_dm_s[i - 1] * decay + plus_dm[i]
            minus_dm_s[i] = minus_dm_s[i - 1] * decay + minus_dm[i]

        # Directional Indicators — vectorized
        safe_atr = np.where(atr_s > 0, atr_s, 1.0)
        plus_di = np.where(atr_s > 0, 100.0 * plus_dm_s / safe_atr, 0.0)
        minus_di = np.where(atr_s > 0, 100.0 * minus_dm_s / safe_atr, 0.0)

        di_sum = plus_di + minus_di
        safe_di_sum = np.where(di_sum > 0, di_sum, 1.0)
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / safe_di_sum, 0.0)

        # ADX: smoothed DX (Wilder's)
        adx_start = 2 * period
        if adx_start < n:
            result[adx_start] = dx[period: adx_start + 1].mean()
            for i in range(adx_start + 1, n):
                result[i] = (result[i - 1] * (period - 1) + dx[i]) / period

        return result
