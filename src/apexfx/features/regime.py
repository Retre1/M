"""Market regime classification combining Hurst, volatility, and trend strength."""

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
        result = pd.DataFrame(index=bars.index)

        high = bars["high"].values
        low = bars["low"].values
        close = bars["close"].values
        open_ = bars["open"].values

        # Realized volatility (Garman-Klass)
        result["realized_vol"] = garman_klass_volatility(
            open_, high, low, close, self._vol_window
        )

        # Trend strength (ADX-like computation)
        result["trend_strength"] = self._compute_adx(high, low, close, self._adx_period)

        # Regime classification
        # Use hurst_exponent if available in bars, otherwise use trend_strength only
        hurst = bars.get("hurst_exponent", pd.Series(np.full(n, 0.5), index=bars.index))
        hurst_values = hurst.values

        regime_trending = np.zeros(n)
        regime_mean_reverting = np.zeros(n)
        regime_flat = np.zeros(n)
        regime_label = np.ones(n)  # default: flat (1)

        for i in range(n):
            h = hurst_values[i] if not np.isnan(hurst_values[i]) else 0.5
            ts = result["trend_strength"].iloc[i]

            if h > 0.55 and not np.isnan(ts) and ts > self._trend_threshold:
                regime_trending[i] = 1.0
                regime_label[i] = 2  # trending
            elif h < 0.45:
                regime_mean_reverting[i] = 1.0
                regime_label[i] = 0  # mean-reverting
            else:
                regime_flat[i] = 1.0
                regime_label[i] = 1  # flat

        result["regime_trending"] = regime_trending
        result["regime_mean_reverting"] = regime_mean_reverting
        result["regime_flat"] = regime_flat
        result["regime_label"] = regime_label

        return result

    @staticmethod
    def _compute_adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Compute Average Directional Index (ADX) as trend strength measure."""
        n = len(high)
        result = np.full(n, np.nan)

        if n < period * 2:
            return result

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            if up > down and up > 0:
                plus_dm[i] = up
            if down > up and down > 0:
                minus_dm[i] = down

        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        atr_s = np.zeros(n)
        plus_dm_s = np.zeros(n)
        minus_dm_s = np.zeros(n)

        atr_s[period] = np.sum(tr[1 : period + 1])
        plus_dm_s[period] = np.sum(plus_dm[1 : period + 1])
        minus_dm_s[period] = np.sum(minus_dm[1 : period + 1])

        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
            plus_dm_s[i] = plus_dm_s[i - 1] - plus_dm_s[i - 1] / period + plus_dm[i]
            minus_dm_s[i] = minus_dm_s[i - 1] - minus_dm_s[i - 1] / period + minus_dm[i]

        # Directional Indicators
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if atr_s[i] > 0:
                plus_di[i] = 100 * plus_dm_s[i] / atr_s[i]
                minus_di[i] = 100 * minus_dm_s[i] / atr_s[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX: smoothed DX
        adx_start = 2 * period
        if adx_start < n:
            result[adx_start] = np.mean(dx[period : adx_start + 1])
            for i in range(adx_start + 1, n):
                result[i] = (result[i - 1] * (period - 1) + dx[i]) / period

        return result
