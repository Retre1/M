"""Rolling Hurst exponent via Rescaled Range (R/S) analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor


class HurstExtractor(BaseFeatureExtractor):
    """
    Computes the Hurst exponent using R/S analysis.
    H > 0.5: trending (persistent) market
    H < 0.5: mean-reverting (anti-persistent) market
    H ≈ 0.5: random walk
    """

    def __init__(
        self,
        window: int = 252,
        min_lag: int = 2,
        max_lag: int = 20,
        trend_threshold: float = 0.55,
        reversion_threshold: float = 0.45,
    ) -> None:
        self._window = window
        self._min_lag = min_lag
        self._max_lag = max_lag
        self._trend_threshold = trend_threshold
        self._reversion_threshold = reversion_threshold

    @property
    def feature_names(self) -> list[str]:
        return ["hurst_exponent", "hurst_regime"]

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        n = len(bars)
        result = pd.DataFrame(index=bars.index)
        result["hurst_exponent"] = np.nan
        result["hurst_regime"] = np.nan

        prices = bars["close"].values
        log_returns = np.diff(np.log(prices))

        for i in range(self._window, n - 1):
            window_returns = log_returns[i - self._window : i]
            h = self._compute_hurst(window_returns)

            result.iloc[i + 1, result.columns.get_loc("hurst_exponent")] = h

            if h > self._trend_threshold:
                regime = 2  # trending
            elif h < self._reversion_threshold:
                regime = 0  # mean-reverting
            else:
                regime = 1  # random walk

            result.iloc[i + 1, result.columns.get_loc("hurst_regime")] = regime

        return result

    def _compute_hurst(self, series: np.ndarray) -> float:
        """Compute Hurst exponent via R/S analysis."""
        n = len(series)
        if n < self._max_lag * 2:
            return 0.5

        lags = range(self._min_lag, min(self._max_lag + 1, n // 2))
        rs_values = []
        lag_values = []

        for lag in lags:
            rs = self._rescaled_range(series, lag)
            if rs > 0:
                rs_values.append(np.log(rs))
                lag_values.append(np.log(lag))

        if len(rs_values) < 3:
            return 0.5

        # Linear regression: log(R/S) = H * log(lag) + c
        lag_arr = np.array(lag_values)
        rs_arr = np.array(rs_values)

        n_pts = len(lag_arr)
        sum_x = lag_arr.sum()
        sum_y = rs_arr.sum()
        sum_xy = (lag_arr * rs_arr).sum()
        sum_x2 = (lag_arr**2).sum()

        denom = n_pts * sum_x2 - sum_x**2
        if abs(denom) < 1e-10:
            return 0.5

        hurst = (n_pts * sum_xy - sum_x * sum_y) / denom
        return float(np.clip(hurst, 0.0, 1.0))

    @staticmethod
    def _rescaled_range(series: np.ndarray, lag: int) -> float:
        """Compute the rescaled range for a given lag."""
        n = len(series)
        n_chunks = n // lag

        if n_chunks == 0:
            return 0.0

        rs_sum = 0.0
        count = 0

        for i in range(n_chunks):
            chunk = series[i * lag : (i + 1) * lag]
            mean = chunk.mean()
            std = chunk.std(ddof=1)

            if std < 1e-10:
                continue

            cumdev = np.cumsum(chunk - mean)
            r = cumdev.max() - cumdev.min()
            rs_sum += r / std
            count += 1

        return rs_sum / count if count > 0 else 0.0
