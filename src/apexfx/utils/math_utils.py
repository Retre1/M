"""Shared mathematical utilities.

Vectorized where possible. Functions that require Wilder's-style
recursive smoothing retain a single scalar loop (unavoidable),
but all element-wise operations are vectorized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score — vectorized via pandas rolling."""
    s = pd.Series(data, dtype=np.float64)
    rolling_mean = s.rolling(window, min_periods=window).mean().shift(1)
    rolling_std = s.rolling(window, min_periods=window).std(ddof=1).shift(1)

    std_vals = np.maximum(rolling_std.values, 1e-10)
    mean_vals = rolling_mean.values

    result = np.full_like(data, np.nan, dtype=np.float64)
    valid = ~np.isnan(mean_vals)
    result[valid] = (data[valid] - mean_vals[valid]) / std_vals[valid]
    return result


def ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average — vectorized via pandas."""
    return pd.Series(data, dtype=np.float64).ewm(span=span, adjust=False).mean().values


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series."""
    return np.diff(np.log(prices))


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from price series."""
    return np.diff(prices) / prices[:-1]


def parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """Parkinson volatility estimator — vectorized via pandas rolling."""
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    parkinson_sq = factor * log_hl ** 2

    rolling_mean = pd.Series(parkinson_sq).rolling(window, min_periods=window).mean().values
    result = np.sqrt(rolling_mean * 252)
    return result


def garman_klass_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Garman-Klass volatility estimator — vectorized via pandas rolling."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

    rolling_mean = pd.Series(gk).rolling(window, min_periods=window).mean().values
    result = np.sqrt(rolling_mean * 252)
    return result


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range — vectorized element-wise, Wilder's recursive smoothing.

    The recursive smoothing (EMA-like) is inherently sequential,
    but all element-wise operations are vectorized.
    """
    # True Range: vectorized
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    tr = np.concatenate([[high[0] - low[0]], tr])

    # Wilder's smoothing (recursive — unavoidable)
    result = np.full(len(high), np.nan)
    result[period - 1] = tr[:period].mean()

    decay = (period - 1) / period
    for i in range(period, len(high)):
        result[i] = result[i - 1] * decay + tr[i] / period

    return result
