"""Shared mathematical utilities."""

from __future__ import annotations

import numpy as np


def rolling_zscore(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score."""
    result = np.full_like(data, np.nan, dtype=np.float64)
    for i in range(window, len(data)):
        segment = data[i - window : i]
        mean = np.mean(segment)
        std = np.std(segment, ddof=1)
        if std > 1e-10:
            result[i] = (data[i] - mean) / std
        else:
            result[i] = 0.0
    return result


def ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(data, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series."""
    return np.diff(np.log(prices))


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from price series."""
    return np.diff(prices) / prices[:-1]


def parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """Parkinson volatility estimator (more efficient than close-to-close)."""
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    parkinson_sq = factor * log_hl**2

    result = np.full(len(high), np.nan)
    for i in range(window, len(high)):
        result[i] = np.sqrt(np.mean(parkinson_sq[i - window : i]) * 252)
    return result


def garman_klass_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Garman-Klass volatility estimator."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    result = np.full(len(high), np.nan)
    for i in range(window, len(high)):
        result[i] = np.sqrt(np.mean(gk[i - window : i]) * 252)
    return result


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    tr = np.concatenate([[high[0] - low[0]], tr])

    result = np.full(len(high), np.nan)
    result[period - 1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result
