"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Generate sample OHLCV bar data for testing."""
    np.random.seed(42)
    n = 500
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.001)

    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC"),
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.0005),
        "low": prices - np.abs(np.random.randn(n) * 0.0005),
        "close": prices + np.random.randn(n) * 0.0002,
        "volume": np.random.lognormal(10, 1, n),
        "tick_count": np.random.randint(50, 500, n),
    })


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample return series."""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 + 0.0003
