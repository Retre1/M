"""Shared test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Wednesday 12:00 UTC — mid-week, inside the London/NY overlap. Chosen so that
# neither WeekendGapGuard (blocks Friday from 20:00 UTC and the weekend) nor
# is_forex_market_open() rejects the trade.
TRADING_CLOCK = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


class _FrozenDatetime(datetime):
    """``datetime`` subclass whose ``now()`` is pinned to ``TRADING_CLOCK``.

    Subclassing keeps every other constructor and method intact, so patched
    modules can still build and compare datetimes normally.
    """

    @classmethod
    def now(cls, tz=None):  # noqa: D102 - mirrors datetime.now
        return TRADING_CLOCK if tz is not None else TRADING_CLOCK.replace(tzinfo=None)


@pytest.fixture
def frozen_trading_clock():
    """Pin the wall clock in the modules that gate trading on the current time.

    ``WeekendGapGuard`` and ``LiquidityGuard`` read ``datetime.now(UTC)`` and
    correctly refuse to trade outside market hours. Without this fixture any
    test that goes through them passes Monday to Thursday and fails every
    Friday evening and weekend — the suite's result depends on the day it runs.

    Production behaviour is intentional and left untouched; only the clock the
    tests observe is fixed.
    """
    with (
        patch("apexfx.risk.risk_manager.datetime", _FrozenDatetime),
        patch("apexfx.execution.liquidity_guard.datetime", _FrozenDatetime),
    ):
        yield TRADING_CLOCK


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
