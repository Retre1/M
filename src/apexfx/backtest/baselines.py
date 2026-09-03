"""Baseline strategies the model has to beat before it can claim an edge.

The April audit ran these ad hoc and found the result that governs everything
else: over Feb 2024 - Feb 2026 on EURUSD H4, buy & hold returned **+9.97% at
Sharpe 0.65** while every trend-following variant lost money, and the trained
model returned -0.18% at Sharpe -0.24. The window was a clean uptrend, so any
strategy that shorts pays for it.

Keeping the baselines in the package rather than in a scratch script is the
point: a number is only meaningful next to the alternative it beat. They
implement the same ``Strategy`` protocol as the model wrapper, so they run
through the identical engine, cost model and risk layer — a comparison against
a baseline scored under different assumptions would prove nothing.

Random is included deliberately. It is not a contender; it calibrates the cost
model. The audit measured it at -30.2%, which is the shape of a strategy paying
spread on every bar with no signal. A random baseline that comes out flat means
the costs are not being charged.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


class BuyAndHold:
    """Go long on the first bar and never trade again.

    The benchmark that matters. A system that cannot beat it is not earning its
    complexity, whatever its Sharpe looks like in isolation.
    """

    name = "buy_and_hold"

    def __init__(self, size: float = 1.0) -> None:
        self._size = size

    def reset(self) -> None:
        pass

    def on_bar(self, features: pd.Series, bar: pd.Series) -> float:  # noqa: ARG002
        return self._size


class MACross:
    """Long above the moving-average cross, short below."""

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = False) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
        self.name = f"ma_cross_{fast}_{slow}" + ("_long" if long_only else "")
        self._fast = fast
        self._slow = slow
        self._long_only = long_only
        self._closes: deque[float] = deque(maxlen=slow)

    def reset(self) -> None:
        self._closes.clear()

    def on_bar(self, features: pd.Series, bar: pd.Series) -> float:  # noqa: ARG002
        self._closes.append(float(bar["close"]))
        if len(self._closes) < self._slow:
            return 0.0

        closes = np.asarray(self._closes, dtype=np.float64)
        fast = closes[-self._fast:].mean()
        slow = closes.mean()

        if fast > slow:
            return 1.0
        return 0.0 if self._long_only else -1.0


class DonchianBreakout:
    """Long on a break of the N-bar high, short on a break of the N-bar low.

    ``long_only`` exists because it was the only baseline in the audit's window
    that finished positive (+1.58%): in an uptrend the short side is what does
    the damage.
    """

    def __init__(self, lookback: int = 20, long_only: bool = False) -> None:
        if lookback < 2:
            raise ValueError(f"lookback must be at least 2, got {lookback}")
        self.name = f"donchian_{lookback}" + ("_long" if long_only else "")
        self._lookback = lookback
        self._long_only = long_only
        self._highs: deque[float] = deque(maxlen=lookback)
        self._lows: deque[float] = deque(maxlen=lookback)
        self._position = 0.0

    def reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self._position = 0.0

    def on_bar(self, features: pd.Series, bar: pd.Series) -> float:  # noqa: ARG002
        close = float(bar["close"])

        # Compare against the channel formed *before* this bar, then extend it.
        # Including the current bar would let the strategy trade on a high it
        # has only just learned about — a lookahead that flatters the result.
        if len(self._highs) == self._lookback:
            if close > max(self._highs):
                self._position = 1.0
            elif close < min(self._lows):
                self._position = 0.0 if self._long_only else -1.0

        self._highs.append(float(bar["high"]))
        self._lows.append(float(bar["low"]))
        return self._position


class RandomStrategy:
    """Uniform random action — a calibration probe, not a contender.

    Seeded, so a comparison run is reproducible.
    """

    name = "random"

    def __init__(self, seed: int = 42, flat_prob: float = 0.0) -> None:
        self._seed = seed
        self._flat_prob = flat_prob
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def on_bar(self, features: pd.Series, bar: pd.Series) -> float:  # noqa: ARG002
        if self._flat_prob and self._rng.random() < self._flat_prob:
            return 0.0
        return float(self._rng.choice([-1.0, 1.0]))


def default_baselines() -> list:
    """The comparison set from the audit's reality check, in that order."""
    return [
        BuyAndHold(),
        MACross(fast=20, slow=50),
        DonchianBreakout(lookback=20),
        DonchianBreakout(lookback=55),
        DonchianBreakout(lookback=20, long_only=True),
        RandomStrategy(seed=42),
    ]
