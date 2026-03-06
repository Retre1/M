"""Structure break and key level detection features.

Implements the professional trader's approach:
- Swing high/low identification via fractal method
- Structure break detection (Break of Structure — BOS)
- Trend structure scoring (HH+HL vs LH+LL)
- Level confluence scoring
- Breakout volume confirmation
- Retest detection (price returns to broken level)

These features let the RL agent LEARN when structure breaks
confirm a fundamental bias — the core of the described strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class StructureExtractor(BaseFeatureExtractor):
    """Extracts 8 structure/level features for each bar.

    Features:
        swing_high_distance: Distance to last swing high / ATR
        swing_low_distance: Distance to last swing low / ATR
        structure_break_bull: 1.0 if price broke above previous swing high (BOS)
        structure_break_bear: 1.0 if price broke below previous swing low (BOS)
        structure_trend: +1 HH+HL, -1 LH+LL, 0 mixed
        level_confluence: Count of S/R levels within 1 ATR
        breakout_strength: Volume at bar / avg volume (breakout confirmation)
        retest_signal: 1.0 if price retested broken level from opposite side
    """

    def __init__(
        self,
        swing_period: int = 5,
        atr_period: int = 14,
        confluence_atr_mult: float = 1.0,
        volume_avg_window: int = 20,
        retest_atr_mult: float = 0.5,
        max_swing_memory: int = 10,
    ) -> None:
        self._swing_period = swing_period
        self._atr_period = atr_period
        self._confluence_atr_mult = confluence_atr_mult
        self._volume_avg_window = volume_avg_window
        self._retest_atr_mult = retest_atr_mult
        self._max_swing_memory = max_swing_memory

    @property
    def feature_names(self) -> list[str]:
        return [
            "swing_high_distance",
            "swing_low_distance",
            "structure_break_bull",
            "structure_break_bear",
            "structure_trend",
            "level_confluence",
            "breakout_strength",
            "retest_signal",
        ]

    def extract(
        self,
        bars: pd.DataFrame,
        ticks: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        n = len(bars)
        result = pd.DataFrame(index=bars.index)
        for col in self.feature_names:
            result[col] = 0.0

        if n < self._swing_period * 2 + 1:
            return result

        high = bars["high"].values.astype(float)
        low = bars["low"].values.astype(float)
        close = bars["close"].values.astype(float)
        volume = bars["volume"].values.astype(float) if "volume" in bars.columns else np.ones(n)

        # Precompute ATR
        atr_values = self._compute_atr(high, low, close, self._atr_period)

        # Precompute volume moving average
        vol_ma = self._rolling_mean(volume, self._volume_avg_window)

        # Detect all swing highs and lows
        swing_highs = self._detect_swing_highs(high, self._swing_period)
        swing_lows = self._detect_swing_lows(low, self._swing_period)

        # Running lists of recent swing levels
        recent_highs: list[tuple[int, float]] = []  # (bar_idx, price)
        recent_lows: list[tuple[int, float]] = []
        broken_levels: list[tuple[float, int, str]] = []  # (price, break_idx, "bull"/"bear")

        min_start = self._swing_period + self._atr_period
        for i in range(min_start, n):
            current_atr = atr_values[i]
            if np.isnan(current_atr) or current_atr <= 0:
                continue

            current_price = close[i]
            current_high = high[i]
            current_low = low[i]

            # Update swing lists (only add confirmed swings, i.e. from swing_period bars ago)
            check_idx = i - self._swing_period
            if check_idx >= 0:
                if swing_highs[check_idx]:
                    recent_highs.append((check_idx, high[check_idx]))
                    if len(recent_highs) > self._max_swing_memory:
                        recent_highs.pop(0)
                if swing_lows[check_idx]:
                    recent_lows.append((check_idx, low[check_idx]))
                    if len(recent_lows) > self._max_swing_memory:
                        recent_lows.pop(0)

            # 1. Swing high distance (normalized by ATR)
            if recent_highs:
                last_sh = recent_highs[-1][1]
                dist = (current_price - last_sh) / current_atr
                result.iat[i, 0] = float(np.clip(dist, -10.0, 10.0))

            # 2. Swing low distance (normalized by ATR)
            if recent_lows:
                last_sl = recent_lows[-1][1]
                dist = (current_price - last_sl) / current_atr
                result.iat[i, 1] = float(np.clip(dist, -10.0, 10.0))

            # 3 & 4. Structure break detection
            bull_break = 0.0
            bear_break = 0.0

            if recent_highs:
                last_sh_price = recent_highs[-1][1]
                # Bullish BOS: current high breaks above the last swing high
                if current_high > last_sh_price and (i == min_start or high[i - 1] <= last_sh_price):
                    bull_break = 1.0
                    broken_levels.append((last_sh_price, i, "bull"))

            if recent_lows:
                last_sl_price = recent_lows[-1][1]
                # Bearish BOS: current low breaks below the last swing low
                if current_low < last_sl_price and (i == min_start or low[i - 1] >= last_sl_price):
                    bear_break = 1.0
                    broken_levels.append((last_sl_price, i, "bear"))

            result.iat[i, 2] = bull_break
            result.iat[i, 3] = bear_break

            # 5. Structure trend: HH+HL = bullish, LH+LL = bearish
            trend = self._compute_structure_trend(recent_highs, recent_lows)
            result.iat[i, 4] = trend

            # 6. Level confluence: how many S/R levels within confluence_atr_mult * ATR
            confluence = self._compute_confluence(
                current_price, recent_highs, recent_lows, current_atr,
            )
            result.iat[i, 5] = confluence

            # 7. Breakout strength: volume / avg volume
            if vol_ma[i] > 0:
                strength = volume[i] / vol_ma[i]
                result.iat[i, 6] = float(np.clip(strength, 0.0, 5.0))

            # 8. Retest signal
            retest = self._detect_retest(
                current_price, current_atr, i, broken_levels,
            )
            result.iat[i, 7] = retest

            # Cleanup old broken levels
            broken_levels = [
                (p, idx, d) for p, idx, d in broken_levels
                if i - idx < 100
            ]

        return result

    @staticmethod
    def _detect_swing_highs(high: np.ndarray, period: int) -> np.ndarray:
        """Detect fractal swing highs: bar is highest in [i-period, i+period]."""
        n = len(high)
        is_swing = np.zeros(n, dtype=bool)
        for i in range(period, n - period):
            left = high[i - period:i]
            right = high[i + 1:i + period + 1]
            if high[i] > left.max() and high[i] > right.max():
                is_swing[i] = True
        return is_swing

    @staticmethod
    def _detect_swing_lows(low: np.ndarray, period: int) -> np.ndarray:
        """Detect fractal swing lows: bar is lowest in [i-period, i+period]."""
        n = len(low)
        is_swing = np.zeros(n, dtype=bool)
        for i in range(period, n - period):
            left = low[i - period:i]
            right = low[i + 1:i + period + 1]
            if low[i] < left.min() and low[i] < right.min():
                is_swing[i] = True
        return is_swing

    @staticmethod
    def _compute_structure_trend(
        recent_highs: list[tuple[int, float]],
        recent_lows: list[tuple[int, float]],
    ) -> float:
        """Determine trend structure from swing sequence.

        +1.0: Higher Highs + Higher Lows (bullish structure)
        -1.0: Lower Highs + Lower Lows (bearish structure)
         0.0: Mixed or insufficient data
        """
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return 0.0

        hh = recent_highs[-1][1] > recent_highs[-2][1]  # higher high
        hl = recent_lows[-1][1] > recent_lows[-2][1]     # higher low
        lh = recent_highs[-1][1] < recent_highs[-2][1]   # lower high
        ll = recent_lows[-1][1] < recent_lows[-2][1]     # lower low

        if hh and hl:
            return 1.0  # bullish structure
        elif lh and ll:
            return -1.0  # bearish structure
        return 0.0  # mixed

    def _compute_confluence(
        self,
        current_price: float,
        recent_highs: list[tuple[int, float]],
        recent_lows: list[tuple[int, float]],
        current_atr: float,
    ) -> float:
        """Count how many S/R levels are within confluence_atr_mult * ATR."""
        threshold = current_atr * self._confluence_atr_mult
        count = 0

        levels = [p for _, p in recent_highs] + [p for _, p in recent_lows]
        for level in levels:
            if abs(current_price - level) <= threshold:
                count += 1

        return float(min(count, 10))  # cap at 10

    def _detect_retest(
        self,
        current_price: float,
        current_atr: float,
        current_idx: int,
        broken_levels: list[tuple[float, int, str]],
    ) -> float:
        """Detect if price is retesting a recently broken level.

        After a bullish break, a retest means price comes back down
        to the broken resistance (now support) from above.
        """
        threshold = current_atr * self._retest_atr_mult

        for level_price, break_idx, direction in reversed(broken_levels):
            bars_since = current_idx - break_idx
            if bars_since < 2 or bars_since > 50:
                continue

            distance = abs(current_price - level_price)
            if distance <= threshold:
                # Bullish break retest: price came back down to level
                if direction == "bull" and current_price >= level_price:
                    return 1.0
                # Bearish break retest: price came back up to level
                if direction == "bear" and current_price <= level_price:
                    return 1.0

        return 0.0

    @staticmethod
    def _compute_atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int,
    ) -> np.ndarray:
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr_vals = np.full(n, np.nan)
        if n >= period:
            atr_vals[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period
        return atr_vals

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        result = np.full(n, np.nan)
        if n >= window:
            cumsum = np.cumsum(values)
            result[window - 1] = cumsum[window - 1] / window
            for i in range(window, n):
                result[i] = (cumsum[i] - cumsum[i - window]) / window
        return result
