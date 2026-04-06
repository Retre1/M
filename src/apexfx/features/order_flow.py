"""Order flow imbalance (delta) computation from tick data.

Vectorized: tick classification uses np.sign + ffill pattern,
divergence detection uses rolling max/min via pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor


class OrderFlowExtractor(BaseFeatureExtractor):
    """
    Computes order flow imbalance metrics from tick data.
    Uses the tick rule: uptick = buy, downtick = sell.
    """

    def __init__(self, delta_windows: list[int] | None = None) -> None:
        self._windows = delta_windows or [10, 50, 100]

    @property
    def feature_names(self) -> list[str]:
        names = ["delta", "delta_pct", "cumulative_delta", "delta_divergence"]
        for w in self._windows:
            names.extend([f"delta_ma_{w}", f"delta_pct_ma_{w}"])
        return names

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        if ticks is not None and not ticks.empty:
            return self._compute_from_ticks(bars, ticks)
        return self._estimate_from_bars(bars)

    def _compute_from_ticks(self, bars: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
        """Compute exact delta from tick data — vectorized tick classification."""
        result = pd.DataFrame(index=bars.index)

        prices = ticks["last"].values if "last" in ticks.columns else ticks["bid"].values
        volumes = ticks["volume"].values if "volume" in ticks.columns else np.ones(len(ticks))

        # Vectorized tick direction: sign of price changes, forward-filled
        price_diff = np.diff(prices, prepend=prices[0])
        raw_direction = np.sign(price_diff)
        # Forward-fill zeros (unchanged prices keep previous direction)
        tick_direction = pd.Series(raw_direction).replace(0, np.nan).ffill().fillna(0).values

        buy_volume = np.where(tick_direction > 0, volumes, 0.0)
        sell_volume = np.where(tick_direction < 0, volumes, 0.0)

        # Aggregate to bar level using searchsorted (vectorized bin assignment)
        tick_times = ticks["time"].values
        bar_times = bars["time"].values

        # Assign each tick to a bar via searchsorted
        bar_indices = np.searchsorted(bar_times, tick_times, side="right") - 1
        bar_indices = np.clip(bar_indices, 0, len(bars) - 1)

        deltas = np.zeros(len(bars))
        total_volumes = np.zeros(len(bars))
        np.add.at(deltas, bar_indices, buy_volume - sell_volume)
        np.add.at(total_volumes, bar_indices, volumes)

        result["delta"] = deltas
        result["delta_pct"] = np.where(total_volumes > 0, deltas / total_volumes, 0.0)
        result["cumulative_delta"] = np.cumsum(deltas)
        result["delta_divergence"] = self._compute_divergence(
            bars["close"].values, result["cumulative_delta"].values
        )

        for w in self._windows:
            result[f"delta_ma_{w}"] = result["delta"].rolling(w, min_periods=1).mean()
            result[f"delta_pct_ma_{w}"] = result["delta_pct"].rolling(w, min_periods=1).mean()

        return result

    def _estimate_from_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Estimate delta from OHLCV bars when tick data is unavailable."""
        result = pd.DataFrame(index=bars.index)

        close = bars["close"].values.astype(np.float64)
        high = bars["high"].values.astype(np.float64)
        low = bars["low"].values.astype(np.float64)
        volume = bars["volume"].values.astype(np.float64)

        bar_range = high - low
        safe_range = np.where(bar_range > 0, bar_range, 1.0)
        close_position = np.where(
            bar_range > 0,
            (close - low) / safe_range * 2 - 1,
            0.0,
        )

        estimated_delta = close_position * volume
        safe_volume = np.where(volume == 0, 1.0, volume)

        result["delta"] = estimated_delta
        result["delta_pct"] = estimated_delta / safe_volume
        result["cumulative_delta"] = np.cumsum(estimated_delta)
        result["delta_divergence"] = self._compute_divergence(
            close, result["cumulative_delta"].values
        )

        for w in self._windows:
            result[f"delta_ma_{w}"] = result["delta"].rolling(w, min_periods=1).mean()
            result[f"delta_pct_ma_{w}"] = result["delta_pct"].rolling(w, min_periods=1).mean()

        return result

    @staticmethod
    def _compute_divergence(
        price: np.ndarray, cum_delta: np.ndarray, lookback: int = 20
    ) -> np.ndarray:
        """Detect divergence between price and cumulative delta — vectorized."""
        n = len(price)
        divergence = np.zeros(n)

        if n <= lookback:
            return divergence

        # Rolling max/min of price and delta (excluding current bar)
        price_s = pd.Series(price)
        delta_s = pd.Series(cum_delta)

        # Rolling max/min over [i-lookback : i] (shifted to exclude current)
        roll_price_max = price_s.rolling(lookback, min_periods=1).max().shift(1).values
        roll_price_min = price_s.rolling(lookback, min_periods=1).min().shift(1).values
        roll_delta_max = delta_s.rolling(lookback, min_periods=1).max().shift(1).values
        roll_delta_min = delta_s.rolling(lookback, min_periods=1).min().shift(1).values

        # Bearish divergence: price new high but delta doesn't confirm
        bearish = (price >= roll_price_max) & (cum_delta < roll_delta_max)
        divergence[bearish & (np.arange(n) >= lookback)] = -1.0

        # Bullish divergence: price new low but delta doesn't confirm
        bullish = (price <= roll_price_min) & (cum_delta > roll_delta_min)
        divergence[bullish & (np.arange(n) >= lookback)] = 1.0

        return divergence
