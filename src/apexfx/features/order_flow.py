"""Order flow imbalance (delta) computation from tick data."""

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
        n = len(bars)
        result = pd.DataFrame(index=bars.index)

        if ticks is not None and not ticks.empty:
            result = self._compute_from_ticks(bars, ticks)
        else:
            result = self._estimate_from_bars(bars)

        return result

    def _compute_from_ticks(self, bars: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
        """Compute exact delta from tick data."""
        result = pd.DataFrame(index=bars.index)

        # Classify ticks using tick rule
        prices = ticks["last"].values if "last" in ticks.columns else ticks["bid"].values
        volumes = ticks["volume"].values if "volume" in ticks.columns else np.ones(len(ticks))

        tick_direction = np.zeros(len(prices))
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                tick_direction[i] = 1  # Buy
            elif prices[i] < prices[i - 1]:
                tick_direction[i] = -1  # Sell
            else:
                tick_direction[i] = tick_direction[i - 1]  # Same as previous

        buy_volume = np.where(tick_direction > 0, volumes, 0)
        sell_volume = np.where(tick_direction < 0, volumes, 0)

        # Aggregate to bar level using time alignment
        tick_times = ticks["time"].values
        bar_times = bars["time"].values

        deltas = np.zeros(len(bars))
        total_volumes = np.zeros(len(bars))

        for i in range(len(bars) - 1):
            mask = (tick_times >= bar_times[i]) & (tick_times < bar_times[i + 1])
            deltas[i] = buy_volume[mask].sum() - sell_volume[mask].sum()
            total_volumes[i] = volumes[mask].sum()

        # Last bar
        mask = tick_times >= bar_times[-1]
        deltas[-1] = buy_volume[mask].sum() - sell_volume[mask].sum()
        total_volumes[-1] = volumes[mask].sum()

        result["delta"] = deltas
        result["delta_pct"] = np.where(total_volumes > 0, deltas / total_volumes, 0)
        result["cumulative_delta"] = np.cumsum(deltas)

        # Delta divergence: price makes new high but delta doesn't confirm
        result["delta_divergence"] = self._compute_divergence(
            bars["close"].values, result["cumulative_delta"].values
        )

        # Moving averages
        for w in self._windows:
            result[f"delta_ma_{w}"] = result["delta"].rolling(w, min_periods=1).mean()
            result[f"delta_pct_ma_{w}"] = result["delta_pct"].rolling(w, min_periods=1).mean()

        return result

    def _estimate_from_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Estimate delta from OHLCV bars when tick data is unavailable."""
        result = pd.DataFrame(index=bars.index)

        close = bars["close"].values
        open_ = bars["open"].values
        high = bars["high"].values
        low = bars["low"].values
        volume = bars["volume"].values

        # Estimate using close position within bar range
        bar_range = high - low
        close_position = np.where(
            bar_range > 0,
            (close - low) / bar_range * 2 - 1,  # -1 to +1
            0,
        )

        estimated_delta = close_position * volume
        total_volume = volume.copy()
        total_volume[total_volume == 0] = 1

        result["delta"] = estimated_delta
        result["delta_pct"] = estimated_delta / total_volume
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
        """Detect divergence between price and cumulative delta."""
        divergence = np.zeros(len(price))

        for i in range(lookback, len(price)):
            price_window = price[i - lookback : i + 1]
            delta_window = cum_delta[i - lookback : i + 1]

            # Bearish divergence: price new high, delta not confirming
            if price_window[-1] >= np.max(price_window[:-1]):
                if delta_window[-1] < np.max(delta_window[:-1]):
                    divergence[i] = -1

            # Bullish divergence: price new low, delta not confirming
            if price_window[-1] <= np.min(price_window[:-1]):
                if delta_window[-1] > np.min(delta_window[:-1]):
                    divergence[i] = 1

        return divergence
