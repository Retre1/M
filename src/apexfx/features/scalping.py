"""Scalping feature extractor for M1 medium-frequency trading.

Computes 10 features tailored for scalping on the M1 timeframe:
micro-momentum, tick velocity, ATR, session scoring, volume surge,
bid-ask pressure, and mean-reversion z-score.

Works with both time bars and volume bars (requires 'close' column at minimum).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ScalpingExtractor:
    """Extract 10 scalping features from M1 bar data.

    Features (10 dims):
        0. normalized_spread      -- current spread / ATR (low = tight)
        1. micro_momentum_5       -- 5-bar return (close[0] - close[-5]) / ATR
        2. micro_momentum_10      -- 10-bar return / ATR
        3. micro_momentum_20      -- 20-bar return / ATR
        4. tick_velocity           -- bars per minute (volume of bars in lookback)
        5. m1_atr_norm             -- 14-bar ATR on M1, normalized by close price
        6. time_of_day_score       -- 1.0 London/NY overlap, 0.5 London/NY solo, 0.2 Asian
        7. volume_surge            -- current volume / average volume ratio
        8. bid_ask_pressure        -- estimated from (close - low) / (high - low) * 2 - 1
        9. mean_reversion_signal   -- z-score of close on M1 (rolling 20-bar)

    Parameters
    ----------
    lookback : int
        Number of bars for rolling calculations (default 20).
    """

    # Session hours in UTC
    _LONDON_START = 7
    _LONDON_END = 16
    _NY_START = 12
    _NY_END = 21
    _OVERLAP_START = 12
    _OVERLAP_END = 16
    _TOKYO_START = 0
    _TOKYO_END = 9

    _FEATURE_NAMES = [
        "normalized_spread",
        "micro_momentum_5",
        "micro_momentum_10",
        "micro_momentum_20",
        "tick_velocity",
        "m1_atr_norm",
        "time_of_day_score",
        "volume_surge",
        "bid_ask_pressure",
        "mean_reversion_signal",
    ]

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    @property
    def feature_names(self) -> list[str]:
        """Return list of 10 feature names."""
        return list(self._FEATURE_NAMES)

    def extract(
        self,
        bars: pd.DataFrame,
        current_spread: float | None = None,
    ) -> dict[str, float]:
        """Extract scalping features from the most recent bars.

        Parameters
        ----------
        bars : pd.DataFrame
            M1 OHLCV bars. Must have columns: 'close'. Optional: 'high', 'low',
            'volume', 'tick_volume'. Index should be DatetimeIndex for
            time-of-day scoring. Needs at least ``lookback`` rows.
        current_spread : float or None
            Current bid-ask spread in price units. If None, estimated from
            high-low range of the last bar.

        Returns
        -------
        dict[str, float]
            Dictionary with 10 feature values.
        """
        n = len(bars)
        close = bars["close"].values

        # --- ATR (14-bar True Range on M1) ---
        atr = self._compute_atr(bars, period=14)
        atr = max(atr, 1e-10)  # avoid division by zero

        # --- Normalized spread ---
        if current_spread is not None:
            normalized_spread = current_spread / atr
        elif "high" in bars.columns and "low" in bars.columns:
            # Estimate spread from last bar's high-low range
            last_range = float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
            normalized_spread = last_range / atr
        else:
            normalized_spread = 0.5  # neutral default

        # --- Micro-momentum (returns normalized by ATR) ---
        micro_momentum_5 = self._momentum(close, 5, atr)
        micro_momentum_10 = self._momentum(close, 10, atr)
        micro_momentum_20 = self._momentum(close, 20, atr)

        # --- Tick velocity (bars per minute) ---
        tick_velocity = self._tick_velocity(bars)

        # --- M1 ATR normalized by close price ---
        m1_atr_norm = atr / max(float(close[-1]), 1e-10)

        # --- Time of day score ---
        time_of_day_score = self._time_of_day_score(bars)

        # --- Volume surge ---
        volume_surge = self._volume_surge(bars)

        # --- Bid-ask pressure ---
        bid_ask_pressure = self._bid_ask_pressure(bars)

        # --- Mean reversion signal (z-score) ---
        mean_reversion_signal = self._mean_reversion_zscore(close)

        return {
            "normalized_spread": float(np.clip(normalized_spread, 0.0, 10.0)),
            "micro_momentum_5": float(np.clip(micro_momentum_5, -5.0, 5.0)),
            "micro_momentum_10": float(np.clip(micro_momentum_10, -5.0, 5.0)),
            "micro_momentum_20": float(np.clip(micro_momentum_20, -5.0, 5.0)),
            "tick_velocity": float(np.clip(tick_velocity, 0.0, 10.0)),
            "m1_atr_norm": float(np.clip(m1_atr_norm, 0.0, 0.1)),
            "time_of_day_score": float(time_of_day_score),
            "volume_surge": float(np.clip(volume_surge, 0.0, 10.0)),
            "bid_ask_pressure": float(np.clip(bid_ask_pressure, -1.0, 1.0)),
            "mean_reversion_signal": float(np.clip(mean_reversion_signal, -5.0, 5.0)),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
        """Compute Average True Range over the last ``period`` bars."""
        if len(bars) < 2:
            if "high" in bars.columns and "low" in bars.columns:
                return float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
            return 1e-10

        high = bars["high"].values if "high" in bars.columns else bars["close"].values
        low = bars["low"].values if "low" in bars.columns else bars["close"].values
        close = bars["close"].values

        n = len(bars)
        start = max(1, n - period)
        true_ranges = []
        for i in range(start, n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            true_ranges.append(tr)

        return float(np.mean(true_ranges)) if true_ranges else 1e-10

    @staticmethod
    def _momentum(close: np.ndarray, period: int, atr: float) -> float:
        """Compute return over ``period`` bars, normalized by ATR."""
        if len(close) <= period:
            return 0.0
        return float((close[-1] - close[-1 - period]) / atr)

    @staticmethod
    def _tick_velocity(bars: pd.DataFrame) -> float:
        """Estimate bars per minute from the time index."""
        if not isinstance(bars.index, pd.DatetimeIndex) or len(bars) < 2:
            return 1.0  # default: 1 bar per minute for M1

        time_span = (bars.index[-1] - bars.index[0]).total_seconds()
        if time_span <= 0:
            return 1.0
        minutes = time_span / 60.0
        return float(len(bars) / minutes)

    @classmethod
    def _time_of_day_score(cls, bars: pd.DataFrame) -> float:
        """Score based on current trading session.

        Returns:
            1.0 during London/NY overlap (best liquidity)
            0.5 during London or NY solo sessions
            0.2 during Asian/Tokyo session
            0.1 outside major sessions
        """
        if not isinstance(bars.index, pd.DatetimeIndex):
            return 0.5  # neutral default

        hour = bars.index[-1].hour
        return cls._score_hour(hour)

    @classmethod
    def _score_hour(cls, hour: int) -> float:
        """Score a UTC hour for scalping suitability."""
        if cls._OVERLAP_START <= hour < cls._OVERLAP_END:
            return 1.0
        if cls._LONDON_START <= hour < cls._LONDON_END:
            return 0.5
        if cls._NY_START <= hour < cls._NY_END:
            return 0.5
        if cls._TOKYO_START <= hour < cls._TOKYO_END:
            return 0.2
        return 0.1

    @staticmethod
    def _volume_surge(bars: pd.DataFrame) -> float:
        """Current volume / average volume ratio."""
        vol_col = None
        for col in ("volume", "tick_volume", "real_volume"):
            if col in bars.columns:
                vol_col = col
                break

        if vol_col is None:
            return 1.0  # neutral if no volume data

        volumes = bars[vol_col].values
        if len(volumes) < 2:
            return 1.0

        avg_vol = float(np.mean(volumes[:-1]))
        if avg_vol <= 0:
            return 1.0
        return float(volumes[-1] / avg_vol)

    @staticmethod
    def _bid_ask_pressure(bars: pd.DataFrame) -> float:
        """Estimate buying/selling pressure from candle position.

        Formula: (close - low) / (high - low) * 2 - 1
        Result in [-1, 1]: +1 = close at high (buyers), -1 = close at low (sellers).
        """
        if "high" not in bars.columns or "low" not in bars.columns:
            return 0.0

        h = float(bars["high"].iloc[-1])
        l = float(bars["low"].iloc[-1])
        c = float(bars["close"].iloc[-1])

        hl_range = h - l
        if hl_range <= 0:
            return 0.0

        return float((c - l) / hl_range * 2.0 - 1.0)

    def _mean_reversion_zscore(self, close: np.ndarray) -> float:
        """Z-score of the latest close relative to rolling mean/std."""
        period = min(self.lookback, len(close))
        if period < 3:
            return 0.0

        window = close[-period:]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std <= 0:
            return 0.0
        return float((close[-1] - mean) / std)
