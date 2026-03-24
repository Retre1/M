"""Multi-Timeframe Observation Builder.

Constructs the MTF observation dict from three timeframe slices.
Computes mtf_context features (cross-timeframe signals) that help
the agent understand inter-timeframe relationships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.data.mtf_aligner import MTFSlice
from apexfx.utils.time_utils import encode_time_features, get_session_id


class MTFObservationBuilder:
    """Builds the MTF observation dictionary for the Gymnasium environment.

    Observation keys:
        d1_market_features: (d1_lookback * n_market_features,) — D1 market data
        d1_time_features:   (d1_lookback * n_time_features,) — D1 time encoding
        h1_market_features: (h1_lookback * n_market_features,) — H1 market data
        h1_time_features:   (h1_lookback * n_time_features,) — H1 time encoding
        m5_market_features: (m5_lookback * n_market_features,) — M5 market data
        m5_time_features:   (m5_lookback * n_time_features,) — M5 time encoding
        trend_features:     (n_trend_features,) — from H1
        reversion_features: (n_reversion_features,) — from H1
        regime_features:    (n_regime_features,) — from H1
        position_state:     (4,) — position/PnL/time
        mtf_context:        (6,) — cross-timeframe signals
    """

    def __init__(
        self,
        n_market_features: int = 30,
        n_trend_features: int = 8,
        n_reversion_features: int = 8,
        n_regime_features: int = 6,
        n_time_features: int = 5,
        d1_lookback: int = 5,
        h1_lookback: int = 20,
        m5_lookback: int = 20,
    ) -> None:
        self.n_market_features = n_market_features
        self.n_trend_features = n_trend_features
        self.n_reversion_features = n_reversion_features
        self.n_regime_features = n_regime_features
        self.n_time_features = n_time_features
        self.d1_lookback = d1_lookback
        self.h1_lookback = h1_lookback
        self.m5_lookback = m5_lookback

        # Column mappings (same as ObservationBuilder for consistency)
        self.market_columns: list[str] = []
        self.trend_columns: list[str] = [
            "hurst_exponent", "trend_strength", "realized_vol",
            "wavelet_trend", "fft_dominant_period",
            "delta_ma_50", "regime_trending", "poc_distance",
        ]
        self.reversion_columns: list[str] = [
            "close_zscore", "hvn_distance", "volume_profile_skew",
            "delta_pct", "delta_divergence", "regime_mean_reverting",
            "nearest_support_distance", "nearest_resistance_distance",
        ]
        self.regime_columns: list[str] = [
            "hurst_exponent", "realized_vol", "trend_strength",
            "regime_trending", "regime_mean_reverting", "regime_flat",
        ]

    def build(
        self,
        mtf_slice: MTFSlice,
        h1_features: pd.DataFrame,
        h1_idx: int,
        position: float,
        unrealized_pnl: float,
        time_in_position: float,
        portfolio_value: float,
        initial_balance: float = 100_000.0,
    ) -> dict[str, np.ndarray]:
        """Build MTF observation from aligned timeframe slices.

        Parameters
        ----------
        mtf_slice : MTFSlice
            Aligned D1/H1/M5 data slices from MTFDataAligner.
        h1_features : pd.DataFrame
            Full H1 feature DataFrame (for trend/reversion/regime extraction).
        h1_idx : int
            Current H1 index.
        position, unrealized_pnl, time_in_position, portfolio_value : float
            Trading state for position_state vector.
        initial_balance : float
            Initial portfolio balance.

        Returns
        -------
        dict[str, np.ndarray]
            Observation dictionary matching the MTF observation space.
        """
        # Market features for each timeframe
        d1_market = self._extract_market_window(mtf_slice.d1, self.d1_lookback)
        h1_market = self._extract_market_window(mtf_slice.h1, self.h1_lookback)
        m5_market = self._extract_market_window(mtf_slice.m5, self.m5_lookback)

        # Time features for each timeframe
        d1_time = self._extract_time_window(mtf_slice.d1, self.d1_lookback)
        h1_time = self._extract_time_window(mtf_slice.h1, self.h1_lookback)
        m5_time = self._extract_time_window(mtf_slice.m5, self.m5_lookback)

        # Scalar features from H1 (latest bar)
        trend = self._extract_latest(h1_features, h1_idx, self.trend_columns, self.n_trend_features)
        reversion = self._extract_latest(h1_features, h1_idx, self.reversion_columns, self.n_reversion_features)
        regime = self._extract_latest(h1_features, h1_idx, self.regime_columns, self.n_regime_features)

        # Position state
        position_state = np.array([
            position,
            unrealized_pnl / (initial_balance + 1e-10),
            min(time_in_position / 100.0, 1.0),
            portfolio_value / (initial_balance + 1e-10) - 1.0,
        ], dtype=np.float32)

        # MTF context — cross-timeframe signals
        mtf_context = self._compute_mtf_context(mtf_slice)

        return {
            "d1_market_features": d1_market.flatten(),
            "d1_time_features": d1_time.flatten(),
            "h1_market_features": h1_market.flatten(),
            "h1_time_features": h1_time.flatten(),
            "m5_market_features": m5_market.flatten(),
            "m5_time_features": m5_time.flatten(),
            "trend_features": trend,
            "reversion_features": reversion,
            "regime_features": regime,
            "position_state": position_state,
            "mtf_context": mtf_context,
        }

    def _extract_market_window(
        self,
        df: pd.DataFrame,
        lookback: int,
    ) -> np.ndarray:
        """Extract and pad market features for a timeframe window."""
        # Identify market columns (exclude OHLCV and metadata)
        exclude = {"time", "open", "high", "low", "close", "volume",
                    "tick_count", "regime_label", "hurst_regime", "regime"}

        if not self.market_columns:
            self.market_columns = [c for c in df.columns if c not in exclude][
                : self.n_market_features
            ]

        # Use available columns, falling back to OHLCV if no features
        avail_cols = [c for c in self.market_columns if c in df.columns]
        if not avail_cols:
            # Fallback: use OHLCV as basic features
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            avail_cols = ohlcv_cols

        data = df[avail_cols].values.astype(np.float32) if len(avail_cols) > 0 else np.zeros((len(df), 1), dtype=np.float32)

        # Pad columns to n_market_features
        if data.shape[1] < self.n_market_features:
            padding = np.zeros(
                (data.shape[0], self.n_market_features - data.shape[1]),
                dtype=np.float32,
            )
            data = np.hstack([data, padding])
        else:
            data = data[:, : self.n_market_features]

        # Pad rows to lookback
        if len(data) < lookback:
            row_pad = np.zeros(
                (lookback - len(data), self.n_market_features),
                dtype=np.float32,
            )
            data = np.vstack([row_pad, data])

        # Replace NaN
        data = np.nan_to_num(data, nan=0.0, posinf=5.0, neginf=-5.0)

        return data

    def _extract_time_window(
        self,
        df: pd.DataFrame,
        lookback: int,
    ) -> np.ndarray:
        """Extract time features (sin/cos encoding) for a timeframe window."""
        result = np.zeros((lookback, self.n_time_features), dtype=np.float32)

        n_bars = min(len(df), lookback)
        start_row = lookback - n_bars

        for i in range(n_bars):
            row_idx = len(df) - n_bars + i
            if row_idx >= 0 and row_idx < len(df) and "time" in df.columns:
                dt = df.iloc[row_idx]["time"]
                try:
                    time_enc = encode_time_features(dt)
                    session = get_session_id(dt)
                    result[start_row + i, :4] = time_enc
                    result[start_row + i, 4] = session / 5.0
                except Exception:
                    pass

        return result

    def _extract_latest(
        self,
        features: pd.DataFrame,
        idx: int,
        columns: list[str],
        expected_size: int,
    ) -> np.ndarray:
        """Extract latest scalar features from a DataFrame."""
        result = np.zeros(expected_size, dtype=np.float32)
        if idx >= len(features):
            return result
        row = features.iloc[idx]
        for i, col in enumerate(columns[:expected_size]):
            if col in features.columns:
                val = row[col]
                result[i] = 0.0 if (isinstance(val, float) and np.isnan(val)) else float(val)
        return result

    def _compute_mtf_context(self, mtf_slice: MTFSlice) -> np.ndarray:
        """Compute cross-timeframe context features.

        Returns 6 features:
        [0] d1_trend      — D1 price direction (close[-1]/close[-2] - 1)
        [1] d1_volatility  — D1 ATR proxy (high-low range / close)
        [2] h1_d1_alignment — H1-D1 trend agreement (same sign = positive)
        [3] m5_momentum    — M5 short-term momentum
        [4] m5_reversal    — M5 reversal signal (RSI-like, oversold → positive)
        [5] coherence      — Cross-TF coherence score (all aligned = 1.0)
        """
        ctx = np.zeros(6, dtype=np.float32)

        # D1 trend
        d1_trend = self._compute_trend(mtf_slice.d1)
        ctx[0] = d1_trend

        # D1 volatility (ATR proxy)
        ctx[1] = self._compute_volatility(mtf_slice.d1)

        # H1 trend for alignment
        h1_trend = self._compute_trend(mtf_slice.h1)

        # H1-D1 alignment: +1 if same direction, -1 if opposite
        if abs(d1_trend) > 1e-6 and abs(h1_trend) > 1e-6:
            ctx[2] = np.sign(d1_trend) * np.sign(h1_trend)
        else:
            ctx[2] = 0.0

        # M5 momentum (short-term price change)
        m5_trend = self._compute_trend(mtf_slice.m5)
        ctx[3] = m5_trend

        # M5 reversal signal
        ctx[4] = self._compute_reversal(mtf_slice.m5)

        # Cross-TF coherence: all 3 agree → 1.0, disagree → 0.0
        signs = [
            np.sign(d1_trend) if abs(d1_trend) > 1e-6 else 0,
            np.sign(h1_trend) if abs(h1_trend) > 1e-6 else 0,
            np.sign(m5_trend) if abs(m5_trend) > 1e-6 else 0,
        ]
        nonzero = [s for s in signs if s != 0]
        if len(nonzero) >= 2:
            # Coherence: fraction of timeframes agreeing with majority
            majority = np.sign(sum(nonzero))
            ctx[5] = sum(1 for s in nonzero if s == majority) / len(nonzero)
        else:
            ctx[5] = 0.5  # neutral

        return ctx

    @staticmethod
    def _compute_trend(df: pd.DataFrame) -> float:
        """Simple trend measure: (last close - first close) / first close."""
        if len(df) < 2 or "close" not in df.columns:
            return 0.0
        closes = df["close"].values
        first = closes[0]
        last = closes[-1]
        if abs(first) < 1e-10:
            return 0.0
        return float(np.clip((last - first) / first, -0.1, 0.1))

    @staticmethod
    def _compute_volatility(df: pd.DataFrame) -> float:
        """ATR-like volatility: mean((high-low)/close)."""
        if len(df) < 1 or not all(c in df.columns for c in ("high", "low", "close")):
            return 0.0
        ranges = (df["high"].values - df["low"].values) / (df["close"].values + 1e-10)
        return float(np.clip(np.mean(ranges), 0, 0.1))

    @staticmethod
    def _compute_reversal(df: pd.DataFrame) -> float:
        """M5 reversal signal based on price z-score within the window.

        Positive = oversold (buy signal), negative = overbought (sell signal).
        """
        if len(df) < 5 or "close" not in df.columns:
            return 0.0
        closes = df["close"].values
        mean = np.mean(closes)
        std = np.std(closes)
        if std < 1e-10:
            return 0.0
        z = (closes[-1] - mean) / std
        # Invert: negative z = oversold → positive reversal signal
        return float(np.clip(-z / 3.0, -1.0, 1.0))
