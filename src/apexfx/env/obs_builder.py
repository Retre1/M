"""Observation space construction for the Gymnasium environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from apexfx.utils.time_utils import encode_time_features, get_session_id

if TYPE_CHECKING:
    import pandas as pd


class ObservationBuilder:
    """
    Constructs the observation dict from current market state.
    Handles lookback window slicing, normalization, and time encoding.
    """

    def __init__(
        self,
        n_market_features: int = 30,
        n_trend_features: int = 8,
        n_reversion_features: int = 8,
        n_regime_features: int = 6,
        n_time_features: int = 5,
        lookback: int = 100,
    ) -> None:
        self.n_market_features = n_market_features
        self.n_trend_features = n_trend_features
        self.n_reversion_features = n_reversion_features
        self.n_regime_features = n_regime_features
        self.n_time_features = n_time_features
        self.lookback = lookback

        # Feature column mappings
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
        features: pd.DataFrame,
        current_idx: int,
        position: float,
        unrealized_pnl: float,
        time_in_position: float,
        portfolio_value: float,
        initial_balance: float = 100_000,
    ) -> dict[str, np.ndarray]:
        """
        Build observation dict from feature DataFrame and current state.

        Returns dict with keys matching the gymnasium spaces.Dict.
        """
        start_idx = max(0, current_idx - self.lookback + 1)
        window = features.iloc[start_idx : current_idx + 1]

        # Market features: full lookback window
        market_cols = [c for c in features.columns if c not in [
            "time", "open", "high", "low", "close", "volume", "tick_count",
            "regime_label", "hurst_regime",
        ]]
        if not self.market_columns:
            self.market_columns = market_cols[:self.n_market_features]

        market_data = window[self.market_columns[:self.n_market_features]].values.astype(np.float32)

        # Pad if not enough history
        if len(market_data) < self.lookback:
            padding = np.zeros(
                (self.lookback - len(market_data), market_data.shape[1]),
                dtype=np.float32,
            )
            market_data = np.vstack([padding, market_data])

        # Replace NaN with 0
        market_data = np.nan_to_num(market_data, nan=0.0, posinf=5.0, neginf=-5.0)

        # Trend features: latest values
        trend_data = self._extract_latest(features, current_idx, self.trend_columns,
                                          self.n_trend_features)

        # Reversion features: latest values
        reversion_data = self._extract_latest(features, current_idx, self.reversion_columns,
                                              self.n_reversion_features)

        # Regime features: latest values
        regime_data = self._extract_latest(features, current_idx, self.regime_columns,
                                           self.n_regime_features)

        # Time features: sinusoidal encoding for each step in lookback
        time_data = np.zeros((self.lookback, self.n_time_features), dtype=np.float32)
        for i, idx in enumerate(range(start_idx, current_idx + 1)):
            if idx < len(features) and "time" in features.columns:
                dt = features.iloc[idx]["time"]
                time_enc = encode_time_features(dt)
                session = get_session_id(dt)
                time_data[self.lookback - len(window) + i, :4] = time_enc
                time_data[self.lookback - len(window) + i, 4] = session / 5.0  # normalize

        # Position state
        position_state = np.array([
            position,
            unrealized_pnl / (initial_balance + 1e-10),
            min(time_in_position / 100.0, 1.0),  # normalize
            portfolio_value / (initial_balance + 1e-10) - 1.0,  # relative to start
        ], dtype=np.float32)

        return {
            "market_features": market_data.flatten(),
            "time_features": time_data.flatten(),
            "trend_features": trend_data,
            "reversion_features": reversion_data,
            "regime_features": regime_data,
            "position_state": position_state,
        }

    def _extract_latest(
        self,
        features: pd.DataFrame,
        idx: int,
        columns: list[str],
        expected_size: int,
    ) -> np.ndarray:
        """Extract the latest values for given columns, padding if needed."""
        result = np.zeros(expected_size, dtype=np.float32)
        row = features.iloc[idx]
        for i, col in enumerate(columns[:expected_size]):
            if col in features.columns:
                val = row[col]
                result[i] = 0.0 if (isinstance(val, float) and np.isnan(val)) else float(val)
        return result
