"""L2 Order Book microstructure features.

Extracts institutional flow signals from order book data:
- Bid/ask imbalance reveals buying/selling pressure
- VPIN (Volume-synchronized PIN) measures flow toxicity
- Depth ratio shows support/resistance strength
- Order flow imbalance tracks cumulative delta

For backtesting (no real L2 data): estimates features from OHLCV using
tick-rule and volume-clock approximations.
For live trading: uses real L2 snapshots from MT5 DOM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class OrderBookExtractor(BaseFeatureExtractor):
    """L2 order book microstructure features.

    8 features capturing market microstructure signals that reveal
    institutional flow, liquidity conditions, and trading pressure.
    """

    def __init__(
        self,
        depth_levels: int = 5,
        vpin_window: int = 50,
        flow_window: int = 20,
    ) -> None:
        self._depth_levels = depth_levels
        self._vpin_window = vpin_window
        self._flow_window = flow_window
        self._live_book: dict | None = None

    @property
    def feature_names(self) -> list[str]:
        return [
            "bid_ask_imbalance",     # (bid_vol - ask_vol) / (bid_vol + ask_vol)
            "weighted_mid_price",    # Volume-weighted mid price deviation
            "depth_ratio",           # Top N bid depth / Top N ask depth
            "spread_bps",            # Current spread in basis points
            "order_flow_imbalance",  # Cumulative delta (buy vol - sell vol)
            "volume_at_best",        # Volume at best bid+ask (normalized)
            "book_pressure",         # Normalized pressure score [-1, 1]
            "trade_flow_toxicity",   # VPIN-like flow toxicity measure
        ]

    def extract(self, data: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        """Extract from OHLCV data (synthetic L2 estimation for backtesting).

        Uses established microstructure proxies:
        - Tick rule: classify trades as buyer/seller initiated
        - Volume clock: synchronize on volume rather than time
        - VPIN: Volume-synchronized Probability of Informed trading
        """
        n = len(data)
        result = pd.DataFrame(index=data.index)

        close = data["close"].values
        open_ = data["open"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values if "volume" in data.columns else np.ones(n)

        # --- 1. Bid-Ask Imbalance (from close-open direction + volume) ---
        # Positive close-open → buying pressure, negative → selling pressure
        direction = np.sign(close - open_)
        buy_vol = np.where(direction > 0, volume, volume * 0.3)
        sell_vol = np.where(direction < 0, volume, volume * 0.3)
        total_vol = buy_vol + sell_vol + 1e-10
        result["bid_ask_imbalance"] = (buy_vol - sell_vol) / total_vol

        # --- 2. Weighted Mid Price deviation ---
        # How far close is from the midpoint of high-low range
        mid = (high + low) / 2.0
        hl_range = high - low + 1e-10
        result["weighted_mid_price"] = (close - mid) / hl_range

        # --- 3. Depth Ratio (rolling up-volume / down-volume) ---
        up_vol = np.where(close > open_, volume, 0.0)
        down_vol = np.where(close < open_, volume, 0.0)

        depth_ratio = np.ones(n)
        for i in range(self._flow_window, n):
            up_sum = np.sum(up_vol[i - self._flow_window:i]) + 1e-10
            down_sum = np.sum(down_vol[i - self._flow_window:i]) + 1e-10
            depth_ratio[i] = up_sum / down_sum
        # Normalize to [-1, 1] range
        result["depth_ratio"] = np.clip((depth_ratio - 1.0) / (depth_ratio + 1.0), -1, 1)

        # --- 4. Spread (basis points from high-low as proxy) ---
        # In backtesting, true spread is unknown; use intrabar range as estimate
        result["spread_bps"] = (high - low) / (close + 1e-10) * 10_000

        # --- 5. Order Flow Imbalance (cumulative tick-rule signed volume) ---
        # Tick rule: if close > prev_close → buy, else sell
        tick_direction = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                tick_direction[i] = 1.0
            elif close[i] < close[i - 1]:
                tick_direction[i] = -1.0
            else:
                tick_direction[i] = tick_direction[i - 1]  # unchanged

        signed_volume = tick_direction * volume
        ofi = np.zeros(n)
        for i in range(self._flow_window, n):
            window = signed_volume[i - self._flow_window:i]
            total = np.sum(np.abs(window)) + 1e-10
            ofi[i] = np.sum(window) / total
        result["order_flow_imbalance"] = ofi

        # --- 6. Volume at Best (normalized current volume) ---
        vol_ma = pd.Series(volume).rolling(self._flow_window, min_periods=1).mean().values
        result["volume_at_best"] = np.clip(volume / (vol_ma + 1e-10) - 1.0, -2.0, 5.0)

        # --- 7. Book Pressure (composite score) ---
        # Combines imbalance, depth ratio, and flow into single [-1, 1] score
        result["book_pressure"] = np.clip(
            0.4 * result["bid_ask_imbalance"].values
            + 0.3 * result["depth_ratio"].values
            + 0.3 * ofi,
            -1.0, 1.0,
        )

        # --- 8. Trade Flow Toxicity (VPIN approximation) ---
        # VPIN = |Buy Volume - Sell Volume| / Total Volume, rolling
        vpin = np.zeros(n)
        for i in range(self._vpin_window, n):
            bv = np.sum(buy_vol[i - self._vpin_window:i])
            sv = np.sum(sell_vol[i - self._vpin_window:i])
            tv = bv + sv + 1e-10
            vpin[i] = abs(bv - sv) / tv
        result["trade_flow_toxicity"] = vpin

        # Fill NaN with 0
        result = result.fillna(0.0)

        return result

    def set_live_book(self, book_snapshot: dict) -> None:
        """Set real L2 order book snapshot for live feature extraction.

        Args:
            book_snapshot: {"bids": [(price, volume), ...], "asks": [(price, volume), ...]}
        """
        self._live_book = book_snapshot

    def extract_live(self) -> np.ndarray:
        """Extract features from real L2 order book snapshot."""
        if self._live_book is None:
            return np.zeros(len(self.feature_names))

        bids = self._live_book.get("bids", [])
        asks = self._live_book.get("asks", [])

        if not bids or not asks:
            return np.zeros(len(self.feature_names))

        # Top N levels
        top_bids = bids[:self._depth_levels]
        top_asks = asks[:self._depth_levels]

        bid_vol = sum(v for _, v in top_bids)
        ask_vol = sum(v for _, v in top_asks)
        total_vol = bid_vol + ask_vol + 1e-10

        # 1. Imbalance
        imbalance = (bid_vol - ask_vol) / total_vol

        # 2. Weighted mid
        best_bid = top_bids[0][0]
        best_ask = top_asks[0][0]
        mid = (best_bid + best_ask) / 2.0

        # Volume-weighted mid
        bid_vwap = sum(p * v for p, v in top_bids) / (bid_vol + 1e-10)
        ask_vwap = sum(p * v for p, v in top_asks) / (ask_vol + 1e-10)
        wt_mid = (bid_vwap * ask_vol + ask_vwap * bid_vol) / total_vol
        wt_mid_dev = (wt_mid - mid) / (mid + 1e-10)

        # 3. Depth ratio
        depth_ratio = (bid_vol - ask_vol) / total_vol

        # 4. Spread
        spread_bps = (best_ask - best_bid) / mid * 10_000

        # 5-8 need time series context; return 0 for single snapshot
        return np.array([
            imbalance, wt_mid_dev, depth_ratio, spread_bps,
            0.0, bid_vol + ask_vol, imbalance, 0.0,
        ], dtype=np.float32)
