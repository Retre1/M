"""Dynamic position sizing: Kelly criterion + volatility scaling + hard caps."""

from __future__ import annotations

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class PositionSizer:
    """
    Computes position size based on:
    1. Model confidence (|action| magnitude)
    2. Kelly-inspired scaling (half-Kelly for conservatism)
    3. Volatility inverse scaling (higher vol → smaller position)
    4. Hard caps (never exceed max_position_pct of portfolio)
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        kelly_fraction: float = 0.5,
        min_trades_for_kelly: int = 30,
        vol_lookback_bars: int = 20,
        min_lot_size: float = 0.01,
        contract_size: float = 100_000.0,
    ) -> None:
        self._max_pct = max_position_pct
        self._kelly_frac = kelly_fraction
        self._min_kelly_trades = min_trades_for_kelly
        self._vol_lookback = vol_lookback_bars
        self._min_lot = min_lot_size
        self._contract_size = contract_size

        self._trade_wins: int = 0
        self._trade_losses: int = 0
        self._avg_win: float = 0.0
        self._avg_loss: float = 0.0

    def update_trade_stats(self, trade_return: float) -> None:
        """Update running trade statistics for Kelly computation.

        Args:
            trade_return: Return on risk (PnL / notional value), NOT absolute PnL.
                          e.g. +0.02 for a 2% gain, -0.01 for a 1% loss.
                          Using returns instead of absolute PnL ensures Kelly fraction
                          is correctly computed regardless of position size.
        """
        if trade_return > 0:
            self._trade_wins += 1
            n = self._trade_wins
            self._avg_win = self._avg_win * ((n - 1) / n) + trade_return / n
        elif trade_return < 0:
            self._trade_losses += 1
            n = self._trade_losses
            self._avg_loss = self._avg_loss * ((n - 1) / n) + abs(trade_return) / n

    def compute(
        self,
        action: float,
        portfolio_value: float,
        current_price: float,
        current_atr: float | None = None,
        historical_atr: float | None = None,
    ) -> float:
        """
        Compute position size in lots.

        Args:
            action: model output in [-1, 1]
            portfolio_value: current portfolio value
            current_price: current market price
            current_atr: current ATR (for volatility scaling)
            historical_atr: historical average ATR (baseline for scaling)

        Returns:
            Position size in lots (always positive; direction from action sign)
        """
        confidence = abs(action)

        # 1. Kelly fraction
        kelly = self._compute_kelly()

        # 2. Volatility adjustment
        vol_scalar = self._volatility_adjustment(current_atr, historical_atr)

        # 3. Raw position value
        max_value = portfolio_value * self._max_pct
        raw_value = confidence * kelly * vol_scalar * max_value

        # 4. Hard cap
        capped_value = min(raw_value, max_value)

        # Convert to lots
        lot_value = current_price * self._contract_size
        if lot_value <= 0:
            return 0.0

        lots = capped_value / lot_value
        lots = self._round_to_lot_step(lots)

        return lots

    def _compute_kelly(self) -> float:
        """Compute half-Kelly criterion fraction."""
        total_trades = self._trade_wins + self._trade_losses

        if total_trades < self._min_kelly_trades:
            # Not enough data: use a conservative default
            return self._kelly_frac * 0.5  # Quarter Kelly until sufficient data

        win_rate = self._trade_wins / total_trades
        if self._avg_loss <= 0 or win_rate <= 0:
            return self._kelly_frac * 0.5

        # Kelly: f = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
        b = self._avg_win / self._avg_loss
        q = 1.0 - win_rate
        kelly_full = (win_rate * b - q) / b

        # Apply fraction (half Kelly by default)
        kelly = kelly_full * self._kelly_frac

        # Clamp to [0, 1]
        return float(np.clip(kelly, 0.0, 1.0))

    @staticmethod
    def _volatility_adjustment(
        current_atr: float | None, historical_atr: float | None
    ) -> float:
        """Inverse volatility scaling: higher vol → smaller position."""
        if current_atr is None or historical_atr is None:
            return 1.0
        if current_atr <= 0 or historical_atr <= 0:
            return 1.0

        # Ratio: if current vol is 2x historical, scale position to 50%
        ratio = historical_atr / current_atr
        return float(np.clip(ratio, 0.25, 2.0))

    def _round_to_lot_step(self, lots: float) -> float:
        """Round to the nearest valid lot step."""
        if lots < self._min_lot:
            return 0.0
        return round(lots / self._min_lot) * self._min_lot
