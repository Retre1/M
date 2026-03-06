"""Core Gymnasium environment for Forex trading with continuous action space.

Changes from original:
- FIX: Drawdown termination threshold raised to 15% (was same as reward penalty = double punishment)
- ADD: Realistic slippage model (size-dependent + volatility-dependent)
- ADD: Variable spread model (session-based + volatility-scaled)
- ADD: Z-Score forwarding to QuantumZScoreReward
- ADD: Adaptive trailing stop-loss (ATR-based)
- ADD: Trade return tracking for proper Kelly criterion
- FIX: Look-ahead bias — actions now execute at next-bar open, not current close
- FIX: historical_atr computed as rolling median instead of fallback to current
- ADD: Hold reward — bonus for maintaining profitable positions
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from apexfx.env.obs_builder import ObservationBuilder
from apexfx.env.reward import BaseRewardFunction, HoldAwareReward, QuantumZScoreReward, TradingReward


class SpreadModel:
    """Realistic spread modeling: varies by session and volatility."""

    def __init__(
        self,
        base_spread_pips: float = 1.0,
        pip_value: float = 0.0001,
    ) -> None:
        self._base = base_spread_pips
        self._pip = pip_value
        # Session multipliers (empirical)
        self._session_mult = {
            "overlap": 0.8,
            "london": 1.0,
            "new_york": 1.0,
            "tokyo": 1.5,
            "sydney": 2.0,
            "off_hours": 3.0,
        }

    def get_spread(
        self,
        hour: int,
        current_atr: float | None = None,
        historical_atr: float | None = None,
    ) -> float:
        session = self._get_session(hour)
        mult = self._session_mult.get(session, 1.5)

        # Volatility scaling
        vol_mult = 1.0
        if current_atr is not None and historical_atr is not None and historical_atr > 0:
            vol_mult = max(1.0, current_atr / historical_atr)

        spread_pips = self._base * mult * vol_mult
        return spread_pips * self._pip

    @staticmethod
    def _get_session(hour: int) -> str:
        if 12 <= hour <= 16:
            return "overlap"
        if 7 <= hour <= 16:
            return "london"
        if 12 <= hour <= 21:
            return "new_york"
        if 0 <= hour <= 9:
            return "tokyo"
        return "off_hours"


class SlippageModel:
    """Realistic slippage: increases with volume and volatility."""

    def __init__(self, pip_value: float = 0.0001) -> None:
        self._pip = pip_value

    def compute(
        self,
        volume: float,
        current_atr: float | None = None,
        historical_atr: float | None = None,
    ) -> float:
        # Size impact: larger orders get more slippage
        size_impact = 0.1 * np.log1p(max(0, volume * 10))  # pips

        # Volatility impact
        vol_impact = 0.0
        if current_atr is not None and historical_atr is not None and historical_atr > 0:
            vol_ratio = current_atr / historical_atr
            if vol_ratio > 1.0:
                vol_impact = 0.2 * (vol_ratio - 1.0)

        slippage_pips = size_impact + vol_impact
        return slippage_pips * self._pip


class AdaptiveStopLoss:
    """ATR-based adaptive trailing stop-loss with break-even logic.

    Break-even: after price moves breakeven_atr_mult * ATR in our favor,
    the stop is locked at the entry price (no worse than flat).
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        min_stop_pips: float = 10.0,
        pip_value: float = 0.0001,
        breakeven_atr_mult: float = 1.5,
    ) -> None:
        self._atr_mult = atr_multiplier
        self._min_stop = min_stop_pips
        self._pip = pip_value
        self._breakeven_atr_mult = breakeven_atr_mult
        self._stop_price: float | None = None
        self._best_price: float | None = None
        self._entry_price: float | None = None
        self._breakeven_activated: bool = False

    def reset(self) -> None:
        self._stop_price = None
        self._best_price = None
        self._entry_price = None
        self._breakeven_activated = False

    def set_entry(self, entry_price: float) -> None:
        """Set entry price for break-even calculation."""
        self._entry_price = entry_price
        self._breakeven_activated = False

    @property
    def breakeven_activated(self) -> bool:
        return self._breakeven_activated

    def update(
        self,
        current_price: float,
        direction: int,
        current_atr: float | None,
    ) -> bool:
        """Update trailing stop. Returns True if stop is hit."""
        if direction == 0 or current_atr is None:
            self.reset()
            return False

        stop_distance = max(
            current_atr * self._atr_mult,
            self._min_stop * self._pip,
        )

        if self._best_price is None:
            self._best_price = current_price

        # --- Break-even logic ---
        if (
            not self._breakeven_activated
            and self._entry_price is not None
            and current_atr > 0
        ):
            profit_distance = (current_price - self._entry_price) * direction
            if profit_distance >= current_atr * self._breakeven_atr_mult:
                self._breakeven_activated = True
                # Lock stop at entry price
                if direction > 0:
                    if self._stop_price is None or self._stop_price < self._entry_price:
                        self._stop_price = self._entry_price
                else:
                    if self._stop_price is None or self._stop_price > self._entry_price:
                        self._stop_price = self._entry_price

        # --- Normal trailing logic ---
        if direction > 0:  # Long
            self._best_price = max(self._best_price, current_price)
            new_stop = self._best_price - stop_distance
            # Respect break-even floor
            if self._breakeven_activated and self._entry_price is not None:
                new_stop = max(new_stop, self._entry_price)
            if self._stop_price is None:
                self._stop_price = new_stop
            else:
                self._stop_price = max(self._stop_price, new_stop)
            return current_price <= self._stop_price
        else:  # Short
            self._best_price = min(self._best_price, current_price)
            new_stop = self._best_price + stop_distance
            # Respect break-even ceiling
            if self._breakeven_activated and self._entry_price is not None:
                new_stop = min(new_stop, self._entry_price)
            if self._stop_price is None:
                self._stop_price = new_stop
            else:
                self._stop_price = min(self._stop_price, new_stop)
            return current_price >= self._stop_price

    @property
    def stop_price(self) -> float | None:
        return self._stop_price


class PartialFillModel:
    """Simulates partial fills based on volume and liquidity.

    In real markets, large orders may only partially fill. This model
    penalizes backtesting P&L to reflect realistic fill expectations.
    """

    # Session liquidity multipliers (higher = better fills)
    SESSION_FILL_MULT: dict[str, float] = {
        "overlap": 1.0,     # Best liquidity
        "london": 0.95,
        "new_york": 0.95,
        "tokyo": 0.85,
        "sydney": 0.75,
        "off_hours": 0.60,
    }

    def __init__(
        self,
        base_fill_rate: float = 0.95,
        volume_impact: float = 0.3,
        session_impact: bool = True,
    ) -> None:
        """Initialize partial fill model.

        Args:
            base_fill_rate: Base fill rate (0-1) for small orders.
            volume_impact: How much large volume reduces fill rate.
                0.0 = no impact, 1.0 = maximum impact.
            session_impact: Whether to apply session-based liquidity penalties.
        """
        self._base_rate = np.clip(base_fill_rate, 0.5, 1.0)
        self._volume_impact = np.clip(volume_impact, 0.0, 1.0)
        self._session_impact = session_impact

    def simulate_fill(
        self,
        requested_volume: float,
        hour: int = 12,
        current_atr: float | None = None,
    ) -> tuple[float, float]:
        """Simulate a partial fill.

        Args:
            requested_volume: Requested volume in lots.
            hour: UTC hour for session-based adjustment.
            current_atr: Current ATR (higher vol → worse fills).

        Returns:
            (filled_volume, fill_rate) tuple.
        """
        if requested_volume <= 0:
            return 0.0, 0.0

        fill_rate = self._base_rate

        # Volume penalty: larger orders get worse fills
        # f(v) = base * exp(-impact * v) — decays exponentially with size
        if self._volume_impact > 0 and requested_volume > 0.1:
            volume_penalty = np.exp(-self._volume_impact * (requested_volume - 0.1))
            fill_rate *= max(0.5, volume_penalty)

        # Session penalty: illiquid sessions get worse fills
        if self._session_impact:
            session = SpreadModel._get_session(hour)
            session_mult = self.SESSION_FILL_MULT.get(session, 0.75)
            fill_rate *= session_mult

        # ATR penalty: higher volatility → worse fills (wider spreads, more rejections)
        if current_atr is not None and current_atr > 0.001:
            # Penalize when ATR > 1% of price (rough proxy)
            vol_penalty = max(0.8, 1.0 - 0.5 * max(0, current_atr - 0.001))
            fill_rate *= vol_penalty

        fill_rate = float(np.clip(fill_rate, 0.3, 1.0))
        filled_volume = round(requested_volume * fill_rate, 2)

        return filled_volume, fill_rate


class ForexTradingEnv(gym.Env):
    """
    Gymnasium environment for algorithmic Forex trading.

    Observation: Dict with market_features, time_features, trend/reversion/regime features,
                 and position state.
    Action: Box(-1, 1) — continuous action where:
        +1.0 = maximum long
         0.0 = neutral / exit
        -1.0 = maximum short
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100_000.0,
        max_position_pct: float = 0.10,
        transaction_cost_pips: float = 1.0,
        pip_value: float = 0.0001,
        contract_size: float = 100_000.0,
        episode_length: int = 2000,
        n_market_features: int = 30,
        n_trend_features: int = 8,
        n_reversion_features: int = 8,
        n_regime_features: int = 6,
        n_time_features: int = 5,
        n_fundamental_features: int = 8,
        n_structure_features: int = 8,
        lookback: int = 100,
        reward_fn: BaseRewardFunction | None = None,
        max_drawdown_pct: float = 0.15,
        render_mode: str | None = None,
        use_realistic_costs: bool = True,
        execution_delay_bars: int = 1,
        max_position_layers: int = 3,
        breakeven_atr_mult: float = 1.5,
        partial_fill_model: PartialFillModel | None = None,
    ) -> None:
        super().__init__()

        self._data = data
        self._initial_balance = initial_balance
        self._max_position_pct = max_position_pct
        self._base_transaction_cost = transaction_cost_pips * pip_value
        self._pip_value = pip_value
        self._contract_size = contract_size
        self._episode_length = episode_length
        self._max_drawdown_pct = max_drawdown_pct
        self._lookback = lookback
        self.render_mode = render_mode
        self._use_realistic_costs = use_realistic_costs
        self._execution_delay = execution_delay_bars

        self._reward_fn = reward_fn or QuantumZScoreReward()

        self._obs_builder = ObservationBuilder(
            n_market_features=n_market_features,
            n_trend_features=n_trend_features,
            n_reversion_features=n_reversion_features,
            n_regime_features=n_regime_features,
            n_time_features=n_time_features,
            n_fundamental_features=n_fundamental_features,
            n_structure_features=n_structure_features,
            lookback=lookback,
        )

        # Realistic cost models
        self._spread_model = SpreadModel(
            base_spread_pips=transaction_cost_pips, pip_value=pip_value
        )
        self._slippage_model = SlippageModel(pip_value=pip_value)
        self._stop_loss = AdaptiveStopLoss(
            pip_value=pip_value,
            breakeven_atr_mult=breakeven_atr_mult,
        )
        self._max_layers = max_position_layers
        self._partial_fill_model = partial_fill_model

        # --- Spaces ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        market_flat_size = lookback * n_market_features
        time_flat_size = lookback * n_time_features

        self.observation_space = spaces.Dict({
            "market_features": spaces.Box(-np.inf, np.inf, shape=(market_flat_size,), dtype=np.float32),
            "time_features": spaces.Box(-np.inf, np.inf, shape=(time_flat_size,), dtype=np.float32),
            "trend_features": spaces.Box(-np.inf, np.inf, shape=(n_trend_features,), dtype=np.float32),
            "reversion_features": spaces.Box(-np.inf, np.inf, shape=(n_reversion_features,), dtype=np.float32),
            "regime_features": spaces.Box(-np.inf, np.inf, shape=(n_regime_features,), dtype=np.float32),
            "fundamental_features": spaces.Box(-np.inf, np.inf, shape=(n_fundamental_features,), dtype=np.float32),
            "structure_features": spaces.Box(-np.inf, np.inf, shape=(n_structure_features,), dtype=np.float32),
            "position_state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        })

        # --- Precompute historical ATR for realistic cost modeling ---
        self._historical_atr: float | None = self._precompute_historical_atr()

        # --- State ---
        self._current_idx: int = 0
        self._start_idx: int = 0
        self._portfolio_value: float = initial_balance
        self._cash: float = initial_balance
        self._position: float = 0.0  # lots
        self._position_direction: int = 0  # -1, 0, +1
        self._entry_price: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._time_in_position: int = 0
        self._peak_value: float = initial_balance
        self._total_trades: int = 0
        self._trade_returns: list[float] = []
        self._equity_curve: list[float] = []
        # Pending action queue for execution delay simulation
        self._action_queue: list[float] = []
        # Position layer tracking for pyramiding
        self._position_layers: list[dict] = []  # [{size, entry_price, entry_idx}]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        n = len(self._data)
        max_start = n - self._episode_length - self._lookback
        if max_start <= self._lookback:
            self._start_idx = self._lookback
        else:
            self._start_idx = self.np_random.integers(self._lookback, max_start)

        self._current_idx = self._start_idx
        self._portfolio_value = self._initial_balance
        self._cash = self._initial_balance
        self._position = 0.0
        self._position_direction = 0
        self._entry_price = 0.0
        self._unrealized_pnl = 0.0
        self._time_in_position = 0
        self._peak_value = self._initial_balance
        self._total_trades = 0
        self._trade_returns = []
        self._equity_curve = [self._initial_balance]
        self._reward_fn.reset()
        self._stop_loss.reset()
        self._action_queue = []
        self._position_layers = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_value = float(np.clip(action[0], -1.0, 1.0))

        prev_portfolio = self._portfolio_value

        # --- Advance market FIRST (look-ahead bias fix) ---
        # Action is decided based on current bar, but executed after delay
        self._current_idx += 1

        # --- Execution delay: queue action, execute delayed one ---
        self._action_queue.append(action_value)
        if self._current_idx < len(self._data):
            # Set Z-Score for reward before execution
            if isinstance(self._reward_fn, QuantumZScoreReward):
                z_score = self._get_current_zscore()
                self._reward_fn.set_zscore(z_score)
            elif isinstance(self._reward_fn, HoldAwareReward):
                z_score = self._get_current_zscore()
                self._reward_fn.set_zscore(z_score)

            # Execute the delayed action (or current if no delay configured)
            if len(self._action_queue) > self._execution_delay:
                delayed_action = self._action_queue.pop(0)
            else:
                delayed_action = 0.0  # hold flat until delay buffer fills
            self._execute_action(delayed_action)

        # --- Update P&L at this bar's close ---
        self._update_pnl()

        # --- Check adaptive stop-loss ---
        if self._position_direction != 0 and self._current_idx < len(self._data):
            current_price = self._data.iloc[self._current_idx]["close"]
            current_atr = self._get_current_atr()
            if self._stop_loss.update(current_price, self._position_direction, current_atr):
                self._close_position(current_price)

        # --- Feed position state to reward function ---
        if isinstance(self._reward_fn, TradingReward):
            # Check fundamental/structure features for reward signals
            news_active = False
            structure_aligned = False
            if self._current_idx < len(self._data):
                row = self._data.iloc[self._current_idx]
                if "news_impact_active" in row.index:
                    news_active = float(row.get("news_impact_active", 0)) > 0.5
                if "structure_break_bull" in row.index and "structure_break_bear" in row.index:
                    bull_break = float(row.get("structure_break_bull", 0)) > 0.5
                    bear_break = float(row.get("structure_break_bear", 0)) > 0.5
                    if self._position_direction > 0 and bull_break:
                        structure_aligned = True
                    elif self._position_direction < 0 and bear_break:
                        structure_aligned = True

            self._reward_fn.set_trade_info(
                action=action_value,
                direction=self._position_direction,
                unrealized_pnl=self._unrealized_pnl,
                time_in_position=self._time_in_position,
                news_active=news_active,
                structure_aligned=structure_aligned,
            )
            # Feed ATR for volatility-adjusted reward
            self._reward_fn.set_atr(self._get_current_atr())
        elif isinstance(self._reward_fn, HoldAwareReward):
            self._reward_fn.set_position_info(
                direction=self._position_direction,
                unrealized_pnl=self._unrealized_pnl,
                time_in_position=self._time_in_position,
            )

        # --- Compute reward ---
        reward = self._reward_fn.compute(self._portfolio_value, prev_portfolio)

        # --- Check termination ---
        terminated = False
        truncated = False

        # Max drawdown termination (raised to 15% to avoid double penalty with reward)
        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value
        if drawdown > self._max_drawdown_pct:
            terminated = True

        # Bankruptcy
        if self._portfolio_value <= 0:
            terminated = True
            reward = -10.0

        # Episode length
        steps_done = self._current_idx - self._start_idx
        if steps_done >= self._episode_length:
            truncated = True

        # Data boundary
        if self._current_idx >= len(self._data) - 1:
            truncated = True

        self._equity_curve.append(self._portfolio_value)

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _get_current_zscore(self) -> float:
        """Extract current price Z-Score from features if available."""
        if self._current_idx < len(self._data):
            row = self._data.iloc[self._current_idx]
            # Try common z-score column names
            for col in ("z_score", "price_z_score", "rolling_z_score"):
                if col in row.index:
                    val = row[col]
                    if not np.isnan(val):
                        return float(val)
        return 0.0

    def _precompute_historical_atr(self) -> float | None:
        """Precompute median ATR across the dataset for realistic cost modeling."""
        for col in ("atr", "realized_vol"):
            if col in self._data.columns:
                values = self._data[col].dropna()
                if len(values) > 0:
                    return float(values.median())
        # Fallback: compute from OHLC if available
        if all(c in self._data.columns for c in ("high", "low", "close")):
            high = self._data["high"].values
            low = self._data["low"].values
            close = self._data["close"].values
            tr = np.zeros(len(high))
            tr[0] = high[0] - low[0]
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )
            if len(tr) >= 14:
                # Rolling ATR median
                atr_vals = pd.Series(tr).rolling(14).mean().dropna()
                if len(atr_vals) > 0:
                    return float(atr_vals.median())
        return None

    def _get_current_atr(self) -> float | None:
        """Get current ATR from features if available."""
        if self._current_idx < len(self._data):
            row = self._data.iloc[self._current_idx]
            for col in ("atr", "realized_vol"):
                if col in row.index:
                    val = row[col]
                    if not np.isnan(val):
                        return float(val)
        return None

    def _get_transaction_cost(self, volume: float) -> float:
        """Compute realistic transaction cost including spread and slippage."""
        if not self._use_realistic_costs:
            return volume * self._base_transaction_cost * self._contract_size

        # Get bar hour for spread session estimation
        hour = 12  # default London
        if self._current_idx < len(self._data):
            row = self._data.iloc[self._current_idx]
            if "time" in row.index:
                try:
                    hour = pd.Timestamp(row["time"]).hour
                except Exception:
                    pass

        current_atr = self._get_current_atr()
        historical_atr = self._historical_atr or current_atr

        spread = self._spread_model.get_spread(hour, current_atr, historical_atr)
        slippage = self._slippage_model.compute(volume, current_atr, historical_atr)

        total_cost_per_unit = spread + slippage
        return volume * total_cost_per_unit * self._contract_size

    def _execute_action(self, action: float) -> None:
        """Translate continuous action into position changes.

        Uses the bar's open price for execution to prevent look-ahead bias.
        In real trading, an order placed after observing bar N would fill
        at bar N+1's open, not its close.
        """
        row = self._data.iloc[self._current_idx]
        current_price = row["open"] if "open" in row.index else row["close"]

        if abs(action) < 0.05:
            target_direction = 0
            target_size = 0.0
        else:
            target_direction = 1 if action > 0 else -1
            max_lots = (self._portfolio_value * self._max_position_pct) / (
                current_price * self._contract_size
            )
            target_size = abs(action) * max_lots

        # Close existing position if direction changes
        if self._position_direction != 0 and target_direction != self._position_direction:
            self._close_position(current_price)

        # Open or adjust position
        if target_direction != 0:
            if self._position_direction == 0:
                self._open_position(target_direction, target_size, current_price)
            elif self._position_direction == target_direction:
                size_diff = target_size - self._position
                if abs(size_diff) > 0.001:
                    self._adjust_position(size_diff, current_price)
        elif self._position_direction != 0:
            self._close_position(current_price)

    def _open_position(self, direction: int, size: float, price: float) -> None:
        """Open a new position."""
        # Apply partial fill model if enabled
        if self._partial_fill_model is not None:
            hour = 12
            if self._current_idx < len(self._data):
                row = self._data.iloc[self._current_idx]
                if "time" in row.index:
                    try:
                        hour = pd.Timestamp(row["time"]).hour
                    except Exception:
                        pass
            filled_size, fill_rate = self._partial_fill_model.simulate_fill(
                size, hour, self._get_current_atr()
            )
            if filled_size < 0.01:
                return  # Fill too small, skip
            size = filled_size

        cost = self._get_transaction_cost(size)
        self._cash -= cost
        self._position = size
        self._position_direction = direction
        self._entry_price = price
        self._time_in_position = 0
        self._total_trades += 1
        self._stop_loss.reset()
        self._stop_loss.set_entry(price)
        self._position_layers = [{"size": size, "entry_price": price, "entry_idx": self._current_idx}]

    def _close_position(self, price: float) -> None:
        """Close the current position."""
        if self._position <= 0:
            return

        price_diff = price - self._entry_price
        pnl = price_diff * self._position * self._contract_size * self._position_direction
        cost = self._get_transaction_cost(self._position)

        self._cash += pnl - cost

        # Record trade return (for Kelly criterion)
        notional = self._entry_price * self._position * self._contract_size
        trade_return = pnl / (notional + 1e-10)
        self._trade_returns.append(trade_return)

        # Reset position state
        self._position = 0.0
        self._position_direction = 0
        self._entry_price = 0.0
        self._unrealized_pnl = 0.0
        self._time_in_position = 0
        self._stop_loss.reset()
        self._position_layers = []

    def _adjust_position(self, size_diff: float, price: float) -> None:
        """Adjust an existing position's size."""
        cost = self._get_transaction_cost(abs(size_diff))
        self._cash -= cost
        self._position = max(0, self._position + size_diff)

        if size_diff > 0 and len(self._position_layers) < self._max_layers:
            # Adding to position — track as new layer
            self._position_layers.append({
                "size": size_diff,
                "entry_price": price,
                "entry_idx": self._current_idx,
            })
            # Update weighted avg entry price for stop-loss
            total_size = sum(l["size"] for l in self._position_layers)
            if total_size > 0:
                self._entry_price = (
                    sum(l["size"] * l["entry_price"] for l in self._position_layers)
                    / total_size
                )

    def _update_pnl(self) -> None:
        """Update unrealized P&L and portfolio value."""
        if self._position > 0 and self._current_idx < len(self._data):
            current_price = self._data.iloc[self._current_idx]["close"]
            price_diff = current_price - self._entry_price
            self._unrealized_pnl = (
                price_diff * self._position * self._contract_size * self._position_direction
            )
            self._time_in_position += 1
        else:
            self._unrealized_pnl = 0.0

        self._portfolio_value = self._cash + self._unrealized_pnl
        self._peak_value = max(self._peak_value, self._portfolio_value)

    def _get_observation(self) -> dict[str, np.ndarray]:
        # Compute position management signals for observation
        current_atr = self._get_current_atr() or 1e-10
        current_price = 0.0
        if self._current_idx < len(self._data):
            current_price = self._data.iloc[self._current_idx]["close"]

        distance_to_stop = 0.0
        if self._stop_loss.stop_price is not None and current_atr > 0:
            distance_to_stop = abs(current_price - self._stop_loss.stop_price) / current_atr

        avg_entry_distance = 0.0
        if self._entry_price > 0 and current_atr > 0:
            avg_entry_distance = (
                (current_price - self._entry_price) * self._position_direction / current_atr
            )

        return self._obs_builder.build(
            features=self._data,
            current_idx=self._current_idx,
            position=self._position * self._position_direction,
            unrealized_pnl=self._unrealized_pnl,
            time_in_position=float(self._time_in_position),
            portfolio_value=self._portfolio_value,
            initial_balance=self._initial_balance,
            n_layers=len(self._position_layers),
            breakeven_active=self._stop_loss.breakeven_activated,
            distance_to_stop=distance_to_stop,
            avg_entry_distance=avg_entry_distance,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "position": self._position * self._position_direction,
            "unrealized_pnl": self._unrealized_pnl,
            "drawdown": (self._peak_value - self._portfolio_value) / self._peak_value,
            "total_trades": self._total_trades,
            "step": self._current_idx - self._start_idx,
        }
