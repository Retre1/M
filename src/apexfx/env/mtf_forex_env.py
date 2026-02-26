"""Multi-Timeframe Forex Trading Environment.

Extends ForexTradingEnv to observe D1 + H1 + M5 simultaneously.
Steps on H1 bars (primary timeframe), but the observation includes
context from D1 (higher TF trend) and M5 (lower TF detail).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from apexfx.data.mtf_aligner import MTFDataAligner
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.mtf_obs_builder import MTFObservationBuilder

if TYPE_CHECKING:
    import pandas as pd

    from apexfx.env.reward import BaseRewardFunction


class MTFForexTradingEnv(ForexTradingEnv):
    """Multi-Timeframe Forex trading environment.

    Inherits all trading mechanics from ForexTradingEnv (position management,
    costs, stop-loss, reward). Overrides observation construction to include
    D1 and M5 context alongside H1.

    The environment steps through H1 bars. At each step, it looks up the
    corresponding D1 and M5 slices via MTFDataAligner.
    """

    def __init__(
        self,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        m5_data: pd.DataFrame,
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
        d1_lookback: int = 5,
        h1_lookback: int = 20,
        m5_lookback: int = 20,
        reward_fn: BaseRewardFunction | None = None,
        max_drawdown_pct: float = 0.15,
        render_mode: str | None = None,
        use_realistic_costs: bool = True,
    ) -> None:
        # Store MTF-specific params before calling super().__init__
        self._d1_data = d1_data
        self._m5_data = m5_data
        self._d1_lookback = d1_lookback
        self._h1_lookback = h1_lookback
        self._m5_lookback = m5_lookback

        # Initialize base env with H1 as primary data
        super().__init__(
            data=h1_data,
            initial_balance=initial_balance,
            max_position_pct=max_position_pct,
            transaction_cost_pips=transaction_cost_pips,
            pip_value=pip_value,
            contract_size=contract_size,
            episode_length=episode_length,
            n_market_features=n_market_features,
            n_trend_features=n_trend_features,
            n_reversion_features=n_reversion_features,
            n_regime_features=n_regime_features,
            n_time_features=n_time_features,
            lookback=h1_lookback,
            reward_fn=reward_fn,
            max_drawdown_pct=max_drawdown_pct,
            render_mode=render_mode,
            use_realistic_costs=use_realistic_costs,
        )

        # MTF aligner for cross-timeframe lookups
        self._aligner = MTFDataAligner(
            d1_data=d1_data,
            h1_data=h1_data,
            m5_data=m5_data,
            d1_lookback=d1_lookback,
            h1_lookback=h1_lookback,
            m5_lookback=m5_lookback,
        )

        # MTF observation builder
        self._mtf_obs_builder = MTFObservationBuilder(
            n_market_features=n_market_features,
            n_trend_features=n_trend_features,
            n_reversion_features=n_reversion_features,
            n_regime_features=n_regime_features,
            n_time_features=n_time_features,
            d1_lookback=d1_lookback,
            h1_lookback=h1_lookback,
            m5_lookback=m5_lookback,
        )

        # Override observation space to MTF layout
        n_mtf_context = 6
        d1_market_size = d1_lookback * n_market_features
        d1_time_size = d1_lookback * n_time_features
        h1_market_size = h1_lookback * n_market_features
        h1_time_size = h1_lookback * n_time_features
        m5_market_size = m5_lookback * n_market_features
        m5_time_size = m5_lookback * n_time_features

        box = spaces.Box
        inf = np.inf
        f32 = np.float32
        self.observation_space = spaces.Dict({
            "d1_market_features": box(-inf, inf, shape=(d1_market_size,), dtype=f32),
            "d1_time_features": box(-inf, inf, shape=(d1_time_size,), dtype=f32),
            "h1_market_features": box(-inf, inf, shape=(h1_market_size,), dtype=f32),
            "h1_time_features": box(-inf, inf, shape=(h1_time_size,), dtype=f32),
            "m5_market_features": box(-inf, inf, shape=(m5_market_size,), dtype=f32),
            "m5_time_features": box(-inf, inf, shape=(m5_time_size,), dtype=f32),
            "trend_features": box(-inf, inf, shape=(n_trend_features,), dtype=f32),
            "reversion_features": box(-inf, inf, shape=(n_reversion_features,), dtype=f32),
            "regime_features": box(-inf, inf, shape=(n_regime_features,), dtype=f32),
            "position_state": box(-inf, inf, shape=(4,), dtype=f32),
            "mtf_context": box(-inf, inf, shape=(n_mtf_context,), dtype=f32),
        })

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset with MTF-aware start index."""
        # Use grandparent reset for seed handling
        gym.Env.reset(self, seed=seed)

        n = len(self._data)
        min_start = self._aligner.min_h1_idx
        max_start = n - self._episode_length - 1

        if max_start <= min_start:
            self._start_idx = min_start
        else:
            self._start_idx = self.np_random.integers(min_start, max_start)

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

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Override to build MTF observation instead of single-TF."""
        # Get aligned slices from all three timeframes
        mtf_slice = self._aligner.get_slice(self._current_idx)

        return self._mtf_obs_builder.build(
            mtf_slice=mtf_slice,
            h1_features=self._data,
            h1_idx=self._current_idx,
            position=self._position * self._position_direction,
            unrealized_pnl=self._unrealized_pnl,
            time_in_position=float(self._time_in_position),
            portfolio_value=self._portfolio_value,
            initial_balance=self._initial_balance,
        )
