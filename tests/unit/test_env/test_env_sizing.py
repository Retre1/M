"""The env sizes through PositionSizer, and the reward scale that follows.

The env used to size positions itself, as a share of notional:

    max_lots = equity * max_position_pct / (price * contract_size)

which is 0.091 lots on EURUSD and risks about 0.02% of equity against a 2xATR
stop. The backtest went through ``PositionSizer`` and the env did not, so the
agent was trained at one exposure scale and measured at another.

Fixing that raises the median position 13.5x, and the two reward terms that
multiply a fraction of equity move with it while ``trade_cost_weight`` and the
inactivity ramp stay flat constants. Those weights are divided by the same
measured factor, which is what these tests hold in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.reward_v5 import RARARewardV5Config

EQUITY = 100_000.0
CONTRACT = 100_000.0


def _frame(n: int = 1500, atr: float = 0.0010, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0.0, 0.0012, n)))
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close * 1.0008, "low": close * 0.9992,
        "close": close, "volume": np.full(n, 500),
    })
    for column in ("hurst_exponent", "trend_strength", "realized_vol", "close_zscore"):
        df[column] = 0.5
    df["atr"] = atr
    return df


def _run(df: pd.DataFrame, steps: int = 300, **kwargs) -> list[float]:
    env = ForexTradingEnv(data=df, episode_length=steps + 50, lookback=50, **kwargs)
    env.reset()
    sizes: list[float] = []
    for i in range(steps):
        action = np.array([1.0 if (i // 40) % 2 == 0 else -1.0], dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        if env._position > 0:
            sizes.append(float(env._position))
        if terminated or truncated:
            break
    return sizes


class TestEnvSizesFromRisk:
    def test_exposure_is_far_above_the_old_notional_cap(self):
        """0.091 lots was the old ceiling regardless of volatility."""
        sizes = _run(_frame())
        assert np.median(sizes) > 0.5

    def test_a_wider_stop_buys_fewer_lots(self):
        """Volatility now reaches size through the stop distance."""
        calm = np.median(_run(_frame(atr=0.0005)))
        wild = np.median(_run(_frame(atr=0.0030)))
        assert calm > wild

    def test_risk_at_the_stop_is_roughly_constant_across_volatility(self):
        """The point of sizing from risk: the loss if stopped does not depend
        on how wide the stop had to be."""
        def risk(atr: float) -> float:
            lots = np.median(_run(_frame(atr=atr)))
            return lots * CONTRACT * 2.0 * atr

        assert risk(0.0005) == pytest.approx(risk(0.0030), rel=0.15)

    def test_the_leverage_cap_binds(self):
        """A near-zero stop would otherwise ask for an unbounded position.

        Leverage is measured against equity *at the time of the position*, not
        against the starting balance: the cap is a fraction of current
        portfolio value, and equity drifts during an episode.
        """
        cap = 3.0
        env = ForexTradingEnv(
            data=_frame(atr=1e-7), episode_length=350, lookback=50, max_leverage=cap,
        )
        env.reset()
        worst = 0.0
        for i in range(300):
            action = np.array([1.0 if (i // 40) % 2 == 0 else -1.0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            if env._position > 0:
                notional = env._position * env._entry_price * CONTRACT
                worst = max(worst, notional / env._portfolio_value)
            if terminated or truncated:
                break
        assert worst > 0.0, "the cap was never exercised"
        assert worst <= cap * 1.02

    def test_a_lower_risk_budget_gives_a_smaller_position(self):
        big = np.median(_run(_frame(), risk_per_trade_pct=0.02))
        small = np.median(_run(_frame(), risk_per_trade_pct=0.005))
        assert small < big

    def test_kelly_statistics_are_fed_back(self):
        """Without this the sizer never leaves its warm-up quarter-Kelly."""
        env = ForexTradingEnv(data=_frame(), episode_length=400, lookback=50)
        env.reset()
        for i in range(300):
            env.step(np.array([1.0 if (i // 30) % 2 == 0 else -1.0], dtype=np.float32))
        sizer = env._position_sizer
        assert sizer._trade_wins + sizer._trade_losses > 0


class TestRewardScaleSurvivesTheSizingChange:
    """The weights multiply a fraction of equity, so they move with size."""

    CFG = RARARewardV5Config()
    OLD_WEIGHT, OLD_LOTS = 4_000.0, 0.0909
    NEW_LOTS = 1.23  # measured median after the change

    @classmethod
    def _realized_term(cls, lots: float, weight: float, move_pips: float = 20) -> float:
        pnl = lots * CONTRACT * move_pips * 0.0001
        return (pnl / EQUITY) * weight

    def test_a_typical_trade_scores_what_it_scored_before(self):
        before = self._realized_term(self.OLD_LOTS, self.OLD_WEIGHT)
        after = self._realized_term(self.NEW_LOTS, self.CFG.realized_pnl_weight)
        assert after == pytest.approx(before, rel=0.05)

    def test_the_inactivity_ramp_keeps_its_relative_weight(self):
        """Flat constant against a term that would otherwise grow 13.5x."""
        ramp = self.CFG.inactivity_weight * (
            self.CFG.max_inactivity_bars - self.CFG.inactivity_grace
        )
        before = self._realized_term(self.OLD_LOTS, self.OLD_WEIGHT) / ramp
        after = self._realized_term(self.NEW_LOTS, self.CFG.realized_pnl_weight) / ramp
        assert after == pytest.approx(before, rel=0.05)

    def test_the_trade_cost_keeps_its_relative_weight(self):
        before = self._realized_term(self.OLD_LOTS, self.OLD_WEIGHT)
        after = self._realized_term(self.NEW_LOTS, self.CFG.realized_pnl_weight)
        assert (after / self.CFG.trade_cost_weight) == pytest.approx(
            before / self.CFG.trade_cost_weight, rel=0.05,
        )

    def test_the_clip_stays_out_of_reach(self):
        """Left at 4000 the clip would bite past ~100 pips and truncate the
        outcomes most worth learning from."""
        pips_to_clip = (
            self.CFG.reward_clip / self.CFG.realized_pnl_weight * EQUITY
        ) / (self.NEW_LOTS * CONTRACT * 0.0001)
        assert pips_to_clip > 500

    def test_the_two_pnl_weights_keep_their_ratio(self):
        assert (self.CFG.realized_pnl_weight
                / self.CFG.unrealized_delta_weight) == pytest.approx(8.0)
