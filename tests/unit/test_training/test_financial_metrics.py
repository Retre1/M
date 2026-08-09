"""Gate 0 — training metrics must measure money, not the shaped reward.

Runs 1-6 were scored by ``mean/std`` over the reward series and a "profit
factor" summed from episode rewards. Both are blind to trading:

* PF read 0.0 in every stage of every run, because ``ep_rew_mean`` was always
  negative and the positive-reward set was therefore empty.
* Retuning reward weights between runs moved "Sharpe" without the policy
  changing, so -147 -> -58 -> -52 compared different scales.

These tests fail if either property comes back.
"""

from __future__ import annotations

import numpy as np
import pytest

from apexfx.training.trainer_v2 import BARS_PER_YEAR_H1, CurriculumV2Callback, _finite


def _make_callback(**kwargs) -> CurriculumV2Callback:
    """A callback detached from SB3, exercising metric collection alone.

    ``model`` is normally injected by ``init_callback``; ``_read_entropy``
    reads it and falls back to a neutral value when it is None.
    """
    cb = CurriculumV2Callback(curriculum=None, stage_idx=0, **kwargs)
    cb.model = None
    return cb


def _feed(cb: CurriculumV2Callback, *, bar_returns, trade_returns, episode_reward):
    """Push one finished episode through the callback's collectors."""
    cb._episode_rewards.append(episode_reward)
    cb._bar_returns.append(np.asarray(bar_returns, dtype=np.float64))
    cb._trade_returns.append(np.asarray(trade_returns, dtype=np.float64))
    cb._episode_trade_counts.append(len(trade_returns))


class TestProfitFactorReflectsTrading:
    def test_losing_but_trading_agent_gets_pf_between_zero_and_one(self):
        """The exact case runs 1-6 could not express.

        A negative episode reward with real winning and losing trades must give
        0 < PF < 1. The old implementation returned exactly 0.0 here.
        """
        cb = _make_callback()
        _feed(
            cb,
            bar_returns=[0.001, -0.002, 0.0015, -0.003],
            trade_returns=[0.01, -0.02, 0.015, -0.03],
            episode_reward=-250.0,
        )
        pf = cb.get_metrics()["profit_factor"]
        assert 0.0 < pf < 1.0, f"expected a losing-but-trading PF, got {pf}"

    def test_profitable_agent_gets_pf_above_one(self):
        cb = _make_callback()
        _feed(
            cb,
            bar_returns=[0.004, -0.001, 0.003],
            trade_returns=[0.04, -0.01, 0.03],
            episode_reward=-10.0,  # reward can still be negative — PF must not care
        )
        assert cb.get_metrics()["profit_factor"] > 1.0

    def test_pf_is_zero_only_when_no_trade_won(self):
        cb = _make_callback()
        _feed(
            cb,
            bar_returns=[-0.001, -0.002],
            trade_returns=[-0.01, -0.02],
            episode_reward=-100.0,
        )
        assert cb.get_metrics()["profit_factor"] == pytest.approx(0.0)

    def test_negative_episode_rewards_do_not_force_pf_to_zero(self):
        """Guards the specific defect: every episode reward negative, PF > 0."""
        cb = _make_callback()
        for _ in range(5):
            _feed(
                cb,
                bar_returns=[0.002, -0.001],
                trade_returns=[0.02, -0.01],
                episode_reward=-2646.0,  # the value runs 1-3 actually reported
            )
        metrics = cb.get_metrics()
        assert metrics["ep_rew_mean"] < 0
        assert metrics["profit_factor"] > 0.0


class TestSharpeIsFinancial:
    def test_sharpe_ignores_reward_scale(self):
        """Rescaling the reward must not move the financial Sharpe.

        Between runs the reward weights changed by factors of 4 and 5. If
        Sharpe still tracked rewards, those retunes would masquerade as
        learning progress.
        """
        bars = [0.001, -0.0005, 0.002, -0.001, 0.0015]
        trades = [0.01, -0.005]

        small = _make_callback()
        _feed(small, bar_returns=bars, trade_returns=trades, episode_reward=-80.0)

        large = _make_callback()
        _feed(large, bar_returns=bars, trade_returns=trades, episode_reward=-2813.0)

        assert small.get_metrics()["sharpe"] == pytest.approx(
            large.get_metrics()["sharpe"],
        )

    def test_sharpe_tracks_returns(self):
        """Same volatility, opposite drift — Sharpe must order them.

        The series must actually vary: a constant one has zero standard
        deviation and an undefined Sharpe, which ``sharpe_ratio`` reports as
        0.0 for both sides.
        """
        shape = [0.001, 0.003, 0.001, 0.003]

        winner = _make_callback()
        _feed(winner, bar_returns=shape, trade_returns=[0.02], episode_reward=0.0)

        loser = _make_callback()
        _feed(
            loser,
            bar_returns=[-r for r in shape],
            trade_returns=[-0.02],
            episode_reward=0.0,
        )

        assert winner.get_metrics()["sharpe"] > 0 > loser.get_metrics()["sharpe"]

    def test_sharpe_magnitude_is_plausible(self):
        """A financial Sharpe lives in single digits, not the hundred-thousands.

        The runs reported values around -700000; anything of that order means
        the reward series is being measured again.
        """
        cb = _make_callback()
        rng = np.random.default_rng(0)
        _feed(
            cb,
            bar_returns=rng.normal(0.0001, 0.002, 500),
            trade_returns=rng.normal(0.0, 0.01, 30),
            episode_reward=-500.0,
        )
        assert abs(cb.get_metrics()["sharpe"]) < 100

    def test_reward_sharpe_is_kept_under_its_own_name(self):
        """Old logs stay interpretable, but the name says what it measures."""
        cb = _make_callback()
        for reward in (-2813.0, -2646.0, -2700.0):
            _feed(cb, bar_returns=[0.001], trade_returns=[0.01], episode_reward=reward)
        metrics = cb.get_metrics()
        assert "reward_sharpe" in metrics
        assert metrics["reward_sharpe"] != metrics["sharpe"]


class TestTradeActivityIsVisible:
    def test_trade_counts_are_reported(self):
        cb = _make_callback()
        _feed(cb, bar_returns=[0.001], trade_returns=[0.01] * 7, episode_reward=-5.0)
        _feed(cb, bar_returns=[0.001], trade_returns=[0.01] * 3, episode_reward=-5.0)
        metrics = cb.get_metrics()
        assert metrics["n_trades"] == 10
        assert metrics["trades_per_episode"] == pytest.approx(5.0)

    def test_flat_agent_is_distinguishable_from_a_trading_one(self):
        """Run 1's agent never traded; nothing in its metrics said so."""
        flat = _make_callback()
        _feed(flat, bar_returns=[0.0] * 50, trade_returns=[], episode_reward=-2646.0)
        assert flat.get_metrics()["n_trades"] == 0


class TestEmptyAndDegenerateInputs:
    def test_no_episodes_yet_does_not_raise(self):
        assert _make_callback().get_metrics()["sharpe"] == pytest.approx(0.0)

    def test_metrics_are_all_finite(self):
        """inf leaking into early-stop deltas or TensorBoard is a defect."""
        cb = _make_callback()
        _feed(cb, bar_returns=[0.001] * 5, trade_returns=[0.01, 0.02], episode_reward=1.0)
        for key, value in cb.get_metrics().items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_finite_helper_maps_infinities(self):
        assert _finite(float("inf")) == 999.0
        assert _finite(float("-inf")) == -999.0
        assert _finite(float("nan")) == 0.0


class TestAnnualisation:
    def test_h1_bars_use_hourly_annualisation(self):
        """252 is the daily factor; H1 bars need 24x that."""
        assert BARS_PER_YEAR_H1 == 24 * 252

    def test_periods_per_year_changes_the_scale(self):
        bars = [0.001, -0.0005, 0.002, -0.001]
        trades = [0.01]

        hourly = _make_callback(periods_per_year=BARS_PER_YEAR_H1)
        _feed(hourly, bar_returns=bars, trade_returns=trades, episode_reward=0.0)

        daily = _make_callback(periods_per_year=252)
        _feed(daily, bar_returns=bars, trade_returns=trades, episode_reward=0.0)

        ratio = hourly.get_metrics()["sharpe"] / daily.get_metrics()["sharpe"]
        assert ratio == pytest.approx(np.sqrt(24), rel=1e-6)
