"""Tests for StrategyFilter.training_mode.

Regression tests for AUDIT_REPORT.md finding #2: on synthetic training data
the fundamental / structure features are zero or noise, so rules 4 (min
bias), 5 (structure confirm) and 6 (direction alignment) blocked ~100% of
entries and collapsed the policy (Stage 3 ``profit_factor=0.0`` in logs).

In ``training_mode``:
    - Rules 4, 5, 6 are bypassed (new entries no longer blocked by these).
    - Rules 1 (news blackout), 2 (event imminent), 3 (conflicting_signals
      force-close) and 7 (pre-news scaling) remain active.

At inference time ``training_mode`` must default to False.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from apexfx.env.trade_filter import StrategyFilter
from apexfx.env.wrappers import TradeFilterWrapper


def _obs(
    *,
    news: float = 0.0,
    time_to_event: float = 1.0,
    bias: float = 0.0,
    conflicting: float = 0.0,
    break_bull: float = 0.0,
    break_bear: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build an observation dict with the fields the filter reads."""
    fund = np.zeros(10, dtype=np.float64)
    fund[StrategyFilter._IDX_NEWS_IMPACT_ACTIVE] = news
    fund[StrategyFilter._IDX_TIME_TO_NEXT_EVENT] = time_to_event
    fund[StrategyFilter._IDX_FUNDAMENTAL_BIAS] = bias
    fund[StrategyFilter._IDX_CONFLICTING_SIGNALS] = conflicting

    struct = np.zeros(10, dtype=np.float64)
    struct[StrategyFilter._IDX_STRUCTURE_BREAK_BULL] = break_bull
    struct[StrategyFilter._IDX_STRUCTURE_BREAK_BEAR] = break_bear

    return {"fundamental_features": fund, "structure_features": struct}


class TestDefaultStrict:
    """Live-mode (default) must keep every rule active."""

    def test_default_is_strict(self) -> None:
        f = StrategyFilter()
        assert f._training_mode is False

    def test_strict_blocks_insufficient_bias(self) -> None:
        """Rule 4 fires when bias is below threshold and we try to enter."""
        f = StrategyFilter()
        obs = _obs(bias=0.0)  # no bias
        decision = f.check(obs, proposed_action=0.5, current_position=0.0)
        assert decision.allowed is False
        assert decision.reason == "insufficient_bias"

    def test_strict_blocks_no_structure(self) -> None:
        """Rule 5 fires: bias present but no structure break."""
        f = StrategyFilter()
        obs = _obs(bias=0.4, break_bull=0.0, break_bear=0.0)
        decision = f.check(obs, proposed_action=0.5, current_position=0.0)
        assert decision.allowed is False
        assert decision.reason == "no_structure_confirm"

    def test_strict_blocks_against_bias(self) -> None:
        """Rule 6 fires: long action against strong short bias."""
        f = StrategyFilter()
        # Strong bias + matching bullish structure, but proposed short.
        obs = _obs(bias=-0.8, break_bull=1.0)
        decision = f.check(obs, proposed_action=0.5, current_position=0.0)
        # Structure is bullish and the action is long, so rule 5 passes.
        # But bias is strongly negative so rule 6 should block.
        assert decision.allowed is False
        # Either direction-mismatch (rule 5) or against-bias (rule 6) may fire;
        # the point is that strict mode BLOCKS on these synthetic features.
        assert decision.reason in {"structure_direction_mismatch", "against_fundamental_bias"}


class TestTrainingMode:
    """In training_mode rules 4/5/6 are bypassed."""

    def test_training_mode_allows_no_bias(self) -> None:
        f = StrategyFilter(training_mode=True)
        obs = _obs(bias=0.0)
        decision = f.check(obs, proposed_action=0.5, current_position=0.0)
        assert decision.allowed is True
        assert decision.reason == ""

    def test_training_mode_allows_no_structure(self) -> None:
        f = StrategyFilter(training_mode=True)
        obs = _obs(bias=0.0, break_bull=0.0, break_bear=0.0)
        decision = f.check(obs, proposed_action=-0.5, current_position=0.0)
        assert decision.allowed is True

    def test_training_mode_allows_against_bias(self) -> None:
        f = StrategyFilter(training_mode=True)
        obs = _obs(bias=-0.9)  # strong short bias
        decision = f.check(obs, proposed_action=0.7, current_position=0.0)  # long
        assert decision.allowed is True

    # --- Safety rules must stay active even in training_mode ---

    def test_training_mode_still_blocks_news(self) -> None:
        """Rule 2: news blackout remains enforced."""
        f = StrategyFilter(training_mode=True)
        obs = _obs(news=1.0)
        decision = f.check(obs, proposed_action=0.5, current_position=0.0)
        assert decision.allowed is False
        assert decision.reason == "news_blackout"

    def test_training_mode_still_blocks_event_imminent(self) -> None:
        """Rule 3: imminent event remains enforced."""
        f = StrategyFilter(training_mode=True)
        obs = _obs(time_to_event=0.001)  # below threshold
        decision = f.check(obs, proposed_action=0.5, current_position=0.0)
        assert decision.allowed is False
        assert decision.reason == "event_imminent"

    def test_training_mode_still_force_closes_on_conflict(self) -> None:
        """Rule 1: conflicting signals still force-close an open position."""
        f = StrategyFilter(training_mode=True)
        obs = _obs(conflicting=1.0)
        decision = f.check(obs, proposed_action=0.0, current_position=0.5)
        assert decision.force_close is True

    def test_training_mode_allows_majority_of_random_entries(self) -> None:
        """Smoke: on random-ish synthetic obs, training_mode should let most actions through."""
        rng = np.random.RandomState(0)
        f = StrategyFilter(training_mode=True)
        allowed = 0
        n = 200
        for _ in range(n):
            obs = _obs(
                news=float(rng.uniform(0.0, 0.3)),     # below blackout
                time_to_event=float(rng.uniform(0.2, 1.0)),  # not imminent
                bias=float(rng.uniform(-0.1, 0.1)),     # below min threshold
                conflicting=0.0,
                break_bull=0.0,
                break_bear=0.0,
            )
            decision = f.check(obs, proposed_action=float(rng.uniform(-1, 1)), current_position=0.0)
            if decision.allowed:
                allowed += 1
        # Strict mode would reject ~all of these; training_mode should accept
        # essentially every one since the blocking rules are rules 4/5/6.
        assert allowed / n >= 0.95


class TestWrapperPlumbing:
    """TradeFilterWrapper must plumb training_mode through to the filter."""

    def test_wrapper_default_filter_strict(self) -> None:
        # Any tiny env works — we only check the filter attribute.
        env = gym.make("CartPole-v1")
        wrapper = TradeFilterWrapper(env)
        assert wrapper._filter._training_mode is False
        env.close()

    def test_wrapper_training_mode_flag(self) -> None:
        env = gym.make("CartPole-v1")
        wrapper = TradeFilterWrapper(env, training_mode=True)
        assert wrapper._filter._training_mode is True
        env.close()

    def test_wrapper_override_wins(self) -> None:
        """Explicit wrapper-level training_mode overrides a pre-built filter."""
        env = gym.make("CartPole-v1")
        f = StrategyFilter(training_mode=False)
        wrapper = TradeFilterWrapper(env, strategy_filter=f, training_mode=True)
        assert wrapper._filter._training_mode is True
        env.close()
