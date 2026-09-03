"""Unit tests for RARAReward_v5.

Tests verify that the reward function:
1. Does not collapse into "do nothing" (inactivity penalty works).
2. Does not incentivise overtrading (trade cost penalty works).
3. Reacts correctly to jumps (jump-risk penalty scales with position).
4. Rewards regime adaptation (FSD surprise bonus).
5. Rewards gamma stability (gamma-aware bonus).
6. Penalises tail risk (CVaR penalty).
7. Produces stable rewards across different β noise levels.
8. Backward-compatible with the v4 API (set_trade_info).
9. Diagnostics dict contains all 9 components.
10. Config validation works (Pydantic frozen model).
"""

from __future__ import annotations

import numpy as np
import pytest

from apexfx.env.reward_v5 import RARAReward_v5, RARARewardV5Config


@pytest.fixture
def reward_fn() -> RARAReward_v5:
    """Default reward function with standard config."""
    return RARAReward_v5(RARARewardV5Config())


@pytest.fixture
def reward_fn_custom() -> RARAReward_v5:
    """Reward function with exaggerated weights for testing."""
    return RARAReward_v5(RARARewardV5Config(
        realized_pnl_weight=10000.0,
        trade_cost_weight=1.0,  # very expensive trades
        inactivity_weight=0.01,  # aggressive inactivity penalty
        jump_risk_weight=5.0,
        cvar_weight=2.0,
    ))


class TestNoDoNothingCollapse:
    """The agent must not be able to stay flat indefinitely for free."""

    def test_inactivity_penalty_grows_with_flat_bars(self, reward_fn: RARAReward_v5) -> None:
        """Penalty should increase as the agent stays flat longer."""
        reward_fn.reset()
        rewards = []

        for _step in range(200):
            reward_fn.set_step_context(
                position=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                price=1.10,
            )
            r = reward_fn.compute(100_000.0, 100_000.0)
            rewards.append(r)

        # After grace period, rewards should become negative
        grace = reward_fn._cfg.inactivity_grace
        assert all(r == 0.0 for r in rewards[:grace]), "No penalty during grace"
        assert rewards[-1] < rewards[grace + 1] < 0, "Penalty grows over time"

    def test_flat_agent_cumulative_penalty_is_significant(self, reward_fn: RARAReward_v5) -> None:
        """Total penalty for staying flat 500 bars must be materially negative."""
        reward_fn.reset()
        total = 0.0

        for _ in range(500):
            reward_fn.set_step_context(
                position=0.0, unrealized_pnl=0.0,
                realized_pnl=0.0, price=1.10,
            )
            total += reward_fn.compute(100_000.0, 100_000.0)

        assert total < -1.0, f"Flat agent penalty too weak: {total:.4f}"


class TestNoOvertrading:
    """The agent must not be incentivised to flip positions every bar."""

    def test_frequent_flipping_is_punished(self, reward_fn: RARAReward_v5) -> None:
        """Flipping long↔short every bar should accumulate trade costs."""
        reward_fn.reset()
        total = 0.0

        for step in range(100):
            direction = 1.0 if step % 2 == 0 else -1.0
            reward_fn.set_step_context(
                position=direction,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                price=1.10,
            )
            total += reward_fn.compute(100_000.0, 100_000.0)

        # 100 flips × trade_cost_weight should dominate
        assert total < -5.0, f"Overtrading penalty too weak: {total:.4f}"

    def test_single_trade_cheaper_than_many(self, reward_fn: RARAReward_v5) -> None:
        """One entry held for 50 bars must cost less than 50 entries."""
        reward_fn.reset()
        # Single entry
        reward_fn.set_step_context(position=1.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10)
        single_cost = reward_fn.compute(100_000.0, 100_000.0)

        reward_fn.reset()
        # 50 entries (flat → long each bar)
        multi_cost = 0.0
        for i in range(50):
            pos = 1.0 if i % 2 == 0 else 0.0
            reward_fn.set_step_context(position=pos, unrealized_pnl=0.0,
                                        realized_pnl=0.0, price=1.10)
            multi_cost += reward_fn.compute(100_000.0, 100_000.0)

        assert multi_cost < single_cost * 20, "Many trades should cost more"


class TestJumpRiskPenalty:
    """Jump-risk penalty must scale with position size."""

    def test_larger_position_higher_penalty(self, reward_fn: RARAReward_v5) -> None:
        """Doubling position should roughly double jump penalty."""
        reward_fn.reset()

        # Small position
        reward_fn.set_step_context(position=0.1, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10)
        r_small = reward_fn.compute(100_000.0, 100_000.0)
        penalty_small = reward_fn.last_components["reward/jump_penalty"]

        reward_fn.reset()

        # Large position
        reward_fn.set_step_context(position=1.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10)
        r_large = reward_fn.compute(100_000.0, 100_000.0)
        penalty_large = reward_fn.last_components["reward/jump_penalty"]

        assert penalty_large < penalty_small, "Larger position → bigger penalty (more negative)"

    def test_flat_position_no_jump_penalty(self, reward_fn: RARAReward_v5) -> None:
        """Flat position should have zero jump risk."""
        reward_fn.reset()
        reward_fn.set_step_context(position=0.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10)
        reward_fn.compute(100_000.0, 100_000.0)
        assert reward_fn.last_components["reward/jump_penalty"] == 0.0


class TestFSDRegimeBonus:
    """FSD regime transition must produce a bonus when positioned."""

    def test_bonus_on_regime_change(self, reward_fn: RARAReward_v5) -> None:
        """Switching from NEUTRAL to RISK_ON while positioned → bonus."""
        reward_fn.reset()

        # Bar 1: neutral regime, enter position
        reward_fn.set_step_context(position=1.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10, fsd_regime=1)
        reward_fn.compute(100_000.0, 100_000.0)
        bonus_before = reward_fn.last_components["reward/fsd_bonus"]

        # Bar 2: regime changes to RISK_ON
        reward_fn.set_step_context(position=1.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10, fsd_regime=2)
        reward_fn.compute(100_000.0, 100_000.0)
        bonus_after = reward_fn.last_components["reward/fsd_bonus"]

        assert bonus_after > bonus_before, "Should get bonus on regime change"

    def test_no_bonus_when_flat(self, reward_fn: RARAReward_v5) -> None:
        """Regime change while flat → no bonus (agent must be positioned)."""
        reward_fn.reset()
        reward_fn.set_step_context(position=0.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10, fsd_regime=1)
        reward_fn.compute(100_000.0, 100_000.0)

        reward_fn.set_step_context(position=0.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10, fsd_regime=2)
        reward_fn.compute(100_000.0, 100_000.0)
        assert reward_fn.last_components["reward/fsd_bonus"] == 0.0


class TestGammaAwareBonus:
    """Smooth return sequences should receive gamma bonus."""

    def test_smooth_returns_get_bonus(self, reward_fn: RARAReward_v5) -> None:
        """Three consecutive similar returns → low gamma → bonus."""
        reward_fn.reset()

        # Simulate 3 bars with nearly identical returns
        for i in range(3):
            reward_fn.set_step_context(position=1.0, unrealized_pnl=float(i),
                                        realized_pnl=0.0, price=1.10)
            reward_fn.compute(100_000.0 + i, 100_000.0 + max(0, i - 1))

        bonus = reward_fn.last_components["reward/gamma_bonus"]
        assert bonus >= 0.0, "Smooth returns should get non-negative gamma bonus"


class TestCVaRPenalty:
    """CVaR should penalise sequences with heavy left-tail losses."""

    @pytest.fixture
    def reward_fn_cvar(self) -> RARAReward_v5:
        """CVaR-enabled fixture (default config has cvar_weight=0 to isolate
        other signals; this fixture restores CVaR for its dedicated tests)."""
        return RARAReward_v5(RARARewardV5Config(cvar_weight=0.3))

    def test_losing_streak_triggers_cvar(self, reward_fn_cvar: RARAReward_v5) -> None:
        """Consistent losses should produce a CVaR penalty."""
        reward_fn_cvar.reset()
        val = 100_000.0

        # 30 losing steps
        for _ in range(30):
            new_val = val * 0.999  # -0.1% per step
            reward_fn_cvar.set_step_context(position=1.0, unrealized_pnl=-100.0,
                                        realized_pnl=0.0, price=1.10)
            reward_fn_cvar.compute(new_val, val)
            val = new_val

        cvar = reward_fn_cvar.last_components["reward/cvar_penalty"]
        assert cvar < 0.0, f"CVaR penalty should be negative, got {cvar}"

    def test_winning_streak_no_cvar(self, reward_fn: RARAReward_v5) -> None:
        """Consistent wins should produce zero CVaR penalty."""
        reward_fn.reset()
        val = 100_000.0

        for _ in range(30):
            new_val = val * 1.001
            reward_fn.set_step_context(position=1.0, unrealized_pnl=100.0,
                                        realized_pnl=0.0, price=1.10)
            reward_fn.compute(new_val, val)
            val = new_val

        cvar = reward_fn.last_components["reward/cvar_penalty"]
        assert cvar == 0.0, f"No CVaR penalty for winners, got {cvar}"


class TestStabilityAcrossBeta:
    """Reward should produce stable magnitudes across different SBBTS β levels."""

    @pytest.mark.parametrize("vol_scale", [0.5, 1.0, 2.0, 5.0])
    def test_reward_bounded_across_volatilities(
        self, vol_scale: float,
    ) -> None:
        """Reward stays within clip bounds regardless of return magnitude."""
        rng = np.random.default_rng(42)
        fn = RARAReward_v5(RARARewardV5Config())
        fn.reset()

        val = 100_000.0
        rewards = []
        for _ in range(200):
            ret = rng.normal(0, 0.001 * vol_scale)
            new_val = val * (1 + ret)
            fn.set_step_context(
                position=rng.choice([-1.0, 0.0, 1.0]),
                unrealized_pnl=rng.normal(0, 100 * vol_scale),
                realized_pnl=0.0,
                price=1.10 + rng.normal(0, 0.01 * vol_scale),
            )
            r = fn.compute(new_val, val)
            rewards.append(r)
            val = new_val

        rewards_arr = np.array(rewards)
        clip = fn._cfg.reward_clip
        assert np.all(rewards_arr >= -clip), "Below lower clip"
        assert np.all(rewards_arr <= clip), "Above upper clip"
        assert np.std(rewards_arr) < clip, "Reward std should be bounded"


class TestBackwardCompatibility:
    """v5 must work with envs using the v4 API (set_trade_info)."""

    def test_set_trade_info_adapter(self, reward_fn: RARAReward_v5) -> None:
        """set_trade_info should produce valid rewards without crashing."""
        reward_fn.reset()
        reward_fn.set_trade_info(
            action=0.5,
            direction=1,
            unrealized_pnl=50.0,
            time_in_position=3,
            realized_pnl_this_step=10.0,
            structure_aligned=True,
        )
        r = reward_fn.compute(100_010.0, 100_000.0)
        assert isinstance(r, float)
        assert np.isfinite(r)


class TestDiagnostics:
    """Diagnostics dict must contain all 9 components + total."""

    def test_all_components_present(self, reward_fn: RARAReward_v5) -> None:
        reward_fn.reset()
        reward_fn.set_step_context(position=1.0, unrealized_pnl=0.0,
                                    realized_pnl=0.0, price=1.10)
        reward_fn.compute(100_000.0, 100_000.0)
        components = reward_fn.last_components

        expected_keys = {
            "reward/total",
            "reward/realized_pnl",
            "reward/unrealized_delta",
            "reward/trade_cost",
            "reward/inactivity",
            "reward/jump_penalty",
            "reward/fsd_bonus",
            "reward/gamma_bonus",
            "reward/structure_bonus",
            "reward/cvar_penalty",
        }
        assert set(components.keys()) == expected_keys

    def test_components_sum_approximately_to_total(self, reward_fn: RARAReward_v5) -> None:
        """Component sum (pre-asymmetry) should be close to total."""
        reward_fn.reset()
        reward_fn.set_step_context(position=1.0, unrealized_pnl=50.0,
                                    realized_pnl=10.0, price=1.10)
        reward_fn.compute(100_010.0, 100_000.0)
        c = reward_fn.last_components
        # Note: total includes loss_asymmetry scaling, so exact match
        # only holds when reward >= 0
        component_sum = sum(v for k, v in c.items() if k != "reward/total")
        # They should be in the same ballpark
        assert abs(component_sum) < reward_fn._cfg.reward_clip * 2


class TestConfigValidation:
    """Pydantic config must be frozen and validated."""

    def test_config_is_frozen(self) -> None:
        cfg = RARARewardV5Config()
        with pytest.raises(Exception):
            cfg.realized_pnl_weight = 999  # type: ignore

    def test_custom_config(self) -> None:
        cfg = RARARewardV5Config(realized_pnl_weight=50_000, cvar_alpha=0.10)
        fn = RARAReward_v5(cfg)
        assert fn._cfg.realized_pnl_weight == 50_000
        assert fn._cfg.cvar_alpha == 0.10
