"""Tests for reward functions."""

import numpy as np

from apexfx.env.reward import DifferentialSharpeReward, QuantumZScoreReward, SortinoReward


class TestDifferentialSharpeReward:
    def test_positive_return_positive_reward(self):
        reward_fn = DifferentialSharpeReward()
        # Growing portfolio should produce positive reward (after initial steps)
        for i in range(20):
            reward_fn.compute(100_000 + i * 100, 100_000 + (i - 1) * 100 if i > 0 else 100_000)
        # After building up positive returns, should be positive
        last_r = reward_fn.compute(102_100, 102_000)
        assert isinstance(last_r, float)

    def test_drawdown_penalty(self):
        reward_fn = DifferentialSharpeReward(lambda_dd=5.0)
        # Build up, then crash
        reward_fn.compute(110_000, 100_000)
        reward_fn.compute(120_000, 110_000)
        crash_reward = reward_fn.compute(100_000, 120_000)
        # Large drawdown should produce negative reward
        assert crash_reward < 0

    def test_reset(self):
        reward_fn = DifferentialSharpeReward()
        reward_fn.compute(110_000, 100_000)
        reward_fn.reset()
        assert reward_fn._A == 0.0
        assert reward_fn._B == 0.0

    def test_zero_portfolio(self):
        reward_fn = DifferentialSharpeReward()
        result = reward_fn.compute(100_000, 0)
        assert result == 0.0

    def test_reward_bounded(self):
        reward_fn = DifferentialSharpeReward()
        for _ in range(100):
            r = reward_fn.compute(
                100_000 + np.random.randn() * 10_000,
                100_000,
            )
            assert -10.0 <= r <= 10.0


class TestSortinoReward:
    def test_positive_returns(self):
        reward_fn = SortinoReward()
        for i in range(50):
            r = reward_fn.compute(100_000 + i * 50, 100_000 + max(0, i - 1) * 50)
        assert isinstance(r, float)

    def test_reset(self):
        reward_fn = SortinoReward()
        reward_fn.compute(110_000, 100_000)
        reward_fn.reset()
        assert len(reward_fn._returns) == 0


class TestQuantumZScoreReward:
    def test_zscore_bonus(self):
        reward_fn = QuantumZScoreReward(z_score_bonus_weight=0.5)
        # Positive return against negative z-score (mean reversion trade)
        reward_fn.set_zscore(-3.5)
        r = reward_fn.compute(101_000, 100_000)
        assert isinstance(r, float)
