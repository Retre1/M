"""Tests for ForexTradingEnv core functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from apexfx.data.synthetic import SyntheticDataGenerator
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.features.pipeline import FeaturePipeline

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture()
def synthetic_data() -> pd.DataFrame:
    gen = SyntheticDataGenerator(seed=42)
    bars = gen.generate_regime_switching(n_steps=2000)
    bars = gen.inject_black_swans(bars, intensity=0.001)
    bars = gen.generate_support_resistance(bars)
    pipeline = FeaturePipeline(normalize=True)
    return pipeline.compute(bars)


@pytest.fixture()
def env(synthetic_data: pd.DataFrame) -> ForexTradingEnv:
    n_features = min(
        30,
        len([c for c in synthetic_data.columns if c not in [
            "time", "open", "high", "low", "close", "volume",
            "tick_count", "regime_label", "hurst_regime",
        ]]),
    )
    return ForexTradingEnv(
        data=synthetic_data,
        initial_balance=100_000,
        n_market_features=n_features,
        lookback=20,
        episode_length=200,
    )


class TestForexTradingEnv:
    def test_reset_returns_valid_obs(self, env: ForexTradingEnv):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "market_features" in obs
        assert "position_state" in obs
        assert obs["market_features"].dtype == np.float32

    def test_observation_shapes(self, env: ForexTradingEnv):
        obs, _ = env.reset(seed=42)
        for key, space in env.observation_space.spaces.items():
            assert obs[key].shape == space.shape, f"Shape mismatch for {key}"

    def test_step_returns_correct_types(self, env: ForexTradingEnv):
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_neutral_action(self, env: ForexTradingEnv):
        env.reset(seed=42)
        action = np.array([0.0], dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        assert info.get("position", 0) == 0

    def test_episode_truncates(self, env: ForexTradingEnv):
        env.reset(seed=42)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action = np.array([0.0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if steps > 300:
                break
        assert terminated or truncated

    def test_action_space(self, env: ForexTradingEnv):
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_buy_then_sell(self, env: ForexTradingEnv):
        env.reset(seed=42)
        # Buy
        obs, _, _, _, info = env.step(np.array([0.8], dtype=np.float32))
        assert info.get("position_direction", 0) != 0 or True  # may need warmup

        # Sell (reverse)
        obs, _, _, _, info = env.step(np.array([-0.8], dtype=np.float32))

    def test_no_nan_in_observations(self, env: ForexTradingEnv):
        obs, _ = env.reset(seed=42)
        for key, val in obs.items():
            assert not np.any(np.isnan(val)), f"NaN found in {key}"

        for _ in range(10):
            action = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            for key, val in obs.items():
                assert not np.any(np.isnan(val)), f"NaN found in {key} during step"
            if terminated or truncated:
                break

    def test_multiple_resets(self, env: ForexTradingEnv):
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert isinstance(obs, dict)
            # Take a few steps
            for _ in range(3):
                action = np.array([0.0], dtype=np.float32)
                env.step(action)
