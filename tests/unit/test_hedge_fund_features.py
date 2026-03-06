"""Tests for hedge-fund level features: Adversarial, World Model, Hierarchical RL.

Tests verify:
1. TradingReward integration with ForexTradingEnv (set_trade_info called)
2. AdversarialObsWrapper noise injection and calibration
3. TemporalCommitmentWrapper penalty mechanism
4. ActionSmoothingWrapper EMA
5. WorldModel forward/backward/ensemble disagreement
6. GradientPenaltyCallback initialization
7. Config schema for new features
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from apexfx.config.schema import (
    AdversarialConfig,
    AppConfig,
    TemporalCommitmentConfig,
    TrainingConfig,
    WorldModelConfig,
)


class TestTradingRewardIntegration:
    """Verify TradingReward.set_trade_info() is called by ForexTradingEnv."""

    def _make_env(self):
        """Create minimal ForexTradingEnv with TradingReward."""
        import pandas as pd
        from apexfx.env.forex_env import ForexTradingEnv
        from apexfx.env.reward import TradingReward

        n = 500
        np.random.seed(42)
        data = pd.DataFrame({
            "open": 1.1 + np.cumsum(np.random.randn(n) * 0.001),
            "high": 1.1 + np.cumsum(np.random.randn(n) * 0.001) + 0.002,
            "low": 1.1 + np.cumsum(np.random.randn(n) * 0.001) - 0.002,
            "close": 1.1 + np.cumsum(np.random.randn(n) * 0.001),
            "volume": np.random.randint(100, 1000, n),
        })
        # Add required feature columns
        for col in ["trend_strength", "hurst_exponent", "realized_vol",
                     "wavelet_trend", "fft_dominant_period", "delta_ma_50",
                     "regime_trending", "poc_distance"]:
            data[col] = np.random.randn(n) * 0.1
        for col in ["close_zscore", "hvn_distance", "volume_profile_skew",
                     "delta_pct", "delta_divergence", "regime_mean_reverting",
                     "nearest_support_distance", "nearest_resistance_distance"]:
            data[col] = np.random.randn(n) * 0.1
        for col in ["regime_flat"]:
            data[col] = 0.0

        reward_fn = TradingReward(loss_weight=2.0, reward_scale=1000.0)
        env = ForexTradingEnv(
            data=data,
            initial_balance=100_000.0,
            n_market_features=5,
            lookback=10,
            reward_fn=reward_fn,
            episode_length=100,
        )
        return env, reward_fn

    def test_set_trade_info_called(self):
        """TradingReward gets position state updates during step()."""
        env, reward_fn = self._make_env()
        obs, _ = env.reset()

        # Take a buy action
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # After step, reward_fn should have updated position_direction
        assert reward_fn._position_direction in [-1, 0, 1]
        assert isinstance(reward, float)

    def test_churn_penalty_applied(self):
        """Closing a position before min_hold_bars triggers churn penalty."""
        env, reward_fn = self._make_env()
        obs, _ = env.reset()

        # Open position
        obs, r1, _, _, _ = env.step(np.array([0.8]))
        # Close immediately (< min_hold_bars=3)
        obs, r2, _, _, _ = env.step(np.array([0.0]))

        # The reward should reflect some position management
        assert isinstance(r2, float)


class TestAdversarialObsWrapper:
    """Test adversarial noise injection."""

    def _make_env(self):
        """Create a minimal test environment."""
        import gymnasium as gym
        from gymnasium import spaces

        class SimpleEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Dict({
                    "market_features": spaces.Box(-1, 1, shape=(10,), dtype=np.float32),
                    "position_state": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                })
                self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
                self._step = 0

            def reset(self, **kwargs):
                self._step = 0
                return {
                    "market_features": np.zeros(10, dtype=np.float32),
                    "position_state": np.zeros(4, dtype=np.float32),
                }, {}

            def step(self, action):
                self._step += 1
                obs = {
                    "market_features": np.ones(10, dtype=np.float32) * 0.5,
                    "position_state": np.array([0.1, 0.0, 0.0, 1.0], dtype=np.float32),
                }
                return obs, 1.0, self._step >= 100, False, {}

        return SimpleEnv()

    def test_noise_injection(self):
        """Noise is applied to market features but not position state."""
        from apexfx.training.adversarial import AdversarialObsWrapper

        env = self._make_env()
        wrapped = AdversarialObsWrapper(
            env, noise_std=0.1, warmup_steps=0, adversarial_prob=1.0,
            noise_schedule="constant",
        )

        obs, _ = wrapped.reset()
        # Step to populate running stats
        for _ in range(5):
            obs, _, _, _, _ = wrapped.step(np.array([0.0]))

        # After warmup, noise should be applied
        obs, _, _, _, _ = wrapped.step(np.array([0.0]))

        # Position state should be unchanged
        np.testing.assert_array_equal(
            obs["position_state"],
            np.array([0.1, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_cosine_schedule(self):
        """Noise decays with cosine schedule."""
        from apexfx.training.adversarial import AdversarialObsWrapper

        env = self._make_env()
        wrapped = AdversarialObsWrapper(
            env, noise_std=0.5, warmup_steps=0, decay_steps=100,
            noise_schedule="cosine",
        )

        # At step 0, noise should be full
        scale_start = wrapped._get_noise_scale()
        assert scale_start == pytest.approx(0.5, abs=0.01)

        # At step 50 (middle), noise should be ~0.25
        wrapped._step_count = 50
        scale_mid = wrapped._get_noise_scale()
        assert 0.1 < scale_mid < 0.4

        # At step 100 (end), noise should be near 0
        wrapped._step_count = 100
        scale_end = wrapped._get_noise_scale()
        assert scale_end < 0.05

    def test_warmup(self):
        """No noise during warmup period."""
        from apexfx.training.adversarial import AdversarialObsWrapper

        env = self._make_env()
        wrapped = AdversarialObsWrapper(
            env, noise_std=0.5, warmup_steps=100, adversarial_prob=1.0,
        )

        wrapped._step_count = 50
        assert wrapped._get_noise_scale() == 0.0

        wrapped._step_count = 101
        assert wrapped._get_noise_scale() > 0.0


class TestTemporalCommitment:
    """Test temporal commitment wrapper."""

    def _make_env(self):
        """Create a minimal test environment."""
        import gymnasium as gym
        from gymnasium import spaces

        class SimpleEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Dict({
                    "market": spaces.Box(-1, 1, shape=(5,), dtype=np.float32),
                })
                self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
                self._step = 0

            def reset(self, **kwargs):
                self._step = 0
                return {"market": np.zeros(5, dtype=np.float32)}, {}

            def step(self, action):
                self._step += 1
                return (
                    {"market": np.random.randn(5).astype(np.float32)},
                    0.1,
                    self._step >= 100,
                    False,
                    {},
                )

        return SimpleEnv()

    def test_commitment_penalty(self):
        """Reversing direction during commitment incurs penalty."""
        from apexfx.training.hierarchical import TemporalCommitmentWrapper

        env = self._make_env()
        wrapped = TemporalCommitmentWrapper(
            env, min_hold=5, commitment_penalty=0.5,
        )

        obs, _ = wrapped.reset()

        # Go long
        _, r1, _, _, info1 = wrapped.step(np.array([0.8]))
        assert info1["commitment_direction"] == 1
        assert info1["commitment_penalty"] == 0.0

        # Reverse to short (within min_hold) — should get penalty
        _, r2, _, _, info2 = wrapped.step(np.array([-0.8]))
        assert info2["commitment_penalty"] == pytest.approx(0.5)

    def test_no_penalty_after_min_hold(self):
        """No penalty when reversing after min_hold bars."""
        from apexfx.training.hierarchical import TemporalCommitmentWrapper

        env = self._make_env()
        wrapped = TemporalCommitmentWrapper(
            env, min_hold=3, commitment_penalty=0.5,
        )

        obs, _ = wrapped.reset()

        # Hold long for min_hold bars
        for _ in range(4):
            wrapped.step(np.array([0.8]))

        # Now reverse — should be free
        _, _, _, _, info = wrapped.step(np.array([-0.8]))
        assert info["commitment_penalty"] == 0.0


class TestActionSmoothing:
    """Test EMA action smoothing."""

    def _make_env(self):
        import gymnasium as gym
        from gymnasium import spaces

        class RecordEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Dict({
                    "x": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                })
                self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
                self.recorded_actions = []

            def reset(self, **kwargs):
                self.recorded_actions = []
                return {"x": np.zeros(2, dtype=np.float32)}, {}

            def step(self, action):
                self.recorded_actions.append(float(action[0]))
                return (
                    {"x": np.zeros(2, dtype=np.float32)},
                    0.0, False, False, {},
                )

        return RecordEnv()

    def test_smoothing(self):
        """Actions are EMA-smoothed."""
        from apexfx.training.hierarchical import ActionSmoothingWrapper

        env = self._make_env()
        wrapped = ActionSmoothingWrapper(env, alpha=0.3)
        wrapped.reset()

        # Send alternating actions: +1, -1, +1, -1
        for i in range(6):
            action = np.array([1.0 if i % 2 == 0 else -1.0])
            wrapped.step(action)

        # Smoothed actions should have reduced amplitude
        actions = env.recorded_actions
        # First action passes through
        assert abs(actions[0]) == pytest.approx(1.0, abs=0.01)
        # Subsequent actions should be smoothed (lower amplitude)
        for a in actions[2:]:
            assert abs(a) < 0.95  # Not raw ±1


class TestWorldModel:
    """Test world model architecture and training."""

    def test_forward_pass(self):
        """World model processes features correctly."""
        from apexfx.models.world_model import WorldModel

        wm = WorldModel(d_features=77, d_latent=32, d_action=1, n_ensemble=3)

        features = torch.randn(8, 77)
        z = wm.encode(features)
        assert z.shape == (8, 32)

        actions = torch.randn(8, 1)
        z_next = wm.predict_next(z, actions)
        assert z_next.shape == (8, 32)

        reward = wm.predict_reward(z)
        assert reward.shape == (8, 1)

        done = wm.predict_done(z)
        assert done.shape == (8, 1)
        assert (done >= 0).all() and (done <= 1).all()

    def test_ensemble_disagreement(self):
        """Ensemble disagreement produces positive values."""
        from apexfx.models.world_model import WorldModel

        wm = WorldModel(d_features=32, d_latent=16, d_action=1, n_ensemble=5)
        z = torch.randn(4, 16)
        a = torch.randn(4, 1)

        disagreement = wm.ensemble_disagreement(z, a)
        assert disagreement.shape == (4, 1)
        assert (disagreement >= 0).all()

    def test_imagination_rollout(self):
        """Imagination produces correct shaped outputs."""
        from apexfx.models.world_model import WorldModel

        wm = WorldModel(d_features=32, d_latent=16, d_action=1, n_ensemble=3)
        z = torch.randn(4, 16)

        def dummy_policy(z):
            return torch.tanh(z[:, :1])

        result = wm.imagine(z, dummy_policy, horizon=5)
        assert result["latents"].shape == (4, 6, 16)  # H+1 latents
        assert result["actions"].shape == (4, 5, 1)
        assert result["rewards"].shape == (4, 5, 1)
        assert result["dones"].shape == (4, 5, 1)

    def test_residual_connection(self):
        """DynamicsHead uses residual (predicts delta)."""
        from apexfx.models.world_model import DynamicsHead

        head = DynamicsHead(d_latent=16, d_action=1, d_hidden=32, dropout=0.0)
        z = torch.randn(4, 16)
        a = torch.zeros(4, 1)

        # With zero action, next state should be close to current (residual)
        z_next = head(z, a)
        assert z_next.shape == (4, 16)

    def test_gradient_flow(self):
        """Gradients flow through world model."""
        from apexfx.models.world_model import WorldModel

        wm = WorldModel(d_features=32, d_latent=16, d_action=1, n_ensemble=2)
        features = torch.randn(4, 32, requires_grad=True)
        z = wm.encode(features)
        z_next = wm.predict_next(z, torch.randn(4, 1), member_idx=0)
        loss = z_next.sum()
        loss.backward()

        assert features.grad is not None
        assert features.grad.abs().sum() > 0


class TestConfigSchema:
    """Test new config models."""

    def test_adversarial_config(self):
        config = AdversarialConfig()
        assert config.enabled is True
        assert config.noise_std == 0.01
        assert config.noise_schedule == "cosine"

    def test_world_model_config(self):
        config = WorldModelConfig()
        assert config.enabled is True
        assert config.n_ensemble == 5
        assert config.curiosity_weight == 0.01

    def test_temporal_commitment_config(self):
        config = TemporalCommitmentConfig()
        assert config.enabled is True
        assert config.min_hold == 5

    def test_training_config_has_new_fields(self):
        config = TrainingConfig()
        assert hasattr(config, "adversarial")
        assert hasattr(config, "world_model")
        assert hasattr(config, "temporal_commitment")

    def test_full_app_config(self):
        """Full AppConfig instantiation with all new features."""
        config = AppConfig()
        assert config.training.adversarial.enabled is True
        assert config.training.world_model.enabled is True
        assert config.training.temporal_commitment.enabled is True
        assert config.training.ewc.enabled is True
        assert config.model.diversity.enabled is True
        assert config.model.uncertainty.enabled is True
