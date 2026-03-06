"""Gymnasium environment wrappers for observation processing and trade filtering."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from apexfx.env.trade_filter import FilterDecision, StrategyFilter


class FlattenDictObservation(gym.ObservationWrapper):
    """Flatten Dict observation into a single Box for algorithms that need it."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)

        total_size = 0
        for key, space in env.observation_space.spaces.items():
            assert isinstance(space, spaces.Box), f"Space {key} must be Box"
            total_size += int(np.prod(space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        parts = [observation[key].flatten() for key in sorted(observation.keys())]
        return np.concatenate(parts).astype(np.float32)


class NormalizeReward(gym.Wrapper):
    """Running mean/std reward normalization."""

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self._return = 0.0
        self._ret_rms_mean = 0.0
        self._ret_rms_var = 1.0
        self._count = 0

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._return = self._return * self.gamma + reward
        self._update_stats(self._return)

        std = max(np.sqrt(self._ret_rms_var), self.epsilon)
        normalized_reward = reward / std

        if terminated or truncated:
            self._return = 0.0

        return obs, float(normalized_reward), terminated, truncated, info

    def _update_stats(self, value: float) -> None:
        self._count += 1
        delta = value - self._ret_rms_mean
        self._ret_rms_mean += delta / self._count
        delta2 = value - self._ret_rms_mean
        self._ret_rms_var += (delta * delta2 - self._ret_rms_var) / self._count


class MonitorWrapper(gym.Wrapper):
    """Records episode statistics for logging."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._episode_count: int = 0
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_length = 0
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._episode_return += reward
        self._episode_length += 1

        if terminated or truncated:
            self._episode_count += 1
            self.episode_returns.append(self._episode_return)
            self.episode_lengths.append(self._episode_length)
            info["episode"] = {
                "r": self._episode_return,
                "l": self._episode_length,
                "count": self._episode_count,
            }

        return obs, reward, terminated, truncated, info


class TradeFilterWrapper(gym.Wrapper):
    """Applies StrategyFilter rules during RL training.

    Intercepts the agent's action and applies rule-based filters:
    - Blocks entries during news blackouts
    - Forces exits on conflicting signals
    - Requires minimum fundamental bias / structure confirmation
    - Reduces position when news is approaching

    This teaches the agent that certain actions are forbidden,
    so it learns to avoid those situations naturally during training.
    """

    def __init__(self, env: gym.Env, strategy_filter: StrategyFilter | None = None) -> None:
        super().__init__(env)
        self._filter = strategy_filter or StrategyFilter()
        self._last_obs: dict[str, np.ndarray] | None = None
        self._filter_stats = {
            "total_actions": 0,
            "blocked": 0,
            "force_closed": 0,
            "scaled": 0,
        }

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs if isinstance(obs, dict) else None
        return obs, info

    def step(self, action: np.ndarray | float) -> tuple[Any, float, bool, bool, dict]:
        # Convert to float
        action_val = float(action) if np.isscalar(action) else float(action.item())
        original_action = action_val

        self._filter_stats["total_actions"] += 1

        # Apply filter if we have observation data
        decision = FilterDecision(allowed=True, scale=1.0, reason="", force_close=False)
        if self._last_obs is not None and isinstance(self._last_obs, dict):
            # Get current position from position_state
            current_position = 0.0
            pos_state = self._last_obs.get("position_state")
            if pos_state is not None and len(pos_state) > 0:
                current_position = float(pos_state[0])

            decision = self._filter.check(self._last_obs, action_val, current_position)

            if decision.force_close:
                # Force close: set action to 0 (neutral)
                action_val = 0.0
                self._filter_stats["force_closed"] += 1
            elif not decision.allowed:
                # Block new entry: keep current position unchanged
                # If we have a position, maintain it; if flat, stay flat
                if abs(current_position) > 0.01:
                    # Maintain current direction/size — just pass current position through
                    action_val = current_position
                else:
                    action_val = 0.0
                self._filter_stats["blocked"] += 1
            elif decision.scale < 1.0:
                action_val = action_val * decision.scale
                self._filter_stats["scaled"] += 1

        # Pass modified action to env
        if np.isscalar(action):
            modified_action = np.float32(action_val)
        else:
            modified_action = np.array([action_val], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(modified_action)

        # Track observation for next step
        self._last_obs = obs if isinstance(obs, dict) else None

        # Add filter info to info dict
        if decision.reason:
            info["filter_reason"] = decision.reason
            info["filter_original_action"] = original_action

        return obs, reward, terminated, truncated, info

    @property
    def filter_stats(self) -> dict[str, int]:
        """Return filter statistics for monitoring."""
        return dict(self._filter_stats)
