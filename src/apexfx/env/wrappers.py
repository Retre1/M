"""Gymnasium environment wrappers for observation processing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
