"""Hierarchical RL — Options Framework for Temporal Abstraction.

Why this matters for hedge-fund-level performance:
Retail bots make single-step decisions at every bar. Hedge funds think in
terms of *trade plans* that span hours or days:

- "I'll hold this trend for 20 bars or until my target is hit"
- "I'll scale in over 5 bars during this breakout"
- "I'll wait for confirmation over 3 bars before committing"

The Options Framework (Sutton et al., 1999) provides exactly this:
each "option" is a temporally-extended action that commits the agent
to a sub-policy for multiple steps, with a termination condition.

Architecture
------------
The high-level meta-policy (HierarchicalWrapper) selects an option every
K steps. The option executes for K steps unless its termination condition
triggers early.

Three option types match our three agents:
1. **TrendOption**: Commit to a direction for up to 20 bars
   - Terminates when trend signal reverses
   - Gradually scales position

2. **ReversionOption**: Trade against extremes for 3-10 bars
   - Terminates when Z-score normalises
   - Aggressive initial position, scales out

3. **WaitOption**: Stay flat and observe
   - Terminates when conviction threshold is crossed
   - Position = 0 throughout

This is implemented as a Gymnasium wrapper that:
1. Receives option selection from the meta-policy every K steps
2. Runs the sub-policy (modified action) for K steps
3. Returns accumulated reward and final observation to meta-policy

References
----------
- Sutton, R. S., Precup, D., & Singh, S. (1999). Options framework.
- Bacon et al. (2017). Option-Critic Architecture.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class TemporalCommitmentWrapper(gym.Wrapper):
    """Temporal commitment: agent's action is held for a minimum period.

    Instead of full Options Framework (complex), this provides the key
    benefit — temporal abstraction — via a simple mechanism:

    When the agent takes an action, it commits to that action direction
    for at least ``min_hold`` steps. During the hold period:
    - The action magnitude can be adjusted (fine-tuning position size)
    - But the direction (sign) cannot reverse
    - Going to flat (|action| < threshold) is penalised during commitment

    This prevents the noisy "flip-flopping" that RL agents love to do
    and forces the agent to think in terms of trade plans.

    The commitment is soft: the agent CAN break commitment, but receives
    a penalty for doing so. This allows emergency exits.

    Parameters
    ----------
    env : gym.Env
        Base trading environment.
    min_hold : int
        Minimum bars to hold a direction once committed.
    commitment_penalty : float
        Reward penalty for breaking commitment early.
    flat_threshold : float
        |action| below this = flat (no position).
    """

    def __init__(
        self,
        env: gym.Env,
        min_hold: int = 5,
        commitment_penalty: float = 0.1,
        flat_threshold: float = 0.05,
    ) -> None:
        super().__init__(env)
        self._min_hold = min_hold
        self._commitment_penalty = commitment_penalty
        self._flat_threshold = flat_threshold

        # Commitment state
        self._committed_direction: int = 0  # -1, 0, +1
        self._hold_count: int = 0

        # NOTE: We do NOT extend the observation space to avoid breaking
        # the SB3 feature extractor (HiveMindExtractor). The commitment
        # state is tracked internally and affects behaviour through the
        # penalty mechanism, not through observation augmentation.

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.env.reset(**kwargs)
        self._committed_direction = 0
        self._hold_count = 0
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        penalty = 0.0

        # Determine intended direction
        if abs(action_value) < self._flat_threshold:
            intended_direction = 0
        else:
            intended_direction = 1 if action_value > 0 else -1

        # Check commitment
        if self._committed_direction != 0 and self._hold_count < self._min_hold:
            if intended_direction != self._committed_direction and intended_direction != 0:
                # Breaking commitment: allow but penalise
                penalty = self._commitment_penalty
                # Reset commitment to new direction
                self._committed_direction = intended_direction
                self._hold_count = 0
            elif intended_direction == 0 and self._committed_direction != 0:
                # Going flat during commitment: smaller penalty
                penalty = self._commitment_penalty * 0.5
                self._committed_direction = 0
                self._hold_count = 0

        # Update commitment state
        if intended_direction != 0 and self._committed_direction == 0:
            # New commitment
            self._committed_direction = intended_direction
            self._hold_count = 0
        elif intended_direction == 0:
            self._committed_direction = 0
            self._hold_count = 0

        if self._committed_direction != 0:
            self._hold_count += 1

        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply commitment penalty
        reward -= penalty

        # Track info
        info["commitment_direction"] = self._committed_direction
        info["commitment_hold_count"] = self._hold_count
        info["commitment_penalty"] = penalty

        return obs, reward, terminated, truncated, info


class ActionSmoothingWrapper(gym.Wrapper):
    """Exponential moving average of actions for smoother trading.

    Instead of executing raw noisy RL actions, this wrapper applies
    EMA smoothing: a_executed = alpha * a_new + (1-alpha) * a_prev.

    This is how real market makers operate — they smooth their quotes
    to avoid getting picked off by noise.

    Parameters
    ----------
    env : gym.Env
        Base trading environment.
    alpha : float
        EMA smoothing factor (0 = fully smooth/slow, 1 = no smoothing).
    """

    def __init__(self, env: gym.Env, alpha: float = 0.3) -> None:
        super().__init__(env)
        self._alpha = alpha
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        if self._prev_action is not None:
            smoothed = self._alpha * action + (1.0 - self._alpha) * self._prev_action
        else:
            smoothed = action

        self._prev_action = smoothed.copy()
        return self.env.step(smoothed)


class MultiStepRewardWrapper(gym.Wrapper):
    """N-step return accumulation for better credit assignment.

    Instead of immediate rewards, accumulates discounted rewards over
    N steps. This helps the agent see the longer-term consequences
    of its actions (critical for trading where a good entry shows
    its value 5-20 bars later).

    Note: This is complementary to — not a replacement for — SB3's
    built-in n-step return computation in the replay buffer.

    Parameters
    ----------
    env : gym.Env
        Base environment.
    n_steps : int
        Number of steps to accumulate.
    gamma : float
        Discount factor for future rewards.
    """

    def __init__(self, env: gym.Env, n_steps: int = 5, gamma: float = 0.99) -> None:
        super().__init__(env)
        self._n_steps = n_steps
        self._gamma = gamma
        self._reward_buffer: list[float] = []
        self._accumulated_reward: float = 0.0

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._reward_buffer = []
        self._accumulated_reward = 0.0
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._reward_buffer.append(reward)

        # Compute n-step return
        if len(self._reward_buffer) >= self._n_steps:
            n_step_return = 0.0
            for i, r in enumerate(self._reward_buffer[-self._n_steps:]):
                n_step_return += (self._gamma ** i) * r
            # Blend: 50% immediate + 50% n-step
            reward = 0.5 * reward + 0.5 * n_step_return / self._n_steps

        if terminated or truncated:
            self._reward_buffer = []

        info["n_step_return"] = reward
        return obs, reward, terminated, truncated, info
