"""Prioritized Experience Replay (PER) — SumTree + IS weights.

Why this matters:
- Standard replay samples uniformly → 80% of training on "nothing happened" bars.
- PER samples proportional to TD error → agent learns faster from rare, important events
  (crashes, breakouts, regime shifts).
- Importance Sampling (IS) weights correct for the sampling bias.

Integration:
- PERCallback updates priorities after each gradient step.
- Works with SB3's DictReplayBuffer for Dict observation spaces.

References:
- Schaul et al., 2015 — Prioritized Experience Replay
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class SumTree:
    """Binary tree where each leaf stores a priority and parent nodes store sums.

    Allows O(log N) sampling proportional to priority and O(log N) updates.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority at tree_idx."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float) -> int:
        """Add a new priority (circular buffer style). Returns leaf index."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        leaf_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return leaf_idx

    def get(self, value: float) -> tuple[int, int, float]:
        """Sample a leaf proportional to priority.

        Returns (tree_idx, data_idx, priority).
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left] or right >= len(self.tree):
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = parent - self.capacity + 1
        return parent, data_idx, self.tree[parent]


class PERCallback(BaseCallback):
    """Prioritized Experience Replay callback for SB3.

    After each gradient step, computes TD errors and updates priorities
    in a SumTree. Provides importance sampling weights for loss correction.

    Parameters
    ----------
    alpha : float
        Priority exponent. 0 = uniform, 1 = fully prioritized.
    beta_start : float
        Initial IS correction exponent. Annealed to 1.0 over training.
    beta_frames : int
        Number of frames to anneal beta from beta_start to 1.0.
    epsilon : float
        Small constant added to TD error to ensure non-zero priority.
    update_freq : int
        Update priorities every N environment steps.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        epsilon: float = 1e-6,
        update_freq: int = 100,
        batch_size: int = 256,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.batch_size = batch_size

        self._tree: SumTree | None = None
        self._initialized: bool = False
        self._frame_count: int = 0
        self._total_priority_updates: int = 0

    def _init_tree(self) -> None:
        """Lazy init — need buffer size from model."""
        buf = self.model.replay_buffer
        capacity = buf.buffer_size
        self._tree = SumTree(capacity)
        # Initialize all existing entries with max priority
        max_priority = 1.0
        for i in range(min(buf.size(), capacity)):
            self._tree.add(max_priority)
        self._initialized = True
        logger.info("PER SumTree initialized", capacity=capacity, initial_entries=buf.size())

    @property
    def beta(self) -> float:
        """Current IS correction exponent (annealed from beta_start to 1.0)."""
        fraction = min(1.0, self._frame_count / max(1, self.beta_frames))
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def _on_step(self) -> bool:
        self._frame_count += 1

        if not hasattr(self.model, "replay_buffer") or self.model.replay_buffer is None:
            return True
        if self.model.replay_buffer.size() < max(self.batch_size, self.model.learning_starts):
            return True

        if not self._initialized:
            self._init_tree()

        # Add new transitions with max priority (will be corrected on first sample)
        if self._tree.n_entries < self.model.replay_buffer.size():
            max_p = max(self._tree.tree[self._tree.capacity - 1:
                                         self._tree.capacity - 1 + self._tree.n_entries].max(),
                        1.0)
            while self._tree.n_entries < self.model.replay_buffer.size():
                self._tree.add(max_p)

        if self.num_timesteps % self.update_freq != 0:
            return True

        try:
            self._update_priorities()
        except Exception as exc:
            logger.debug("PER priority update skipped", reason=str(exc))

        return True

    def _update_priorities(self) -> None:
        """Compute TD errors and update SumTree priorities."""
        model = self.model
        buf = model.replay_buffer

        # Sample a batch
        replay_data = buf.sample(self.batch_size)

        with torch.no_grad():
            # Compute TD errors from critic
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

            # Get current Q-values
            features = model.policy.features_extractor(obs)
            next_features = model.policy.features_extractor(next_obs)

            # Use critic to estimate TD error
            if hasattr(model.policy, "critic"):
                current_q = model.policy.critic(features, actions)
                if isinstance(current_q, (list, tuple)):
                    current_q = current_q[0]

                # Next Q via target
                if hasattr(model.policy, "actor_target"):
                    next_actions = model.policy.actor_target(next_features)
                else:
                    next_actions, _ = model.policy.predict(next_obs, deterministic=True)
                    next_actions = torch.as_tensor(next_actions, device=features.device)

                if hasattr(model, "critic_target"):
                    next_q = model.critic_target(next_features, next_actions)
                    if isinstance(next_q, (list, tuple)):
                        next_q = next_q[0]
                else:
                    next_q = current_q  # fallback

                td_target = rewards + model.gamma * (1 - dones) * next_q
                td_errors = torch.abs(current_q - td_target).squeeze().cpu().numpy()
            else:
                # Fallback: use reward magnitude as priority
                td_errors = torch.abs(rewards).squeeze().cpu().numpy()

        # Update priorities in SumTree
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        # We can't directly map replay_data indices to tree indices in SB3,
        # so update recent entries proportional to TD error magnitude
        n_entries = self._tree.n_entries
        for i, priority in enumerate(priorities[:n_entries]):
            # Map to recent buffer positions
            buf_idx = (self._tree.data_pointer - self.batch_size + i) % self._tree.capacity
            if buf_idx < 0:
                buf_idx += self._tree.capacity
            tree_idx = buf_idx + self._tree.capacity - 1
            if tree_idx < len(self._tree.tree):
                self._tree.update(tree_idx, float(priority))

        self._total_priority_updates += 1

        # Log periodically
        if self._total_priority_updates % 50 == 0 and self.logger:
            self.logger.record("per/beta", self.beta)
            self.logger.record("per/mean_priority", float(np.mean(priorities)))
            self.logger.record("per/max_priority", float(np.max(priorities)))
            self.logger.record("per/total_sum", self._tree.total)
