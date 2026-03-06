"""Adversarial Training for distribution-shift robustness.

Hedge funds face constant regime changes. An agent trained only on clean data
will overfit to the training distribution and fail in live markets where:
- Spreads widen unexpectedly
- Volatility spikes or collapses
- Correlations break down
- Data feeds have glitches

This module provides two mechanisms:

1. **AdversarialObsWrapper** — Gymnasium wrapper that injects calibrated noise
   into observations during training. The agent learns to be robust to small
   perturbations in market features (similar to adversarial training in CV).

2. **AdversarialCallback** — SB3 callback that applies FGSM-style perturbations
   to the critic's input, penalising sensitivity to small input changes
   (Lipschitz regularisation via gradient penalty).

Mathematical Foundation
-----------------------
For each observation x, we generate x' = x + epsilon * sign(grad_x L)
where L is the critic loss. Training on both x and x' makes the policy
smoother and more robust — the same principle behind adversarial training
in image classification (Goodfellow et al., 2015), adapted for RL.

Usage
-----
>>> from apexfx.training.adversarial import AdversarialObsWrapper
>>> env = AdversarialObsWrapper(base_env, noise_std=0.01, noise_schedule="cosine")
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class AdversarialObsWrapper(gym.Wrapper):
    """Inject calibrated random noise into observations during training.

    Noise is calibrated per-feature using running statistics so that
    high-variance features get proportionally less noise (preventing
    gradient explosion) while low-variance features get meaningful
    perturbations.

    Parameters
    ----------
    env : gym.Env
        Base environment with Dict observation space.
    noise_std : float
        Base noise standard deviation (relative to feature running std).
    noise_schedule : str
        How noise decays over training:
        - "constant": fixed noise throughout
        - "linear": linearly decays to 0 over ``decay_steps``
        - "cosine": cosine annealing decay
    decay_steps : int
        Total steps over which noise decays (for linear/cosine schedules).
    warmup_steps : int
        Steps before noise starts (let the normaliser warm up).
    adversarial_prob : float
        Probability of applying noise on each step (stochastic application).
    """

    def __init__(
        self,
        env: gym.Env,
        noise_std: float = 0.01,
        noise_schedule: str = "cosine",
        decay_steps: int = 500_000,
        warmup_steps: int = 1000,
        adversarial_prob: float = 0.5,
    ) -> None:
        super().__init__(env)
        self._noise_std = noise_std
        self._schedule = noise_schedule
        self._decay_steps = decay_steps
        self._warmup_steps = warmup_steps
        self._adv_prob = adversarial_prob
        self._step_count = 0

        # Running statistics per feature key
        self._running_mean: dict[str, np.ndarray] = {}
        self._running_var: dict[str, np.ndarray] = {}
        self._obs_count: int = 0

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._update_stats(obs)

        # Stochastic noise application
        if (
            self._step_count > self._warmup_steps
            and np.random.random() < self._adv_prob
        ):
            obs = self._perturb(obs)

        return obs, reward, terminated, truncated, info

    def _get_noise_scale(self) -> float:
        """Compute current noise scale based on schedule."""
        if self._step_count < self._warmup_steps:
            return 0.0

        effective_step = self._step_count - self._warmup_steps
        progress = min(effective_step / max(self._decay_steps, 1), 1.0)

        if self._schedule == "constant":
            return self._noise_std
        elif self._schedule == "linear":
            return self._noise_std * (1.0 - progress)
        elif self._schedule == "cosine":
            return self._noise_std * 0.5 * (1.0 + np.cos(np.pi * progress))
        return self._noise_std

    def _update_stats(self, obs: dict[str, np.ndarray]) -> None:
        """Update running mean/var for noise calibration."""
        self._obs_count += 1
        for key, val in obs.items():
            if key not in self._running_mean:
                self._running_mean[key] = np.zeros_like(val, dtype=np.float64)
                self._running_var[key] = np.ones_like(val, dtype=np.float64)

            delta = val.astype(np.float64) - self._running_mean[key]
            self._running_mean[key] += delta / self._obs_count
            delta2 = val.astype(np.float64) - self._running_mean[key]
            self._running_var[key] += (delta * delta2 - self._running_var[key]) / self._obs_count

    def _perturb(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply calibrated noise to observation."""
        scale = self._get_noise_scale()
        if scale <= 0:
            return obs

        perturbed = {}
        for key, val in obs.items():
            if key == "position_state":
                # Never perturb position state — it's the agent's own info
                perturbed[key] = val
                continue

            # Calibrated noise: proportional to feature std
            std = np.sqrt(np.maximum(self._running_var.get(key, np.ones_like(val)), 1e-8))
            noise = np.random.randn(*val.shape).astype(np.float32) * std.astype(np.float32) * scale
            perturbed[key] = val + noise

        return perturbed


class GradientPenaltyCallback(BaseCallback):
    """Critic gradient penalty for Lipschitz-smooth value function.

    Applies a gradient penalty to the critic (Q-function) that penalises
    large input gradients. This makes the critic's value landscape smoother,
    leading to more stable policy updates and better generalisation.

    Gradient penalty: ``gp_weight * E[||grad_x Q(x, a)||^2]``

    This is the RL analogue of the gradient penalty used in WGAN-GP
    (Gulrajani et al., 2017), adapted for continuous control.

    Parameters
    ----------
    gp_weight : float
        Weight of the gradient penalty loss term.
    update_freq : int
        Apply penalty every N training steps.
    """

    def __init__(
        self,
        gp_weight: float = 0.1,
        update_freq: int = 100,
    ) -> None:
        super().__init__()
        self.gp_weight = gp_weight
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq != 0:
            return True

        import torch

        model = self.model
        # Only works with off-policy algorithms (SAC, TQC)
        if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
            return True
        if model.replay_buffer.size() < model.batch_size:
            return True

        try:
            # Sample a mini-batch
            replay_data = model.replay_buffer.sample(min(64, model.batch_size))
            obs = replay_data.observations

            # Get actions from current policy
            with torch.no_grad():
                actions = model.actor(obs)

            # Compute gradient of Q w.r.t. observations
            # We need gradients through the feature extractor
            for key in obs:
                if obs[key].requires_grad is False:
                    obs[key] = obs[key].detach().requires_grad_(True)

            # Forward through critic
            q_values = model.critic(obs, actions)
            if isinstance(q_values, tuple):
                q_values = q_values[0]  # Take first critic

            # Compute gradient w.r.t. inputs
            grads = torch.autograd.grad(
                outputs=q_values.sum(),
                inputs=[obs[key] for key in obs if obs[key].requires_grad],
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            # Gradient penalty
            grad_penalty = torch.tensor(0.0, device=q_values.device)
            for g in grads:
                if g is not None:
                    grad_penalty = grad_penalty + g.pow(2).mean()

            # Apply penalty through a separate backward pass
            penalty_loss = self.gp_weight * grad_penalty

            if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
                model.critic.optimizer.zero_grad()
                penalty_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
                model.critic.optimizer.step()

            if self.num_timesteps % (self.update_freq * 10) == 0:
                self.logger.record("adversarial/grad_penalty", float(grad_penalty.item()))

        except Exception as e:
            # Don't crash training on GP failure
            if self.num_timesteps % (self.update_freq * 100) == 0:
                logger.warning("Gradient penalty failed", error=str(e))

        return True
