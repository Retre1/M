"""DML (Differential Machine Learning) network for jump-diffusion pricing.

Based on: "Differential Machine Learning: 0DTE pricing with stochastic
volatility and jumps"

Implements a neural network that learns both function values AND their
partial derivatives (sensitivities/Greeks), providing:
1. More data-efficient learning via differential regularization
2. Built-in sensitivity analysis for risk management
3. Jump-diffusion dynamics via Merton model parametrization

The DML network can be used as:
- A value function approximator with built-in Greeks
- A world model component that predicts price dynamics + sensitivities
- A risk model for position sizing based on learned derivatives
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DMLConfig:
    """Configuration for DML network."""

    n_hidden: int = 4
    d_hidden: int = 256
    differential_weight: float = 0.1
    jump_intensity: float = 0.05
    jump_mean: float = -0.01
    jump_std: float = 0.02
    lr: float = 1e-3


class DMLNetwork(nn.Module):
    """Differential Machine Learning network.

    Learns f(x) and ∂f/∂x simultaneously. The differential regularization
    term encourages the network's automatic derivatives to match target
    sensitivities, improving sample efficiency and generalization.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, config: DMLConfig | None = None):
        super().__init__()
        self.config = config or DMLConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(self.config.n_hidden):
            layers.extend([nn.Linear(d_in, self.config.d_hidden), nn.SiLU()])
            d_in = self.config.d_hidden
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()
        logger.info(
            "DML network initialized",
            input_dim=input_dim,
            output_dim=output_dim,
            n_params=sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute f(x)."""
        return self.net(x)

    def forward_with_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both f(x) and ∂f/∂x.

        Args:
            x: Input tensor of shape (batch, input_dim). Requires grad.

        Returns:
            Tuple of (values, gradients) where gradients has shape
            (batch, output_dim, input_dim).
        """
        x = x.requires_grad_(True)
        y = self.net(x)

        grads = []
        for i in range(self.output_dim):
            grad_i = torch.autograd.grad(
                y[:, i].sum(),
                x,
                create_graph=True,
                retain_graph=True,
            )[0]
            grads.append(grad_i)

        gradients = torch.stack(grads, dim=1)  # (batch, output_dim, input_dim)
        return y, gradients

    def differential_loss(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        dy_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined value + differential loss.

        L = MSE(f(x), y) + λ * MSE(∂f/∂x, dy)

        Args:
            x: Input features.
            y_target: Target values.
            dy_target: Target derivatives (optional). If None, only value loss.

        Returns:
            Combined scalar loss.
        """
        y_pred, dy_pred = self.forward_with_grad(x)

        value_loss = nn.functional.mse_loss(y_pred, y_target)

        if dy_target is not None:
            diff_loss = nn.functional.mse_loss(dy_pred, dy_target)
            total = value_loss + self.config.differential_weight * diff_loss
        else:
            total = value_loss

        return total


class MertonJumpDiffusion:
    """Merton jump-diffusion model for generating training targets.

    dS/S = (μ - λk)dt + σdW + J dN

    where J ~ LogNormal(jump_mean, jump_std), N is Poisson(λ).
    Provides analytical option prices and Greeks as DML training targets.
    """

    def __init__(self, config: DMLConfig | None = None) -> None:
        self.config = config or DMLConfig()

    def simulate(
        self,
        s0: float,
        mu: float,
        sigma: float,
        n_steps: int,
        n_paths: int,
        dt: float = 1 / 252,
    ) -> np.ndarray:
        """Simulate Merton jump-diffusion paths.

        Returns:
            Array of shape (n_paths, n_steps) with price paths.
        """
        cfg = self.config
        rng = np.random.default_rng()

        # Mean jump size compensation
        k = np.exp(cfg.jump_mean + 0.5 * cfg.jump_std**2) - 1.0

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = s0

        for t in range(1, n_steps):
            # Diffusion
            z = rng.standard_normal(n_paths)
            diffusion = (mu - cfg.jump_intensity * k - 0.5 * sigma**2) * dt + sigma * np.sqrt(
                dt
            ) * z

            # Jumps
            n_jumps = rng.poisson(cfg.jump_intensity * dt, n_paths)
            jump_sizes = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jumps = rng.normal(cfg.jump_mean, cfg.jump_std, n_jumps[i])
                    jump_sizes[i] = np.sum(jumps)

            paths[:, t] = paths[:, t - 1] * np.exp(diffusion + jump_sizes)

        return paths
