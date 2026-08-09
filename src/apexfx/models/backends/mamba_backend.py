"""Mamba-style Selective State Space dynamics backend.

Pure PyTorch implementation of the S6 selective scan mechanism
(Gu & Dao, 2023). No external mamba_ssm dependency required.

The selective SSM learns input-dependent transition matrices,
making it well-suited for non-stationary forex dynamics where
regime changes alter the underlying state evolution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from apexfx.models.config import MambaBackendConfig


class SelectiveSSM(nn.Module):
    """Single selective state space layer (S6 block).

    Implements discretised state space: h_t = A_bar h_{t-1} + B_bar x_t
    where A_bar, B_bar are input-dependent (selective).
    """

    def __init__(self, d_inner: int, d_state: int, dt_rank: int,
                 dt_min: float, dt_max: float) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Input-dependent projections for selectivity
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # dt projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialise dt bias for stable discretisation
        dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.bias, math.log(dt_min), math.log(dt_max))

        # A parameter (diagonal, log-parameterised for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan over input sequence.

        Args:
            x: (batch, d_inner) — single step input.

        Returns:
            (batch, d_inner) output.
        """
        x.shape[0]

        # Project to get dt, B, C (input-dependent)
        x_dbl = self.x_proj(x)
        dt_rank = x_dbl.shape[-1] - self.d_state * 2
        dt, B, C = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)

        # dt: (batch, dt_rank) → (batch, d_inner), then softplus for positivity
        dt = F.softplus(self.dt_proj(dt))

        # Discretise: A_bar = exp(A * dt), B_bar = B * dt
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # (batch, d_inner, d_state)
        B_bar = B.unsqueeze(1) * dt.unsqueeze(-1)  # (batch, d_inner, d_state)

        # Single-step state update (no sequence dimension — used per-step in dynamics)
        # h = A_bar * h_prev + B_bar * x  (elementwise for diagonal A)
        # For single-step: just compute the output projection
        # y = C @ (B_bar * x_expanded) + D * x
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.d_state)
        y = (C.unsqueeze(1) * B_bar * x_expanded).sum(dim=-1) + self.D * x

        return y


class MambaBlock(nn.Module):
    """Full Mamba block: conv1d → SSM → output gate."""

    def __init__(self, d_model: int, cfg: MambaBackendConfig) -> None:
        super().__init__()
        d_inner = d_model * cfg.expand_factor
        dt_rank = cfg.dt_rank if cfg.dt_rank > 0 else max(1, math.ceil(d_model / 16))

        # Input projection (split into two paths)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Local convolution (causal, depthwise)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1, groups=d_inner, bias=True,
        )

        # SSM core
        self.ssm = SelectiveSSM(d_inner, cfg.d_state, dt_rank, cfg.dt_min, cfg.dt_max)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            (batch, d_model)
        """
        # Split into SSM path and gate path
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1d expects (batch, channels, length) — treat as length-1 sequence
        x_conv = x_ssm.unsqueeze(-1)
        x_conv = self.conv1d(x_conv)[..., :1].squeeze(-1)
        x_conv = F.silu(x_conv)

        # Selective SSM
        y = self.ssm(x_conv)

        # Gate and project back
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class MambaBackend(nn.Module):
    """Mamba dynamics backend for WorldModelHybrid.

    Maps (z_t, a_t) → z_{t+1} using selective state space layers
    with residual connections.
    """

    def __init__(self, d_latent: int, d_action: int, cfg: MambaBackendConfig) -> None:
        super().__init__()
        d_model = d_latent + d_action

        self.input_proj = nn.Linear(d_latent + d_action, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        self.mamba = MambaBlock(d_model, cfg)
        self.norm_out = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, d_latent)
        self.residual = nn.Linear(d_latent, d_latent, bias=False)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent: z_{t+1} = residual(z_t) + mamba(z_t, a_t)."""
        x = torch.cat([z, action], dim=-1)
        x = self.norm_in(self.input_proj(x))
        x = x + self.mamba(x)  # residual around mamba block
        x = self.norm_out(x)
        delta = self.output_proj(x)
        return self.residual(z) + delta
