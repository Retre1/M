"""Gated Residual Network — fundamental building block of the TFT."""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.models.components.layers import GateAddNorm


class GatedResidualNetwork(nn.Module):
    """
    GRN: Linear → ELU → Linear → GLU gate → Dropout → Add & LayerNorm.

    The GLU (Gated Linear Unit) allows the network to suppress
    irrelevant parts of the transformation.

    Optionally accepts a context vector (e.g., static covariates)
    that modulates the transformation.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int | None = None,
        d_output: int | None = None,
        dropout: float = 0.1,
        d_context: int | None = None,
    ) -> None:
        super().__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden or d_input
        self.d_output = d_output or d_input

        # Skip connection projection if dimensions differ
        if self.d_input != self.d_output:
            self.skip_proj = nn.Linear(self.d_input, self.d_output)
        else:
            self.skip_proj = None

        # Primary layers
        self.fc1 = nn.Linear(self.d_input, self.d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(self.d_hidden, self.d_output)

        # Optional context vector
        if d_context is not None:
            self.context_proj = nn.Linear(d_context, self.d_hidden, bias=False)
        else:
            self.context_proj = None

        # Gate + Add + Norm
        self.gate_add_norm = GateAddNorm(self.d_output, self.d_output, dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, [time,] d_input)
            context: optional (batch, [time,] d_context)

        Returns:
            (batch, [time,] d_output)
        """
        # Skip connection
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        # Primary path
        hidden = self.fc1(x)

        # Context modulation
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)

        # Gate, add residual, normalize
        return self.gate_add_norm(hidden, residual)
