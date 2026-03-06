"""Shared building blocks: GLU, Gated Linear Unit, LayerNorm wrappers."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU activation: splits input in half, applies sigmoid gate to one half."""

    def __init__(self, d_input: int, d_output: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_input, d_output * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * self.sigmoid(b)


class GateAddNorm(nn.Module):
    """GLU → Dropout → Add (residual) → LayerNorm."""

    def __init__(self, d_input: int, d_output: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(d_input, d_output)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.glu(x))
        return self.layer_norm(out + residual)


class TimeDistributed(nn.Module):
    """Apply a module to each time step independently."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        batch, time, features = x.shape
        x_flat = x.reshape(batch * time, features)
        out = self.module(x_flat)
        return out.reshape(batch, time, -1)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Supports splitting into ``encode()`` (hidden layers) and ``decode()``
    (output head) for cross-agent attention integration.  The standard
    ``forward()`` is equivalent to ``decode(encode(x))``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.1,
        output_activation: str | None = None,
    ) -> None:
        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        act_cls = activations.get(activation, nn.ReLU)

        # --- Hidden layers with LayerNorm for gradient stability ---
        hidden_layers: list[nn.Module] = []
        prev_size = input_size
        for h_size in hidden_sizes:
            hidden_layers.extend([
                nn.Linear(prev_size, h_size),
                nn.LayerNorm(h_size),
                act_cls(),
                nn.Dropout(dropout),
            ])
            prev_size = h_size

        self.hidden_net = nn.Sequential(*hidden_layers)
        self.hidden_dim: int = prev_size  # last hidden size

        # --- Output head ---
        output_layers: list[nn.Module] = [nn.Linear(prev_size, output_size)]
        if output_activation:
            out_act = activations.get(output_activation)
            if out_act:
                output_layers.append(out_act())

        self.output_head = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.hidden_net(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate hidden representation ``(batch, hidden_dim)``."""
        return self.hidden_net(x)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map hidden representation to output ``(batch, output_size)``."""
        return self.output_head(hidden)
