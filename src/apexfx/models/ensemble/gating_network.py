"""Gating Network (Meta-Controller) — dynamically reweights agents based on market regime."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class GatingNetwork(nn.Module):
    """
    Meta-Controller that observes market regime and TFT context
    to produce mixing weights for each specialist agent.

    The gating network decides in real-time how much to trust
    the trend agent vs. the reversion agent based on current
    market conditions.
    """

    def __init__(
        self,
        d_regime: int = 6,
        d_context: int = 64,
        n_agents: int = 2,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents

        hidden = hidden_sizes or [64, 32]
        d_input = d_regime + d_context

        layers: list[nn.Module] = []
        prev = d_input
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_agents))

        self.encoder = nn.Sequential(*layers)

        # Temperature parameter for softmax sharpness (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        regime_features: torch.Tensor,
        tft_context: torch.Tensor,
        agent_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            regime_features: (batch, d_regime) — Hurst, volatility, trend strength,
                             regime one-hot encoding
            tft_context: (batch, d_context) — TFT encoded market state
            agent_outputs: (batch, n_agents) — proposed actions from each agent

        Returns:
            weights: (batch, n_agents) — softmax mixing weights summing to 1
            combined_action: (batch, 1) — weighted sum of agent actions
        """
        combined_input = torch.cat([regime_features, tft_context], dim=-1)
        logits = self.encoder(combined_input)

        # Temperature-scaled softmax
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        weights = F.softmax(logits / temp, dim=-1)  # (batch, n_agents)

        # Weighted combination of agent outputs
        combined_action = (weights * agent_outputs).sum(dim=-1, keepdim=True)

        return weights, combined_action
