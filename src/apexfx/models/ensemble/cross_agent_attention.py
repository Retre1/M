"""Cross-Agent Attention — enables specialist agents to communicate before acting.

Before producing final actions each agent's hidden representation attends to
every other agent's hidden through multi-head self-attention.  This allows
agents to coordinate:

* The Trend agent can see the Breakout agent's strong setup and amplify.
* The Reversion agent can see the Trend agent is committed and reduce its
  contrarian signal.
* The Breakout agent can see the Reversion agent detects an anomaly and
  avoid false breakouts.

A **gated residual** connection blends original and attended representations
so the model can gracefully fall back to independent behaviour when
cross-agent context is unhelpful.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAgentAttention(nn.Module):
    """Multi-head self-attention across agent hidden representations.

    Because agents may have different hidden dimensions (e.g. Trend=64,
    Breakout=32) every agent's hidden is first projected to a common
    ``d_attn`` space, attention is computed, and the enriched vector is
    projected back to each agent's original dimensionality.

    Parameters
    ----------
    agent_hidden_dims : list[int]
        Hidden dimensionality of each agent (in agent order).
    d_attn : int
        Common attention dimension.  Defaults to 64.
    n_heads : int
        Number of attention heads.  ``d_attn`` must be divisible by ``n_heads``.
    dropout : float
        Dropout on attention weights and the gate network.
    """

    def __init__(
        self,
        agent_hidden_dims: list[int],
        d_attn: int = 64,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_agents = len(agent_hidden_dims)
        self.d_attn = d_attn

        # --- Per-agent projections to common d_attn space ---
        self.in_projections = nn.ModuleList([
            nn.Linear(d, d_attn) if d != d_attn else nn.Identity()
            for d in agent_hidden_dims
        ])

        # --- Per-agent back-projections to original dim ---
        self.out_projections = nn.ModuleList([
            nn.Linear(d_attn, d) if d != d_attn else nn.Identity()
            for d in agent_hidden_dims
        ])

        # --- Multi-head self-attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(d_attn)

        # --- Gated residual: learn how much cross-agent context to use ---
        # sigmoid gate ∈ [0, 1]:  0 → ignore other agents,  1 → fully attend
        self.gate = nn.Sequential(
            nn.Linear(d_attn * 2, d_attn),
            nn.SiLU(),
            nn.Linear(d_attn, d_attn),
            nn.Sigmoid(),
        )

    def forward(
        self,
        agent_hiddens: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run cross-agent self-attention.

        Parameters
        ----------
        agent_hiddens : list of (batch, d_i) tensors
            Penultimate hidden from each specialist agent.

        Returns
        -------
        enriched_hiddens : list of (batch, d_i) tensors
            Enriched hidden representations (same dims as input).
        attention_weights : (batch, n_agents, n_agents)
            Who-attends-to-whom matrix (interpretable).
        """
        # 1. Project to common space  →  (batch, n_agents, d_attn)
        projected = torch.stack(
            [proj(h) for proj, h in zip(self.in_projections, agent_hiddens)],
            dim=1,
        )

        # 2. Self-attention across agents
        attended, attn_weights = self.attention(projected, projected, projected)

        # 3. Gated residual — let model control information flow
        gate_input = torch.cat([projected, attended], dim=-1)
        gate_value = self.gate(gate_input)
        fused = self.layer_norm(projected + gate_value * attended)

        # 4. Back-project each agent to its original dim
        enriched: list[torch.Tensor] = []
        for i, out_proj in enumerate(self.out_projections):
            enriched.append(out_proj(fused[:, i, :]))

        return enriched, attn_weights
