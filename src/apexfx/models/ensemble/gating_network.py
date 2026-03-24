"""Gating Network (Meta-Controller) — dynamically reweights agents based on market regime.

Architecture: **two-stream fusion** with agent-aware routing + meta-confidence gate.

Stream 1 — *Context stream*: encodes market regime + TFT state to determine
which *type* of market this is (trending / mean-reverting / breakout).

Stream 2 — *Agent-aware stream*: also sees what agents are proposing so
the meta-controller can evaluate whether the specialist signals agree or
conflict before assigning weights.

Key improvement over softmax gating: **sigmoid per-agent weights + meta-confidence**.
Softmax forces sum=1, meaning someone always dominates. Sigmoid allows ALL
agents to be suppressed when the meta-controller is uncertain → the model
can choose to NOT trade. The meta-confidence gate scales the final action
by overall conviction (0 = "I don't know" → flat, 1 = "high conviction").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    """Two-stream agent-aware meta-controller with sigmoid gating and confidence.

    Improvements over softmax-only gating:

    1. **Sigmoid per-agent weights** — each agent independently [0,1], allows
       "none of the above" (all low) when market is unclear.
    2. **Meta-confidence gate** — scalar [0,1] that scales the final action.
       When all agents disagree or market is ambiguous, confidence → 0 → flat.
    3. **Agent-aware routing** — sees agent proposals when computing weights.
    4. **Learnable log-temperature** — stable optimisation via log-parameterisation.
    """

    def __init__(
        self,
        d_regime: int = 6,
        d_context: int = 64,
        n_agents: int = 3,
        d_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents

        d_main_input = d_regime + d_context
        d_agent_input = d_main_input + n_agents  # appends agent_outputs

        # --- Stream 1: context (regime understanding) ---
        self.context_stream = nn.Sequential(
            nn.Linear(d_main_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # --- Stream 2: agent-aware (what are specialists proposing?) ---
        self.agent_stream = nn.Sequential(
            nn.Linear(d_agent_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # --- Fusion → per-agent sigmoid weights ---
        self.fusion = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_agents),
        )

        # --- Meta-confidence gate: "how sure are we overall?" ---
        # Inputs: fused hidden + agent agreement signal
        self.confidence_gate = nn.Sequential(
            nn.Linear(d_hidden * 2 + n_agents, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # --- Learnable log-temperature (for sigmoid sharpness) ---
        self.log_temperature = nn.Parameter(torch.zeros(1))

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
            weights: (batch, n_agents) — sigmoid weights (each in [0,1] independently)
            combined_action: (batch, 1) — weighted sum × confidence
        """
        main_input = torch.cat([regime_features, tft_context], dim=-1)

        # Stream 1 — "what regime are we in?"
        ctx_hidden = self.context_stream(main_input)

        # Stream 2 — "what are the agents saying?"
        agent_input = torch.cat([main_input, agent_outputs], dim=-1)
        agt_hidden = self.agent_stream(agent_input)

        # Fuse both streams
        fused = torch.cat([ctx_hidden, agt_hidden], dim=-1)

        # Per-agent sigmoid weights (temperature-scaled)
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        raw_logits = self.fusion(fused)
        weights = torch.sigmoid(raw_logits / temperature)  # (batch, n_agents) each in [0,1]

        # Normalize weights so they sum to 1 for combining actions
        # but preserve the sigmoid magnitudes for confidence signal
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_weights = weights / weight_sum

        # Weighted combination of agent outputs
        raw_combined = (normalized_weights * agent_outputs).sum(dim=-1, keepdim=True)

        # Agent agreement signal: how aligned are the agents?
        # High agreement → agents point same direction → trade confidently
        agent_agreement = agent_outputs.std(dim=-1, keepdim=True)  # (batch, 1)
        # Pad to n_agents for concat
        agreement_features = torch.cat([
            agent_agreement,
            agent_outputs.mean(dim=-1, keepdim=True),
            weight_sum / self.n_agents,  # average sigmoid activation
        ], dim=-1)  # (batch, 3)

        # Meta-confidence: scalar [0, 1] — overall conviction
        confidence = self.confidence_gate(
            torch.cat([fused, agreement_features], dim=-1)
        )  # (batch, 1)

        # Final action = weighted_combination × confidence
        # When confidence is low → action shrinks toward 0 (neutral/flat)
        combined_action = raw_combined * confidence

        return weights, combined_action
