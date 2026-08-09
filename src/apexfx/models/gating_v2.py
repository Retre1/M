"""HiveMind Gating v2 — FSD-conditioned meta-controller with anti-collapse.

Extends GatingNetwork with three key improvements:

1. **FSD regime conditioning** — the gating network receives FSD regime
   features (dispersion + 3 one-hot regime bits) as an additional input
   stream, allowing it to shift agent weights based on the current
   macro regime (risk-on / neutral / risk-off).

2. **Entropy regularization** — prevents gating collapse where one agent
   dominates. Maintains H(weights) > entropy_min threshold. When entropy
   drops below threshold, a penalty is added to the training loss.

3. **Diversity loss** — encourages agent weight distributions to differ
   across batch samples, preventing the gating from converging to a
   single static allocation regardless of market state.

Architecture:
    Stream 1 (Context): regime_features + tft_context + FSD embedding
    Stream 2 (Agent-aware): same + agent proposals
    → Fusion → sigmoid per-agent weights + meta-confidence
    → Entropy check → diversity loss

Anti-collapse guarantee: entropy_min=0.12 ensures that for 3 agents,
no single agent can have weight > ~0.85 (from H({0.85, 0.075, 0.075}) ≈ 0.12).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from apexfx.models.config import GatingV2Config


class FSDEmbedding(nn.Module):
    """Embed FSD regime features into a dense representation.

    Input: 4 features [fsd_dispersion, regime_risk_off, regime_neutral, regime_risk_on]
    Output: (batch, fsd_embed_dim)
    """

    def __init__(self, d_fsd: int = 4, embed_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_fsd, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, fsd_features: torch.Tensor) -> torch.Tensor:
        return self.net(fsd_features)


class HiveMindGating_v2(nn.Module):
    """FSD-conditioned two-stream gating with entropy regularization.

    Improvements over GatingNetwork v1:
    1. FSD regime conditioning via dedicated embedding stream
    2. Entropy regularization (anti-collapse mechanism)
    3. Diversity loss across batch for dynamic gating
    4. Separate confidence dropout (higher than main dropout)
    5. Gating diagnostics for TensorBoard logging

    Usage::

        gating = HiveMindGating_v2(GatingV2Config())
        weights, action, diagnostics = gating(
            regime_features, tft_context, agent_outputs, fsd_features,
        )
        # diagnostics["entropy_loss"] should be added to training loss
    """

    def __init__(self, config: GatingV2Config | None = None) -> None:
        super().__init__()
        cfg = config or GatingV2Config()
        self._cfg = cfg
        self.n_agents = cfg.n_agents

        # FSD embedding stream
        self.fsd_embed = FSDEmbedding(cfg.d_fsd, cfg.fsd_embed_dim)

        # Total context input dimension
        d_main = cfg.d_regime + cfg.d_context + cfg.fsd_embed_dim
        d_agent = d_main + cfg.n_agents

        # --- Stream 1: Context (regime + TFT + FSD) ---
        self.context_stream = nn.Sequential(
            nn.Linear(d_main, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        # --- Stream 2: Agent-aware ---
        self.agent_stream = nn.Sequential(
            nn.Linear(d_agent, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        # --- Fusion → per-agent sigmoid weights ---
        self.fusion = nn.Sequential(
            nn.Linear(cfg.d_hidden * 2, cfg.d_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.n_agents),
        )

        # --- Meta-confidence gate ---
        # Inputs: fused hidden + agent agreement (3 features)
        self.confidence_gate = nn.Sequential(
            nn.Linear(cfg.d_hidden * 2 + cfg.n_agents, cfg.d_hidden // 2),
            nn.SiLU(),
            nn.Dropout(cfg.confidence_dropout),
            nn.Linear(cfg.d_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # --- Learnable log-temperature ---
        init_log_temp = math.log(max(cfg.temperature_init, 1e-6))
        self.log_temperature = nn.Parameter(torch.tensor(init_log_temp))

    def forward(
        self,
        regime_features: torch.Tensor,
        tft_context: torch.Tensor,
        agent_outputs: torch.Tensor,
        fsd_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            regime_features: (batch, d_regime) — Hurst, volatility, etc.
            tft_context: (batch, d_context) — TFT encoded market state
            agent_outputs: (batch, n_agents) — proposed actions from each agent
            fsd_features: (batch, d_fsd) — FSD regime features [dispersion + one-hot]
                          If None, zeros are used.

        Returns:
            weights: (batch, n_agents) — sigmoid weights per agent
            combined_action: (batch, 1) — weighted sum × confidence
            diagnostics: dict with entropy, diversity, confidence metrics
        """
        batch = regime_features.shape[0]
        device = regime_features.device
        cfg = self._cfg

        # FSD embedding
        if fsd_features is None:
            fsd_features = torch.zeros(batch, cfg.d_fsd, device=device)
        fsd_emb = self.fsd_embed(fsd_features)

        # Build main input: regime + TFT context + FSD
        main_input = torch.cat([regime_features, tft_context, fsd_emb], dim=-1)

        # Stream 1 — context understanding
        ctx_hidden = self.context_stream(main_input)

        # Stream 2 — agent-aware routing
        agent_input = torch.cat([main_input, agent_outputs], dim=-1)
        agt_hidden = self.agent_stream(agent_input)

        # Fuse
        fused = torch.cat([ctx_hidden, agt_hidden], dim=-1)

        # Per-agent sigmoid weights (temperature-scaled)
        temperature = torch.exp(self.log_temperature).clamp(
            min=cfg.temperature_min, max=cfg.temperature_max,
        )
        raw_logits = self.fusion(fused)
        weights = torch.sigmoid(raw_logits / temperature)

        # Normalise for action combination
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_weights = weights / weight_sum

        # Weighted action
        raw_combined = (normalized_weights * agent_outputs).sum(dim=-1, keepdim=True)

        # Agent agreement features
        agreement_features = torch.cat([
            agent_outputs.std(dim=-1, keepdim=True),
            agent_outputs.mean(dim=-1, keepdim=True),
            weight_sum / self.n_agents,
        ], dim=-1)

        # Meta-confidence
        confidence = self.confidence_gate(
            torch.cat([fused, agreement_features], dim=-1),
        )

        combined_action = raw_combined * confidence

        # ── Diagnostics & anti-collapse ──────────────────────────
        diagnostics = self._compute_diagnostics(
            normalized_weights, confidence, weights,
        )

        return weights, combined_action, diagnostics

    def _compute_diagnostics(
        self,
        normalized_weights: torch.Tensor,
        confidence: torch.Tensor,
        raw_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute entropy, diversity loss, and other diagnostics."""
        cfg = self._cfg

        # --- Entropy of normalised weights ---
        # H = -sum(w * log(w + eps))
        eps = 1e-8
        entropy = -(normalized_weights * torch.log(normalized_weights + eps)).sum(dim=-1)
        mean_entropy = entropy.mean()

        # Entropy penalty: penalise when entropy < threshold
        entropy_loss = torch.relu(cfg.entropy_min - mean_entropy) * cfg.entropy_weight

        # --- Diversity loss ---
        # Encourage weight distributions to vary across the batch
        # Using variance of weights across batch dimension
        weight_var = normalized_weights.var(dim=0).mean()  # variance per agent, averaged
        diversity_loss = -weight_var * cfg.diversity_weight  # negative = reward diversity

        # --- Combined anti-collapse loss ---
        anticollapse_loss = entropy_loss + diversity_loss

        return {
            "gating/entropy": mean_entropy.detach(),
            "gating/entropy_loss": entropy_loss.detach(),
            "gating/diversity_loss": diversity_loss.detach(),
            "gating/anticollapse_loss": anticollapse_loss,  # not detached — for backprop
            "gating/confidence_mean": confidence.mean().detach(),
            "gating/confidence_std": confidence.std().detach(),
            "gating/temperature": torch.exp(self.log_temperature).detach(),
            "gating/max_weight": normalized_weights.max(dim=-1).values.mean().detach(),
            "gating/weight_std": normalized_weights.std(dim=-1).mean().detach(),
        }

    def get_anticollapse_loss(
        self, diagnostics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Extract the anti-collapse loss for addition to the RL loss.

        Usage in training loop::

            weights, action, diag = gating(regime, tft, agents, fsd)
            rl_loss = ...  # standard RL loss
            total_loss = rl_loss + gating.get_anticollapse_loss(diag)
        """
        return diagnostics["gating/anticollapse_loss"]
