"""Cross-Timeframe Fusion module.

Combines encoded states from D1, H1, M5 via attention-weighted fusion
into a single fused representation for the downstream agents and gating.

Architecture:
    3 × (batch, d_model) → attention weights → weighted sum → MLP → (batch, d_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTimeframeFusion(nn.Module):
    """Attention-weighted fusion of multi-timeframe TFT outputs.

    Takes 3 encoded states (one per timeframe) and produces a single
    fused state using learned attention weights + residual MLP.

    The attention mechanism learns which timeframe is most important
    for the current market context (e.g., D1 dominates in trending,
    M5 dominates in scalping).
    """

    def __init__(
        self,
        d_model: int = 32,
        n_timeframes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_timeframes = n_timeframes

        # Attention: each TF state → scalar attention logit
        self.attention_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # Post-fusion MLP with residual
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Residual projection (in case input dim differs)
        self.residual_proj = nn.Linear(d_model, d_model)

        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(d_model)

        # MTF context integration (optional 6-dim context vector)
        self.context_proj = nn.Linear(6, d_model)

    def forward(
        self,
        d1_state: torch.Tensor,
        h1_state: torch.Tensor,
        m5_state: torch.Tensor,
        mtf_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse three timeframe states into one.

        Parameters
        ----------
        d1_state : (batch, d_model) — D1 encoded state
        h1_state : (batch, d_model) — H1 encoded state
        m5_state : (batch, d_model) — M5 encoded state
        mtf_context : (batch, 6) — optional cross-TF context features

        Returns
        -------
        fused_state : (batch, d_model) — fused representation
        attention_weights : (batch, 3) — per-timeframe attention weights
        """
        batch = d1_state.size(0)

        # Stack: (batch, 3, d_model)
        states = torch.stack([d1_state, h1_state, m5_state], dim=1)

        # Compute attention logits: (batch, 3, 1)
        attn_logits = self.attention_proj(states)

        # Softmax over timeframes: (batch, 3, 1)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Weighted sum: (batch, d_model)
        weighted = (states * attn_weights).sum(dim=1)

        # Integrate MTF context if provided
        if mtf_context is not None:
            context_emb = self.context_proj(mtf_context)  # (batch, d_model)
            weighted = weighted + context_emb

        # Residual MLP
        fused = self.fusion_mlp(weighted)
        fused = fused + self.residual_proj(weighted)
        fused = self.layer_norm(fused)

        # Return attention weights as (batch, 3) for interpretability
        return fused, attn_weights.squeeze(-1)
