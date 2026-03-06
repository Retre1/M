"""Mean-reversion agent for Z-Score anomaly detection entries.

Supports cross-agent attention: ``encode()`` returns a hidden representation
that participates in multi-agent attention, then ``act()`` produces the final
gated action from the (optionally enriched) hidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.models.components.layers import MLP


class ReversionAgent(nn.Module):
    """
    Mean-reversion specialist agent.
    Activates when Z-Score exceeds thresholds (|Z| > 3.0).
    Uses a confidence gate that approaches zero near normal Z-Score ranges.
    """

    def __init__(
        self,
        d_tft: int = 64,
        d_reversion_features: int = 8,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
        z_score_threshold: float = 3.0,
    ) -> None:
        super().__init__()

        self.z_score_threshold = z_score_threshold
        self.d_input = d_tft + d_reversion_features
        hidden = hidden_sizes or [128, 128, 64]

        self.feature_net = MLP(
            input_size=self.d_input,
            hidden_sizes=hidden,
            output_size=1,
            activation="relu",
            dropout=dropout,
        )

        # Confidence gate network: learns when to activate
        self.confidence_gate = nn.Sequential(
            nn.Linear(d_reversion_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.tanh = nn.Tanh()

    # ------------------------------------------------------------------
    # Cross-agent attention API
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self.feature_net.hidden_dim

    def encode(
        self,
        tft_state: torch.Tensor,
        reversion_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return hidden representation *before* action head and gating.

        Returns
        -------
        hidden : (batch, hidden_dim)
        """
        # Clip z_score (index 0) to prevent gradient explosion from extreme values
        reversion_features = reversion_features.clone()
        reversion_features[:, 0] = reversion_features[:, 0].clamp(-5.0, 5.0)

        combined = torch.cat([tft_state, reversion_features], dim=-1)
        return self.feature_net.encode(combined)

    def act(
        self,
        hidden: torch.Tensor,
        specialist_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce gated action from (optionally enriched) hidden.

        Parameters
        ----------
        hidden : (batch, hidden_dim)
            Penultimate hidden — may be enriched by cross-agent attention.
        specialist_features : (batch, d_reversion_features)
            Needed for the confidence gate.  Falls back to ones if ``None``.

        Returns
        -------
        action : (batch, 1) in [-1, 1], gated by confidence
        """
        raw_action = self.tanh(self.feature_net.decode(hidden))

        if specialist_features is not None:
            confidence = self.confidence_gate(specialist_features)
            return raw_action * confidence
        return raw_action

    # ------------------------------------------------------------------
    # Standard forward (backward-compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        tft_state: torch.Tensor,
        reversion_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tft_state: (batch, d_tft) — encoded market state from TFT
            reversion_features: (batch, d_reversion_features) — reversion indicators
                Expected: [z_score, bollinger_pct, hvn_distance, rsi,
                           stoch_k, volume_profile_skew, regime_mean_reverting, position_state]

        Returns:
            action: (batch, 1) — scalar in [-1, 1], gated by confidence
                Near 0 when Z-Score is in normal range.
                Significant magnitude only during anomalous conditions.
        """
        hidden = self.encode(tft_state, reversion_features)
        return self.act(hidden, specialist_features=reversion_features)
