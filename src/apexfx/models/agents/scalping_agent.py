"""Scalping agent for M1 medium-frequency trading (10-100 trades/day).

Targets 3-5 pip moves with tight stops. Uses spread and momentum gates
to activate only when conditions are favorable for scalping: tight spreads
and clear micro-momentum signals.

Supports cross-agent attention: ``encode()`` returns a hidden representation
that participates in multi-agent attention, then ``act()`` produces the final
dual-gated action from the (optionally enriched) hidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.models.components.layers import MLP


class ScalpingAgent(nn.Module):
    """Scalping specialist agent for M1 timeframe.

    Detects and trades micro-momentum setups on tight spreads.
    Uses two gates to filter signals:

    1. **Spread Gate** (sigmoid): only activates when spread is tight.
       Input feature[0] = normalized spread (low = tight = good).
    2. **Momentum Gate** (sigmoid): activates on micro-momentum signals.
       Input features[1:3] = micro_momentum_5 and micro_momentum_10.

    Action semantics:
        +1.0: strong long scalp (micro-momentum up + tight spread)
        -1.0: strong short scalp (micro-momentum down + tight spread)
         0.0: no scalp (wide spread, no momentum, or conflicting signals)
    """

    def __init__(
        self,
        d_tft: int = 64,
        d_scalp_features: int = 10,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_input = d_tft + d_scalp_features
        hidden = hidden_sizes or [128, 64, 32]

        self.feature_net = MLP(
            input_size=self.d_input,
            hidden_sizes=hidden,
            output_size=1,
            activation="relu",
            dropout=dropout,
        )

        # Spread gate: activates only when spread is tight.
        # Feature[0] = normalized_spread (low value = tight = good for scalping).
        # The gate learns to suppress action when spread is wide.
        self.spread_gate = nn.Sequential(
            nn.Linear(d_scalp_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Momentum gate: activates on micro-momentum signals.
        # Features[1:3] = micro_momentum_5, micro_momentum_10.
        # The gate learns to fire only when short-term momentum is clear.
        self.momentum_gate = nn.Sequential(
            nn.Linear(d_scalp_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.tanh = nn.Tanh()

    # ------------------------------------------------------------------
    # Cross-agent attention API
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the penultimate hidden representation."""
        return self.feature_net.hidden_dim

    def encode(
        self,
        tft_state: torch.Tensor,
        scalp_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return hidden representation *before* action head and gating.

        Parameters
        ----------
        tft_state : (batch, d_tft)
            Encoded market state from TFT.
        scalp_features : (batch, d_scalp_features)
            Scalping-specific features (10 dims).

        Returns
        -------
        hidden : (batch, hidden_dim)
        """
        combined = torch.cat([tft_state, scalp_features], dim=-1)
        return self.feature_net.encode(combined)

    def act(
        self,
        hidden: torch.Tensor,
        specialist_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce dual-gated action from (optionally enriched) hidden.

        Parameters
        ----------
        hidden : (batch, hidden_dim)
            Penultimate hidden -- may be enriched by cross-agent attention.
        specialist_features : (batch, d_scalp_features)
            Needed for spread/momentum gates. Falls back to ones if ``None``.

        Returns
        -------
        action : (batch, 1) in [-1, 1], gated by spread x momentum
        """
        raw_action = self.tanh(self.feature_net.decode(hidden))

        if specialist_features is not None:
            spread_confidence = self.spread_gate(specialist_features)
            momentum_confidence = self.momentum_gate(specialist_features)
            gate = spread_confidence * momentum_confidence
            return raw_action * gate
        return raw_action

    # ------------------------------------------------------------------
    # Standard forward (backward-compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        tft_state: torch.Tensor,
        scalp_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tft_state: (batch, d_tft) -- encoded market state from TFT
            scalp_features: (batch, d_scalp_features) -- scalping indicators
                Expected features (10 dims):
                [normalized_spread, micro_momentum_5, micro_momentum_10,
                 micro_momentum_20, tick_velocity, m1_atr_norm,
                 time_of_day_score, volume_surge, bid_ask_pressure,
                 mean_reversion_signal]

        Returns:
            action: (batch, 1) -- scalar in [-1, 1], gated by spread x momentum
                Only significant when spread is tight and micro-momentum is clear.
        """
        hidden = self.encode(tft_state, scalp_features)
        return self.act(hidden, specialist_features=scalp_features)
