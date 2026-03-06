"""Trend-following agent operating on H1/D1 timeframe features.

Supports cross-agent attention: ``encode()`` returns a hidden representation
that participates in multi-agent attention, then ``act()`` produces the final
action from the (optionally enriched) hidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.models.components.layers import MLP


class TrendAgent(nn.Module):
    """
    Trend-following specialist agent.
    Operates on TFT-encoded representation + trend-specific features.
    Outputs a scalar action in [-1, 1] representing trade direction and confidence.
    """

    def __init__(
        self,
        d_tft: int = 64,
        d_trend_features: int = 8,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_input = d_tft + d_trend_features
        hidden = hidden_sizes or [128, 128, 64]

        self.feature_net = MLP(
            input_size=self.d_input,
            hidden_sizes=hidden,
            output_size=1,
            activation="relu",
            dropout=dropout,
        )

        # Tanh output to bound action to [-1, 1]
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
        trend_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return hidden representation *before* action head.

        Returns
        -------
        hidden : (batch, hidden_dim)
        """
        combined = torch.cat([tft_state, trend_features], dim=-1)
        return self.feature_net.encode(combined)

    def act(
        self,
        hidden: torch.Tensor,
        specialist_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce action from (optionally enriched) hidden.

        Parameters
        ----------
        hidden : (batch, hidden_dim)
            Penultimate hidden — may be enriched by cross-agent attention.
        specialist_features : ignored (present for uniform API)

        Returns
        -------
        action : (batch, 1) in [-1, 1]
        """
        return self.tanh(self.feature_net.decode(hidden))

    # ------------------------------------------------------------------
    # Standard forward (backward-compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        tft_state: torch.Tensor,
        trend_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tft_state: (batch, d_tft) — encoded market state from TFT
            trend_features: (batch, d_trend_features) — trend indicators
                Expected features: [hurst, adx, ma_slope_20, ma_slope_50,
                                    momentum_10, momentum_30, regime_trending, position_state]

        Returns:
            action: (batch, 1) — scalar in [-1, 1]
                +1.0: strong buy signal (confident uptrend)
                -1.0: strong sell signal (confident downtrend)
                 0.0: no trend detected / neutral
        """
        hidden = self.encode(tft_state, trend_features)
        return self.act(hidden)
