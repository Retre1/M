"""Breakout agent — trades support/resistance level breakouts with volume confirmation."""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.models.components.layers import MLP


class BreakoutAgent(nn.Module):
    """
    Breakout specialist agent.

    Detects and trades breakouts from consolidation ranges.
    Uses support/resistance proximity, volume surge, and volatility squeeze
    as activation signals. Only fires when price crosses a key level
    with volume confirmation.

    Action semantics:
        +1.0: strong bullish breakout (break above resistance + volume)
        -1.0: strong bearish breakout (break below support + volume)
         0.0: no breakout detected (in consolidation or false breakout)
    """

    def __init__(
        self,
        d_tft: int = 64,
        d_breakout_features: int = 10,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
        squeeze_threshold: float = 0.3,
    ) -> None:
        super().__init__()

        self.squeeze_threshold = squeeze_threshold
        self.d_input = d_tft + d_breakout_features
        hidden = hidden_sizes or [128, 64, 32]

        self.feature_net = MLP(
            input_size=self.d_input,
            hidden_sizes=hidden,
            output_size=1,
            activation="relu",
            dropout=dropout,
        )

        # Breakout confidence gate: activates only during volatility squeeze → expansion
        self.breakout_gate = nn.Sequential(
            nn.Linear(d_breakout_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Volume confirmation gate: requires volume surge for breakout
        self.volume_gate = nn.Sequential(
            nn.Linear(d_breakout_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.tanh = nn.Tanh()

    def forward(
        self,
        tft_state: torch.Tensor,
        breakout_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tft_state: (batch, d_tft) — encoded market state from TFT
            breakout_features: (batch, d_breakout_features) — breakout indicators
                Expected features:
                [resistance_distance, support_distance, range_width,
                 bollinger_bandwidth, volume_ratio, atr_ratio,
                 bars_in_range, false_breakout_count,
                 momentum_10, regime_flat]

        Returns:
            action: (batch, 1) — scalar in [-1, 1], gated by breakout + volume
                Only significant when a breakout is detected with confirmation.
        """
        combined = torch.cat([tft_state, breakout_features], dim=-1)

        # Base directional signal
        raw_action = self.tanh(self.feature_net(combined))

        # Breakout gate: is this a breakout condition?
        breakout_confidence = self.breakout_gate(breakout_features)

        # Volume gate: is there volume confirmation?
        volume_confirmation = self.volume_gate(breakout_features)

        # Combined gate: both conditions must be met
        gate = breakout_confidence * volume_confirmation

        return raw_action * gate
