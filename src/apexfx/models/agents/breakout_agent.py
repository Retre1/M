"""Breakout agent — trades support/resistance level breakouts with volume confirmation.

Supports cross-agent attention: ``encode()`` returns a hidden representation
that participates in multi-agent attention, then ``act()`` produces the final
dual-gated action from the (optionally enriched) hidden.
"""

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

    # ------------------------------------------------------------------
    # Cross-agent attention API
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self.feature_net.hidden_dim

    def encode(
        self,
        tft_state: torch.Tensor,
        breakout_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return hidden representation *before* action head and gating.

        Returns
        -------
        hidden : (batch, hidden_dim)
        """
        combined = torch.cat([tft_state, breakout_features], dim=-1)
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
            Penultimate hidden — may be enriched by cross-agent attention.
        specialist_features : (batch, d_breakout_features)
            Needed for breakout/volume gates.  Falls back to ones if ``None``.
            Expected features include bollinger_bandwidth at index 3.

        Returns
        -------
        action : (batch, 1) in [-1, 1], gated by breakout × volume × squeeze
        """
        raw_action = self.tanh(self.feature_net.decode(hidden))

        if specialist_features is not None:
            breakout_confidence = self.breakout_gate(specialist_features)
            volume_confirmation = self.volume_gate(specialist_features)

            # Volatility squeeze filter: only fire breakouts after tight consolidation
            # bollinger_bandwidth is expected at index 3 of breakout_features
            # Low bandwidth = squeeze = breakout likely; high bandwidth = already expanded
            squeeze_mask = torch.ones_like(breakout_confidence)
            if specialist_features.shape[-1] > 3:
                bandwidth = specialist_features[:, 3:4]  # bollinger_bandwidth
                # Suppress signals when bandwidth is high (no squeeze)
                # squeeze_mask ≈ 1 when bandwidth < threshold, ≈ 0 when bandwidth >> threshold
                squeeze_mask = torch.sigmoid(
                    (self.squeeze_threshold - bandwidth) * 10.0
                )

            gate = breakout_confidence * volume_confirmation * squeeze_mask
            return raw_action * gate
        return raw_action

    # ------------------------------------------------------------------
    # Standard forward (backward-compatible)
    # ------------------------------------------------------------------

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
        hidden = self.encode(tft_state, breakout_features)
        return self.act(hidden, specialist_features=breakout_features)
