"""Temporal Fusion Transformer — full assembly.

Architecture flow:
1. Input embedding (continuous → linear, categorical → embedding)
2. Variable Selection (encoder + decoder sides)
3. LSTM Encoder → processes historical sequence
4. LSTM Decoder → processes known future inputs
5. Static enrichment via GRN
6. Temporal self-attention (interpretable multi-head)
7. Position-wise feed-forward (GRN)
8. Output: learned market state representation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from apexfx.models.components.layers import GateAddNorm
from apexfx.models.components.positional import SinusoidalPositionalEncoding
from apexfx.models.tft.attention import InterpretableMultiHeadAttention, create_causal_mask
from apexfx.models.tft.grn import GatedResidualNetwork
from apexfx.models.tft.vsn import VariableSelectionNetwork


@dataclass
class TFTOutput:
    """Output from the Temporal Fusion Transformer."""
    encoded_state: torch.Tensor       # (batch, d_model) — final encoded representation
    temporal_features: torch.Tensor    # (batch, seq, d_model) — per-timestep features
    attention_weights: torch.Tensor    # (batch, seq, seq) — interpretable attention
    variable_importance: torch.Tensor  # (batch, seq, n_vars) — variable selection weights


class TemporalFusionTransformer(nn.Module):
    """
    Custom TFT designed to produce a learned market state representation
    for downstream RL agents. NOT a forecasting model — outputs an
    encoding, not future predictions.
    """

    def __init__(
        self,
        n_continuous_vars: int,
        n_known_future_vars: int = 5,
        n_static_vars: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_continuous_vars = n_continuous_vars
        self.n_known_future_vars = n_known_future_vars
        self.n_static_vars = n_static_vars

        # --- Input Embedding ---
        # Each continuous variable gets its own linear projection to d_model
        self.continuous_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_continuous_vars)
        ])

        # Known future variable embeddings
        self.future_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_known_future_vars)
        ])

        # Static variable embedding (e.g., symbol identity)
        if n_static_vars > 0:
            self.static_embeddings = nn.ModuleList([
                nn.Linear(1, d_model) for _ in range(n_static_vars)
            ])
            self.static_grn = GatedResidualNetwork(
                d_input=d_model * n_static_vars,
                d_hidden=d_model,
                d_output=d_model,
                dropout=dropout,
            )
        else:
            self.static_embeddings = None
            self.static_grn = None

        # --- Variable Selection Networks ---
        self.encoder_vsn = VariableSelectionNetwork(
            d_model=d_model,
            n_vars=n_continuous_vars,
            dropout=dropout,
            d_context=d_model if n_static_vars > 0 else None,
        )

        if n_known_future_vars > 0:
            self.decoder_vsn = VariableSelectionNetwork(
                d_model=d_model,
                n_vars=n_known_future_vars,
                dropout=dropout,
                d_context=d_model if n_static_vars > 0 else None,
            )
        else:
            self.decoder_vsn = None

        # --- LSTM Encoder/Decoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Post-LSTM gate
        self.encoder_gate = GateAddNorm(d_model, d_model, dropout)
        self.decoder_gate = GateAddNorm(d_model, d_model, dropout)

        # --- Static Enrichment ---
        self.static_enrichment_grn = GatedResidualNetwork(
            d_input=d_model,
            d_hidden=d_model,
            d_output=d_model,
            dropout=dropout,
            d_context=d_model if n_static_vars > 0 else None,
        )

        # --- Temporal Self-Attention ---
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(
            n_heads=n_heads,
            d_model=d_model,
            dropout=dropout,
        )
        self.attention_gate = GateAddNorm(d_model, d_model, dropout)

        # --- Position-wise Feed-Forward ---
        self.ff_grn = GatedResidualNetwork(
            d_input=d_model,
            d_hidden=d_model * 4,
            d_output=d_model,
            dropout=dropout,
        )
        self.ff_gate = GateAddNorm(d_model, d_model, dropout)

        # --- Output projection ---
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_past: torch.Tensor,
        x_future: torch.Tensor | None = None,
        x_static: torch.Tensor | None = None,
    ) -> TFTOutput:
        """
        Args:
            x_past: (batch, seq_past, n_continuous_vars) — historical features
            x_future: (batch, seq_future, n_known_future_vars) — known future features
            x_static: (batch, n_static_vars) — static covariates (optional)

        Returns:
            TFTOutput with encoded state, temporal features, attention weights,
            and variable importance scores.
        """
        batch = x_past.size(0)

        # --- Static context ---
        static_context = None
        if self.static_embeddings is not None and x_static is not None:
            static_emb = [
                emb(x_static[:, i : i + 1])
                for i, emb in enumerate(self.static_embeddings)
            ]
            static_flat = torch.cat(static_emb, dim=-1)  # (batch, n_static * d_model)
            static_context = self.static_grn(static_flat)  # (batch, d_model)

        # --- Encoder: embed and select past variables ---
        past_emb = torch.stack([
            emb(x_past[:, :, i : i + 1])
            for i, emb in enumerate(self.continuous_embeddings)
        ], dim=2)  # (batch, seq, n_vars, d_model)

        encoder_input, var_weights = self.encoder_vsn(past_emb, static_context)
        # encoder_input: (batch, seq, d_model)

        # --- LSTM Encoder ---
        encoder_output, (h_n, c_n) = self.encoder_lstm(encoder_input)
        encoder_output = self.encoder_gate(encoder_output, encoder_input)

        # --- Decoder ---
        if self.decoder_vsn is not None and x_future is not None:
            future_emb = torch.stack([
                emb(x_future[:, :, i : i + 1])
                for i, emb in enumerate(self.future_embeddings)
            ], dim=2)  # (batch, seq_future, n_future_vars, d_model)

            decoder_input, _ = self.decoder_vsn(future_emb, static_context)
            decoder_output, _ = self.decoder_lstm(decoder_input, (h_n, c_n))
            decoder_output = self.decoder_gate(decoder_output, decoder_input)

            # Concatenate encoder and decoder outputs
            temporal = torch.cat([encoder_output, decoder_output], dim=1)
        else:
            temporal = encoder_output

        # --- Static Enrichment ---
        temporal = self.static_enrichment_grn(temporal, static_context)

        # --- Temporal Self-Attention ---
        temporal = self.positional_encoding(temporal)
        seq_len = temporal.size(1)
        causal_mask = create_causal_mask(seq_len, temporal.device)

        attn_output, attn_weights = self.attention(
            temporal, temporal, temporal, mask=causal_mask
        )
        temporal = self.attention_gate(attn_output, temporal)

        # --- Feed-Forward ---
        ff_output = self.ff_grn(temporal)
        temporal = self.ff_gate(ff_output, temporal)

        # --- Output ---
        # Use the last timestep as the encoded state
        encoded_state = self.output_proj(temporal[:, -1, :])  # (batch, d_model)

        return TFTOutput(
            encoded_state=encoded_state,
            temporal_features=temporal,
            attention_weights=attn_weights,
            variable_importance=var_weights,
        )
