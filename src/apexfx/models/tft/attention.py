"""Interpretable Multi-Head Attention for the Temporal Fusion Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights.

    Key difference from standard MHA: attention weights are shared across
    heads for interpretability — you can directly read which time steps
    the model attends to. Uses separate value heads but a single set of
    attention weights.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: combines multi-head outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # For interpretable attention: weight-sharing across heads
        self.W_h = nn.Linear(self.d_k, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: optional (batch, seq_q, seq_k) or (seq_q, seq_k) boolean mask

        Returns:
            output: (batch, seq_q, d_model)
            attention_weights: (batch, seq_q, seq_k) — interpretable weights
        """
        batch = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self._split_heads(self.W_q(query))  # (batch, n_heads, seq_q, d_k)
        K = self._split_heads(self.W_k(key))    # (batch, n_heads, seq_k, d_k)
        V = self._split_heads(self.W_v(value))  # (batch, n_heads, seq_k, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, n_heads, seq_q, seq_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention weights per head
        attn_weights_per_head = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_q, seq_k)
        attn_weights_per_head = self.dropout(attn_weights_per_head)

        # Interpretable attention: average across heads for a single set of weights
        attn_weights = attn_weights_per_head.mean(dim=1)  # (batch, seq_q, seq_k)

        # Weighted sum of values per head
        head_outputs = torch.matmul(attn_weights_per_head, V)  # (batch, n_heads, seq_q, d_k)

        # Combine heads via interpretable weight-sharing
        # head_outputs: (batch, n_heads, seq_q, d_k) → (batch, seq_q, n_heads, d_k)
        head_outputs = head_outputs.permute(0, 2, 1, 3)

        # Weight each head's contribution via W_h for interpretability
        # This produces a single d_k-dimensional output per position
        head_weights = self.W_h(head_outputs)  # (batch, seq_q, n_heads, 1)
        head_weights = F.softmax(head_weights, dim=2)
        output = (head_outputs * head_weights).sum(dim=2)  # (batch, seq_q, d_k)

        # Project back to d_model
        # We need to go from d_k to d_model
        # Pad or project
        output = self._concat_heads_interpretable(head_outputs, batch)
        output = self.W_o(output)

        return output, attn_weights

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _concat_heads_interpretable(
        self, head_outputs: torch.Tensor, batch: int
    ) -> torch.Tensor:
        """(batch, seq_q, n_heads, d_k) → (batch, seq_q, d_model)"""
        batch, seq_q, n_heads, d_k = head_outputs.shape
        return head_outputs.reshape(batch, seq_q, n_heads * d_k)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower-triangular) attention mask."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
