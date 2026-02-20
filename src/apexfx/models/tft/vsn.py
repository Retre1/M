"""Variable Selection Network — selects and weights input variables by importance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from apexfx.models.tft.grn import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """
    VSN applies per-variable GRN transformations, then produces
    softmax-weighted variable importance scores.
    Provides interpretability: inspect which features the model considers important.
    """

    def __init__(
        self,
        d_model: int,
        n_vars: int,
        dropout: float = 0.1,
        d_context: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars

        # Per-variable GRNs: transform each variable independently
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(
                d_input=d_model,
                d_hidden=d_model,
                d_output=d_model,
                dropout=dropout,
            )
            for _ in range(n_vars)
        ])

        # Joint GRN for variable importance weights
        self.weight_grn = GatedResidualNetwork(
            d_input=d_model * n_vars,
            d_hidden=d_model,
            d_output=n_vars,
            dropout=dropout,
            d_context=d_context,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, n_vars, d_model) — each variable embedded to d_model
            context: optional (batch, time, d_context) or (batch, d_context)

        Returns:
            selected: (batch, time, d_model) — weighted sum of variable embeddings
            weights: (batch, time, n_vars) — variable importance weights
        """
        batch, time, n_vars, d_model = x.shape
        assert n_vars == self.n_vars, f"Expected {self.n_vars} vars, got {n_vars}"

        # Transform each variable
        transformed = []
        for i in range(n_vars):
            var_input = x[:, :, i, :]  # (batch, time, d_model)
            var_out = self.var_grns[i](var_input)  # (batch, time, d_model)
            transformed.append(var_out)

        # Stack: (batch, time, n_vars, d_model)
        transformed_stack = torch.stack(transformed, dim=2)

        # Flatten all variables for the weight GRN
        flat = x.reshape(batch, time, n_vars * d_model)  # (batch, time, n_vars * d_model)

        # Expand context if needed
        if context is not None and context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, time, -1)

        # Compute importance weights
        weight_logits = self.weight_grn(flat, context)  # (batch, time, n_vars)
        weights = F.softmax(weight_logits, dim=-1)  # (batch, time, n_vars)

        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # (batch, time, n_vars, 1)
        selected = (transformed_stack * weights_expanded).sum(dim=2)  # (batch, time, d_model)

        return selected, weights
