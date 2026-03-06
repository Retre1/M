"""Epistemic uncertainty estimation via MC Dropout.

During inference the model is uncertain about market states it has not seen
during training.  MC Dropout provides a principled Bayesian approximation:
run *N* stochastic forward passes with dropout **enabled**, then measure
the disagreement across samples.

High variance → model is uncertain → automatically reduce position size.

Usage
-----
>>> estimator = UncertaintyEstimator(n_samples=10, uncertainty_weight=0.5)
>>> result = estimator.estimate(hive_mind, market_features, time_features, ...)
>>> print(result.uncertainty_score, result.position_scale)

The estimator is **inference-only** — it adds no overhead during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class UncertaintyResult:
    """Aggregated uncertainty metrics from MC Dropout sampling."""

    # --- Actions ---
    action_mean: torch.Tensor         # (batch, 1) mean action across N samples
    action_std: torch.Tensor          # (batch, 1) std of action
    uncertainty_score: torch.Tensor   # (batch, 1) normalised ∈ [0, 1]
    position_scale: torch.Tensor      # (batch, 1) ∈ [min_scale, 1.0]

    # --- Per-agent uncertainty ---
    trend_std: torch.Tensor           # (batch, 1)
    reversion_std: torch.Tensor       # (batch, 1)
    breakout_std: torch.Tensor        # (batch, 1)

    # --- Gating uncertainty ---
    gating_mean: torch.Tensor         # (batch, n_agents) mean gating weights
    gating_std: torch.Tensor          # (batch, n_agents) std of gating weights
    gating_entropy: torch.Tensor      # (batch,) Shannon entropy of mean weights

    # --- Meta ---
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Scalar summary for logging (averages over batch)."""
        return {
            "action_mean": self.action_mean.mean().item(),
            "action_std": self.action_std.mean().item(),
            "uncertainty_score": self.uncertainty_score.mean().item(),
            "position_scale": self.position_scale.mean().item(),
            "trend_std": self.trend_std.mean().item(),
            "reversion_std": self.reversion_std.mean().item(),
            "breakout_std": self.breakout_std.mean().item(),
            "gating_entropy": self.gating_entropy.mean().item(),
            "n_samples": self.n_samples,
        }


# ------------------------------------------------------------------
# MC Dropout helpers
# ------------------------------------------------------------------


def _enable_mc_dropout(model: nn.Module) -> None:
    """Switch **only** Dropout layers to training mode (MC Dropout).

    All other layers (BatchNorm, LayerNorm, etc.) stay in eval mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _disable_mc_dropout(model: nn.Module) -> None:
    """Restore all Dropout layers to eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


# ------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------


class UncertaintyEstimator:
    """MC Dropout uncertainty estimator for the HiveMind ensemble.

    Parameters
    ----------
    n_samples : int
        Number of stochastic forward passes (higher = more accurate,
        but linearly more compute).  10 is a good default.
    uncertainty_weight : float
        Controls how aggressively position is scaled down.
        ``position_scale = clamp(1 - weight × uncertainty, min_scale, 1)``.
    min_position_scale : float
        Floor for position scaling — even in extreme uncertainty, never go
        below this fraction of the base position.
    """

    def __init__(
        self,
        n_samples: int = 10,
        uncertainty_weight: float = 0.5,
        min_position_scale: float = 0.1,
    ) -> None:
        self.n_samples = n_samples
        self.uncertainty_weight = uncertainty_weight
        self.min_position_scale = min_position_scale

    @torch.no_grad()
    def estimate(
        self,
        hive_mind: nn.Module,
        *forward_args,
        **forward_kwargs,
    ) -> UncertaintyResult:
        """Run N MC Dropout forward passes and aggregate uncertainty.

        Parameters
        ----------
        hive_mind : HiveMind or MTFHiveMind
            The ensemble model (must have ``forward()`` returning an output
            dataclass with ``action``, ``trend_action``, ``reversion_action``,
            ``breakout_action``, ``gating_weights`` attributes).
        *forward_args, **forward_kwargs :
            Arguments passed directly to ``hive_mind.forward()``.

        Returns
        -------
        UncertaintyResult
        """
        was_training = hive_mind.training
        hive_mind.eval()
        _enable_mc_dropout(hive_mind)

        # --- Collect N stochastic samples ---
        actions: list[torch.Tensor] = []
        trend_actions: list[torch.Tensor] = []
        rev_actions: list[torch.Tensor] = []
        brk_actions: list[torch.Tensor] = []
        gating_weights: list[torch.Tensor] = []

        for _ in range(self.n_samples):
            out = hive_mind(*forward_args, **forward_kwargs)
            actions.append(out.action)
            trend_actions.append(out.trend_action)
            rev_actions.append(out.reversion_action)
            brk_actions.append(out.breakout_action)
            gating_weights.append(out.gating_weights)

        # --- Restore model state ---
        if was_training:
            hive_mind.train()
        else:
            _disable_mc_dropout(hive_mind)

        # --- Aggregate statistics ---
        action_stack = torch.stack(actions)         # (N, batch, 1)
        action_mean = action_stack.mean(dim=0)
        action_std = action_stack.std(dim=0)

        trend_std = torch.stack(trend_actions).std(dim=0)
        rev_std = torch.stack(rev_actions).std(dim=0)
        brk_std = torch.stack(brk_actions).std(dim=0)

        gating_stack = torch.stack(gating_weights)  # (N, batch, n_agents)
        gating_mean = gating_stack.mean(dim=0)
        gating_std = gating_stack.std(dim=0)

        # Shannon entropy of mean gating: higher entropy = less decisive
        gating_entropy = -(
            gating_mean * torch.log(gating_mean + 1e-8)
        ).sum(dim=-1)

        # --- Uncertainty score ---
        # Normalised action std relative to the action range [-1, 1].
        # Max possible std for uniform on [-1,1] ≈ 0.577.  We clamp to [0,1].
        uncertainty_score = (action_std / 0.577).clamp(0.0, 1.0)

        # --- Position scaling ---
        position_scale = (
            1.0 - self.uncertainty_weight * uncertainty_score
        ).clamp(min=self.min_position_scale)

        return UncertaintyResult(
            action_mean=action_mean,
            action_std=action_std,
            uncertainty_score=uncertainty_score,
            position_scale=position_scale,
            trend_std=trend_std,
            reversion_std=rev_std,
            breakout_std=brk_std,
            gating_mean=gating_mean,
            gating_std=gating_std,
            gating_entropy=gating_entropy,
            n_samples=self.n_samples,
        )

    def estimate_from_extractor(
        self,
        extractor: nn.Module,
        observations: dict[str, torch.Tensor],
    ) -> UncertaintyResult:
        """Convenience: run MC Dropout on a HiveMindExtractor.

        Useful when you have SB3-format observations (flat dict).  Calls
        ``estimate()`` on the inner ``extractor.hive_mind`` after unpacking
        the observation tensors.

        Parameters
        ----------
        extractor : HiveMindExtractor or MTFHiveMindExtractor
        observations : dict of tensors (SB3 replay buffer format)
        """
        hm = extractor.hive_mind

        # Detect MTF vs single-TF from observation keys
        if "h1_market_features" in observations:
            return self._estimate_mtf(hm, extractor, observations)
        return self._estimate_single_tf(hm, extractor, observations)

    # ------------------------------------------------------------------
    # Private helpers for obs unpacking
    # ------------------------------------------------------------------

    def _estimate_single_tf(
        self,
        hm: nn.Module,
        ext: nn.Module,
        obs: dict[str, torch.Tensor],
    ) -> UncertaintyResult:
        batch = obs["market_features"].shape[0]
        market = obs["market_features"].reshape(
            batch, ext.seq_len, ext.n_continuous_vars,
        )
        time_feat = obs["time_features"].reshape(
            batch, ext.seq_len, ext.n_known_future_vars,
        )
        return self.estimate(
            hm,
            market,
            time_feat,
            obs["trend_features"],
            obs["reversion_features"],
            obs["regime_features"],
            breakout_features=obs.get("breakout_features"),
        )

    def _estimate_mtf(
        self,
        hm: nn.Module,
        ext: nn.Module,
        obs: dict[str, torch.Tensor],
    ) -> UncertaintyResult:
        batch = obs["h1_market_features"].shape[0]
        return self.estimate(
            hm,
            obs["d1_market_features"].reshape(batch, ext.d1_lookback, ext.n_continuous_vars),
            obs["d1_time_features"].reshape(batch, ext.d1_lookback, ext.n_known_future_vars),
            obs["h1_market_features"].reshape(batch, ext.h1_lookback, ext.n_continuous_vars),
            obs["h1_time_features"].reshape(batch, ext.h1_lookback, ext.n_known_future_vars),
            obs["m5_market_features"].reshape(batch, ext.m5_lookback, ext.n_continuous_vars),
            obs["m5_time_features"].reshape(batch, ext.m5_lookback, ext.n_known_future_vars),
            obs["trend_features"],
            obs["reversion_features"],
            obs["regime_features"],
            mtf_context=obs.get("mtf_context"),
            breakout_features=obs.get("breakout_features"),
        )
