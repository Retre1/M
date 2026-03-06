"""Elastic Weight Consolidation (EWC) for continual RL learning.

When the market regime shifts, the model must adapt to new conditions without
forgetting previously learned strategies.  EWC adds a quadratic penalty that
keeps important parameters (measured by Fisher Information) close to their
previous optimal values.

Supports **Online EWC** (running Fisher average across stages) which is more
memory-efficient than storing separate Fisher matrices per task.

Usage
-----
>>> reg = EWCRegularizer(lambda_ewc=5000.0)
>>> # After stage N completes:
>>> reg.compute_fisher(model, replay_buffer, n_samples=2000)
>>> # During stage N+1 — penalty prevents forgetting:
>>> penalty = reg.penalty(model.policy.features_extractor)
>>> penalty.backward()

Integration: EWCCallback applies the penalty every ``update_freq`` RL steps
via a dedicated optimiser (same pattern as DiversityCallback).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class EWCRegularizer:
    """Diagonal Fisher-based EWC with online averaging.

    Parameters
    ----------
    lambda_ewc : float
        Penalty strength.  Higher values → stronger memory retention but
        slower adaptation.  Typical range: 1000–10000 for RL.
    gamma_ewc : float
        Online EWC decay factor.  When a new Fisher is computed, the stored
        Fisher is updated as ``F = gamma * F_old + (1 - gamma) * F_new``.
        Set to 0.0 for standard (non-online) EWC.
    """

    def __init__(
        self,
        lambda_ewc: float = 5000.0,
        gamma_ewc: float = 0.9,
    ) -> None:
        self.lambda_ewc = lambda_ewc
        self.gamma_ewc = gamma_ewc

        self._fisher: dict[str, torch.Tensor] | None = None
        self._optimal_params: dict[str, torch.Tensor] | None = None
        self._n_consolidations: int = 0

    @property
    def has_fisher(self) -> bool:
        """True once at least one consolidation has been performed."""
        return self._fisher is not None

    # ------------------------------------------------------------------
    # Fisher computation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_extractor(model) -> nn.Module | None:
        """Resolve the feature extractor from an SB3 model."""
        # MultiInputPolicy / custom policy → top-level features_extractor
        ext = getattr(model.policy, "features_extractor", None)
        if ext is not None:
            return ext
        # MlpPolicy → actor has its own extractor
        actor = getattr(model.policy, "actor", None)
        if actor is not None:
            ext = getattr(actor, "features_extractor", None)
            if ext is not None:
                return ext
        return None

    def consolidate(
        self,
        model,
        n_samples: int = 2000,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Compute diagonal Fisher Information on the feature extractor.

        Uses the **empirical Fisher** approximation: squared gradients of the
        feature-extractor output w.r.t. its parameters, averaged over replay
        buffer samples.

        Parameters
        ----------
        model : SAC | PPO
            SB3 model with ``policy.features_extractor``.
        n_samples : int
            Number of replay-buffer samples for Fisher estimation.
        batch_size : int
            Mini-batch size for each Fisher accumulation step.

        Returns
        -------
        dict with summary statistics for logging.
        """
        if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
            logger.warning("No replay buffer — skipping Fisher computation")
            return {"fisher_status": "no_buffer"}

        buf_size = model.replay_buffer.size()
        if buf_size < batch_size:
            logger.warning("Replay buffer too small for Fisher", size=buf_size)
            return {"fisher_status": "buffer_too_small"}

        extractor = self._get_extractor(model)
        if extractor is None:
            logger.warning("No features extractor found — skipping Fisher")
            return {"fisher_status": "no_extractor"}

        was_training = extractor.training
        extractor.train()  # need dropout active for realistic gradients

        # Accumulate squared gradients
        new_fisher: dict[str, torch.Tensor] = {}
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param.data)

        if not new_fisher:
            logger.warning("Feature extractor has no trainable params — skipping Fisher")
            if not was_training:
                extractor.eval()
            return {"fisher_status": "no_trainable_params"}

        n_batches = max(1, n_samples // batch_size)
        actual_samples = 0

        # Try to get actor for proper Fisher computation
        actor = getattr(model.policy, "actor", None)

        for _ in range(n_batches):
            replay_data = model.replay_buffer.sample(batch_size)
            obs = replay_data.observations
            actions = replay_data.actions

            extractor.zero_grad()
            if actor is not None:
                actor.zero_grad()

            # Enable grads for this forward pass
            with torch.enable_grad():
                features = extractor(obs)

                if actor is not None:
                    # Proper Fisher: use actor log-probability as surrogate
                    # This measures parameter importance for actual decisions
                    try:
                        mean_actions = actor.action_dist.proba_distribution(
                            *actor.get_action_dist_params(features)
                        ).log_prob(actions)
                        surrogate = -mean_actions.sum()
                    except Exception:
                        # Fallback: use squared feature output
                        surrogate = features.pow(2).sum()
                else:
                    # Fallback for PPO or when actor is not accessible
                    surrogate = features.pow(2).sum()

                surrogate.backward()

            for name, param in extractor.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += param.grad.data.pow(2) * batch_size

            actual_samples += batch_size

        # Average
        for name in new_fisher:
            new_fisher[name] /= actual_samples

        # Online EWC: exponential moving average of Fisher
        if self._fisher is not None and self.gamma_ewc > 0:
            for name in new_fisher:
                if name in self._fisher:
                    new_fisher[name] = (
                        self.gamma_ewc * self._fisher[name]
                        + (1.0 - self.gamma_ewc) * new_fisher[name]
                    )

        self._fisher = new_fisher

        # Store optimal parameters (snapshot)
        self._optimal_params = {}
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                self._optimal_params[name] = param.data.clone()

        self._n_consolidations += 1

        if not was_training:
            extractor.eval()

        # Summary stats
        fisher_norms = {
            name: f.norm().item() for name, f in self._fisher.items()
        }
        mean_norm = sum(fisher_norms.values()) / max(len(fisher_norms), 1)
        max_norm = max(fisher_norms.values()) if fisher_norms else 0.0
        n_params = sum(f.numel() for f in self._fisher.values())

        logger.info(
            "EWC Fisher consolidation complete",
            n_consolidations=self._n_consolidations,
            n_params=n_params,
            mean_fisher_norm=round(mean_norm, 6),
            max_fisher_norm=round(max_norm, 6),
            n_samples=actual_samples,
            online_gamma=self.gamma_ewc,
        )

        return {
            "fisher_status": "ok",
            "n_params": n_params,
            "mean_fisher_norm": mean_norm,
            "max_fisher_norm": max_norm,
            "n_consolidations": self._n_consolidations,
        }

    # ------------------------------------------------------------------
    # Penalty computation
    # ------------------------------------------------------------------

    def penalty(self, extractor: nn.Module) -> torch.Tensor:
        """Compute EWC penalty: λ/2 * Σ F_i * (θ_i - θ*_i)².

        Returns a scalar tensor with grad attached (backprop-ready).
        Returns zero tensor if no Fisher has been computed yet.
        """
        if self._fisher is None or self._optimal_params is None:
            return torch.tensor(0.0, requires_grad=True)

        penalty = torch.tensor(0.0, device=next(extractor.parameters()).device)

        for name, param in extractor.named_parameters():
            if name in self._fisher and name in self._optimal_params:
                fisher_diag = self._fisher[name].to(param.device)
                optimal = self._optimal_params[name].to(param.device)
                penalty = penalty + (fisher_diag * (param - optimal).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * penalty

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise for checkpointing."""
        return {
            "fisher": self._fisher,
            "optimal_params": self._optimal_params,
            "n_consolidations": self._n_consolidations,
            "lambda_ewc": self.lambda_ewc,
            "gamma_ewc": self.gamma_ewc,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self._fisher = state["fisher"]
        self._optimal_params = state["optimal_params"]
        self._n_consolidations = state["n_consolidations"]
        self.lambda_ewc = state["lambda_ewc"]
        self.gamma_ewc = state["gamma_ewc"]
