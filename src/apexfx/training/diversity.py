"""Ensemble diversity regularization — prevents agent collapse and encourages specialization.

Implements two complementary diversity mechanisms:

1. **Action Variance Penalty**: penalises when all agents produce identical outputs.
   Encourages each specialist to develop a unique trading signal.

2. **Pairwise Correlation Penalty**: penalises high absolute Pearson correlation
   between agent output streams within each mini-batch.

3. **Gating Entropy Bonus**: encourages the gating network to utilise all agents
   instead of collapsing onto a single specialist.

Applied as a periodic auxiliary optimisation step via ``DiversityCallback``
(see ``callbacks.py``), separate from the main RL training loop.
"""

from __future__ import annotations

import torch

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Loss computation
# ------------------------------------------------------------------


def compute_diversity_loss(
    agent_actions: list[torch.Tensor],
    gating_weights: torch.Tensor,
    diversity_weight: float = 0.01,
    entropy_weight: float = 0.1,
) -> torch.Tensor:
    """Compute the combined ensemble diversity loss.

    The loss has three terms, all differentiable:

    * **-action_variance** — we *maximise* cross-agent variance so that
      agents produce spread-out signals.
    * **+correlation_penalty** — we *minimise* the absolute Pearson
      correlation between each pair of agent outputs.
    * **-gating_entropy** — we *maximise* the Shannon entropy of gating
      weights to prevent weight collapse.

    Parameters
    ----------
    agent_actions : list of (batch, 1) tensors
        Actions from each specialist agent.
    gating_weights : (batch, n_agents)
        Softmax weights from the gating network.
    diversity_weight : float
        Coefficient for variance & correlation terms.
    entropy_weight : float
        Coefficient for gating entropy term.

    Returns
    -------
    loss : scalar tensor  (lower → more diverse)
    """
    device = gating_weights.device

    # --- 1. Action variance (cross-agent) ---
    actions = torch.cat(agent_actions, dim=-1)  # (batch, n_agents)
    action_variance = actions.var(dim=-1).mean()

    # --- 2. Pairwise correlation penalty ---
    n_agents = len(agent_actions)
    correlation_penalty = torch.tensor(0.0, device=device, requires_grad=True)
    n_pairs = 0
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            ai = agent_actions[i].squeeze(-1)
            aj = agent_actions[j].squeeze(-1)
            ai_c = ai - ai.mean()
            aj_c = aj - aj.mean()
            numer = (ai_c * aj_c).mean()
            denom = ai_c.std() * aj_c.std() + 1e-8
            corr = numer / denom
            correlation_penalty = correlation_penalty + corr.abs()
            n_pairs += 1
    if n_pairs > 0:
        correlation_penalty = correlation_penalty / n_pairs

    # --- 3. Gating entropy ---
    entropy = -(gating_weights * torch.log(gating_weights + 1e-8)).sum(dim=-1).mean()
    max_entropy = torch.log(torch.tensor(float(n_agents), device=device))
    normalised_entropy = entropy / (max_entropy + 1e-8)

    # --- Combined ---
    loss = (
        -diversity_weight * action_variance
        + diversity_weight * correlation_penalty
        - entropy_weight * normalised_entropy
    )
    return loss


# ------------------------------------------------------------------
# Non-differentiable diagnostics
# ------------------------------------------------------------------


class DiversityMetrics:
    """Compute human-readable diversity diagnostics (detached, no grad)."""

    @staticmethod
    @torch.no_grad()
    def compute(
        agent_actions: list[torch.Tensor],
        gating_weights: torch.Tensor,
    ) -> dict[str, float]:
        """Return a flat dict of diversity statistics for logging.

        Keys
        ----
        action_variance        – mean variance across agents per sample
        mean_abs_correlation   – mean |ρ| between agent pairs
        gating_entropy         – raw Shannon entropy of gating weights
        gating_entropy_norm    – entropy ÷ max possible entropy
        gating_max_weight      – mean of per-sample max gating weight
        agent_utilisation      – fraction of agents with mean weight > 0.1
        """
        actions = torch.cat(agent_actions, dim=-1)
        n_agents = actions.shape[-1]

        # Variance
        action_var = actions.var(dim=-1).mean().item()

        # Pairwise |correlation|
        corrs: list[float] = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                ai = agent_actions[i].squeeze(-1)
                aj = agent_actions[j].squeeze(-1)
                ai_c = ai - ai.mean()
                aj_c = aj - aj.mean()
                numer = (ai_c * aj_c).mean()
                denom = ai_c.std() * aj_c.std() + 1e-8
                corrs.append((numer / denom).abs().item())
        mean_corr = sum(corrs) / max(len(corrs), 1)

        # Gating entropy
        ent = -(gating_weights * torch.log(gating_weights + 1e-8)).sum(-1).mean().item()
        max_ent = float(torch.log(torch.tensor(float(n_agents))).item())

        # Agent utilisation
        mean_w = gating_weights.mean(dim=0)
        util = (mean_w > 0.1).float().mean().item()

        return {
            "action_variance": action_var,
            "mean_abs_correlation": mean_corr,
            "gating_entropy": ent,
            "gating_entropy_norm": ent / (max_ent + 1e-8),
            "gating_max_weight": gating_weights.max(dim=-1).values.mean().item(),
            "agent_utilisation": util,
        }
