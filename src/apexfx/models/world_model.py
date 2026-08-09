"""World Model Hybrid — Pluggable-backend dynamics with SBBTS imagination.

Replaces the original GRU-ensemble WorldModel with a modular architecture
supporting multiple dynamics backends:

    1. **Mamba** (Selective SSM) — default, best for sequential forex data
    2. **Tensor Networks** (MPS/PEPS) — compact, captures local correlations
    3. **Quantum Feature Maps** — exponentially rich non-linear encoding
    4. **Hybrid** — learned fusion of multiple backends

Key additions over v1:
    - SBBTS-based imagination rollouts (Schrödinger noise injection)
    - DML-pricer integration for jump-risk estimation in latent space
    - Per-backend ensemble for uncertainty estimation
    - Backward-compatible with WorldModelCallback API

References:
    - Mamba (Gu & Dao, 2023)
    - Dreamer (Hafner et al., 2020)
    - DML pricing (Paper 5)
    - SBBTS augmentation (Paper 4)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback

from apexfx.models.config import (
    BackendType,
    WorldModelHybridConfig,
)
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ── Helper: legacy DynamicsHead kept for backward compat ─────────────

class DynamicsHead(nn.Module):
    """Single GRU-based dynamics model: (z_t, a_t) → z_{t+1}.

    Kept as fallback when no backend is configured.
    """

    def __init__(self, d_latent: int, d_action: int,
                 d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )
        self.residual = nn.Linear(d_latent, d_latent, bias=False)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        delta = self.net(x)
        return self.residual(z) + delta


# ── Backend factory ──────────────────────────────────────────────────

def _build_backend(
    backend_type: BackendType,
    d_latent: int,
    d_action: int,
    cfg: WorldModelHybridConfig,
) -> nn.Module:
    """Instantiate a dynamics backend from config."""
    if backend_type == BackendType.MAMBA:
        from apexfx.models.backends.mamba_backend import MambaBackend
        return MambaBackend(d_latent, d_action, cfg.mamba)

    if backend_type == BackendType.TENSOR_NETWORK:
        from apexfx.models.backends.tensor_network_backend import TensorNetworkBackend
        return TensorNetworkBackend(d_latent, d_action, cfg.tensor_network)

    if backend_type == BackendType.QUANTUM_FEATURE_MAP:
        from apexfx.models.backends.quantum_feature_map_backend import QuantumFeatureMapBackend
        return QuantumFeatureMapBackend(d_latent, d_action, cfg.quantum_feature_map)

    if backend_type == BackendType.HYBRID:
        # Hybrid: build all sub-backends listed in config
        sub_backends = nn.ModuleList([
            _build_backend(bt, d_latent, d_action, cfg)
            for bt in cfg.hybrid.backends
        ])
        return HybridDynamics(sub_backends, d_latent, cfg.hybrid.fusion)

    raise ValueError(f"Unknown backend: {backend_type}")


class HybridDynamics(nn.Module):
    """Learned fusion of multiple dynamics backends."""

    def __init__(
        self,
        backends: nn.ModuleList,
        d_latent: int,
        fusion: str = "learned_gate",
    ) -> None:
        super().__init__()
        self.backends = backends
        self.fusion = fusion
        n = len(backends)

        if fusion == "learned_gate":
            self.gate = nn.Sequential(
                nn.Linear(d_latent, n),
                nn.Softmax(dim=-1),
            )
        elif fusion == "concat":
            self.proj = nn.Linear(d_latent * n, d_latent)
        elif fusion == "attention":
            self.attn = nn.MultiheadAttention(d_latent, num_heads=1, batch_first=True)
        self.residual = nn.Linear(d_latent, d_latent, bias=False)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = [b(z, action) for b in self.backends]
        stacked = torch.stack(preds, dim=1)  # (batch, n_backends, d_latent)

        if self.fusion == "learned_gate":
            weights = self.gate(z)  # (batch, n_backends)
            out = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion == "concat":
            out = self.proj(torch.cat(preds, dim=-1))
        elif self.fusion == "attention":
            q = z.unsqueeze(1)  # (batch, 1, d_latent)
            out, _ = self.attn(q, stacked, stacked)
            out = out.squeeze(1)
        else:
            out = stacked.mean(dim=1)

        return out


# ── WorldModelHybrid ─────────────────────────────────────────────────

class WorldModelHybrid(nn.Module):
    """Learned dynamics model with pluggable backends.

    Architecture:
        Encoder → Latent z_t
        Backend ensemble: (z_t, a_t) → z_{t+1}  (N members for uncertainty)
        Reward head: z_t → r_t
        Done head: z_t → p(done)
        DML head (optional): z_t → jump risk estimate

    Parameters
    ----------
    d_features : int
        Dimension of the SB3 feature extractor output.
    config : WorldModelHybridConfig
        Full configuration. Defaults to Mamba backend.
    """

    def __init__(
        self,
        d_features: int = 77,
        config: WorldModelHybridConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Backward compat: accept old-style kwargs (d_latent, d_action, etc.)
        if config is None and kwargs:
            compat_args = {}
            for key in ("d_latent", "d_action", "d_hidden", "n_ensemble", "dropout"):
                if key in kwargs:
                    compat_args[key] = kwargs[key]
            cfg = WorldModelHybridConfig(**compat_args) if compat_args else WorldModelHybridConfig()
        else:
            cfg = config or WorldModelHybridConfig()
        self._cfg = cfg
        self.d_latent = cfg.d_latent
        self.d_features = d_features
        self.n_ensemble = cfg.n_ensemble

        # Encoder: features → latent
        self.encoder = nn.Sequential(
            nn.Linear(d_features, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_latent),
        )

        # Dynamics ensemble (each member uses the selected backend)
        self.dynamics = nn.ModuleList([
            _build_backend(cfg.backend, cfg.d_latent, cfg.d_action, cfg)
            for _ in range(cfg.n_ensemble)
        ])

        # Reward predictor: z_t → r_t
        self.reward_head = nn.Sequential(
            nn.Linear(cfg.d_latent, cfg.d_hidden // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden // 2, 1),
        )

        # Done predictor: z_t → probability of episode termination
        self.done_head = nn.Sequential(
            nn.Linear(cfg.d_latent, cfg.d_hidden // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # Optional DML jump-risk head
        self.dml_head: nn.Module | None = None
        if cfg.dml_pricer:
            self.dml_head = nn.Sequential(
                nn.Linear(cfg.d_latent, cfg.d_hidden // 2),
                nn.SiLU(),
                nn.Linear(cfg.d_hidden // 2, 3),  # [jump_prob, jump_mean, jump_std]
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features to latent state."""
        return self.encoder(features)

    def predict_next(
        self, z: torch.Tensor, action: torch.Tensor,
        member_idx: int | None = None,
    ) -> torch.Tensor:
        """Predict next latent state.

        If member_idx is None, returns ensemble mean.
        """
        if member_idx is not None:
            return self.dynamics[member_idx](z, action)
        predictions = torch.stack([d(z, action) for d in self.dynamics])
        return predictions.mean(dim=0)

    def predict_reward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        return self.reward_head(z)

    def predict_done(self, z: torch.Tensor) -> torch.Tensor:
        """Predict termination probability from latent state."""
        return self.done_head(z)

    def predict_jump_risk(self, z: torch.Tensor) -> torch.Tensor | None:
        """Predict jump-diffusion parameters from latent state.

        Returns (batch, 3): [jump_probability, jump_mean, jump_std]
        or None if DML head is disabled.
        """
        if self.dml_head is None:
            return None
        raw = self.dml_head(z)
        # Constrain: prob in [0,1], std > 0
        return torch.stack([
            torch.sigmoid(raw[:, 0]),
            raw[:, 1],
            F.softplus(raw[:, 2]),
        ], dim=-1)

    def ensemble_disagreement(
        self, z: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute disagreement across ensemble members (curiosity signal)."""
        predictions = torch.stack([d(z, action) for d in self.dynamics])
        return predictions.std(dim=0).mean(dim=-1, keepdim=True)

    def imagine(
        self,
        z_start: torch.Tensor,
        policy_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: int | None = None,
        sbbts_noise_scale: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        """Imagination rollout with optional SBBTS noise injection.

        Parameters
        ----------
        z_start : torch.Tensor
            Starting latent state (batch, d_latent).
        policy_fn : callable
            Maps latent → action.
        horizon : int
            Steps to imagine. Defaults to config value.
        sbbts_noise_scale : float
            Scale of Schrödinger-Bass noise added to imagined transitions.
            Set to 0 to disable.

        Returns
        -------
        dict with imagined latents, actions, rewards, dones, jump_risk.
        """
        H = horizon or self._cfg.imagination_horizon

        z = z_start
        latents = [z]
        actions_list: list[torch.Tensor] = []
        rewards_list: list[torch.Tensor] = []
        dones_list: list[torch.Tensor] = []
        jump_risks: list[torch.Tensor] = []

        for _step in range(H):
            with torch.no_grad():
                a = policy_fn(z)
            z_next = self.predict_next(z, a)

            # SBBTS noise injection: adds stochastic perturbation
            # inspired by Schrödinger-Bass dynamics for more realistic
            # imagined trajectories
            if self._cfg.sbbts_imagination and sbbts_noise_scale > 0:
                noise = torch.randn_like(z_next) * sbbts_noise_scale
                # Scale noise by ensemble disagreement (more noise where uncertain)
                with torch.no_grad():
                    uncertainty = self.ensemble_disagreement(z, a)
                noise = noise * (1.0 + uncertainty)
                z_next = z_next + noise

            r = self.predict_reward(z_next)
            d = self.predict_done(z_next)

            latents.append(z_next)
            actions_list.append(a)
            rewards_list.append(r)
            dones_list.append(d)

            # Jump risk from DML head
            jr = self.predict_jump_risk(z_next)
            if jr is not None:
                jump_risks.append(jr)

            z = z_next

        result: dict[str, torch.Tensor] = {
            "latents": torch.stack(latents, dim=1),
            "actions": torch.stack(actions_list, dim=1),
            "rewards": torch.stack(rewards_list, dim=1),
            "dones": torch.stack(dones_list, dim=1),
        }
        if jump_risks:
            result["jump_risk"] = torch.stack(jump_risks, dim=1)

        return result


# ── Backward-compatible aliases ──────────────────────────────────────

# Allow existing code to import WorldModel without changes
WorldModel = WorldModelHybrid


# ── WorldModelCallback (updated for v2) ─────────────────────────────

class WorldModelCallback(BaseCallback):
    """Train the WorldModelHybrid alongside RL, provide curiosity bonus.

    Updates from v1:
    - Uses WorldModelHybrid with pluggable backends
    - Trains DML jump-risk head alongside dynamics
    - Logs per-backend and per-ensemble metrics

    Parameters
    ----------
    d_features : int
        Feature extractor output dimension.
    config : WorldModelHybridConfig | None
        World model configuration. Uses defaults if None.
    """

    def __init__(
        self,
        d_features: int = 77,
        config: WorldModelHybridConfig | None = None,
    ) -> None:
        super().__init__()
        self._config = config or WorldModelHybridConfig()
        self.d_features = d_features

        self._world_model: WorldModelHybrid | None = None
        self._optimizer: torch.optim.Adam | None = None
        self._total_wm_loss: float = 0.0
        self._wm_updates: int = 0
        self._last_curiosity_bonus: float = 0.0

    def _init_world_model(self) -> None:
        """Lazy init — need device and actual feature dim from the RL model."""
        device = next(self.model.policy.parameters()).device

        # Resolve actual features-extractor output dim (may differ from the
        # 77 default, e.g. CombinedExtractor over Dict obs spaces).
        extractor = (
            self.model.policy.features_extractor
            or getattr(self.model.policy.actor, "features_extractor", None)
            or getattr(self.model.policy.critic, "features_extractor", None)
        )
        if extractor is not None and hasattr(extractor, "features_dim"):
            resolved = int(extractor.features_dim)
            if resolved != self.d_features:
                logger.info(
                    "World model feature dim auto-resolved",
                    requested=self.d_features, actual=resolved,
                )
                self.d_features = resolved

        self._world_model = WorldModelHybrid(
            d_features=self.d_features,
            config=self._config,
        ).to(device)
        self._optimizer = torch.optim.Adam(
            self._world_model.parameters(), lr=self._config.lr,
        )
        n_params = sum(p.numel() for p in self._world_model.parameters())
        logger.info(
            "WorldModelHybrid initialized",
            backend=self._config.backend.value,
            n_params=n_params,
            n_ensemble=self._config.n_ensemble,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps % self._config.update_freq != 0:
            return True

        if self._world_model is None:
            self._init_world_model()

        model = self.model
        if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
            return True
        if model.replay_buffer.size() < max(
            self._config.batch_size, model.learning_starts,
        ):
            return True

        try:
            self._train_world_model()
        except Exception as e:
            if self._wm_updates == 0:
                import traceback as _tb
                logger.warning(
                    "World model training failed",
                    error=str(e),
                    trace=_tb.format_exc(),
                )

        return True

    def _train_world_model(self) -> None:
        """Train world model on replay buffer transitions."""
        model = self.model
        wm = self._world_model
        cfg = self._config

        replay_data = model.replay_buffer.sample(cfg.batch_size)
        obs = replay_data.observations
        actions = replay_data.actions
        next_obs = replay_data.next_observations
        rewards = replay_data.rewards
        dones = replay_data.dones

        # SAC/TQC keep features_extractor on actor/critic, not on the top-level
        # policy (which reports None). Resolve whichever extractor is live.
        extractor = (
            model.policy.features_extractor
            or getattr(model.policy.actor, "features_extractor", None)
            or getattr(model.policy.critic, "features_extractor", None)
        )
        if extractor is None:
            raise RuntimeError("No features_extractor available on policy")

        with torch.no_grad():
            features = extractor(obs)
            next_features = extractor(next_obs)

        z = wm.encode(features)
        with torch.no_grad():
            z_next_actual = wm.encode(next_features)

        # Bootstrap ensemble training
        dynamics_loss = torch.tensor(0.0, device=z.device)
        for _i, dyn in enumerate(wm.dynamics):
            n = z.shape[0]
            idx = torch.randint(0, n, (int(n * 0.8),), device=z.device)
            z_boot = z[idx]
            a_boot = actions[idx]
            z_next_boot = z_next_actual[idx]
            z_next_pred = dyn(z_boot, a_boot)
            dynamics_loss = dynamics_loss + F.mse_loss(z_next_pred, z_next_boot)
        dynamics_loss = dynamics_loss / wm.n_ensemble

        # Reward prediction loss
        reward_pred = wm.predict_reward(z)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # Done prediction loss
        done_pred = wm.predict_done(z)
        done_loss = F.binary_cross_entropy(done_pred, dones.float())

        total_loss = (
            dynamics_loss
            + cfg.reward_pred_weight * reward_loss
            + cfg.done_pred_weight * done_loss
        )

        # DML jump-risk loss (self-supervised: predict next-step return magnitude)
        if wm.dml_head is not None:
            jr = wm.predict_jump_risk(z)
            if jr is not None:
                # Use reward magnitude as proxy for jump events
                jump_target = (rewards.abs() > rewards.std() * 2).float()
                jr_loss = F.binary_cross_entropy(jr[:, 0:1], jump_target)
                total_loss = total_loss + 0.1 * jr_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        self._optimizer.step()

        self._total_wm_loss += total_loss.item()
        self._wm_updates += 1

        # Curiosity bonus
        if cfg.curiosity_weight > 0:
            with torch.no_grad():
                disagreement = wm.ensemble_disagreement(z, actions)
                self._last_curiosity_bonus = (
                    disagreement * cfg.curiosity_weight
                ).mean().item()

        # Logging
        if self._wm_updates % 10 == 0:
            avg_loss = self._total_wm_loss / max(self._wm_updates, 1)
            self.logger.record("world_model/loss", avg_loss)
            self.logger.record("world_model/dynamics_loss", dynamics_loss.item())
            self.logger.record("world_model/reward_loss", reward_loss.item())
            self.logger.record("world_model/backend", cfg.backend.value)

            with torch.no_grad():
                disagreement = wm.ensemble_disagreement(z[:32], actions[:32])
                self.logger.record("world_model/curiosity", disagreement.mean().item())

    @property
    def world_model(self) -> WorldModelHybrid | None:
        return self._world_model
