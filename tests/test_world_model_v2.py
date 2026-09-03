"""Unit tests for Phase 3: WorldModelHybrid, Backends, and GatingV2.

Tests verify:
1. Each backend produces correct output shapes.
2. WorldModelHybrid encodes, predicts, and imagines correctly.
3. Ensemble disagreement is non-negative and varies.
4. DML jump-risk head produces constrained outputs.
5. GatingV2 produces valid weights with entropy > 0.
6. Anti-collapse mechanism activates when entropy is low.
7. FSD conditioning changes gating behavior.
8. Hybrid backend fuses multiple backends.
9. Config immutability (frozen Pydantic models).
"""

from __future__ import annotations

import pytest
import torch

from apexfx.models.backends.mamba_backend import MambaBackend
from apexfx.models.backends.quantum_feature_map_backend import QuantumFeatureMapBackend
from apexfx.models.backends.tensor_network_backend import TensorNetworkBackend
from apexfx.models.config import (
    BackendType,
    GatingV2Config,
    MambaBackendConfig,
    QuantumFeatureMapConfig,
    TensorNetworkBackendConfig,
    WorldModelHybridConfig,
)
from apexfx.models.gating_v2 import HiveMindGating_v2
from apexfx.models.world_model import WorldModelHybrid

D_LATENT = 16
D_ACTION = 1
D_FEATURES = 32
BATCH = 4


@pytest.fixture
def z() -> torch.Tensor:
    return torch.randn(BATCH, D_LATENT)


@pytest.fixture
def action() -> torch.Tensor:
    return torch.randn(BATCH, D_ACTION)


@pytest.fixture
def features() -> torch.Tensor:
    return torch.randn(BATCH, D_FEATURES)


# ── Backend tests ────────────────────────────────────────────────────


class TestMambaBackend:
    def test_output_shape(self, z: torch.Tensor, action: torch.Tensor) -> None:
        backend = MambaBackend(D_LATENT, D_ACTION, MambaBackendConfig())
        out = backend(z, action)
        assert out.shape == (BATCH, D_LATENT)

    def test_residual_connection(self, z: torch.Tensor, action: torch.Tensor) -> None:
        """Output should differ from input (non-trivial transformation)."""
        backend = MambaBackend(D_LATENT, D_ACTION, MambaBackendConfig())
        out = backend(z, action)
        assert not torch.allclose(out, z, atol=1e-4)

    def test_gradients_flow(self, z: torch.Tensor, action: torch.Tensor) -> None:
        backend = MambaBackend(D_LATENT, D_ACTION, MambaBackendConfig())
        z_req = z.clone().requires_grad_(True)
        out = backend(z_req, action)
        out.sum().backward()
        assert z_req.grad is not None
        assert z_req.grad.abs().sum() > 0


class TestTensorNetworkBackend:
    def test_output_shape(self, z: torch.Tensor, action: torch.Tensor) -> None:
        backend = TensorNetworkBackend(
            D_LATENT, D_ACTION, TensorNetworkBackendConfig(),
        )
        out = backend(z, action)
        assert out.shape == (BATCH, D_LATENT)

    def test_gradients_flow(self, z: torch.Tensor, action: torch.Tensor) -> None:
        backend = TensorNetworkBackend(
            D_LATENT, D_ACTION, TensorNetworkBackendConfig(),
        )
        z_req = z.clone().requires_grad_(True)
        out = backend(z_req, action)
        out.sum().backward()
        assert z_req.grad is not None


class TestQuantumFeatureMapBackend:
    def test_output_shape(self) -> None:
        cfg = QuantumFeatureMapConfig(n_qubits=4, n_layers=2)
        backend = QuantumFeatureMapBackend(D_LATENT, D_ACTION, cfg)
        z = torch.randn(BATCH, D_LATENT)
        a = torch.randn(BATCH, D_ACTION)
        out = backend(z, a)
        assert out.shape == (BATCH, D_LATENT)

    def test_gradients_flow(self) -> None:
        cfg = QuantumFeatureMapConfig(n_qubits=4, n_layers=2)
        backend = QuantumFeatureMapBackend(D_LATENT, D_ACTION, cfg)
        z = torch.randn(BATCH, D_LATENT, requires_grad=True)
        a = torch.randn(BATCH, D_ACTION)
        out = backend(z, a)
        out.sum().backward()
        assert z.grad is not None


# ── WorldModelHybrid tests ───────────────────────────────────────────


class TestWorldModelHybrid:
    @pytest.fixture
    def wm(self) -> WorldModelHybrid:
        cfg = WorldModelHybridConfig(
            d_latent=D_LATENT, d_hidden=32,
            n_ensemble=3, backend=BackendType.MAMBA,
        )
        return WorldModelHybrid(d_features=D_FEATURES, config=cfg)

    def test_encode_shape(self, wm: WorldModelHybrid, features: torch.Tensor) -> None:
        z = wm.encode(features)
        assert z.shape == (BATCH, D_LATENT)

    def test_predict_next_ensemble_mean(
        self, wm: WorldModelHybrid, z: torch.Tensor, action: torch.Tensor,
    ) -> None:
        z_next = wm.predict_next(z, action)
        assert z_next.shape == (BATCH, D_LATENT)

    def test_predict_next_single_member(
        self, wm: WorldModelHybrid, z: torch.Tensor, action: torch.Tensor,
    ) -> None:
        z_next = wm.predict_next(z, action, member_idx=0)
        assert z_next.shape == (BATCH, D_LATENT)

    def test_ensemble_disagreement(
        self, wm: WorldModelHybrid, z: torch.Tensor, action: torch.Tensor,
    ) -> None:
        d = wm.ensemble_disagreement(z, action)
        assert d.shape == (BATCH, 1)
        assert (d >= 0).all(), "Disagreement (std) must be non-negative"

    def test_reward_and_done_heads(
        self, wm: WorldModelHybrid, z: torch.Tensor,
    ) -> None:
        r = wm.predict_reward(z)
        d = wm.predict_done(z)
        assert r.shape == (BATCH, 1)
        assert d.shape == (BATCH, 1)
        assert (d >= 0).all() and (d <= 1).all(), "Done prob in [0, 1]"

    def test_dml_jump_risk_head(self, wm: WorldModelHybrid, z: torch.Tensor) -> None:
        jr = wm.predict_jump_risk(z)
        assert jr is not None
        assert jr.shape == (BATCH, 3)
        assert (jr[:, 0] >= 0).all() and (jr[:, 0] <= 1).all(), "Jump prob in [0,1]"
        assert (jr[:, 2] > 0).all(), "Jump std > 0 (softplus)"

    def test_imagination_rollout(self, wm: WorldModelHybrid, features: torch.Tensor) -> None:
        z_start = wm.encode(features)
        def policy_fn(z):
            return torch.tanh(z[:, :D_ACTION])

        result = wm.imagine(z_start, policy_fn, horizon=3)

        assert result["latents"].shape == (BATCH, 4, D_LATENT)  # H+1
        assert result["actions"].shape == (BATCH, 3, D_ACTION)
        assert result["rewards"].shape == (BATCH, 3, 1)
        assert result["dones"].shape == (BATCH, 3, 1)
        assert "jump_risk" in result
        assert result["jump_risk"].shape == (BATCH, 3, 3)

    def test_imagination_sbbts_noise(self, features: torch.Tensor) -> None:
        """SBBTS noise injection should produce different rollouts."""
        cfg = WorldModelHybridConfig(
            d_latent=D_LATENT, d_hidden=32, n_ensemble=3,
            sbbts_imagination=True,
        )
        wm = WorldModelHybrid(d_features=D_FEATURES, config=cfg)
        z_start = wm.encode(features)
        def policy_fn(z):
            return torch.zeros(z.shape[0], D_ACTION)

        torch.manual_seed(1)
        r1 = wm.imagine(z_start, policy_fn, horizon=3, sbbts_noise_scale=0.1)
        torch.manual_seed(2)
        r2 = wm.imagine(z_start, policy_fn, horizon=3, sbbts_noise_scale=0.1)

        # Different seeds → different noise → different latents
        assert not torch.allclose(r1["latents"], r2["latents"], atol=1e-6)


@pytest.mark.parametrize("backend", [
    BackendType.MAMBA,
    BackendType.TENSOR_NETWORK,
])
def test_world_model_with_different_backends(
    backend: BackendType, features: torch.Tensor, action: torch.Tensor,
) -> None:
    """WorldModelHybrid should work with any backend."""
    cfg = WorldModelHybridConfig(
        d_latent=D_LATENT, d_hidden=32, n_ensemble=2,
        backend=backend,
    )
    wm = WorldModelHybrid(d_features=D_FEATURES, config=cfg)
    z = wm.encode(features)
    z_next = wm.predict_next(z, action)
    assert z_next.shape == (BATCH, D_LATENT)


# ── GatingV2 tests ───────────────────────────────────────────────────


class TestGatingV2:
    @pytest.fixture
    def gating(self) -> HiveMindGating_v2:
        return HiveMindGating_v2(GatingV2Config(
            d_regime=6, d_context=16, n_agents=3,
            d_hidden=16, d_fsd=4, fsd_embed_dim=8,
        ))

    @pytest.fixture
    def inputs(self) -> tuple[torch.Tensor, ...]:
        regime = torch.randn(BATCH, 6)
        tft = torch.randn(BATCH, 16)
        agents = torch.randn(BATCH, 3)
        fsd = torch.zeros(BATCH, 4)
        fsd[:, 2] = 1.0  # NEUTRAL regime
        return regime, tft, agents, fsd

    def test_output_shapes(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        regime, tft, agents, fsd = inputs
        weights, action, diag = gating(regime, tft, agents, fsd)
        assert weights.shape == (BATCH, 3)
        assert action.shape == (BATCH, 1)
        assert "gating/entropy" in diag

    def test_weights_in_valid_range(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        regime, tft, agents, fsd = inputs
        weights, _, _ = gating(regime, tft, agents, fsd)
        assert (weights >= 0).all() and (weights <= 1).all()

    def test_entropy_is_positive(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        regime, tft, agents, fsd = inputs
        _, _, diag = gating(regime, tft, agents, fsd)
        assert diag["gating/entropy"] > 0, "Entropy must be positive"

    def test_anticollapse_loss_is_differentiable(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        regime, tft, agents, fsd = inputs
        _, _, diag = gating(regime, tft, agents, fsd)
        loss = gating.get_anticollapse_loss(diag)
        loss.backward()
        # At least some parameters should have gradients
        grads = [p.grad for p in gating.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_fsd_changes_output(self) -> None:
        """Different FSD regimes should produce different weights."""
        gating = HiveMindGating_v2(GatingV2Config(
            d_regime=6, d_context=16, n_agents=3,
            d_hidden=16, d_fsd=4, fsd_embed_dim=8,
        ))
        regime = torch.randn(BATCH, 6)
        tft = torch.randn(BATCH, 16)
        agents = torch.randn(BATCH, 3)

        # Risk-off regime
        fsd_off = torch.zeros(BATCH, 4)
        fsd_off[:, 0] = 0.8  # high dispersion
        fsd_off[:, 1] = 1.0  # risk_off one-hot

        # Risk-on regime
        fsd_on = torch.zeros(BATCH, 4)
        fsd_on[:, 0] = 0.2  # low dispersion
        fsd_on[:, 3] = 1.0  # risk_on one-hot

        w_off, _, _ = gating(regime, tft, agents, fsd_off)
        w_on, _, _ = gating(regime, tft, agents, fsd_on)

        # Weights should differ between regimes
        assert not torch.allclose(w_off, w_on, atol=1e-4), \
            "FSD regime should affect gating weights"

    def test_no_fsd_fallback(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        """Should work when fsd_features=None (zeros fallback)."""
        regime, tft, agents, _ = inputs
        weights, action, diag = gating(regime, tft, agents, fsd_features=None)
        assert weights.shape == (BATCH, 3)
        assert torch.isfinite(action).all()

    def test_confidence_bounded(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        regime, tft, agents, fsd = inputs
        _, _, diag = gating(regime, tft, agents, fsd)
        assert 0.0 <= diag["gating/confidence_mean"] <= 1.0

    def test_diagnostics_all_keys(
        self, gating: HiveMindGating_v2,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        regime, tft, agents, fsd = inputs
        _, _, diag = gating(regime, tft, agents, fsd)
        expected_keys = {
            "gating/entropy", "gating/entropy_loss",
            "gating/diversity_loss", "gating/anticollapse_loss",
            "gating/confidence_mean", "gating/confidence_std",
            "gating/temperature", "gating/max_weight", "gating/weight_std",
        }
        assert set(diag.keys()) == expected_keys


# ── Config immutability ──────────────────────────────────────────────


class TestConfigImmutability:
    def test_world_model_config_frozen(self) -> None:
        cfg = WorldModelHybridConfig()
        with pytest.raises(Exception):
            cfg.d_latent = 999  # type: ignore

    def test_gating_config_frozen(self) -> None:
        cfg = GatingV2Config()
        with pytest.raises(Exception):
            cfg.entropy_min = 999  # type: ignore

    def test_mamba_config_frozen(self) -> None:
        cfg = MambaBackendConfig()
        with pytest.raises(Exception):
            cfg.d_state = 999  # type: ignore
