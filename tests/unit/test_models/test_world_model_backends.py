"""The dynamics backends, and the crash that kept them out of every run.

``WorldModelCallback`` takes ``(d_features, config)``. The call site in
``trainer.py`` passed the training knobs as keyword arguments — a signature from
before the v2 rewrite — and ``world_model.enabled`` defaults to True, so
building the callback list raised TypeError on every training run. The world
model never initialised, and with it none of the pluggable backends.

The two halves came from different branches and were joined by the phase-1
merge. Nothing caught it because no test built the callback list, which is the
same reason the ``TradingReward`` NameError survived that merge. These tests
exist so a third instance of that pattern fails in CI instead of at run time.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from apexfx.config.schema import WorldModelConfig  # noqa: E402
from apexfx.models.config import BackendType, WorldModelHybridConfig  # noqa: E402
from apexfx.models.world_model import (  # noqa: E402
    WorldModelCallback,
    _build_backend,
)
from apexfx.training.trainer import build_world_model_config  # noqa: E402

BACKENDS = [b.value for b in BackendType]


class TestEveryBackendRuns:
    """Each backend maps (latent, action) -> next latent."""

    D_LATENT, D_ACTION, BATCH = 32, 1, 4

    def _backend(self, name: str):
        cfg = WorldModelHybridConfig(backend=BackendType(name))
        return _build_backend(BackendType(name), self.D_LATENT, self.D_ACTION, cfg)

    @pytest.mark.parametrize("name", BACKENDS)
    def test_forward_returns_a_latent(self, name):
        z = torch.randn(self.BATCH, self.D_LATENT)
        a = torch.randn(self.BATCH, self.D_ACTION)
        assert self._backend(name)(z, a).shape == (self.BATCH, self.D_LATENT)

    @pytest.mark.parametrize("name", BACKENDS)
    def test_output_is_finite(self, name):
        z = torch.randn(self.BATCH, self.D_LATENT) * 10
        a = torch.randn(self.BATCH, self.D_ACTION) * 10
        assert torch.isfinite(self._backend(name)(z, a)).all()

    @pytest.mark.parametrize("name", BACKENDS)
    def test_the_backend_is_trainable(self, name):
        """A backend whose parameters get no gradient is decoration."""
        backend = self._backend(name)
        z = torch.randn(self.BATCH, self.D_LATENT)
        a = torch.randn(self.BATCH, self.D_ACTION)
        backend(z, a).sum().backward()

        starved = [
            n for n, p in backend.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)
        ]
        assert starved == [], f"{name}: no gradient reached {starved}"

    @pytest.mark.parametrize("name", BACKENDS)
    def test_the_action_changes_the_prediction(self, name):
        """Dynamics that ignore the action are not dynamics."""
        backend = self._backend(name)
        z = torch.randn(self.BATCH, self.D_LATENT)
        flat = backend(z, torch.zeros(self.BATCH, self.D_ACTION))
        long = backend(z, torch.ones(self.BATCH, self.D_ACTION))
        assert not torch.allclose(flat, long, atol=1e-6)


class TestCallbackAcceptsTheConfigItIsGiven:
    """The regression guard: the call site must match the signature."""

    @pytest.mark.parametrize("name", BACKENDS)
    def test_the_trainer_mapping_builds_a_usable_callback(self, name):
        cfg = build_world_model_config(WorldModelConfig(backend=name))
        callback = WorldModelCallback(d_features=74, config=cfg)
        assert callback._config.backend is BackendType(name)

    def test_training_knobs_survive_the_translation(self):
        app = WorldModelConfig(
            backend="mamba", d_latent=64, d_hidden=256, n_ensemble=7,
            update_freq=50, batch_size=512, lr=1e-4,
            curiosity_weight=0.02, imagination_horizon=5,
        )
        cfg = build_world_model_config(app)
        assert (cfg.d_latent, cfg.d_hidden, cfg.n_ensemble) == (64, 256, 7)
        assert (cfg.update_freq, cfg.batch_size) == (50, 512)
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.curiosity_weight == pytest.approx(0.02)
        assert cfg.imagination_horizon == 5

    def test_an_unknown_backend_is_rejected_at_config_time(self):
        """Better a clear error here than a silent fallback to mamba."""
        with pytest.raises(ValueError):
            build_world_model_config(WorldModelConfig(backend="does_not_exist"))

    def test_the_default_backend_is_mamba(self):
        assert build_world_model_config(WorldModelConfig()).backend is BackendType.MAMBA


class TestShippedConfigIsUsable:
    def test_the_training_yaml_world_model_block_builds(self):
        """configs/training.yaml turns the world model on; it must construct."""
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[3]
        with (root / "configs" / "training.yaml").open() as fh:
            raw = yaml.safe_load(fh)["world_model"]
        app = WorldModelConfig(**raw)
        assert app.enabled
        cfg = build_world_model_config(app)
        WorldModelCallback(d_features=74, config=cfg)


class TestTheSSMActuallyHasState:
    """``A_log`` is the state matrix. It used to have no path to the output.

    ``SelectiveSSM.forward`` computed ``A_bar = exp(A * dt)`` and discarded the
    result, so ``y`` did not depend on ``A`` at all: shifting ``A_log`` by +10
    changed the output by exactly zero, and the parameter received no gradient.
    A "selective state space" backend without its state matrix is a gated
    linear map wearing the name.
    """

    @staticmethod
    def _backend():
        from apexfx.models.backends.mamba_backend import MambaBackend

        cfg = WorldModelHybridConfig()
        backend = MambaBackend(32, 1, cfg.mamba)
        backend.eval()
        return backend

    def test_the_state_matrix_changes_the_prediction(self):
        backend = self._backend()
        z, a = torch.randn(4, 32), torch.randn(4, 1)
        before = backend(z, a).clone()
        with torch.no_grad():
            backend.mamba.ssm.A_log.add_(1.0)
        assert not torch.allclose(before, backend(z, a), atol=1e-8)

    def test_the_state_matrix_is_trained(self):
        backend = self._backend()
        backend(torch.randn(4, 32), torch.randn(4, 1)).sum().backward()
        grad = backend.mamba.ssm.A_log.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0

    def test_the_recurrence_stays_stable(self):
        """A_bar = exp(A·dt) must sit in (0, 1) or the fixed point diverges."""
        backend = self._backend()
        out = backend(torch.randn(8, 32) * 50, torch.randn(8, 1) * 50)
        assert torch.isfinite(out).all()

    def test_a_longer_memory_horizon_changes_the_response(self):
        """A_bar near 1 accumulates, near 0 responds to the current step only."""
        backend = self._backend()
        z, a = torch.randn(4, 32), torch.randn(4, 1)
        with torch.no_grad():
            backend.mamba.ssm.A_log.fill_(-4.0)   # A_bar close to 1
        slow = backend(z, a).clone()
        with torch.no_grad():
            backend.mamba.ssm.A_log.fill_(2.0)    # A_bar close to 0
        fast = backend(z, a)
        assert not torch.allclose(slow, fast, atol=1e-6)


class TestHybridAppliesItsResidual:
    """``HybridDynamics.residual`` was allocated and never used in forward."""

    @staticmethod
    def _hybrid():
        cfg = WorldModelHybridConfig(backend=BackendType.HYBRID)
        return _build_backend(BackendType.HYBRID, 32, 1, cfg)

    def test_the_residual_reaches_the_output(self):
        hybrid = self._hybrid()
        hybrid.eval()
        z, a = torch.randn(4, 32), torch.randn(4, 1)
        before = hybrid(z, a).clone()
        with torch.no_grad():
            hybrid.residual.weight.mul_(3.0)
        assert not torch.allclose(before, hybrid(z, a), atol=1e-8)

    def test_the_residual_is_trained(self):
        hybrid = self._hybrid()
        hybrid(torch.randn(4, 32), torch.randn(4, 1)).sum().backward()
        assert hybrid.residual.weight.grad.abs().sum().item() > 0.0
