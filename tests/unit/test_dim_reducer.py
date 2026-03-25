"""Unit tests for dimensionality reduction module (Sprint 2, Improvement 1B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apexfx.features.dim_reducer import (
    AutoencoderReducer,
    BaseDimReducer,
    DimReducerFactory,
    PCAReducer,
)

_HAS_TORCH = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _HAS_TORCH = False

_requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_data(n_samples: int = 500, n_features: int = 3500, seed: int = 42) -> np.ndarray:
    """Generate random data roughly matching production dimensions."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


# ---------------------------------------------------------------------------
# PCA Reducer
# ---------------------------------------------------------------------------


class TestPCAReducer:
    def test_fit_transform_dimensions(self) -> None:
        """3500 features should reduce to 128 components."""
        X = _random_data(n_samples=300, n_features=3500)
        reducer = PCAReducer(n_components=128)
        out = reducer.fit_transform(X)
        assert out.shape == (300, 128)

    def test_transform_dimensions(self) -> None:
        X = _random_data(n_samples=300, n_features=3500)
        reducer = PCAReducer(n_components=128)
        reducer.fit(X)
        X_new = _random_data(n_samples=50, n_features=3500, seed=99)
        out = reducer.transform(X_new)
        assert out.shape == (50, 128)

    def test_explained_variance_above_threshold(self) -> None:
        """With highly correlated data, explained variance should be high."""
        rng = np.random.default_rng(0)
        # Create data with strong correlations (low intrinsic dim)
        latent = rng.standard_normal((500, 50))
        mixing = rng.standard_normal((50, 200))
        X = latent @ mixing + rng.standard_normal((500, 200)) * 0.1
        reducer = PCAReducer(n_components=50, min_explained_variance=0.90)
        reducer.fit(X)
        total_var = float(np.sum(reducer.explained_variance_ratio))
        assert total_var >= 0.90

    def test_explained_variance_ratio_property(self) -> None:
        X = _random_data(n_samples=200, n_features=500)
        reducer = PCAReducer(n_components=64)
        reducer.fit(X)
        ratios = reducer.explained_variance_ratio
        assert len(ratios) == 64
        assert all(r >= 0 for r in ratios)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X = _random_data(n_samples=200, n_features=500)
        reducer = PCAReducer(n_components=32)
        reducer.fit(X)
        original_out = reducer.transform(X[:10])

        save_path = tmp_path / "pca.joblib"
        reducer.save(save_path)
        assert save_path.exists()

        loaded = PCAReducer(n_components=32)
        loaded.load(save_path)
        assert loaded.is_fitted
        loaded_out = loaded.transform(X[:10])

        np.testing.assert_allclose(original_out, loaded_out, atol=1e-10)

    def test_incremental_fitting(self) -> None:
        """partial_fit on multiple batches should work like batch fit."""
        X = _random_data(n_samples=600, n_features=200)
        reducer = PCAReducer(n_components=32)

        # Fit incrementally in 3 batches
        for i in range(3):
            batch = X[i * 200 : (i + 1) * 200]
            reducer.partial_fit(batch)

        assert reducer.is_fitted
        out = reducer.transform(X[:10])
        assert out.shape == (10, 32)

    def test_n_components_property(self) -> None:
        reducer = PCAReducer(n_components=64)
        assert reducer.n_components == 64

    def test_is_fitted_false_initially(self) -> None:
        reducer = PCAReducer(n_components=128)
        assert reducer.is_fitted is False

    def test_is_fitted_true_after_fit(self) -> None:
        X = _random_data(n_samples=200, n_features=500)
        reducer = PCAReducer(n_components=32)
        reducer.fit(X)
        assert reducer.is_fitted is True

    def test_transform_before_fit_raises(self) -> None:
        reducer = PCAReducer(n_components=128)
        X = _random_data(n_samples=10, n_features=500)
        with pytest.raises(RuntimeError, match="not been fitted"):
            reducer.transform(X)

    def test_explained_variance_before_fit_raises(self) -> None:
        reducer = PCAReducer(n_components=128)
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = reducer.explained_variance_ratio


# ---------------------------------------------------------------------------
# Autoencoder Reducer
# ---------------------------------------------------------------------------


@_requires_torch
class TestAutoencoderReducer:
    def test_fit_transform_dimensions(self) -> None:
        """Input features should compress to n_components."""
        X = _random_data(n_samples=300, n_features=200)
        reducer = AutoencoderReducer(input_dim=200, n_components=32)
        reducer.fit(X, epochs=5, batch_size=64)
        out = reducer.transform(X)
        assert out.shape == (300, 32)

    def test_reconstruction_loss_decreases(self) -> None:
        """After training, reconstruction loss should be finite and reasonable."""
        rng = np.random.default_rng(42)
        # Correlated data so autoencoder can learn
        latent = rng.standard_normal((500, 20))
        mixing = rng.standard_normal((20, 100))
        X = latent @ mixing

        reducer = AutoencoderReducer(input_dim=100, n_components=32)

        # Fit with very few epochs to get initial loss
        reducer.fit(X, epochs=2, batch_size=128)
        loss_early = reducer.reconstruction_loss

        # Re-fit with more epochs
        reducer2 = AutoencoderReducer(input_dim=100, n_components=32)
        reducer2.fit(X, epochs=30, batch_size=128)
        loss_later = reducer2.reconstruction_loss

        assert loss_later is not None
        assert loss_early is not None
        assert loss_later < loss_early

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X = _random_data(n_samples=200, n_features=100)
        reducer = AutoencoderReducer(input_dim=100, n_components=16)
        reducer.fit(X, epochs=3, batch_size=64)
        original_out = reducer.transform(X[:10])

        save_path = tmp_path / "autoencoder.pt"
        reducer.save(save_path)
        assert save_path.exists()

        loaded = AutoencoderReducer(input_dim=100, n_components=16)
        loaded.load(save_path)
        assert loaded.is_fitted
        loaded_out = loaded.transform(X[:10])

        np.testing.assert_allclose(original_out, loaded_out, atol=1e-5)

    def test_n_components_property(self) -> None:
        reducer = AutoencoderReducer(input_dim=500, n_components=64)
        assert reducer.n_components == 64

    def test_is_fitted_false_initially(self) -> None:
        reducer = AutoencoderReducer(input_dim=500, n_components=64)
        assert reducer.is_fitted is False

    def test_is_fitted_true_after_fit(self) -> None:
        X = _random_data(n_samples=200, n_features=100)
        reducer = AutoencoderReducer(input_dim=100, n_components=16)
        reducer.fit(X, epochs=2, batch_size=64)
        assert reducer.is_fitted is True

    def test_transform_before_fit_raises(self) -> None:
        reducer = AutoencoderReducer(input_dim=500, n_components=64)
        X = _random_data(n_samples=10, n_features=500)
        with pytest.raises(RuntimeError, match="not been fitted"):
            reducer.transform(X)

    def test_reconstruction_loss_none_before_fit(self) -> None:
        reducer = AutoencoderReducer(input_dim=500, n_components=64)
        assert reducer.reconstruction_loss is None

    def test_fit_transform_convenience(self) -> None:
        X = _random_data(n_samples=200, n_features=100)
        reducer = AutoencoderReducer(input_dim=100, n_components=16)
        out = reducer.fit_transform(X)
        assert out.shape == (200, 16)
        assert reducer.is_fitted


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestDimReducerFactory:
    def test_create_pca(self) -> None:
        reducer = DimReducerFactory.create(method="pca", n_components=64)
        assert isinstance(reducer, PCAReducer)
        assert reducer.n_components == 64

    @_requires_torch
    def test_create_autoencoder(self) -> None:
        reducer = DimReducerFactory.create(
            method="autoencoder", n_components=32, input_dim=500,
        )
        assert isinstance(reducer, AutoencoderReducer)
        assert reducer.n_components == 32

    def test_create_pca_case_insensitive(self) -> None:
        reducer = DimReducerFactory.create(method="PCA", n_components=64)
        assert isinstance(reducer, PCAReducer)

    def test_create_autoencoder_missing_input_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            DimReducerFactory.create(method="autoencoder", n_components=32)

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            DimReducerFactory.create(method="svd", n_components=32)

    def test_factory_returns_base_type(self) -> None:
        reducer = DimReducerFactory.create(method="pca", n_components=16)
        assert isinstance(reducer, BaseDimReducer)

    def test_factory_kwargs_forwarded(self) -> None:
        reducer = DimReducerFactory.create(
            method="pca", n_components=64, min_explained_variance=0.99,
        )
        assert isinstance(reducer, PCAReducer)
        # The kwarg should have been forwarded (we test indirectly — no error)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_dim_reduction_config_defaults(self) -> None:
        from apexfx.config.schema import DimReductionConfig

        cfg = DimReductionConfig()
        assert cfg.enabled is False
        assert cfg.method == "pca"
        assert cfg.n_components == 128
        assert cfg.min_explained_variance == 0.95
        assert cfg.autoencoder_epochs == 50

    def test_dim_reduction_in_model_config(self) -> None:
        from apexfx.config.schema import ModelConfig

        mc = ModelConfig()
        assert hasattr(mc, "dim_reduction")
        assert mc.dim_reduction.enabled is False
