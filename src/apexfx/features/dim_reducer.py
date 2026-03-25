"""Dimensionality reduction for high-dimensional feature matrices.

Provides PCA and Autoencoder options to compress ~3500 features down to
a configurable number of components (default 128).  Runs AFTER feature
normalization and BEFORE the model sees the data.

Sprint 2 — Improvement 1B.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseDimReducer(ABC):
    """Abstract interface for dimensionality reducers."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the reducer on training data *X* of shape (n_samples, n_features)."""
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project *X* into the reduced space.  Returns (n_samples, n_components)."""
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform."""
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist fitted state to *path*."""
        ...

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Restore fitted state from *path*."""
        ...

    @property
    @abstractmethod
    def n_components(self) -> int:
        """Number of output dimensions."""
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the reducer has been fitted."""
        ...


# ---------------------------------------------------------------------------
# PCA (incremental / online capable)
# ---------------------------------------------------------------------------


class PCAReducer(BaseDimReducer):
    """PCA-based dimensionality reduction using Incremental PCA.

    Supports both batch ``fit`` and incremental ``partial_fit`` for
    online / streaming scenarios.

    Args:
        n_components: Target number of output dimensions.
        min_explained_variance: Informational threshold — logged when
            the cumulative explained variance falls below this value.
    """

    def __init__(
        self,
        n_components: int = 128,
        min_explained_variance: float = 0.95,
    ) -> None:
        self._n_components = n_components
        self._min_explained_variance = min_explained_variance
        self._fitted = False
        # Lazy import to avoid hard dependency at module level
        self._pca = None  # IncrementalPCA instance

    def _ensure_pca(self) -> None:
        if self._pca is None:
            from sklearn.decomposition import IncrementalPCA

            self._pca = IncrementalPCA(n_components=self._n_components)

    # -- Core API ----------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        self._ensure_pca()
        self._pca.fit(X)
        self._fitted = True

        cum_var = float(np.sum(self._pca.explained_variance_ratio_))
        logger.info(
            "PCA fit complete",
            n_components=self._n_components,
            explained_variance=round(cum_var, 4),
            input_dim=X.shape[1],
            n_samples=X.shape[0],
        )
        if cum_var < self._min_explained_variance:
            logger.warning(
                "PCA explained variance below threshold",
                explained_variance=round(cum_var, 4),
                threshold=self._min_explained_variance,
            )

    def partial_fit(self, X: np.ndarray) -> None:
        """Incremental fit on a batch (online / streaming use)."""
        self._ensure_pca()
        self._pca.partial_fit(X)
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCAReducer has not been fitted yet — call fit() first")
        return self._pca.transform(X)

    # -- Properties --------------------------------------------------------

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component explained variance ratio (available after fit)."""
        if not self._fitted:
            raise RuntimeError("PCAReducer has not been fitted yet")
        return self._pca.explained_variance_ratio_

    # -- Persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pca, path)
        logger.info("PCA reducer saved", path=str(path))

    def load(self, path: Path | str) -> None:
        import joblib

        path = Path(path)
        self._pca = joblib.load(path)
        self._n_components = self._pca.n_components
        self._fitted = True
        logger.info("PCA reducer loaded", path=str(path))


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class AutoencoderReducer(BaseDimReducer):
    """Autoencoder-based dimensionality reduction using PyTorch.

    Architecture:
        Encoder: input_dim → 512 → 256 → n_components
        Decoder: n_components → 256 → 512 → input_dim

    Each layer uses BatchNorm + ReLU with dropout.

    Args:
        input_dim: Number of input features (must be known at construction).
        n_components: Bottleneck / latent dimension.
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 128,
        dropout: float = 0.1,
    ) -> None:
        self._input_dim = input_dim
        self._n_components = n_components
        self._dropout = dropout
        self._fitted = False
        self._reconstruction_loss: float | None = None
        self._device: str = "cpu"

        # Lazily built on first use to avoid hard torch dependency at import
        self._model = None

    def _ensure_model(self) -> None:
        """Build the model lazily on first access."""
        if self._model is None:
            self._model = self._build_model()

    def _build_model(self):  # noqa: ANN202 — returns nn.Module
        import torch.nn as nn

        encoder = nn.Sequential(
            nn.Linear(self._input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(256, self._n_components),
        )

        decoder = nn.Sequential(
            nn.Linear(self._n_components, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self._dropout),
            nn.Linear(512, self._input_dim),
        )

        class _Autoencoder(nn.Module):
            def __init__(self, enc: nn.Module, dec: nn.Module) -> None:
                super().__init__()
                self.encoder = enc
                self.decoder = dec

            def forward(self, x):  # noqa: ANN001, ANN201
                z = self.encoder(x)
                return self.decoder(z), z

        return _Autoencoder(encoder, decoder)

    # -- Core API ----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._ensure_model()
        device = self._device
        self._model.to(device)
        self._model.train()

        tensor_x = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        last_loss = float("inf")
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(device)
                reconstructed, _ = self._model(batch)
                loss = criterion(reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            last_loss = epoch_loss / max(n_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.debug(
                    "Autoencoder training",
                    epoch=epoch + 1,
                    loss=round(last_loss, 6),
                )

        self._reconstruction_loss = last_loss
        self._fitted = True

        logger.info(
            "Autoencoder fit complete",
            n_components=self._n_components,
            input_dim=self._input_dim,
            final_loss=round(last_loss, 6),
            epochs=epochs,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "AutoencoderReducer has not been fitted yet — call fit() first"
            )
        import torch

        self._ensure_model()
        self._model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self._device)
            _, z = self._model(tensor_x)
            return z.cpu().numpy()

    # -- Properties --------------------------------------------------------

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def reconstruction_loss(self) -> float | None:
        """Last epoch's mean reconstruction loss (available after fit)."""
        return self._reconstruction_loss

    # -- Persistence -------------------------------------------------------

    def save(self, path: Path | str) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self._model.state_dict(),
            "input_dim": self._input_dim,
            "n_components": self._n_components,
            "dropout": self._dropout,
            "reconstruction_loss": self._reconstruction_loss,
        }
        torch.save(state, path)
        logger.info("Autoencoder reducer saved", path=str(path))

    def load(self, path: Path | str) -> None:
        import torch

        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)

        self._input_dim = state["input_dim"]
        self._n_components = state["n_components"]
        self._dropout = state["dropout"]
        self._reconstruction_loss = state.get("reconstruction_loss")

        # Rebuild model with correct dimensions then load weights
        self._model = self._build_model()
        self._model.load_state_dict(state["model_state_dict"])
        self._model.eval()
        self._fitted = True
        logger.info("Autoencoder reducer loaded", path=str(path))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class DimReducerFactory:
    """Create dimensionality reducers by method name."""

    _METHODS = {"pca", "autoencoder"}

    @staticmethod
    def create(
        method: str,
        n_components: int = 128,
        input_dim: int | None = None,
        **kwargs,
    ) -> BaseDimReducer:
        """Instantiate a reducer.

        Args:
            method: ``"pca"`` or ``"autoencoder"``.
            n_components: Target latent dimension.
            input_dim: Required for ``"autoencoder"`` (original feature count).
            **kwargs: Forwarded to the reducer constructor.

        Returns:
            A concrete :class:`BaseDimReducer` instance.

        Raises:
            ValueError: If *method* is unknown or required args are missing.
        """
        method = method.lower()

        if method == "pca":
            return PCAReducer(n_components=n_components, **kwargs)
        elif method == "autoencoder":
            if input_dim is None:
                raise ValueError(
                    "input_dim is required when method='autoencoder'"
                )
            return AutoencoderReducer(
                input_dim=input_dim,
                n_components=n_components,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown dim reduction method '{method}'. "
                f"Choose from: {sorted(DimReducerFactory._METHODS)}"
            )
