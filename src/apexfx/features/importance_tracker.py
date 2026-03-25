"""Feature importance tracking via EMA of TFT Variable Selection Network weights.

Tracks per-feature importance scores over time using exponential moving averages.
Stores history snapshots for drift detection (used by Improvement 5).
Thread-safe for concurrent updates from multiple inference threads.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

import torch

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureImportanceTracker:
    """Tracks per-feature importance scores from TFT's Variable Selection Network.

    The VSN outputs softmax weights of shape (batch, time, n_vars).
    This tracker maintains an EMA of those weights, averaged across
    batch and time dimensions, to produce a single importance score
    per feature.

    Args:
        feature_names: List of feature names matching the VSN variable order.
        ema_alpha: Smoothing factor for the exponential moving average.
            Smaller values = slower adaptation (more historical weight).
        history_size: Maximum number of EMA snapshots to retain for drift detection.
    """

    def __init__(
        self,
        feature_names: list[str],
        ema_alpha: float = 0.01,
        history_size: int = 1000,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must be a non-empty list")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self._feature_names = list(feature_names)
        self._n_features = len(self._feature_names)
        self._alpha = ema_alpha
        self._history_size = history_size

        # EMA state: None until first update
        self._ema: list[float] | None = None
        self._update_count: int = 0

        # History of EMA snapshots for drift detection
        self._history: deque[list[float]] = deque(maxlen=history_size)

        self._lock = threading.Lock()

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count

    def update(self, importance_tensor: torch.Tensor) -> None:
        """Update EMA with a new importance tensor from the VSN.

        The tensor is detached before processing to avoid gradient impact.

        Accepts shapes:
            - (n_vars,): single sample, single timestep
            - (batch, n_vars): single timestep, averaged across batch
            - (batch, time, n_vars): full VSN output, averaged across batch and time

        Args:
            importance_tensor: Variable importance weights from VSN.
        """
        # Detach and move to CPU
        t = importance_tensor.detach().cpu().float()

        # Reduce to (n_vars,) by averaging over batch and time dims
        if t.dim() == 3:
            # (batch, time, n_vars) -> (n_vars,)
            t = t.mean(dim=(0, 1))
        elif t.dim() == 2:
            # (batch, n_vars) -> (n_vars,)
            t = t.mean(dim=0)
        elif t.dim() == 1:
            pass  # Already (n_vars,)
        else:
            raise ValueError(
                f"Expected 1D, 2D, or 3D tensor, got {t.dim()}D with shape {t.shape}"
            )

        if t.shape[0] != self._n_features:
            raise ValueError(
                f"Tensor has {t.shape[0]} features, expected {self._n_features}"
            )

        values = t.tolist()

        with self._lock:
            if self._ema is None:
                # First update: initialize EMA directly
                self._ema = values
            else:
                # EMA: new = alpha * value + (1 - alpha) * old
                self._ema = [
                    self._alpha * v + (1.0 - self._alpha) * old
                    for v, old in zip(values, self._ema)
                ]

            self._update_count += 1

            # Store snapshot for drift detection
            self._history.append(list(self._ema))

    def get_top_k(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the top K features by importance score (descending).

        Args:
            k: Number of top features to return.

        Returns:
            List of (feature_name, importance_score) tuples, highest first.
        """
        importance = self.get_importance_dict()
        if not importance:
            return []
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def get_bottom_k(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the bottom K features by importance score (ascending).

        Args:
            k: Number of bottom features to return.

        Returns:
            List of (feature_name, importance_score) tuples, lowest first.
        """
        importance = self.get_importance_dict()
        if not importance:
            return []
        sorted_items = sorted(importance.items(), key=lambda x: x[1])
        return sorted_items[:k]

    def get_importance_dict(self) -> dict[str, float]:
        """Return all features with their current EMA importance scores.

        Returns:
            Dict mapping feature name to importance score. Empty if no updates yet.
        """
        with self._lock:
            if self._ema is None:
                return {}
            return dict(zip(self._feature_names, self._ema))

    def save(self, path: Path) -> None:
        """Serialize tracker state to JSON.

        Args:
            path: File path to write JSON state.
        """
        path = Path(path)
        with self._lock:
            state = {
                "feature_names": self._feature_names,
                "ema_alpha": self._alpha,
                "history_size": self._history_size,
                "ema": self._ema,
                "update_count": self._update_count,
                "history": list(self._history),
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(
            "Importance tracker saved",
            path=str(path),
            update_count=self._update_count,
        )

    @classmethod
    def load(cls, path: Path) -> FeatureImportanceTracker:
        """Load tracker state from JSON.

        Args:
            path: File path to read JSON state from.

        Returns:
            Restored FeatureImportanceTracker instance.
        """
        path = Path(path)
        with open(path) as f:
            state = json.load(f)

        tracker = cls(
            feature_names=state["feature_names"],
            ema_alpha=state["ema_alpha"],
            history_size=state["history_size"],
        )
        tracker._ema = state["ema"]
        tracker._update_count = state["update_count"]
        tracker._history = deque(state["history"], maxlen=state["history_size"])

        logger.info(
            "Importance tracker loaded",
            path=str(path),
            update_count=tracker._update_count,
            n_features=tracker._n_features,
        )

        return tracker

    def get_history(self) -> list[list[float]]:
        """Return the history of EMA snapshots (for drift detection).

        Returns:
            List of EMA snapshot lists, oldest first.
        """
        with self._lock:
            return list(self._history)
