"""Adaptive Feature Selection — detects importance drift and auto-disables low-value features.

During live trading, some features become irrelevant as market regimes shift.
This module monitors per-feature importance drift via the FeatureImportanceTracker
and automatically disables features whose importance has decayed below a threshold,
reducing noise and overfitting risk.

Sprint 3, Improvement 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from apexfx.features.importance_tracker import FeatureImportanceTracker
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSelectionResult:
    """Outcome of one adaptive selection evaluation cycle."""

    features_to_disable: list[str]
    features_to_enable: list[str]
    drift_scores: dict[str, float]
    active_count: int
    total_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveFeatureSelector:
    """Monitors feature importance drift and toggles features on/off.

    Args:
        tracker: The FeatureImportanceTracker providing EMA importance scores.
        min_importance: Features below this EMA importance are candidates for disabling.
        drift_window: How many bars back to look for the reference importance snapshot.
        cooldown_bars: Minimum bars between consecutive evaluations.
        max_disable_pct: Maximum fraction of total features that can be disabled per cycle.
        always_on: Feature names that must never be disabled (e.g., regime features).
    """

    def __init__(
        self,
        tracker: FeatureImportanceTracker,
        min_importance: float = 0.001,
        drift_window: int = 500,
        cooldown_bars: int = 100,
        max_disable_pct: float = 0.10,
        always_on: set[str] | None = None,
    ) -> None:
        self._tracker = tracker
        self._min_importance = min_importance
        self._drift_window = drift_window
        self._cooldown_bars = cooldown_bars
        self._max_disable_pct = max_disable_pct
        self._always_on: set[str] = set(always_on) if always_on else set()

        all_features = set(tracker.feature_names)
        self._active_features: set[str] = set(all_features)
        self._disabled_features: set[str] = set()
        self._bars_since_last_eval: int = 0
        self._eval_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Called every bar to increment the internal counter."""
        self._bars_since_last_eval += 1

    def evaluate(self) -> FeatureSelectionResult | None:
        """Evaluate feature importance drift and decide what to toggle.

        Returns ``None`` if the cooldown period has not elapsed.
        """
        if self._bars_since_last_eval < self._cooldown_bars:
            return None

        self._bars_since_last_eval = 0
        self._eval_count += 1

        history = self._tracker.get_history()
        current_importance = self._tracker.get_importance_dict()

        # Nothing to evaluate if tracker has no data yet
        if not current_importance:
            return FeatureSelectionResult(
                features_to_disable=[],
                features_to_enable=[],
                drift_scores={},
                active_count=len(self._active_features),
                total_count=self._tracker.n_features,
            )

        # Reference snapshot: drift_window bars ago (or oldest available)
        ref_index = max(0, len(history) - 1 - self._drift_window)
        ref_snapshot = history[ref_index] if history else None
        feature_names = self._tracker.feature_names

        # Build reference dict
        if ref_snapshot is not None:
            ref_importance = dict(zip(feature_names, ref_snapshot))
        else:
            ref_importance = {}

        # Compute drift scores and candidate lists
        drift_scores: dict[str, float] = {}
        candidates_disable: list[str] = []
        candidates_enable: list[str] = []

        total_features = self._tracker.n_features

        for feat in feature_names:
            current_val = current_importance.get(feat, 0.0)
            ref_val = ref_importance.get(feat, current_val)

            # Drift = fractional change; positive means importance dropped
            if ref_val > 0:
                drift = (ref_val - current_val) / ref_val
            else:
                drift = 0.0
            drift_scores[feat] = drift

            # --- Disable candidates ---
            if feat in self._active_features and feat not in self._always_on:
                # Importance dropped >50% AND current importance < min_importance
                if drift > 0.5 and current_val < self._min_importance:
                    candidates_disable.append(feat)

            # --- Re-enable candidates ---
            if feat in self._disabled_features:
                if current_val > 2.0 * self._min_importance:
                    candidates_enable.append(feat)

        # Safety: cap disables per cycle
        max_disable_count = max(1, int(total_features * self._max_disable_pct))
        candidates_disable = candidates_disable[:max_disable_count]

        # Apply changes
        for feat in candidates_disable:
            self._active_features.discard(feat)
            self._disabled_features.add(feat)

        for feat in candidates_enable:
            self._disabled_features.discard(feat)
            self._active_features.add(feat)

        if candidates_disable:
            logger.info(
                "Features disabled by adaptive selector",
                disabled=candidates_disable,
                eval_count=self._eval_count,
            )
        if candidates_enable:
            logger.info(
                "Features re-enabled by adaptive selector",
                enabled=candidates_enable,
                eval_count=self._eval_count,
            )

        return FeatureSelectionResult(
            features_to_disable=candidates_disable,
            features_to_enable=candidates_enable,
            drift_scores=drift_scores,
            active_count=len(self._active_features),
            total_count=total_features,
        )

    def get_active_features(self) -> set[str]:
        """Return the currently active feature set."""
        return set(self._active_features)

    def get_disabled_features(self) -> set[str]:
        """Return the currently disabled feature set."""
        return set(self._disabled_features)

    def force_enable(self, feature: str) -> None:
        """Manually re-enable a feature."""
        self._disabled_features.discard(feature)
        self._active_features.add(feature)
        logger.info("Feature force-enabled", feature=feature)

    def force_disable(self, feature: str) -> None:
        """Manually disable a feature (respects always_on)."""
        if feature in self._always_on:
            logger.warning(
                "Cannot force-disable always_on feature", feature=feature
            )
            return
        self._active_features.discard(feature)
        self._disabled_features.add(feature)
        logger.info("Feature force-disabled", feature=feature)

    def reset(self) -> None:
        """Re-enable all features and reset evaluation state."""
        all_features = set(self._tracker.feature_names)
        self._active_features = set(all_features)
        self._disabled_features = set()
        self._bars_since_last_eval = 0
        self._eval_count = 0
        logger.info("Adaptive selector reset — all features re-enabled")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist active/disabled feature sets to JSON."""
        path = Path(path)
        state = {
            "active_features": sorted(self._active_features),
            "disabled_features": sorted(self._disabled_features),
            "eval_count": self._eval_count,
            "bars_since_last_eval": self._bars_since_last_eval,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Adaptive selector state saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Restore active/disabled feature sets from JSON."""
        path = Path(path)
        with open(path) as f:
            state = json.load(f)
        self._active_features = set(state["active_features"])
        self._disabled_features = set(state["disabled_features"])
        self._eval_count = state.get("eval_count", 0)
        self._bars_since_last_eval = state.get("bars_since_last_eval", 0)
        logger.info(
            "Adaptive selector state loaded",
            path=str(path),
            active=len(self._active_features),
            disabled=len(self._disabled_features),
        )
