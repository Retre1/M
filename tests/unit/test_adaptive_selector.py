"""Unit tests for AdaptiveFeatureSelector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from apexfx.features.adaptive_selector import (
    AdaptiveFeatureSelector,
    FeatureSelectionResult,
)
from apexfx.features.importance_tracker import FeatureImportanceTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_names() -> list[str]:
    return [f"feat_{i}" for i in range(20)]


@pytest.fixture
def tracker(feature_names: list[str]) -> FeatureImportanceTracker:
    return FeatureImportanceTracker(
        feature_names=feature_names, ema_alpha=0.5, history_size=2000
    )


@pytest.fixture
def selector(tracker: FeatureImportanceTracker) -> AdaptiveFeatureSelector:
    return AdaptiveFeatureSelector(
        tracker=tracker,
        min_importance=0.01,
        drift_window=5,
        cooldown_bars=3,
        max_disable_pct=0.10,
        always_on={"feat_0"},
    )


def _uniform_tensor(n: int, value: float) -> torch.Tensor:
    """Create a 1-D tensor of *n* features all set to *value*."""
    return torch.full((n,), value)


def _make_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values)


# ---------------------------------------------------------------------------
# Test: initial state — all features active
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_all_features_active(self, selector: AdaptiveFeatureSelector, feature_names: list[str]):
        assert selector.get_active_features() == set(feature_names)

    def test_no_features_disabled(self, selector: AdaptiveFeatureSelector):
        assert selector.get_disabled_features() == set()


# ---------------------------------------------------------------------------
# Test: evaluate returns None during cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_returns_none_before_cooldown(self, selector: AdaptiveFeatureSelector, tracker: FeatureImportanceTracker):
        # Feed some data so tracker has state
        tracker.update(_uniform_tensor(20, 0.05))
        selector.tick()
        selector.tick()
        # Only 2 ticks, cooldown is 3
        assert selector.evaluate() is None

    def test_returns_result_after_cooldown(self, selector: AdaptiveFeatureSelector, tracker: FeatureImportanceTracker):
        tracker.update(_uniform_tensor(20, 0.05))
        for _ in range(3):
            selector.tick()
        result = selector.evaluate()
        assert result is not None
        assert isinstance(result, FeatureSelectionResult)


# ---------------------------------------------------------------------------
# Test: evaluate detects low-importance features
# ---------------------------------------------------------------------------

class TestDetectLowImportance:
    def test_disables_drifted_low_importance_features(self, tracker: FeatureImportanceTracker):
        n = 20
        # Start with moderate importance for all features
        high_vals = [0.05] * n
        tracker.update(_make_tensor(high_vals))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=20,  # compare against snapshot 20 bars back
            cooldown_bars=1,
            max_disable_pct=0.50,  # allow disabling many for this test
        )

        # Now push importance of feat_5 very low via many updates
        for _ in range(20):
            vals = [0.05] * n
            vals[5] = 0.0001  # well below min_importance
            tracker.update(_make_tensor(vals))

        selector.tick()
        result = selector.evaluate()
        assert result is not None
        # feat_5 should be flagged for disable (importance dropped >50% and < min)
        assert "feat_5" in result.features_to_disable

    def test_does_not_disable_when_importance_above_threshold(self, tracker: FeatureImportanceTracker):
        n = 20
        # All features stay at decent importance
        for _ in range(10):
            tracker.update(_uniform_tensor(n, 0.05))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=5,
            cooldown_bars=1,
            max_disable_pct=0.50,
        )
        selector.tick()
        result = selector.evaluate()
        assert result is not None
        assert result.features_to_disable == []


# ---------------------------------------------------------------------------
# Test: max_disable_pct limit
# ---------------------------------------------------------------------------

class TestMaxDisablePct:
    def test_caps_disables_per_cycle(self, tracker: FeatureImportanceTracker):
        n = 20
        # Start with high importance
        tracker.update(_uniform_tensor(n, 0.1))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=30,
            cooldown_bars=1,
            max_disable_pct=0.10,  # max 2 out of 20
        )

        # Drive ALL features to very low importance
        for _ in range(30):
            tracker.update(_uniform_tensor(n, 0.0001))

        selector.tick()
        result = selector.evaluate()
        assert result is not None
        # 10% of 20 = 2
        assert len(result.features_to_disable) <= 2


# ---------------------------------------------------------------------------
# Test: always_on features never disabled
# ---------------------------------------------------------------------------

class TestAlwaysOn:
    def test_always_on_never_disabled(self, tracker: FeatureImportanceTracker):
        n = 20
        tracker.update(_uniform_tensor(n, 0.1))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=30,
            cooldown_bars=1,
            max_disable_pct=1.0,  # no cap
            always_on={"feat_0", "feat_1"},
        )

        # Drive everything low
        for _ in range(30):
            tracker.update(_uniform_tensor(n, 0.0001))

        selector.tick()
        result = selector.evaluate()
        assert result is not None
        assert "feat_0" not in result.features_to_disable
        assert "feat_1" not in result.features_to_disable
        assert "feat_0" in selector.get_active_features()
        assert "feat_1" in selector.get_active_features()


# ---------------------------------------------------------------------------
# Test: re-enable when importance recovers
# ---------------------------------------------------------------------------

class TestReEnable:
    def test_reenable_when_importance_recovers(self, tracker: FeatureImportanceTracker):
        n = 20
        # Establish high baseline
        tracker.update(_uniform_tensor(n, 0.1))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=30,
            cooldown_bars=1,
            max_disable_pct=1.0,
        )

        # Drive feat_5 low
        for _ in range(30):
            vals = [0.1] * n
            vals[5] = 0.0001
            tracker.update(_make_tensor(vals))

        selector.tick()
        result = selector.evaluate()
        assert result is not None
        assert "feat_5" in result.features_to_disable
        assert "feat_5" in selector.get_disabled_features()

        # Now recover feat_5 importance above 2 * min_importance = 0.02
        for _ in range(30):
            vals = [0.1] * n
            vals[5] = 0.05  # well above threshold
            tracker.update(_make_tensor(vals))

        selector.tick()
        result2 = selector.evaluate()
        assert result2 is not None
        assert "feat_5" in result2.features_to_enable
        assert "feat_5" in selector.get_active_features()


# ---------------------------------------------------------------------------
# Test: force_enable / force_disable
# ---------------------------------------------------------------------------

class TestForceToggle:
    def test_force_disable(self, selector: AdaptiveFeatureSelector):
        selector.force_disable("feat_5")
        assert "feat_5" in selector.get_disabled_features()
        assert "feat_5" not in selector.get_active_features()

    def test_force_enable(self, selector: AdaptiveFeatureSelector):
        selector.force_disable("feat_5")
        selector.force_enable("feat_5")
        assert "feat_5" in selector.get_active_features()
        assert "feat_5" not in selector.get_disabled_features()

    def test_force_disable_respects_always_on(self, selector: AdaptiveFeatureSelector):
        # feat_0 is always_on
        selector.force_disable("feat_0")
        assert "feat_0" in selector.get_active_features()
        assert "feat_0" not in selector.get_disabled_features()


# ---------------------------------------------------------------------------
# Test: reset re-enables all
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_reenables_all(self, selector: AdaptiveFeatureSelector, feature_names: list[str]):
        selector.force_disable("feat_5")
        selector.force_disable("feat_6")
        assert len(selector.get_disabled_features()) == 2

        selector.reset()
        assert selector.get_active_features() == set(feature_names)
        assert selector.get_disabled_features() == set()


# ---------------------------------------------------------------------------
# Test: save/load roundtrip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, selector: AdaptiveFeatureSelector, tmp_path: Path):
        selector.force_disable("feat_3")
        selector.force_disable("feat_7")

        save_path = tmp_path / "selector_state.json"
        selector.save(save_path)

        assert save_path.exists()
        with open(save_path) as f:
            data = json.load(f)
        assert "feat_3" in data["disabled_features"]
        assert "feat_7" in data["disabled_features"]

        # Create a fresh selector and load
        tracker2 = FeatureImportanceTracker(
            feature_names=[f"feat_{i}" for i in range(20)], ema_alpha=0.5
        )
        selector2 = AdaptiveFeatureSelector(tracker=tracker2, min_importance=0.01)
        selector2.load(save_path)

        assert selector2.get_disabled_features() == {"feat_3", "feat_7"}
        assert "feat_3" not in selector2.get_active_features()
        assert "feat_7" not in selector2.get_active_features()


# ---------------------------------------------------------------------------
# Test: drift_scores calculation
# ---------------------------------------------------------------------------

class TestDriftScores:
    def test_drift_scores_populated(self, tracker: FeatureImportanceTracker):
        n = 20
        for _ in range(5):
            tracker.update(_uniform_tensor(n, 0.05))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=2,
            cooldown_bars=1,
        )
        selector.tick()
        result = selector.evaluate()
        assert result is not None
        assert len(result.drift_scores) == n
        # All features had same importance throughout, so drift should be ~0
        for score in result.drift_scores.values():
            assert abs(score) < 0.01

    def test_drift_scores_reflect_importance_change(self, tracker: FeatureImportanceTracker):
        n = 20
        # Start high
        tracker.update(_uniform_tensor(n, 0.1))

        selector = AdaptiveFeatureSelector(
            tracker=tracker,
            min_importance=0.01,
            drift_window=20,
            cooldown_bars=1,
        )

        # Drive feat_2 low
        for _ in range(20):
            vals = [0.1] * n
            vals[2] = 0.001
            tracker.update(_make_tensor(vals))

        selector.tick()
        result = selector.evaluate()
        assert result is not None
        # feat_2 drift should be positive (importance dropped)
        assert result.drift_scores["feat_2"] > 0.5


# ---------------------------------------------------------------------------
# Test: empty tracker history (no crash)
# ---------------------------------------------------------------------------

class TestEmptyTracker:
    def test_evaluate_with_no_updates(self, feature_names: list[str]):
        tracker = FeatureImportanceTracker(feature_names=feature_names, ema_alpha=0.5)
        selector = AdaptiveFeatureSelector(
            tracker=tracker, min_importance=0.01, cooldown_bars=1
        )
        selector.tick()
        result = selector.evaluate()
        assert result is not None
        assert result.features_to_disable == []
        assert result.features_to_enable == []
        assert result.active_count == len(feature_names)

    def test_evaluate_with_single_update(self, tracker: FeatureImportanceTracker):
        tracker.update(_uniform_tensor(20, 0.05))
        selector = AdaptiveFeatureSelector(
            tracker=tracker, min_importance=0.01, cooldown_bars=1
        )
        selector.tick()
        result = selector.evaluate()
        assert result is not None
        # With only one snapshot, drift is 0 everywhere — no disables
        assert result.features_to_disable == []


# ---------------------------------------------------------------------------
# Test: FeatureSelectionResult dataclass
# ---------------------------------------------------------------------------

class TestFeatureSelectionResult:
    def test_result_fields(self):
        result = FeatureSelectionResult(
            features_to_disable=["a"],
            features_to_enable=["b"],
            drift_scores={"a": 0.8, "b": -0.1},
            active_count=9,
            total_count=10,
        )
        assert result.features_to_disable == ["a"]
        assert result.features_to_enable == ["b"]
        assert result.active_count == 9
        assert result.total_count == 10
        assert result.timestamp is not None
