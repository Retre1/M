"""Unit tests for FeatureImportanceTracker."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import torch

from apexfx.features.importance_tracker import FeatureImportanceTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_names() -> list[str]:
    return [f"feat_{i}" for i in range(10)]


@pytest.fixture
def tracker(feature_names: list[str]) -> FeatureImportanceTracker:
    return FeatureImportanceTracker(feature_names=feature_names, ema_alpha=0.1)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_init_stores_names(self, feature_names: list[str]) -> None:
        tracker = FeatureImportanceTracker(feature_names=feature_names)
        assert tracker.feature_names == feature_names
        assert tracker.n_features == len(feature_names)

    def test_empty_names_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FeatureImportanceTracker(feature_names=[])

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            FeatureImportanceTracker(feature_names=["a"], ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            FeatureImportanceTracker(feature_names=["a"], ema_alpha=1.5)

    def test_no_updates_returns_empty(self, tracker: FeatureImportanceTracker) -> None:
        assert tracker.get_importance_dict() == {}
        assert tracker.get_top_k() == []
        assert tracker.get_bottom_k() == []
        assert tracker.update_count == 0


# ---------------------------------------------------------------------------
# EMA convergence
# ---------------------------------------------------------------------------

class TestEMAConvergence:
    def test_first_update_initializes_directly(
        self, tracker: FeatureImportanceTracker,
    ) -> None:
        # One-hot: feature 0 has all the weight
        t = torch.zeros(10)
        t[0] = 1.0
        tracker.update(t)
        d = tracker.get_importance_dict()
        assert d["feat_0"] == pytest.approx(1.0)
        assert d["feat_1"] == pytest.approx(0.0)

    def test_ema_converges_toward_constant_signal(
        self, feature_names: list[str],
    ) -> None:
        """After many updates with the same input, EMA should converge to it."""
        tracker = FeatureImportanceTracker(feature_names=feature_names, ema_alpha=0.1)

        target = torch.zeros(10)
        target[3] = 0.5
        target[7] = 0.5

        # Feed 200 identical updates
        for _ in range(200):
            tracker.update(target)

        d = tracker.get_importance_dict()
        assert d["feat_3"] == pytest.approx(0.5, abs=0.01)
        assert d["feat_7"] == pytest.approx(0.5, abs=0.01)
        assert d["feat_0"] == pytest.approx(0.0, abs=0.01)

    def test_ema_alpha_1_equals_latest_value(
        self, feature_names: list[str],
    ) -> None:
        """With alpha=1.0, EMA should always equal the latest input."""
        tracker = FeatureImportanceTracker(feature_names=feature_names, ema_alpha=1.0)

        t1 = torch.ones(10) / 10.0
        tracker.update(t1)

        t2 = torch.zeros(10)
        t2[0] = 1.0
        tracker.update(t2)

        d = tracker.get_importance_dict()
        assert d["feat_0"] == pytest.approx(1.0)
        assert d["feat_1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Top-K / Bottom-K ordering
# ---------------------------------------------------------------------------

class TestTopKBottomK:
    def test_top_k_ordering(self, tracker: FeatureImportanceTracker) -> None:
        # Linear importance: feat_0=0, feat_1=1, ..., feat_9=9
        t = torch.arange(10, dtype=torch.float32)
        tracker.update(t)

        top5 = tracker.get_top_k(k=5)
        assert len(top5) == 5
        # Highest first
        names = [name for name, _ in top5]
        assert names == ["feat_9", "feat_8", "feat_7", "feat_6", "feat_5"]

    def test_bottom_k_ordering(self, tracker: FeatureImportanceTracker) -> None:
        t = torch.arange(10, dtype=torch.float32)
        tracker.update(t)

        bottom3 = tracker.get_bottom_k(k=3)
        assert len(bottom3) == 3
        names = [name for name, _ in bottom3]
        assert names == ["feat_0", "feat_1", "feat_2"]

    def test_top_k_larger_than_n_features(
        self, tracker: FeatureImportanceTracker,
    ) -> None:
        t = torch.ones(10)
        tracker.update(t)
        top20 = tracker.get_top_k(k=20)
        assert len(top20) == 10  # Capped at n_features


# ---------------------------------------------------------------------------
# Tensor shape handling
# ---------------------------------------------------------------------------

class TestTensorShapes:
    def test_1d_tensor(self, tracker: FeatureImportanceTracker) -> None:
        t = torch.rand(10)
        tracker.update(t)
        assert tracker.update_count == 1
        assert len(tracker.get_importance_dict()) == 10

    def test_2d_tensor_batch(self, tracker: FeatureImportanceTracker) -> None:
        # (batch=4, n_vars=10)
        t = torch.rand(4, 10)
        tracker.update(t)
        d = tracker.get_importance_dict()
        # Should be the mean across batch dim
        expected = t.mean(dim=0)
        for i, name in enumerate(tracker.feature_names):
            assert d[name] == pytest.approx(expected[i].item(), abs=1e-5)

    def test_3d_tensor_batch_time(self, tracker: FeatureImportanceTracker) -> None:
        # (batch=2, time=5, n_vars=10)
        t = torch.rand(2, 5, 10)
        tracker.update(t)
        d = tracker.get_importance_dict()
        expected = t.mean(dim=(0, 1))
        for i, name in enumerate(tracker.feature_names):
            assert d[name] == pytest.approx(expected[i].item(), abs=1e-5)

    def test_wrong_feature_count_raises(
        self, tracker: FeatureImportanceTracker,
    ) -> None:
        with pytest.raises(ValueError, match="features"):
            tracker.update(torch.rand(5))  # Expected 10

    def test_4d_tensor_raises(self, tracker: FeatureImportanceTracker) -> None:
        with pytest.raises(ValueError, match="1D, 2D, or 3D"):
            tracker.update(torch.rand(2, 3, 4, 10))

    def test_gradient_not_affected(self, tracker: FeatureImportanceTracker) -> None:
        """Updating the tracker should not affect gradients."""
        t = torch.rand(10, requires_grad=True)
        tracker.update(t)
        # The tensor should still have requires_grad
        assert t.requires_grad
        # No error should occur during backward if we use the original tensor
        loss = t.sum()
        loss.backward()
        assert t.grad is not None


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(
        self,
        tracker: FeatureImportanceTracker,
        tmp_path: Path,
    ) -> None:
        # Feed some data
        for i in range(5):
            tracker.update(torch.rand(10))

        save_path = tmp_path / "importance.json"
        tracker.save(save_path)

        assert save_path.exists()

        loaded = FeatureImportanceTracker.load(save_path)

        assert loaded.feature_names == tracker.feature_names
        assert loaded.n_features == tracker.n_features
        assert loaded.update_count == tracker.update_count

        # EMA values should match
        orig_dict = tracker.get_importance_dict()
        loaded_dict = loaded.get_importance_dict()
        for name in tracker.feature_names:
            assert loaded_dict[name] == pytest.approx(orig_dict[name], abs=1e-10)

        # History should match
        assert loaded.get_history() == tracker.get_history()

    def test_save_creates_parent_dirs(self, tracker: FeatureImportanceTracker, tmp_path: Path) -> None:
        tracker.update(torch.rand(10))
        save_path = tmp_path / "nested" / "dir" / "state.json"
        tracker.save(save_path)
        assert save_path.exists()

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            FeatureImportanceTracker.load(Path("/nonexistent/path.json"))

    def test_saved_json_is_valid(
        self,
        tracker: FeatureImportanceTracker,
        tmp_path: Path,
    ) -> None:
        tracker.update(torch.rand(10))
        save_path = tmp_path / "state.json"
        tracker.save(save_path)

        with open(save_path) as f:
            data = json.load(f)

        assert "feature_names" in data
        assert "ema" in data
        assert "history" in data
        assert data["update_count"] == 1


# ---------------------------------------------------------------------------
# History deque max size
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_grows_with_updates(
        self, tracker: FeatureImportanceTracker,
    ) -> None:
        for _ in range(5):
            tracker.update(torch.rand(10))
        assert len(tracker.get_history()) == 5

    def test_history_respects_max_size(self, feature_names: list[str]) -> None:
        tracker = FeatureImportanceTracker(
            feature_names=feature_names,
            history_size=3,
        )
        for _ in range(10):
            tracker.update(torch.rand(10))

        history = tracker.get_history()
        assert len(history) == 3
        assert tracker.update_count == 10

    def test_history_snapshots_are_independent(
        self, tracker: FeatureImportanceTracker,
    ) -> None:
        """Each history entry should be a snapshot, not a reference."""
        tracker.update(torch.ones(10))
        tracker.update(torch.zeros(10))

        history = tracker.get_history()
        assert len(history) == 2
        # First snapshot should still reflect the first update
        assert history[0] != history[1]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_updates(self, feature_names: list[str]) -> None:
        """Multiple threads updating simultaneously should not corrupt state."""
        tracker = FeatureImportanceTracker(
            feature_names=feature_names,
            ema_alpha=0.1,
            history_size=10000,
        )
        n_threads = 8
        updates_per_thread = 100
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(updates_per_thread):
                    tracker.update(torch.rand(10))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent updates: {errors}"
        assert tracker.update_count == n_threads * updates_per_thread

        # EMA should still be valid (all values finite)
        d = tracker.get_importance_dict()
        assert len(d) == 10
        for v in d.values():
            assert isinstance(v, float)
            assert v == v  # Not NaN

    def test_concurrent_read_write(self, feature_names: list[str]) -> None:
        """Concurrent reads and writes should not raise."""
        tracker = FeatureImportanceTracker(
            feature_names=feature_names,
            ema_alpha=0.1,
        )
        # Seed with initial data
        tracker.update(torch.rand(10))

        errors: list[Exception] = []

        def writer() -> None:
            try:
                for _ in range(100):
                    tracker.update(torch.rand(10))
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(100):
                    tracker.get_top_k(5)
                    tracker.get_importance_dict()
                    tracker.get_history()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_importance_tracking_config_defaults(self) -> None:
        from apexfx.config.schema import ImportanceTrackingConfig

        cfg = ImportanceTrackingConfig()
        assert cfg.enabled is False
        assert cfg.ema_alpha == 0.01
        assert cfg.history_size == 1000
        assert cfg.top_k == 5

    def test_importance_tracking_in_model_config(self) -> None:
        from apexfx.config.schema import ModelConfig

        mc = ModelConfig()
        assert hasattr(mc, "importance_tracking")
        assert mc.importance_tracking.enabled is False
