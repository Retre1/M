"""Tests for ModelRegistry — persistent model version tracking and rollback."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from apexfx.training.model_registry import ModelRegistry, ModelVersion


@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    """Create a fresh registry backed by a temporary database."""
    db = tmp_path / "test_registry.db"
    return ModelRegistry(db_path=db)


class TestRegister:
    def test_register_creates_active_version(self, registry: ModelRegistry) -> None:
        v = registry.register("models/v1.zip", validation_sharpe=1.5)
        assert isinstance(v, ModelVersion)
        assert v.version_id == 1
        assert v.is_active is True
        assert v.model_path == "models/v1.zip"
        assert v.validation_sharpe == 1.5
        assert v.validation_return == 0.0

    def test_register_deactivates_previous(self, registry: ModelRegistry) -> None:
        v1 = registry.register("models/v1.zip", validation_sharpe=1.0)
        v2 = registry.register("models/v2.zip", validation_sharpe=1.8)

        # v2 is active, v1 is not
        assert v2.is_active is True
        old = registry.get_version(v1.version_id)
        assert old is not None
        assert old.is_active is False

    def test_register_with_optional_fields(self, registry: ModelRegistry) -> None:
        v = registry.register(
            "models/v1.zip",
            validation_sharpe=2.0,
            validation_return=0.15,
            notes="weekly retrain",
        )
        assert v.validation_return == 0.15
        assert v.notes == "weekly retrain"


class TestGetActive:
    def test_empty_registry_returns_none(self, registry: ModelRegistry) -> None:
        assert registry.get_active() is None

    def test_returns_latest_registered(self, registry: ModelRegistry) -> None:
        registry.register("models/v1.zip", validation_sharpe=1.0)
        registry.register("models/v2.zip", validation_sharpe=1.5)
        active = registry.get_active()
        assert active is not None
        assert active.model_path == "models/v2.zip"
        assert active.is_active is True


class TestRollback:
    def test_rollback_switches_active_version(self, registry: ModelRegistry) -> None:
        v1 = registry.register("models/v1.zip", validation_sharpe=1.0)
        v2 = registry.register("models/v2.zip", validation_sharpe=1.5)

        rolled = registry.rollback(v1.version_id)
        assert rolled.version_id == v1.version_id
        assert rolled.is_active is True

        # v2 should no longer be active
        v2_now = registry.get_version(v2.version_id)
        assert v2_now is not None
        assert v2_now.is_active is False

        # Active should be v1
        active = registry.get_active()
        assert active is not None
        assert active.version_id == v1.version_id

    def test_rollback_nonexistent_raises(self, registry: ModelRegistry) -> None:
        with pytest.raises(ValueError, match="not found"):
            registry.rollback(999)


class TestGetHistory:
    def test_returns_versions_newest_first(self, registry: ModelRegistry) -> None:
        registry.register("models/v1.zip", validation_sharpe=1.0)
        registry.register("models/v2.zip", validation_sharpe=1.5)
        registry.register("models/v3.zip", validation_sharpe=2.0)

        history = registry.get_history()
        assert len(history) == 3
        assert history[0].version_id == 3
        assert history[1].version_id == 2
        assert history[2].version_id == 1

    def test_limit_works(self, registry: ModelRegistry) -> None:
        for i in range(5):
            registry.register(f"models/v{i}.zip", validation_sharpe=float(i))
        history = registry.get_history(limit=2)
        assert len(history) == 2

    def test_empty_history(self, registry: ModelRegistry) -> None:
        assert registry.get_history() == []


class TestDeactivateCurrent:
    def test_deactivate_leaves_no_active(self, registry: ModelRegistry) -> None:
        registry.register("models/v1.zip", validation_sharpe=1.0)
        registry.deactivate_current()
        assert registry.get_active() is None
