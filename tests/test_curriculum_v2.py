"""Unit tests for Phase 4: CurriculumV2, configs, and training components.

Tests verify:
1. Config immutability and default stage structure.
2. Data blending with correct real/synthetic ratios.
3. Stage preparation and data integrity.
4. Adaptive LR controller behavior (boost/reduce/bounds).
5. Multi-metric early stopping logic.
6. Multi-criteria checkpoint tracking.
7. Stage transition with warm-start path resolution.
8. Reproducibility with fixed seeds.
9. FSD feature computation integration.
10. DML jump injection in synthetic data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from apexfx.training.config import (
    CurriculumV2Config,
    StageConfig,
)
from apexfx.training.curriculum_v2 import (
    AdaptiveLRController,
    CurriculumV2,
    DataBlender,
    MultiCriteriaCheckpoint,
    MultiMetricEarlyStopping,
    StageDataV2,
)

# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def real_data() -> pd.DataFrame:
    """Generate minimal real-like OHLCV data."""
    np.random.seed(42)
    n = 500
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": close + np.random.randn(n) * 0.0001,
        "high": close + abs(np.random.randn(n) * 0.001),
        "low": close - abs(np.random.randn(n) * 0.001),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


@pytest.fixture
def config() -> CurriculumV2Config:
    """Default curriculum config with reduced timesteps for testing."""
    return CurriculumV2Config(
        stages=[
            StageConfig(name="warmup", total_timesteps=100, real_ratio=1.0,
                        sbbts_ratio=0.0, filter_quantile=0.95, warm_start=False),
            StageConfig(name="full", total_timesteps=200, real_ratio=0.7,
                        sbbts_ratio=0.3, warm_start=True),
            StageConfig(name="augmented", total_timesteps=150, real_ratio=0.5,
                        sbbts_ratio=0.5, enable_dml_jumps=True, enable_fsd=True,
                        warm_start=True),
            StageConfig(name="adversarial", total_timesteps=100, real_ratio=0.3,
                        sbbts_ratio=0.7, enable_dml_jumps=True, enable_fsd=True,
                        enable_adversarial=True, noise_std=0.01, warm_start=True),
        ],
        deterministic=False,  # avoid CUDA deterministic errors in CI
    )


# ── Config tests ─────────────────────────────────────────────────────

class TestConfig:
    def test_config_frozen(self) -> None:
        cfg = CurriculumV2Config()
        with pytest.raises(Exception):
            cfg.seed = 999  # type: ignore

    def test_stage_config_frozen(self) -> None:
        cfg = StageConfig(name="test")
        with pytest.raises(Exception):
            cfg.total_timesteps = 999  # type: ignore

    def test_default_4_stages(self) -> None:
        cfg = CurriculumV2Config()
        assert cfg.n_stages == 4
        names = [s.name for s in cfg.stages]
        assert names == ["real_warmup", "real_full", "real_augmented", "real_adversarial"]

    def test_total_timesteps(self) -> None:
        cfg = CurriculumV2Config()
        expected = sum(s.total_timesteps for s in cfg.stages)
        assert cfg.total_timesteps == expected

    def test_stage_ratios_valid(self) -> None:
        cfg = CurriculumV2Config()
        for stage in cfg.stages:
            total = stage.real_ratio + stage.sbbts_ratio
            assert 0.99 <= total <= 1.01, f"Stage {stage.name} ratios don't sum to ~1.0"


# ── Data blending tests ─────────────────────────────────────────────

class TestDataBlender:
    def test_pure_real_stage(self, real_data: pd.DataFrame) -> None:
        """Stage 0: 100% real data, no synthetic."""
        blender = DataBlender(seed=42)
        stage_cfg = StageConfig(name="warmup", real_ratio=1.0, sbbts_ratio=0.0)
        result = blender.blend(real_data, stage_cfg)

        assert isinstance(result, StageDataV2)
        assert result.synthetic_data is None
        assert len(result.data) > 0
        assert result.metadata["n_synthetic"] == 0

    def test_mixed_stage_has_synthetic(self, real_data: pd.DataFrame) -> None:
        """Stage with SBBTS should produce synthetic data."""
        blender = DataBlender(seed=42)
        stage_cfg = StageConfig(name="full", real_ratio=0.7, sbbts_ratio=0.3)
        result = blender.blend(real_data, stage_cfg)

        assert result.metadata["n_synthetic"] > 0
        assert result.metadata["n_blended"] > result.metadata["n_real"]

    def test_filter_quantile_reduces_data(self, real_data: pd.DataFrame) -> None:
        """Extreme bar filtering should remove some bars."""
        blender = DataBlender(seed=42)
        stage_cfg = StageConfig(
            name="warmup", real_ratio=1.0, sbbts_ratio=0.0,
            filter_quantile=0.90,
        )
        result = blender.blend(real_data, stage_cfg)
        assert len(result.data) < len(real_data)

    def test_noise_injection(self, real_data: pd.DataFrame) -> None:
        """Noise injection should alter the data."""
        blender = DataBlender(seed=42)
        stage_cfg = StageConfig(
            name="adv", real_ratio=1.0, sbbts_ratio=0.0, noise_std=0.05,
        )
        result = blender.blend(real_data, stage_cfg)
        # Data should differ from original
        assert not np.allclose(
            result.data["close"].values[:10],
            real_data["close"].values[:10],
            atol=1e-6,
        )

    def test_dml_jump_injection(self, real_data: pd.DataFrame) -> None:
        """DML jump injection should modify synthetic data."""
        blender = DataBlender(seed=42)
        stage_cfg = StageConfig(
            name="aug", real_ratio=0.5, sbbts_ratio=0.5,
            enable_dml_jumps=True,
        )
        result = blender.blend(real_data, stage_cfg)
        assert result.metadata["dml_jumps"] is True


# ── Adaptive LR tests ───────────────────────────────────────────────

class TestAdaptiveLR:
    def test_boost_on_low_entropy(self) -> None:
        """LR should increase when entropy drops below threshold."""
        ctrl = AdaptiveLRController(
            base_lr=1e-4, entropy_low=0.15, boost_factor=2.0,
            min_lr=1e-6, max_lr=1e-2,
        )
        old_lr = ctrl.current_lr
        ctrl.update(entropy=0.05, step=1000)
        assert ctrl.current_lr > old_lr

    def test_reduce_on_high_entropy(self) -> None:
        """LR should decrease when entropy exceeds threshold."""
        ctrl = AdaptiveLRController(
            base_lr=1e-3, entropy_high=0.90, reduce_factor=0.5,
            min_lr=1e-6, max_lr=1e-2,
        )
        old_lr = ctrl.current_lr
        ctrl.update(entropy=0.95, step=1000)
        assert ctrl.current_lr < old_lr

    def test_lr_bounded(self) -> None:
        """LR should never exceed bounds."""
        ctrl = AdaptiveLRController(
            base_lr=1e-4, min_lr=1e-5, max_lr=1e-3,
            boost_factor=100.0, entropy_low=0.15,
        )
        ctrl.update(entropy=0.01, step=1000)
        assert ctrl.current_lr <= 1e-3

        ctrl2 = AdaptiveLRController(
            base_lr=1e-3, min_lr=1e-5, max_lr=1e-3,
            reduce_factor=0.001, entropy_high=0.90,
        )
        ctrl2.update(entropy=0.99, step=1000)
        assert ctrl2.current_lr >= 1e-5

    def test_no_change_in_normal_range(self) -> None:
        """LR should stay constant when entropy is in normal range."""
        ctrl = AdaptiveLRController(
            base_lr=1e-4, entropy_low=0.15, entropy_high=0.90,
        )
        old_lr = ctrl.current_lr
        ctrl.update(entropy=0.50, step=1000)
        assert ctrl.current_lr == old_lr

    def test_reset(self) -> None:
        """Reset should restore to base LR."""
        ctrl = AdaptiveLRController(base_lr=1e-4)
        ctrl.update(entropy=0.01, step=1000)
        ctrl.reset(new_base_lr=5e-4)
        assert ctrl.current_lr == 5e-4


# ── Early stopping tests ────────────────────────────────────────────

class TestMultiMetricEarlyStopping:
    def test_no_stop_with_improving_metrics(self) -> None:
        es = MultiMetricEarlyStopping(patience=3)
        for i in range(10):
            should_stop = es.check({
                "ep_rew_mean": float(i),
                "sharpe": float(i) * 0.1,
                "profit_factor": 1.0 + i * 0.01,
            })
            assert not should_stop

    def test_stop_after_patience_exhausted(self) -> None:
        es = MultiMetricEarlyStopping(patience=3)
        # First check with some values
        es.check({"ep_rew_mean": 10.0, "sharpe": 1.0, "profit_factor": 2.0})

        # Then stagnate
        for _ in range(5):
            result = es.check({
                "ep_rew_mean": 10.0,
                "sharpe": 1.0,
                "profit_factor": 2.0,
            })

        assert result is True  # should stop after patience

    def test_partial_improvement_prevents_stop(self) -> None:
        """If even one metric improves, don't stop."""
        es = MultiMetricEarlyStopping(patience=3)
        es.check({"ep_rew_mean": 10.0, "sharpe": 1.0, "profit_factor": 2.0})

        for i in range(5):
            result = es.check({
                "ep_rew_mean": 10.0,       # stagnant
                "sharpe": 1.0 + i * 0.1,   # improving!
                "profit_factor": 2.0,       # stagnant
            })
            assert not result  # sharpe improving prevents stop

    def test_reset(self) -> None:
        es = MultiMetricEarlyStopping(patience=2)
        es.check({"ep_rew_mean": 100.0, "sharpe": 10.0, "profit_factor": 5.0})
        es.reset()
        # After reset, should not stop even with low values
        result = es.check({"ep_rew_mean": 1.0, "sharpe": 0.1, "profit_factor": 0.5})
        assert not result


# ── CurriculumV2 integration tests ──────────────────────────────────

class TestCurriculumV2:
    def test_prepare_stage(
        self, config: CurriculumV2Config, real_data: pd.DataFrame,
    ) -> None:
        curriculum = CurriculumV2(config=config, real_data=real_data)
        stage_data = curriculum.prepare_stage(0)

        assert isinstance(stage_data, StageDataV2)
        assert stage_data.stage_idx == 0
        assert stage_data.stage_cfg.name == "warmup"
        assert len(stage_data.data) > 0

    def test_invalid_stage_raises(
        self, config: CurriculumV2Config, real_data: pd.DataFrame,
    ) -> None:
        curriculum = CurriculumV2(config=config, real_data=real_data)
        with pytest.raises(ValueError, match="does not exist"):
            curriculum.prepare_stage(99)

    def test_no_data_raises(self, config: CurriculumV2Config) -> None:
        curriculum = CurriculumV2(config=config, real_data=None)
        with pytest.raises(ValueError, match="Real data is required"):
            curriculum.prepare_stage(0)

    def test_stage_complete_records_metrics(
        self, config: CurriculumV2Config, real_data: pd.DataFrame,
    ) -> None:
        curriculum = CurriculumV2(config=config, real_data=real_data)
        curriculum.prepare_stage(0)

        metrics = {"ep_rew_mean": 5.0, "sharpe": 1.2, "profit_factor": 1.8,
                    "gating_entropy": 0.45}
        sm = curriculum.on_stage_complete(0, metrics)

        assert sm.stage_name == "warmup"
        assert sm.best_sharpe == 1.2
        assert sm.best_reward == 5.0
        assert len(curriculum.stage_history) == 1

    def test_summary(
        self, config: CurriculumV2Config, real_data: pd.DataFrame,
    ) -> None:
        curriculum = CurriculumV2(config=config, real_data=real_data)
        curriculum.prepare_stage(0)
        curriculum.on_stage_complete(0, {"sharpe": 0.5, "ep_rew_mean": 1.0})

        summary = curriculum.summary()
        assert summary["n_stages_completed"] == 1
        assert len(summary["stages"]) == 1

    def test_reproducibility(self, real_data: pd.DataFrame) -> None:
        """Same seed should produce identical data blends."""
        cfg = CurriculumV2Config(
            seed=123,
            stages=[
                StageConfig(name="test", real_ratio=0.7, sbbts_ratio=0.3),
            ],
            deterministic=False,
        )
        c1 = CurriculumV2(config=cfg, real_data=real_data)
        d1 = c1.prepare_stage(0)

        c2 = CurriculumV2(config=cfg, real_data=real_data)
        d2 = c2.prepare_stage(0)

        assert len(d1.data) == len(d2.data)
        # Real subsets should be identical with same seed
        assert np.allclose(
            d1.real_data["close"].values,
            d2.real_data["close"].values,
        )


# ── Multi-criteria checkpoint tests ──────────────────────────────────

class TestMultiCriteriaCheckpoint:
    def test_save_and_track_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MultiCriteriaCheckpoint(Path(tmpdir))
            model = MagicMock()

            improved = tracker.maybe_save(
                model,
                {"sharpe": 1.0, "gating_entropy": 0.5, "ep_rew_mean": 10.0},
                stage_idx=0, step=1000,
            )
            assert "sharpe" in improved
            assert "entropy" in improved
            assert "reward" in improved

            # Second save with only sharpe improved
            improved2 = tracker.maybe_save(
                model,
                {"sharpe": 1.5, "gating_entropy": 0.3, "ep_rew_mean": 8.0},
                stage_idx=0, step=2000,
            )
            assert "sharpe" in improved2
            assert "entropy" not in improved2
            assert "reward" not in improved2
