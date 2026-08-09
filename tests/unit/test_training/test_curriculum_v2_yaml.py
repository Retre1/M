"""Regression tests for loading CurriculumV2Config from training.yaml.

Run 6 had to be killed and restarted after it emerged that TrainerV2 built
``CurriculumV2Config()`` from hardcoded defaults and never read
``configs/training.yaml``. Nothing failed — the run simply trained with values
nobody had chosen. These tests fail if that reappears.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apexfx.training.config import CurriculumV2Config, load_curriculum_v2_config

REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _write(tmp_path: Path, payload: dict) -> Path:
    (tmp_path / "training.yaml").write_text(yaml.safe_dump(payload))
    return tmp_path


class TestStageMapping:
    def test_yaml_timesteps_reach_the_config(self, tmp_path: Path):
        """The value in the file must be the value the trainer gets."""
        cfg_dir = _write(tmp_path, {
            "curriculum": {"stages": [
                {"name": "real_warmup", "total_timesteps": 12_345},
            ]},
        })
        cfg = load_curriculum_v2_config(cfg_dir)
        assert cfg.n_stages == 1
        assert cfg.stages[0].total_timesteps == 12_345

    def test_defaults_differ_so_the_test_can_fail(self):
        """Guard the guard: 12_345 must not coincide with a default."""
        assert all(s.total_timesteps != 12_345 for s in CurriculumV2Config().stages)

    def test_v2_only_fields_inherit_from_the_matching_default_stage(self, tmp_path: Path):
        """The YAML has no real_ratio/sbbts_ratio; they come from the same-named stage."""
        cfg_dir = _write(tmp_path, {
            "curriculum": {"stages": [
                {"name": "real_full", "total_timesteps": 1_000},
            ]},
        })
        stage = load_curriculum_v2_config(cfg_dir).stages[0]
        default = next(s for s in CurriculumV2Config().stages if s.name == "real_full")
        assert stage.sbbts_ratio == default.sbbts_ratio
        assert stage.real_ratio == default.real_ratio

    def test_nested_augmentation_noise_is_mapped(self, tmp_path: Path):
        cfg_dir = _write(tmp_path, {
            "curriculum": {"stages": [
                {"name": "real_adversarial", "augmentation": {"noise_std": 0.007}},
            ]},
        })
        assert load_curriculum_v2_config(cfg_dir).stages[0].noise_std == pytest.approx(0.007)

    def test_unknown_stage_name_still_loads(self, tmp_path: Path):
        cfg_dir = _write(tmp_path, {
            "curriculum": {"stages": [{"name": "brand_new", "total_timesteps": 42}]},
        })
        stage = load_curriculum_v2_config(cfg_dir).stages[0]
        assert stage.name == "brand_new"
        assert stage.total_timesteps == 42

    def test_stage_without_name_is_rejected(self, tmp_path: Path):
        cfg_dir = _write(tmp_path, {"curriculum": {"stages": [{"total_timesteps": 1}]}})
        with pytest.raises(ValueError, match="name"):
            load_curriculum_v2_config(cfg_dir)


class TestTopLevelMapping:
    def test_ewc_and_early_stopping_are_mapped(self, tmp_path: Path):
        cfg_dir = _write(tmp_path, {
            "ewc": {"lambda_ewc": 1234.0, "gamma_ewc": 0.5},
            "early_stopping": {"patience": 99, "min_delta": 0.25},
            "checkpointing": {"save_freq": 777, "keep_best_n": 9},
        })
        cfg = load_curriculum_v2_config(cfg_dir)
        assert cfg.ewc_lambda == pytest.approx(1234.0)
        assert cfg.ewc_gamma == pytest.approx(0.5)
        assert cfg.early_stopping.patience == 99
        assert cfg.early_stopping.min_delta_reward == pytest.approx(0.25)
        assert cfg.checkpointing.save_freq == 777
        assert cfg.checkpointing.keep_best_n == 9


class TestUnmappedKeysAreReported:
    """A key that looks applied but is not is the bug this loader exists to stop."""

    def test_strict_mode_raises_on_unmapped_key(self, tmp_path: Path):
        cfg_dir = _write(tmp_path, {
            "curriculum": {"stages": [
                {"name": "real_warmup", "augmentation": {"price_shift_std": 0.002}},
            ]},
        })
        with pytest.raises(ValueError, match="price_shift_std"):
            load_curriculum_v2_config(cfg_dir, strict=True)

    def test_non_strict_mode_still_loads(self, tmp_path: Path):
        cfg_dir = _write(tmp_path, {
            "curriculum": {"stages": [
                {"name": "real_warmup", "data_source": "real", "total_timesteps": 5},
            ]},
        })
        assert load_curriculum_v2_config(cfg_dir).stages[0].total_timesteps == 5

    def test_missing_file_falls_back_to_defaults(self, tmp_path: Path):
        assert load_curriculum_v2_config(tmp_path) == CurriculumV2Config()


class TestShippedConfig:
    """The repository's own training.yaml must actually reach the trainer."""

    def test_repo_training_yaml_loads(self):
        cfg = load_curriculum_v2_config(REPO_CONFIGS)
        assert cfg.n_stages == 4
        assert [s.name for s in cfg.stages] == [
            "real_warmup", "real_full", "real_augmented", "real_adversarial",
        ]

    def test_run6_softening_is_in_effect(self):
        """Run 6 halved the adversarial stage and softened its noise.

        NB: this asserts the values, not the wiring. Run 6 was restarted after
        patching config.py by hand, so these defaults were made to match the
        YAML and the assertion would hold even with the loader bypassed.
        ``test_ewc_lambda_proves_yaml_overrides_defaults`` is the wiring proof.
        """
        cfg = load_curriculum_v2_config(REPO_CONFIGS)
        adversarial = next(s for s in cfg.stages if s.name == "real_adversarial")
        assert adversarial.total_timesteps == 500_000
        assert adversarial.noise_std == pytest.approx(0.003)

    def test_ewc_lambda_proves_yaml_overrides_defaults(self):
        """ewc_lambda is 5000 in code and 2000 in YAML — so this can fail.

        If the loader ever stops reaching training.yaml, this is the test that
        catches it against the repository's own shipped config.
        """
        assert CurriculumV2Config().ewc_lambda == pytest.approx(5000.0), (
            "default changed — pick another field where code and YAML differ"
        )
        assert load_curriculum_v2_config(REPO_CONFIGS).ewc_lambda == pytest.approx(2000.0)
