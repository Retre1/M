"""Tests for TFT pre-training opt-out via config.

The 30-epoch supervised TFT pretrain at the start of every training run is
the dominant cost on CPU (~60 min for the smoke config) and adds no value
when verifying env / filter / reward fixes.  ``model.tft.pretrain.enabled``
short-circuits the pretrain so the trainer goes straight to RL.

Defaults must stay backwards-compatible (enabled=True, epochs=30, etc.)
so existing GPU runs are not silently weakened.
"""

from __future__ import annotations

from apexfx.config.schema import TFTConfig, TFTPretrainConfig


class TestTFTPretrainConfigDefaults:
    """Defaults must match the historical hardcoded values."""

    def test_default_enabled_true(self) -> None:
        assert TFTPretrainConfig().enabled is True

    def test_default_epochs_30(self) -> None:
        assert TFTPretrainConfig().epochs == 30

    def test_default_batch_size_128(self) -> None:
        assert TFTPretrainConfig().batch_size == 128

    def test_default_patience_5(self) -> None:
        assert TFTPretrainConfig().patience == 5


class TestTFTConfigCarriesPretrain:
    """TFTConfig must expose the nested pretrain block with defaults."""

    def test_tft_config_has_pretrain(self) -> None:
        cfg = TFTConfig()
        assert isinstance(cfg.pretrain, TFTPretrainConfig)

    def test_tft_config_pretrain_default_enabled(self) -> None:
        assert TFTConfig().pretrain.enabled is True

    def test_pretrain_can_be_disabled(self) -> None:
        cfg = TFTConfig(pretrain=TFTPretrainConfig(enabled=False))
        assert cfg.pretrain.enabled is False

    def test_pretrain_epochs_overridable(self) -> None:
        cfg = TFTConfig(pretrain=TFTPretrainConfig(epochs=2))
        assert cfg.pretrain.epochs == 2
