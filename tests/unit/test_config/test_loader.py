"""Tests for config loading and YAML-to-schema mapping."""

from __future__ import annotations

from apexfx.config.loader import load_config
from apexfx.config.schema import AppConfig


class TestConfigLoading:
    def test_load_config_returns_app_config(self):
        config = load_config("configs")
        assert isinstance(config, AppConfig)

    def test_tft_config_from_yaml(self):
        config = load_config("configs")
        assert config.model.tft.d_model == 32
        assert config.model.tft.n_heads == 4

    def test_trend_agent_config_from_yaml(self):
        """Verify model.yaml trend_agent key maps correctly."""
        config = load_config("configs")
        assert config.model.trend_agent.hidden_sizes == [128, 128, 64]
        assert config.model.trend_agent.dropout == 0.1

    def test_reversion_agent_config_from_yaml(self):
        """Verify model.yaml reversion_agent key maps correctly."""
        config = load_config("configs")
        assert config.model.reversion_agent.hidden_sizes == [128, 128, 64]
        assert config.model.reversion_agent.z_score_threshold == 3.0

    def test_gating_config(self):
        config = load_config("configs")
        assert config.model.gating.n_agents == 3

    def test_rl_config(self):
        config = load_config("configs")
        assert config.model.rl.algorithm == "SAC"
        assert config.model.rl.gamma == 0.99

    def test_risk_config(self):
        config = load_config("configs")
        assert config.risk.max_drawdown_pct > 0
        assert config.risk.daily_var_limit > 0

    def test_mtf_config(self):
        config = load_config("configs")
        assert config.model.mtf.enabled is True
        assert config.model.mtf.lookback.d1 == 5
        assert config.model.mtf.lookback.h1 == 20
        assert config.model.mtf.lookback.m5 == 20

    def test_data_config(self):
        config = load_config("configs")
        assert config.data.feature_window > 0
