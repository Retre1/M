"""Tests for the unified BotConfig — the user-facing API surface."""

from __future__ import annotations

import pytest

from apexfx.aggressive.config import (
    BotConfig,
    Mt5LoginConfig,
    TelegramSettings,
)


# ---------------------------------------------------------------------------


class TestMt5LoginConfig:
    def test_valid_credentials(self) -> None:
        c = Mt5LoginConfig(login=12345, password="p", server="Demo")
        assert c.login == 12345

    def test_zero_login_rejected(self) -> None:
        with pytest.raises(ValueError, match="login"):
            Mt5LoginConfig(login=0, password="p", server="Demo")

    def test_negative_login_rejected(self) -> None:
        with pytest.raises(ValueError, match="login"):
            Mt5LoginConfig(login=-1, password="p", server="Demo")

    def test_empty_password_rejected(self) -> None:
        with pytest.raises(ValueError, match="password"):
            Mt5LoginConfig(login=12345, password="", server="Demo")

    def test_empty_server_rejected(self) -> None:
        with pytest.raises(ValueError, match="server"):
            Mt5LoginConfig(login=12345, password="p", server="")


# ---------------------------------------------------------------------------


class TestTelegramSettings:
    def test_enabled_when_both_set(self) -> None:
        t = TelegramSettings(bot_token="x", chat_id="y")
        assert t.enabled

    def test_disabled_when_empty(self) -> None:
        assert TelegramSettings().enabled is False

    def test_disabled_when_partial(self) -> None:
        assert TelegramSettings(bot_token="x", chat_id="").enabled is False
        assert TelegramSettings(bot_token="", chat_id="y").enabled is False


# ---------------------------------------------------------------------------


def _valid_mt5() -> Mt5LoginConfig:
    return Mt5LoginConfig(login=12345, password="p", server="Demo")


class TestBotConfigValidation:
    def test_minimal_valid(self) -> None:
        # Default symbols, default everything
        config = BotConfig(mt5=_valid_mt5())
        assert "EURUSD" in config.symbols
        assert config.timeframe == "H4"

    def test_empty_symbols_rejected(self) -> None:
        with pytest.raises(ValueError, match="symbols"):
            BotConfig(mt5=_valid_mt5(), symbols=[])

    def test_invalid_timeframe_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeframe"):
            BotConfig(mt5=_valid_mt5(), timeframe="2H")

    def test_invalid_poll_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="poll_interval"):
            BotConfig(mt5=_valid_mt5(), poll_interval_s=0)
        with pytest.raises(ValueError, match="poll_interval"):
            BotConfig(mt5=_valid_mt5(), poll_interval_s=-10)

    def test_invalid_reconnect_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="reconnect_interval"):
            BotConfig(mt5=_valid_mt5(), reconnect_interval_s=0)

    def test_invalid_strategy_params_propagate(self) -> None:
        # exit_period must be < entry_period (TurtleConfig invariant)
        with pytest.raises(ValueError):
            BotConfig(mt5=_valid_mt5(), entry_period=10, exit_period=20)

    def test_invalid_risk_params_propagate(self) -> None:
        # daily_loss_pct must be in (0, 1) (CircuitBreakerConfig invariant)
        with pytest.raises(ValueError):
            BotConfig(mt5=_valid_mt5(), daily_loss_pct=0)
        with pytest.raises(ValueError):
            BotConfig(mt5=_valid_mt5(), daily_loss_pct=1.5)


# ---------------------------------------------------------------------------


class TestBotConfigBuilders:
    def test_build_turtle_config_mirrors_fields(self) -> None:
        c = BotConfig(
            mt5=_valid_mt5(),
            entry_period=25, exit_period=12, atr_period=15,
            risk_per_unit_pct=0.02, max_units=3,
        )
        t = c.build_turtle_config()
        assert t.entry_period == 25
        assert t.exit_period == 12
        assert t.atr_period == 15
        assert t.risk_per_unit_pct == 0.02
        assert t.max_units == 3

    def test_build_breaker_config_mirrors_fields(self) -> None:
        c = BotConfig(
            mt5=_valid_mt5(),
            daily_loss_pct=0.05, weekly_loss_pct=0.15,
            monthly_dd_pct=0.30, max_consecutive_failed_orders=5,
        )
        b = c.build_breaker_config()
        assert b.daily_loss_pct == 0.05
        assert b.weekly_loss_pct == 0.15
        assert b.monthly_dd_pct == 0.30
        assert b.max_consecutive_failed_orders == 5


# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_key_fields(self) -> None:
        c = BotConfig(mt5=_valid_mt5())
        s = c.summary()
        assert "12345" in s
        assert "Demo" in s
        assert "H4" in s
        assert "EURUSD" in s

    def test_summary_indicates_telegram_status(self) -> None:
        off = BotConfig(mt5=_valid_mt5())
        on = BotConfig(
            mt5=_valid_mt5(),
            telegram=TelegramSettings(bot_token="x", chat_id="y"),
        )
        assert "telegram=off" in off.summary()
        assert "telegram=on" in on.summary()


# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_from_env_missing_creds_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("APEXFX_MT5_LOGIN", raising=False)
        with pytest.raises(KeyError):
            BotConfig.from_env()

    def test_from_env_loads_credentials(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APEXFX_MT5_LOGIN", "55555")
        monkeypatch.setenv("APEXFX_MT5_PASSWORD", "secret")
        monkeypatch.setenv("APEXFX_MT5_SERVER", "Demo")
        c = BotConfig.from_env()
        assert c.mt5.login == 55555
        assert c.mt5.password == "secret"

    def test_from_env_loads_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APEXFX_MT5_LOGIN", "1")
        monkeypatch.setenv("APEXFX_MT5_PASSWORD", "p")
        monkeypatch.setenv("APEXFX_MT5_SERVER", "S")
        monkeypatch.setenv("APEXFX_SYMBOLS", "EURUSD,GBPUSD")
        monkeypatch.setenv("APEXFX_TIMEFRAME", "H1")
        monkeypatch.setenv("APEXFX_RISK_PER_UNIT", "0.005")
        monkeypatch.setenv("APEXFX_MAX_UNITS", "2")
        c = BotConfig.from_env()
        assert c.symbols == ["EURUSD", "GBPUSD"]
        assert c.timeframe == "H1"
        assert c.risk_per_unit_pct == 0.005
        assert c.max_units == 2
