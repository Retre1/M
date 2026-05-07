"""Tests for the Telegram notifier."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from apexfx.aggressive.alerts.telegram import (
    NullNotifier,
    TelegramConfig,
    TelegramNotifier,
)


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(bot_token="t", chat_id="c")


@pytest.fixture
def notifier(config: TelegramConfig) -> TelegramNotifier:
    n = TelegramNotifier(config)
    # Replace session with mock — never actually hit the network
    n._session = MagicMock()
    return n


# ---------------------------------------------------------------------------


class TestConfigFromEnv:
    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APEXFX_TELEGRAM_TOKEN", raising=False)
        monkeypatch.delenv("APEXFX_TELEGRAM_CHAT_ID", raising=False)
        assert TelegramConfig.from_env() is None

    def test_returns_none_when_partial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APEXFX_TELEGRAM_TOKEN", "x")
        monkeypatch.delenv("APEXFX_TELEGRAM_CHAT_ID", raising=False)
        assert TelegramConfig.from_env() is None

    def test_returns_config_when_both_set(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APEXFX_TELEGRAM_TOKEN", "tok")
        monkeypatch.setenv("APEXFX_TELEGRAM_CHAT_ID", "chat")
        cfg = TelegramConfig.from_env()
        assert cfg is not None
        assert cfg.bot_token == "tok"
        assert cfg.chat_id == "chat"


# ---------------------------------------------------------------------------


class TestSend:
    def test_success_returns_true(self, notifier: TelegramNotifier) -> None:
        m = MagicMock()
        m.status_code = 200
        notifier._session.post.return_value = m
        assert notifier.send("hello") is True

    def test_http_error_returns_false(self, notifier: TelegramNotifier) -> None:
        m = MagicMock()
        m.status_code = 400
        m.text = "bad request"
        notifier._session.post.return_value = m
        assert notifier.send("hello") is False

    def test_network_error_returns_false(self, notifier: TelegramNotifier) -> None:
        notifier._session.post.side_effect = requests.ConnectionError("no net")
        assert notifier.send("hello") is False

    def test_long_message_truncated(self, notifier: TelegramNotifier) -> None:
        m = MagicMock()
        m.status_code = 200
        notifier._session.post.return_value = m
        notifier.send("x" * 10000)
        # Inspect the actual JSON sent
        sent_text = notifier._session.post.call_args.kwargs["json"]["text"]
        assert len(sent_text) <= 3500
        assert "(truncated)" in sent_text

    def test_chat_id_in_payload(self, notifier: TelegramNotifier) -> None:
        m = MagicMock()
        m.status_code = 200
        notifier._session.post.return_value = m
        notifier.send("hi")
        payload = notifier._session.post.call_args.kwargs["json"]
        assert payload["chat_id"] == "c"
        assert payload["text"] == "hi"


# ---------------------------------------------------------------------------


class TestNotificationTemplates:
    def _captured(self, notifier: TelegramNotifier) -> str:
        m = MagicMock()
        m.status_code = 200
        notifier._session.post.return_value = m
        return ""  # actual text comes from call_args after the action

    def _last_text(self, notifier: TelegramNotifier) -> str:
        return notifier._session.post.call_args.kwargs["json"]["text"]

    def test_entry_template(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_entry(
            symbol="BTC-USDT-SWAP", side="buy", unit=1,
            size=0.05, price=50000.0, sl=48500.0,
        )
        text = self._last_text(notifier)
        assert "Entry" in text
        assert "BTC-USDT-SWAP" in text
        assert "48500" in text  # SL included

    def test_entry_without_sl(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_entry(
            symbol="BTC-USDT-SWAP", side="buy", unit=1,
            size=0.05, price=50000.0, sl=None,
        )
        text = self._last_text(notifier)
        assert "SL" not in text

    def test_pyramid_template(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_pyramid(
            symbol="ETH-USDT-SWAP", side="buy",
            unit=3, size=0.10, price=3000.0,
        )
        text = self._last_text(notifier)
        assert "Pyramid #3" in text
        assert "ETH-USDT-SWAP" in text

    def test_exit_with_pnl(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_exit(
            symbol="BTC-USDT-SWAP", side="buy",
            price=51000.0, reason="donchian_exit", pnl=15.50,
        )
        text = self._last_text(notifier)
        assert "Exit" in text
        assert "donchian_exit" in text
        assert "15.50" in text

    def test_exit_with_negative_pnl(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_exit(
            symbol="BTC-USDT-SWAP", side="buy",
            price=49000.0, reason="hard_stop", pnl=-25.00,
        )
        text = self._last_text(notifier)
        assert "🔴" in text  # negative emoji

    def test_kill_switch_template(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_kill_switch("daily_loss_limit")
        text = self._last_text(notifier)
        assert "KILL SWITCH" in text
        assert "daily_loss_limit" in text

    def test_daily_summary_template(self, notifier: TelegramNotifier) -> None:
        m = MagicMock(); m.status_code = 200
        notifier._session.post.return_value = m
        notifier.notify_daily_summary(
            equity=1042.50, pnl_today=42.50, pnl_pct=0.0425, trades_today=4,
        )
        text = self._last_text(notifier)
        assert "Daily Summary" in text
        assert "1042.50" in text
        assert "4.25" in text


# ---------------------------------------------------------------------------


class TestNullNotifier:
    def test_records_calls(self) -> None:
        n = NullNotifier()
        n.send("hi")
        n.notify_kill_switch("reason")
        assert len(n.sent) == 2
        assert n.sent[0] == "hi"
        assert "kill(reason)" in n.sent[1]

    def test_returns_true(self) -> None:
        n = NullNotifier()
        assert n.send("hi") is True
        assert n.notify_entry(symbol="x", side="y", unit=1, size=1, price=1) is True
