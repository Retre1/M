"""Tests for Phase 6: Alerting system.

Covers:
- AlertManager dispatch, cooldown, deduplication
- TelegramAlerter payload construction
- WebhookAlerter (Slack, Discord, JSON payloads)
- RiskAlertMonitor event detection
- AlertConfig schema defaults
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apexfx.alerts.alert_manager import Alert, AlertLevel, AlertManager
from apexfx.alerts.risk_alerts import RiskAlertMonitor
from apexfx.alerts.telegram_bot import TelegramAlerter
from apexfx.alerts.webhook import WebhookAlerter
from apexfx.config.schema import AlertConfig


# ---------------------------------------------------------------------------
# AlertLevel
# ---------------------------------------------------------------------------

class TestAlertLevel:
    def test_ordering(self):
        assert AlertLevel.INFO < AlertLevel.WARNING
        assert AlertLevel.WARNING < AlertLevel.CRITICAL
        assert AlertLevel.CRITICAL < AlertLevel.EMERGENCY

    def test_values(self):
        assert AlertLevel.INFO == 0
        assert AlertLevel.EMERGENCY == 3


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class TestAlert:
    def test_format_text(self):
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Test Alert",
            message="Something happened",
            metadata={"symbol": "EURUSD"},
        )
        text = alert.format_text()
        assert "Test Alert" in text
        assert "Something happened" in text
        assert "EURUSD" in text

    def test_emoji_mapping(self):
        assert Alert(AlertLevel.INFO, "", "").emoji == "\u2139\ufe0f"
        assert Alert(AlertLevel.WARNING, "", "").emoji == "\u26a0\ufe0f"
        assert Alert(AlertLevel.CRITICAL, "", "").emoji == "\ud83d\udea8"
        assert Alert(AlertLevel.EMERGENCY, "", "").emoji == "\ud83c\udd98"


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestAlertManager:
    @pytest.mark.asyncio
    async def test_send_dispatches_to_channels(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        await mgr.send(AlertLevel.WARNING, "Test", "msg")

        channel.send.assert_called_once()
        assert mgr.stats["sent"] == 1

    @pytest.mark.asyncio
    async def test_suppresses_below_min_level(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.CRITICAL)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        await mgr.send(AlertLevel.WARNING, "Test", "msg")

        channel.send.assert_not_called()
        assert mgr.stats["suppressed"] == 1

    @pytest.mark.asyncio
    async def test_cooldown_deduplication(self):
        mgr = AlertManager(cooldown_s=60, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        await mgr.send(AlertLevel.WARNING, "Same Title", "msg 1")
        await mgr.send(AlertLevel.WARNING, "Same Title", "msg 2")

        # Only first should send (second is within cooldown)
        assert channel.send.call_count == 1
        assert mgr.stats["suppressed"] == 1

    @pytest.mark.asyncio
    async def test_emergency_bypasses_cooldown(self):
        mgr = AlertManager(cooldown_s=60, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        await mgr.send(AlertLevel.EMERGENCY, "Kill", "msg 1")
        await mgr.send(AlertLevel.EMERGENCY, "Kill", "msg 2")

        # Both should send (EMERGENCY bypasses cooldown)
        assert channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        await mgr.info("I", "m")
        await mgr.warning("W", "m")
        await mgr.critical("C", "m")
        await mgr.emergency("E", "m")

        assert channel.send.call_count == 4

    @pytest.mark.asyncio
    async def test_no_channels_logs_warning(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        # Should not raise even with no channels
        await mgr.send(AlertLevel.WARNING, "Test", "no channels")
        assert mgr.stats["total_alerts"] == 1


# ---------------------------------------------------------------------------
# TelegramAlerter
# ---------------------------------------------------------------------------

class TestTelegramAlerter:
    def test_init_requires_token_and_chat_id(self):
        with pytest.raises(ValueError):
            TelegramAlerter("", "123")
        with pytest.raises(ValueError):
            TelegramAlerter("token", "")

    def test_name(self):
        alerter = TelegramAlerter("token", "123")
        assert alerter.name == "telegram"

    @patch("apexfx.alerts.telegram_bot.requests.Session")
    def test_send_sync_constructs_payload(self, mock_session_cls):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        alerter = TelegramAlerter("test_token", "12345")
        alerter._session = mock_session

        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Test",
            message="Critical event",
        )
        result = alerter._send_sync(alert)

        assert result is True
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["chat_id"] == "12345"
        assert "Test" in payload["text"]


# ---------------------------------------------------------------------------
# WebhookAlerter
# ---------------------------------------------------------------------------

class TestWebhookAlerter:
    def test_init_requires_url(self):
        with pytest.raises(ValueError):
            WebhookAlerter("")

    def test_name_includes_format(self):
        assert WebhookAlerter("https://x.com", fmt="slack").name == "webhook:slack"
        assert WebhookAlerter("https://x.com", fmt="discord").name == "webhook:discord"
        assert WebhookAlerter("https://x.com", fmt="json").name == "webhook:json"

    def test_build_slack_payload(self):
        alerter = WebhookAlerter("https://x.com", fmt="slack")
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Warn",
            message="warning msg",
            metadata={"k": "v"},
        )
        payload = alerter._build_payload(alert)
        assert "attachments" in payload
        assert payload["attachments"][0]["color"] == "#ffcc00"

    def test_build_discord_payload(self):
        alerter = WebhookAlerter("https://x.com", fmt="discord")
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Crit",
            message="critical msg",
        )
        payload = alerter._build_payload(alert)
        assert "embeds" in payload
        assert payload["embeds"][0]["color"] == 0xFF6600

    def test_build_json_payload(self):
        alerter = WebhookAlerter("https://x.com", fmt="json")
        alert = Alert(
            level=AlertLevel.INFO,
            title="Info",
            message="info msg",
        )
        payload = alerter._build_payload(alert)
        assert payload["level"] == "INFO"
        assert payload["title"] == "Info"


# ---------------------------------------------------------------------------
# RiskAlertMonitor
# ---------------------------------------------------------------------------

class TestRiskAlertMonitor:
    @pytest.mark.asyncio
    async def test_detects_kill_switch_activation(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        monitor = RiskAlertMonitor(mgr)

        # Mock risk manager with active kill switch
        rm = MagicMock()
        rm.kill_switch.is_active = True
        rm.kill_switch.reason = "Test activation"
        rm.daily_loss_guard.is_triggered = False
        rm.drawdown.is_breached = False
        rm.drawdown.is_critical = False

        await monitor.check_risk_state(rm, portfolio_value=100000)

        # Should have sent emergency alert
        channel.send.assert_called_once()
        alert = channel.send.call_args[0][0]
        assert alert.level == AlertLevel.EMERGENCY

    @pytest.mark.asyncio
    async def test_does_not_repeat_kill_switch_alert(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        monitor = RiskAlertMonitor(mgr)

        rm = MagicMock()
        rm.kill_switch.is_active = True
        rm.kill_switch.reason = "Test"
        rm.daily_loss_guard.is_triggered = False
        rm.drawdown.is_breached = False
        rm.drawdown.is_critical = False

        await monitor.check_risk_state(rm)
        await monitor.check_risk_state(rm)  # Second call

        # Only one alert (not repeated)
        assert channel.send.call_count == 1

    @pytest.mark.asyncio
    async def test_detects_daily_loss(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        monitor = RiskAlertMonitor(mgr)

        rm = MagicMock()
        rm.kill_switch.is_active = False
        rm.daily_loss_guard.is_triggered = True
        rm.daily_loss_guard.daily_loss_pct = 0.025
        rm.drawdown.is_breached = False
        rm.drawdown.is_critical = False

        await monitor.check_risk_state(rm)

        channel.send.assert_called_once()
        alert = channel.send.call_args[0][0]
        assert alert.level == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_mt5_disconnect_reconnect(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        monitor = RiskAlertMonitor(mgr)

        await monitor.on_mt5_disconnect()
        await monitor.on_mt5_reconnect()

        assert channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_urgent_news_alert(self):
        mgr = AlertManager(cooldown_s=0, min_level=AlertLevel.INFO)
        channel = AsyncMock()
        channel.name = "test"
        channel.send = AsyncMock(return_value=True)
        mgr.add_channel(channel)

        monitor = RiskAlertMonitor(mgr)
        await monitor.on_urgent_news("BREAKING: Flash crash", "finnhub")

        channel.send.assert_called_once()


# ---------------------------------------------------------------------------
# AlertConfig
# ---------------------------------------------------------------------------

class TestAlertConfig:
    def test_defaults(self):
        cfg = AlertConfig()
        assert cfg.enabled is False
        assert cfg.min_level == "WARNING"
        assert cfg.telegram_enabled is False
        assert cfg.webhook_enabled is False

    def test_custom(self):
        cfg = AlertConfig(
            enabled=True,
            telegram_enabled=True,
            telegram_bot_token="abc",
            telegram_chat_id="123",
        )
        assert cfg.enabled is True
        assert cfg.telegram_bot_token == "abc"
