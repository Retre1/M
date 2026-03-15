"""Webhook alert channel — sends alerts via HTTP POST (Slack, Discord, PagerDuty, etc.)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import requests

from apexfx.alerts.alert_manager import Alert, AlertLevel
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class WebhookAlerter:
    """Sends alerts to a webhook URL via HTTP POST.

    Supports Slack-compatible JSON payloads and generic JSON.

    Usage:
        # Slack
        alerter = WebhookAlerter(url="https://hooks.slack.com/...", format="slack")

        # Discord
        alerter = WebhookAlerter(url="https://discord.com/api/webhooks/...", format="discord")

        # Generic webhook
        alerter = WebhookAlerter(url="https://example.com/alerts", format="json")
    """

    def __init__(
        self,
        url: str,
        fmt: str = "json",
        timeout_s: float = 10.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not url:
            raise ValueError("Webhook URL is required")
        self._url = url
        self._fmt = fmt
        self._timeout = timeout_s
        self._headers = headers or {}
        self._session: requests.Session | None = None

    @property
    def name(self) -> str:
        return f"webhook:{self._fmt}"

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
            self._session.headers.update(self._headers)
        return self._session

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        if self._fmt == "slack":
            color = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ffcc00",
                AlertLevel.CRITICAL: "#ff6600",
                AlertLevel.EMERGENCY: "#ff0000",
            }.get(alert.level, "#808080")

            fields = [
                {"title": k, "value": str(v), "short": True}
                for k, v in alert.metadata.items()
            ]

            return {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.emoji} {alert.title}",
                        "text": alert.message,
                        "fields": fields,
                        "ts": int(alert.timestamp),
                    }
                ]
            }

        if self._fmt == "discord":
            color_int = {
                AlertLevel.INFO: 0x36A64F,
                AlertLevel.WARNING: 0xFFCC00,
                AlertLevel.CRITICAL: 0xFF6600,
                AlertLevel.EMERGENCY: 0xFF0000,
            }.get(alert.level, 0x808080)

            fields = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in alert.metadata.items()
            ]

            return {
                "embeds": [
                    {
                        "title": f"{alert.emoji} {alert.title}",
                        "description": alert.message,
                        "color": color_int,
                        "fields": fields,
                        "timestamp": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(alert.timestamp)
                        ),
                    }
                ]
            }

        # Generic JSON
        return {
            "level": alert.level.name,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "metadata": alert.metadata,
        }

    async def send(self, alert: Alert) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_sync, alert)

    def _send_sync(self, alert: Alert) -> bool:
        payload = self._build_payload(alert)

        try:
            session = self._get_session()
            resp = session.post(
                self._url,
                data=json.dumps(payload),
                timeout=self._timeout,
            )

            if resp.status_code in (200, 201, 204):
                logger.debug("Webhook alert sent", title=alert.title, fmt=self._fmt)
                return True

            logger.warning(
                "Webhook error",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return False

        except requests.RequestException as e:
            logger.error("Webhook send failed", error=str(e))
            return False

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
