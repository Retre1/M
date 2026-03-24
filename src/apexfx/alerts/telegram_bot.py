"""Telegram alert channel — sends alerts via Telegram Bot API."""

from __future__ import annotations

import asyncio
from typing import Any

import requests

from apexfx.alerts.alert_manager import Alert
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class TelegramAlerter:
    """Sends alerts to a Telegram chat via Bot API.

    Setup:
        1. Create bot via @BotFather -> get token
        2. Start chat with bot or add to group
        3. Get chat_id via https://api.telegram.org/bot<TOKEN>/getUpdates
        4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
    """

    API_BASE = "https://api.telegram.org/bot"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_s: float = 10.0,
        parse_mode: str = "Markdown",
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("Telegram bot_token and chat_id are required")
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout_s
        self._parse_mode = parse_mode
        # Token is NOT embedded in the URL to avoid log leakage
        self._session: requests.Session | None = None

    @property
    def name(self) -> str:
        return "telegram"

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Content-Type"] = "application/json"
        return self._session

    @property
    def _url(self) -> str:
        """Build API URL at request time — never stored as a loggable string."""
        return f"{self.API_BASE}{self._token}/sendMessage"

    async def send(self, alert: Alert) -> bool:
        """Send alert to Telegram (runs in thread pool to avoid blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._send_sync, alert)

    def _send_sync(self, alert: Alert) -> bool:
        """Synchronous send via requests."""
        text = alert.format_text()

        # Telegram message limit is 4096 chars
        if len(text) > 4000:
            text = text[:3997] + "..."

        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": self._parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            session = self._get_session()
            resp = session.post(self._url, json=payload, timeout=self._timeout)

            if resp.status_code == 200:
                logger.debug("Telegram alert sent", title=alert.title)
                return True

            logger.warning(
                "Telegram API error",
                status=resp.status_code,
                body=resp.text[:200],
            )

            # Retry once if rate limited (cap sleep to avoid blocking thread pool)
            if resp.status_code == 429:
                retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                import time

                time.sleep(min(retry_after, 5))
                resp = session.post(self._url, json=payload, timeout=self._timeout)
                return resp.status_code == 200

            return False

        except requests.RequestException as e:
            logger.error("Telegram send failed", error=str(e))
            return False

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
