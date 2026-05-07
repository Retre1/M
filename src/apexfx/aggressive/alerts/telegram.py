"""Telegram alert notifier — every event the user must know about goes here.

What gets a Telegram message
----------------------------
* Entry / pyramid / exit fills (including avg fill price + size)
* Order rejections (with reason)
* Risk-engine trips (kill switch, daily / weekly / monthly limits)
* Webhook auth failures (potential attack)
* Daily PnL summary (sent once at UTC midnight)
* Health check failures (OKX API down / can't read balance)

What does NOT get a Telegram message
------------------------------------
* Per-bar logging (would flood phone)
* Successful API health checks (only failures alert)
* Dedup duplicates (those are normal TV behavior)

Why Telegram and not email/SMS
------------------------------
* Free, no rate limits at retail volume
* Push notifications work cross-platform
* Easy to set up (BotFather → token → done)
* Group support for multi-account / multi-strategy notifications

Bot setup (5 minutes)
---------------------
1. In Telegram, message ``@BotFather`` → ``/newbot`` → choose name
2. Save the token (looks like ``1234567890:AAH...``)
3. Message your new bot anything — this creates a chat
4. Visit ``https://api.telegram.org/bot<TOKEN>/getUpdates`` —
   find your ``chat.id`` in the JSON
5. Set env vars::

       export APEXFX_TELEGRAM_TOKEN="1234567890:AAH..."
       export APEXFX_TELEGRAM_CHAT_ID="-100123..."  # negative for groups

6. Test::

       python -m apexfx.aggressive.alerts.telegram --test
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# Telegram caps message length at 4096 chars; we keep a buffer for the
# rare alert with a long stack-trace.
_MAX_MESSAGE_CHARS = 3500


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelegramConfig:
    """Bot credentials + behaviour toggles."""

    bot_token: str
    chat_id: str
    parse_mode: str = "Markdown"  # "Markdown" or "HTML" or "" for plain
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "TelegramConfig | None":
        """Build from env vars or return None if not configured.

        Returning None instead of raising lets the caller decide whether
        to require alerts (raise) or run silently (continue).  Most
        deployments run with alerts; tests run without.
        """
        token = os.environ.get("APEXFX_TELEGRAM_TOKEN", "").strip()
        chat = os.environ.get("APEXFX_TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat:
            return None
        return cls(bot_token=token, chat_id=chat)


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class TelegramNotifier:
    """Thin wrapper around Telegram's ``sendMessage`` API.

    Failures are non-fatal: if Telegram is down, we log the error and
    move on — losing an alert is preferable to crashing the trading
    bot because of a notification issue.
    """

    BASE_URL = "https://api.telegram.org"

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._url = f"{self.BASE_URL}/bot{config.bot_token}/sendMessage"

    def send(self, text: str) -> bool:
        """Send a message; return True on success.

        Long messages are truncated with a notice — Telegram returns
        400 on >4096 chars so this is a hard requirement, not a
        nice-to-have.
        """
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[: _MAX_MESSAGE_CHARS - 24] + "\n\n…(truncated)"
        payload: dict[str, Any] = {
            "chat_id": self._config.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if self._config.parse_mode:
            payload["parse_mode"] = self._config.parse_mode

        try:
            resp = self._session.post(
                self._url, json=payload, timeout=self._config.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("Telegram send failed (network)", error=str(exc))
            return False

        if resp.status_code != 200:
            logger.warning(
                "Telegram send failed (HTTP)",
                status=resp.status_code, body=resp.text[:200],
            )
            return False

        return True

    # -- Pre-formatted message templates -----------------------------------
    #
    # Keep these as methods rather than format-strings so test assertions
    # can intercept them per-event-type.

    def notify_entry(
        self, *, symbol: str, side: str, unit: int,
        size: float, price: float, sl: float | None = None,
    ) -> bool:
        sl_str = f"\nSL: `{sl}`" if sl is not None else ""
        msg = (
            f"📥 *Entry* {symbol}\n"
            f"Side: `{side}` Unit: `{unit}`\n"
            f"Size: `{size}` @ `{price}`{sl_str}"
        )
        return self.send(msg)

    def notify_pyramid(
        self, *, symbol: str, side: str, unit: int, size: float, price: float,
    ) -> bool:
        msg = (
            f"➕ *Pyramid #{unit}* {symbol}\n"
            f"Side: `{side}` Size: `{size}` @ `{price}`"
        )
        return self.send(msg)

    def notify_exit(
        self, *, symbol: str, side: str, price: float, reason: str,
        pnl: float | None = None,
    ) -> bool:
        pnl_str = ""
        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
            pnl_str = f"\nPnL: {emoji} `{sign}{pnl:.2f}`"
        msg = (
            f"📤 *Exit* {symbol}\n"
            f"Side: `{side}` @ `{price}`\n"
            f"Reason: `{reason}`{pnl_str}"
        )
        return self.send(msg)

    def notify_kill_switch(self, reason: str) -> bool:
        msg = (
            f"🛑 *KILL SWITCH ACTIVATED*\n"
            f"Reason: `{reason}`\n\n"
            f"All trading halted. Investigate before re-arming."
        )
        return self.send(msg)

    def notify_order_rejected(
        self, *, symbol: str, side: str, reason: str,
    ) -> bool:
        msg = (
            f"⚠️ *Order Rejected* {symbol}\n"
            f"Side: `{side}`\nReason: `{reason}`"
        )
        return self.send(msg)

    def notify_daily_summary(
        self, *, equity: float, pnl_today: float, pnl_pct: float,
        trades_today: int,
    ) -> bool:
        sign = "+" if pnl_today >= 0 else ""
        emoji = "🟢" if pnl_today > 0 else ("🔴" if pnl_today < 0 else "⚪")
        msg = (
            f"📊 *Daily Summary*\n"
            f"Equity: `${equity:.2f}`\n"
            f"PnL today: {emoji} `{sign}${pnl_today:.2f} ({sign}{pnl_pct:.2%})`\n"
            f"Trades: `{trades_today}`"
        )
        return self.send(msg)

    def notify_health_failure(self, *, component: str, error: str) -> bool:
        msg = (
            f"❤️‍🩹 *Health Check Failed*\n"
            f"Component: `{component}`\n"
            f"Error: `{error[:300]}`"
        )
        return self.send(msg)


# ---------------------------------------------------------------------------
# No-op notifier for tests / unconfigured deployments
# ---------------------------------------------------------------------------


@dataclass
class NullNotifier:
    """Stand-in when Telegram is not configured.

    Records calls in ``self.sent`` for tests; production code should
    not depend on this attribute.
    """

    sent: list[str] = field(default_factory=list)

    def send(self, text: str) -> bool:
        self.sent.append(text)
        return True

    def notify_entry(self, **kwargs: Any) -> bool:
        return self.send(f"entry({kwargs})")

    def notify_pyramid(self, **kwargs: Any) -> bool:
        return self.send(f"pyramid({kwargs})")

    def notify_exit(self, **kwargs: Any) -> bool:
        return self.send(f"exit({kwargs})")

    def notify_kill_switch(self, reason: str) -> bool:
        return self.send(f"kill({reason})")

    def notify_order_rejected(self, **kwargs: Any) -> bool:
        return self.send(f"rejected({kwargs})")

    def notify_daily_summary(self, **kwargs: Any) -> bool:
        return self.send(f"summary({kwargs})")

    def notify_health_failure(self, **kwargs: Any) -> bool:
        return self.send(f"health({kwargs})")
