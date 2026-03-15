"""Unified alert manager — dispatches alerts to all configured channels."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(IntEnum):
    """Alert severity levels."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3  # Kill switch, equity floor breach


@dataclass
class Alert:
    """Single alert event."""

    level: AlertLevel
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        return {
            AlertLevel.INFO: "\u2139\ufe0f",
            AlertLevel.WARNING: "\u26a0\ufe0f",
            AlertLevel.CRITICAL: "\ud83d\udea8",
            AlertLevel.EMERGENCY: "\ud83c\udd98",
        }.get(self.level, "\u2753")

    def format_text(self) -> str:
        lines = [f"{self.emoji} *{self.title}*", "", self.message]
        if self.metadata:
            lines.append("")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: `{v}`")
        return "\n".join(lines)


class AlertChannel(Protocol):
    """Protocol for alert delivery channels."""

    async def send(self, alert: Alert) -> bool: ...

    @property
    def name(self) -> str: ...


class AlertManager:
    """Central alert dispatcher.

    Deduplicates similar alerts within a cooldown window and dispatches
    to all registered channels (Telegram, Webhook, etc.).
    """

    def __init__(
        self,
        cooldown_s: float = 60.0,
        min_level: AlertLevel = AlertLevel.WARNING,
    ) -> None:
        self._channels: list[AlertChannel] = []
        self._cooldown_s = cooldown_s
        self._min_level = min_level
        self._last_sent: dict[str, float] = {}
        self._stats = {
            "total_alerts": 0,
            "sent": 0,
            "suppressed": 0,
            "failed": 0,
        }

    def add_channel(self, channel: AlertChannel) -> None:
        self._channels.append(channel)
        logger.info("Alert channel registered", channel=channel.name)

    async def send(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        **metadata: str | float | int,
    ) -> None:
        """Send an alert to all channels."""
        self._stats["total_alerts"] += 1

        if level < self._min_level:
            self._stats["suppressed"] += 1
            return

        # Dedup by title
        dedup_key = f"{level}:{title}"
        now = time.time()
        last = self._last_sent.get(dedup_key, 0)

        # Emergency always goes through; others respect cooldown
        if level < AlertLevel.EMERGENCY and (now - last) < self._cooldown_s:
            self._stats["suppressed"] += 1
            logger.debug("Alert suppressed (cooldown)", title=title)
            return

        self._last_sent[dedup_key] = now

        alert = Alert(
            level=level,
            title=title,
            message=message,
            timestamp=now,
            metadata=dict(metadata),
        )

        if not self._channels:
            logger.warning("No alert channels configured", title=title)
            return

        results = await asyncio.gather(
            *(ch.send(alert) for ch in self._channels),
            return_exceptions=True,
        )

        for ch, result in zip(self._channels, results):
            if isinstance(result, Exception):
                logger.error(
                    "Alert channel failed",
                    channel=ch.name,
                    error=str(result),
                )
                self._stats["failed"] += 1
            elif result:
                self._stats["sent"] += 1
            else:
                self._stats["failed"] += 1

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # Convenience methods
    async def info(self, title: str, message: str, **kw: str | float | int) -> None:
        await self.send(AlertLevel.INFO, title, message, **kw)

    async def warning(self, title: str, message: str, **kw: str | float | int) -> None:
        await self.send(AlertLevel.WARNING, title, message, **kw)

    async def critical(self, title: str, message: str, **kw: str | float | int) -> None:
        await self.send(AlertLevel.CRITICAL, title, message, **kw)

    async def emergency(self, title: str, message: str, **kw: str | float | int) -> None:
        await self.send(AlertLevel.EMERGENCY, title, message, **kw)
