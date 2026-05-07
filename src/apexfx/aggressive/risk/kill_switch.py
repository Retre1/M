"""File-based + automatic kill switch.

Why file-based
--------------
On-disk file beats environment-variable / database flag for one reason:
**it survives process crashes and you can flip it from any shell.** If
the bot is misbehaving and you need to stop NOW, ``touch .kill_switch``
in any terminal is faster than logging into a dashboard and clicking
a button — and it works even if the dashboard is down.

The bot reads the file every signal — so the kill takes effect on the
**next** alert, with at most one bar of lag.  At 4H bars that's 4
hours; for safety-critical kills (margin call risk) we also auto-create
the file on a series of triggers below.

Auto-trigger conditions
-----------------------
The kill switch is also created automatically by the circuit breaker
when:

  * Daily loss > 2% of equity (default — configurable)
  * Weekly loss > 5%
  * Three consecutive failed orders (exchange rejecting our requests)
  * Position desync detected (our state ≠ exchange state)

When auto-triggered, the file contains the trigger reason as text,
so post-mortem can read it without tailing logs.

Re-arming
---------
Manual: delete the file and a 24-hour cooldown timer (also a file)
prevents immediate re-trade.  This is anti-tilt: stops you from
"unkilling" right after a series of losses just because you're
emotional.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class KillSwitchState:
    """Current state of the kill switch — what callers see when checking."""

    active: bool
    reason: str = ""
    triggered_at: float = 0.0  # Unix epoch seconds, 0 if not active
    cooldown_remaining_s: float = 0.0  # 0 if not in cooldown


class KillSwitch:
    """File-based kill switch with optional cooldown after re-arming.

    Two on-disk files:

      * ``flag_path``: presence ⇒ kill is active.  Contents = trigger reason.
      * ``cooldown_path``: presence + ts inside ⇒ in cooldown after re-arming.

    Single-process safe.  Multi-process unsafe (multiple bots writing
    the same flag) — but for retail single-VPS deployment this is fine.
    """

    def __init__(
        self,
        flag_path: str | Path = ".kill_switch",
        cooldown_path: str | Path = ".kill_switch_cooldown",
        cooldown_seconds: float = 86400.0,  # 24h
    ) -> None:
        self._flag = Path(flag_path)
        self._cooldown_file = Path(cooldown_path)
        self._cooldown_seconds = cooldown_seconds

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True if either kill flag exists OR we're in re-arm cooldown."""
        return self._flag.exists() or self._cooldown_remaining() > 0

    def state(self) -> KillSwitchState:
        """Detailed state — for logs / dashboards / responses."""
        if self._flag.exists():
            try:
                reason = self._flag.read_text(encoding="utf-8").strip()
            except OSError:
                reason = "(unreadable)"
            try:
                triggered_at = self._flag.stat().st_mtime
            except OSError:
                triggered_at = 0.0
            return KillSwitchState(
                active=True, reason=reason or "manual",
                triggered_at=triggered_at,
            )

        cooldown = self._cooldown_remaining()
        if cooldown > 0:
            return KillSwitchState(
                active=True, reason="re-arm cooldown",
                cooldown_remaining_s=cooldown,
            )

        return KillSwitchState(active=False)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def trigger(self, reason: str) -> None:
        """Activate the kill — auto-trigger path used by circuit breaker.

        Idempotent: if already triggered, does NOT overwrite the original
        reason (first-trigger wins for post-mortem clarity).
        """
        if self._flag.exists():
            logger.info("Kill switch already active, not overwriting",
                        existing_reason=self._read_safe(self._flag))
            return
        self._flag.write_text(reason, encoding="utf-8")
        logger.error("Kill switch TRIGGERED", reason=reason)

    def disarm(self) -> None:
        """Remove the kill flag and start the cooldown timer.

        Use this when you've investigated and decided the bot is safe to
        resume.  The cooldown enforces a wait so that "kill → realize it
        was nothing → unkill in 30s → next bar fires immediately" loop
        can't happen — gives you time to think.
        """
        if not self._flag.exists():
            logger.info("disarm() called but flag not present — no-op")
            return
        try:
            self._flag.unlink()
        except FileNotFoundError:
            pass  # race with another process — ok
        # Start cooldown timer
        self._cooldown_file.write_text(
            f"{time.time() + self._cooldown_seconds:.1f}",
            encoding="utf-8",
        )
        logger.info("Kill switch disarmed; cooldown started",
                    cooldown_seconds=self._cooldown_seconds)

    def force_clear(self) -> None:
        """Remove BOTH flag and cooldown — emergency override.

        Use only when you're sure (e.g. cooldown is preventing testing).
        The fact that this exists is a footgun on purpose: if you reach
        for it, you should have an explicit reason logged.
        """
        for p in (self._flag, self._cooldown_file):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        logger.warning("Kill switch force_clear() called — both files removed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cooldown_remaining(self) -> float:
        """Seconds remaining in cooldown, or 0 if not in cooldown."""
        if not self._cooldown_file.exists():
            return 0.0
        try:
            until = float(self._cooldown_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            # Corrupt cooldown file — treat as expired and clean up
            self._cooldown_file.unlink(missing_ok=True)
            return 0.0
        remaining = until - time.time()
        if remaining <= 0:
            # Expired — clean up so subsequent checks are fast
            self._cooldown_file.unlink(missing_ok=True)
            return 0.0
        return remaining

    @staticmethod
    def _read_safe(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return "(unreadable)"
