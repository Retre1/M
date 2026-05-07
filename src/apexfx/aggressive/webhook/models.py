"""Pydantic models for TradingView webhook alerts.

Why pydantic
------------
TradingView sends arbitrary JSON to our endpoint. Without strict validation
we'd be one ``KeyError`` away from executing a malformed signal. Pydantic
gives us:

  • Field-level validation (types, ranges, enum membership) at parse time
  • Clean error messages when alerts are wrong (logged, never executed)
  • Stable contract between Pine Script alert templates and Python handler

Schema source of truth
----------------------
The JSON shape here MUST match what ``donchian_turtle.pine`` emits in its
``alert_message`` strings.  If you change one, change the other.  The
integration test ``test_pine_alert_format_matches_models`` (in
``test_webhook/test_models.py``) loads a sample alert from disk and parses
it against these models — that's the canary.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertAction(str, Enum):
    """What the strategy wants the bot to do.

    * ``entry``   — open a fresh position (first unit of a Turtle stack)
    * ``pyramid`` — add another unit to an existing position
    * ``exit``    — close all units (Donchian opposite breakout OR hard stop)
    """

    ENTRY = "entry"
    PYRAMID = "pyramid"
    EXIT = "exit"


class AlertSide(str, Enum):
    """Direction of the trade — Pine emits ``long`` / ``short`` strings."""

    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    """Why an exit fired — for logging / post-trade analysis."""

    DONCHIAN_EXIT = "donchian_exit"
    HARD_STOP = "hard_stop"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Alert payload
# ---------------------------------------------------------------------------


class TradingViewAlert(BaseModel):
    """One alert from a TradingView Pine strategy.

    All fields except ``action`` and ``symbol`` are validated by enum or
    range checks — pydantic raises ``ValidationError`` on bad input which
    the webhook server converts into a 400 response (never silently
    accepting garbage).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: AlertAction
    symbol: str = Field(..., min_length=3, max_length=32)
    side: AlertSide
    account: str = Field(..., min_length=1, max_length=64)

    # ----- Entry / pyramid only -----
    unit: int | None = Field(
        default=None, ge=1, le=8,
        description="Which unit in the pyramid stack (1 = first, ≤4 typically).",
    )
    size: float | None = Field(
        default=None, gt=0,
        description="Order quantity in contracts.  Computed by Pine sizing.",
    )
    price: float = Field(..., gt=0, description="Bar-close price at signal.")
    sl: float | None = Field(
        default=None, gt=0,
        description="Stop-loss price for the entry (only on first unit).",
    )

    # ----- Exit only -----
    reason: ExitReason | None = Field(
        default=None,
        description="Required when action == 'exit'; ignored otherwise.",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """TradingView format is e.g. ``OKX:BTCUSDT.P`` — strip exchange prefix
        so downstream code only deals with bare ``BTCUSDT.P`` / ``BTC-USDT-SWAP``."""
        if ":" in v:
            v = v.split(":", 1)[1]
        return v.strip().upper()

    def is_entry_or_pyramid(self) -> bool:
        return self.action in (AlertAction.ENTRY, AlertAction.PYRAMID)

    def is_exit(self) -> bool:
        return self.action is AlertAction.EXIT


# ---------------------------------------------------------------------------
# Signal ID — for idempotency
# ---------------------------------------------------------------------------


class SignalId(BaseModel):
    """Composite key identifying a unique alert.

    TradingView can re-fire the same alert if the user clicks "Trigger now"
    or in the rare case of TV-side retry.  We dedupe on the tuple
    (account, symbol, action, unit, price-bucket, time-bucket).  Identical
    tuples within the dedup window are dropped.

    Bucketing strategy
    ------------------
    * **Price**: 6 significant digits via ``format(price, ".6g")``.  Works
      for BTC at $50,000 (rounds to $0.10) AND SHIB at $0.00003 (rounds
      to $3e-9) without manual scaling.
    * **Time**: integer minute via ``timestamp // 60``.  Two alerts within
      the same wall-clock minute share a bucket — fine for our 4H bars
      because legitimate alerts are spaced at 4-hour boundaries.

    Note: TV doesn't send bar timestamps in alert_message, but we
    reconstruct one server-side from the request arrival time — see
    ``server.py``.
    """

    account: str
    symbol: str
    action: AlertAction
    unit: int | None
    price_bucket: str  # ".6g"-formatted price — same string ⇒ same bucket
    time_bucket: int   # Unix minute (timestamp // 60)

    def to_key(self) -> str:
        unit_part = str(self.unit) if self.unit is not None else "x"
        return (
            f"{self.account}|{self.symbol}|{self.action.value}|"
            f"{unit_part}|{self.price_bucket}|{self.time_bucket}"
        )

    @classmethod
    def from_alert(cls, alert: TradingViewAlert, timestamp_seconds: int) -> "SignalId":
        return cls(
            account=alert.account,
            symbol=alert.symbol,
            action=alert.action,
            unit=alert.unit,
            price_bucket=format(alert.price, ".6g"),
            time_bucket=timestamp_seconds // 60,
        )


# ---------------------------------------------------------------------------
# Server response models
# ---------------------------------------------------------------------------


class WebhookResponse(BaseModel):
    """JSON returned to TradingView after processing.

    TV doesn't read the body but logs the HTTP status and response.  We
    return structured info so a curl/test client can reason about what
    happened.
    """

    status: Literal["accepted", "duplicate", "rejected", "error"]
    signal_id: str | None = None
    message: str = ""
    order_id: str | None = None  # OKX order ID after successful execution
