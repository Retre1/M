"""Flask webhook server for TradingView → OKX integration.

Endpoints
---------
* ``POST /tv-webhook`` — main alert receiver
* ``GET  /health``      — liveness probe (returns 200 + version info)

Why Flask and not FastAPI
-------------------------
Flask is already a transitive dep (via ``dash`` for the dashboard).
Adding FastAPI/Starlette would double the install size for ~10 LOC of
async we don't actually need at retail volume (1-50 alerts/day).
Flask + gunicorn covers this perfectly.

Production deployment
---------------------
Run with::

    gunicorn -w 1 -b 0.0.0.0:8080 \\
        --timeout 30 --keep-alive 5 --log-level info \\
        'apexfx.aggressive.webhook.server:create_app()'

Behind a reverse proxy with TLS (caddy/nginx). TradingView only sends
HTTPS POSTs, so plain HTTP is a no-go.

Single worker
-------------
Use ``--workers 1``.  Multiple workers would each have their own dedup
cache, breaking idempotency.  At retail throughput (one bar = 4h on
H4), a single worker handles 1000× our load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request
from pydantic import ValidationError

from apexfx.aggressive.exchanges.base import Exchange, ExchangeError
from apexfx.aggressive.webhook.auth import verify_shared_secret
from apexfx.aggressive.webhook.dedupe import DedupeCache
from apexfx.aggressive.webhook.handler import HandlerResult, SignalHandler
from apexfx.aggressive.webhook.models import (
    SignalId,
    TradingViewAlert,
    WebhookResponse,
)
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebhookConfig:
    """Server configuration — populated from environment in production.

    All secrets via env vars; **never** hardcode in this module or commit
    to git.  ``.env.example`` documents the required vars.
    """

    shared_secret: str
    secret_header: str = "X-Webhook-Secret"
    dedupe_size: int = 1000
    dedupe_ttl_seconds: float = 300.0
    enable_health_endpoint: bool = True

    @classmethod
    def from_env(cls) -> "WebhookConfig":
        secret = os.environ.get("APEXFX_WEBHOOK_SECRET", "")
        if not secret:
            raise RuntimeError(
                "APEXFX_WEBHOOK_SECRET env var required.  Set it before starting "
                "the server: this is the value TradingView puts in the "
                "X-Webhook-Secret header."
            )
        return cls(
            shared_secret=secret,
            secret_header=os.environ.get("APEXFX_WEBHOOK_HEADER", "X-Webhook-Secret"),
            dedupe_size=int(os.environ.get("APEXFX_DEDUP_SIZE", "1000")),
            dedupe_ttl_seconds=float(os.environ.get("APEXFX_DEDUP_TTL", "300")),
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    handler: SignalHandler,
    config: WebhookConfig,
    dedupe_cache: DedupeCache | None = None,
) -> Flask:
    """Build the Flask app.

    Factory pattern keeps the app testable: each test creates its own
    instance with mock handler / fresh dedup cache.
    """
    app = Flask(__name__)
    cache = dedupe_cache or DedupeCache(
        max_size=config.dedupe_size,
        ttl_seconds=config.dedupe_ttl_seconds,
    )

    @app.route("/tv-webhook", methods=["POST"])
    def tv_webhook() -> tuple[Response, int]:
        return _handle_webhook(request, handler, config, cache)

    if config.enable_health_endpoint:
        @app.route("/health", methods=["GET"])
        def health() -> tuple[Response, int]:
            return jsonify({
                "status": "ok",
                "service": "apexfx-webhook",
                "time": datetime.now(tz=UTC).isoformat(),
            }), 200

    @app.errorhandler(404)
    def not_found(_e):  # type: ignore[no-untyped-def]
        return jsonify({"status": "error", "message": "not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):  # type: ignore[no-untyped-def]
        return jsonify({"status": "error", "message": "method not allowed"}), 405

    return app


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


def _handle_webhook(
    req,  # flask.Request — annotated as Any to avoid type-import dance
    handler: SignalHandler,
    config: WebhookConfig,
    cache: DedupeCache,
) -> tuple[Response, int]:
    """End-to-end processing of one webhook POST.

    Steps (in order, fail-fast):
      1. Auth (shared-secret header)
      2. Parse JSON
      3. Validate via pydantic
      4. Dedupe by SignalId
      5. Hand off to SignalHandler
      6. Build WebhookResponse with status + order_id
    """
    body = req.get_data(as_text=False)

    # 1. Auth
    provided = req.headers.get(config.secret_header, "")
    if not verify_shared_secret(config.shared_secret, provided):
        logger.warning(
            "Webhook auth failed",
            ip=req.remote_addr,
            ua=req.headers.get("User-Agent", "")[:80],
        )
        # Don't reveal which header is missing — opaque 401
        return _resp(WebhookResponse(status="rejected", message="unauthorized"), 401)

    # 2. Parse JSON (raw body, not request.get_json — that would re-read body)
    try:
        import json
        payload = json.loads(body.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Webhook bad JSON", error=str(exc))
        return _resp(WebhookResponse(status="rejected", message=f"bad json: {exc}"), 400)

    # 3. Validate
    try:
        alert = TradingViewAlert.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Webhook validation failed",
                       errors=exc.errors(), payload=payload)
        return _resp(WebhookResponse(
            status="rejected",
            message=f"validation: {exc.errors()[:3]}",  # don't leak full structure
        ), 400)

    # 4. Dedupe
    signal_id = SignalId.from_alert(alert, _now_seconds())
    key = signal_id.to_key()
    if not cache.add_if_new(key):
        logger.info("Duplicate alert dropped", signal_id=key)
        return _resp(WebhookResponse(
            status="duplicate", signal_id=key,
            message="alert already processed within dedup window",
        ), 200)

    # 5. Hand off to handler
    try:
        result: HandlerResult = handler.handle(alert)
    except ExchangeError as exc:
        logger.error("Handler raised ExchangeError", error=str(exc))
        return _resp(WebhookResponse(
            status="error", signal_id=key,
            message=f"exchange error: {exc}",
        ), 502)
    except Exception as exc:  # last-resort guard
        logger.exception("Handler crashed", error=str(exc))
        return _resp(WebhookResponse(
            status="error", signal_id=key,
            message="internal error",
        ), 500)

    # 6. Result
    if result.success and result.order is not None:
        return _resp(WebhookResponse(
            status="accepted", signal_id=key,
            order_id=result.order.order_id, message=result.message,
        ), 200)

    # Risk-rejected or business-logic-rejected
    return _resp(WebhookResponse(
        status="rejected", signal_id=key, message=result.message,
    ), 200)  # 200 because the alert was *received* OK; rejection is policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(payload: WebhookResponse, status: int) -> tuple[Response, int]:
    """Standard JSON response envelope."""
    body = jsonify(payload.model_dump(exclude_none=True))
    return body, status


def _now_seconds() -> int:
    """Unix epoch seconds.  Wrapped for testability (monkeypatch in tests)."""
    return int(datetime.now(tz=UTC).timestamp())


# ---------------------------------------------------------------------------
# Production entry-point — used by systemd / gunicorn
# ---------------------------------------------------------------------------


def create_app_from_env() -> Flask:
    """Production app factory: reads everything from environment vars.

    Wires together:
      * OkxClient with creds from APEXFX_OKX_*
      * SignalHandler with risk gating from CircuitBreaker + KillSwitch
      * WebhookConfig from APEXFX_WEBHOOK_*

    This is what systemd / gunicorn invokes::

        gunicorn -w 1 -b 127.0.0.1:8080 \\
            'apexfx.aggressive.webhook.server:create_app_from_env()'

    Required env vars:
      * APEXFX_WEBHOOK_SECRET   — shared secret with TV
      * APEXFX_OKX_API_KEY      — OKX API key
      * APEXFX_OKX_API_SECRET   — OKX API secret
      * APEXFX_OKX_API_PASSPHRASE — OKX passphrase

    Optional:
      * APEXFX_OKX_DEMO=true (default true — paper mode!)
      * APEXFX_TELEGRAM_TOKEN, APEXFX_TELEGRAM_CHAT_ID — for alerts
      * APEXFX_BREAKER_DAILY_PCT, APEXFX_BREAKER_WEEKLY_PCT — risk thresholds

    Demo defaults to TRUE so a misconfigured deployment doesn't accidentally
    place real orders.  Flip to false explicitly when ready for live.
    """
    from apexfx.aggressive.alerts.telegram import (
        NullNotifier, TelegramConfig, TelegramNotifier,
    )
    from apexfx.aggressive.exchanges.okx_client import OkxClient
    from apexfx.aggressive.risk.circuit_breaker import (
        CircuitBreaker, CircuitBreakerConfig,
    )
    from apexfx.aggressive.risk.kill_switch import KillSwitch
    from apexfx.aggressive.webhook.handler import SignalHandler

    cfg = WebhookConfig.from_env()

    # OKX client — demo by default
    demo = os.environ.get("APEXFX_OKX_DEMO", "true").lower() != "false"
    okx = OkxClient(
        api_key=os.environ["APEXFX_OKX_API_KEY"],
        api_secret=os.environ["APEXFX_OKX_API_SECRET"],
        api_passphrase=os.environ["APEXFX_OKX_API_PASSPHRASE"],
        demo=demo,
    )
    logger.info("OKX client initialized", demo_mode=demo)

    # Risk engine — kill switch is checked on every signal
    kill = KillSwitch()
    breaker_cfg = CircuitBreakerConfig(
        daily_loss_pct=float(os.environ.get("APEXFX_BREAKER_DAILY_PCT", "0.08")),
        weekly_loss_pct=float(os.environ.get("APEXFX_BREAKER_WEEKLY_PCT", "0.20")),
        monthly_dd_pct=float(os.environ.get("APEXFX_BREAKER_MONTHLY_PCT", "0.35")),
    )
    breaker = CircuitBreaker(config=breaker_cfg, kill_switch=kill)

    # Telegram (optional)
    tg_cfg = TelegramConfig.from_env()
    notifier = TelegramNotifier(tg_cfg) if tg_cfg else NullNotifier()
    logger.info("Telegram alerts", enabled=tg_cfg is not None)

    # Risk gate function — closes over kill/breaker
    def risk_check(_alert, exchange):  # type: ignore[no-untyped-def]
        if kill.is_active():
            return f"kill_switch: {kill.state().reason}"
        try:
            balance = exchange.get_balance("USDT")
            trip = breaker.observe_equity(balance.equity)
            if trip.tripped:
                if hasattr(notifier, "notify_kill_switch"):
                    notifier.notify_kill_switch(trip.reason)
                return f"breaker: {trip.reason}"
        except Exception as exc:
            logger.warning("Risk check could not read balance", error=str(exc))
            # Don't block on transient balance read failure
            return None
        return None

    handler = SignalHandler(exchange=okx, risk_check_fn=risk_check)
    return create_app(handler=handler, config=cfg)
