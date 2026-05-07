"""End-to-end tests for the Flask webhook server.

We use Flask's test client (``app.test_client()``) — no real HTTP, no port
binding. Calls travel directly through the WSGI app, which means full
request/response cycle WITHOUT external infra.

Coverage:
  * Auth: valid header → 200, missing/wrong → 401
  * Validation: malformed JSON → 400, schema-invalid → 400
  * Dedup: same alert twice → 200 + 200 with status="duplicate" the 2nd time
  * Handler crash → 502 / 500 with structured error response
  * Health endpoint
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from flask.testing import FlaskClient

from apexfx.aggressive.exchanges.base import (
    ExchangeError,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from apexfx.aggressive.webhook.dedupe import DedupeCache
from apexfx.aggressive.webhook.handler import HandlerResult, SignalHandler
from apexfx.aggressive.webhook.server import WebhookConfig, create_app

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_handler_returning(success: bool, *, order_id: str = "ord-1",
                             reject_reason: str | None = None) -> MagicMock:
    """Mock SignalHandler that returns a canned HandlerResult."""
    handler = MagicMock(spec=SignalHandler)
    if success:
        order = Order(
            order_id=order_id, client_order_id="apx-x",
            symbol="BTC-USDT-SWAP", side=Side.BUY,
            order_type=OrderType.MARKET, status=OrderStatus.OPEN,
            quantity=0.05, filled_quantity=0.0, avg_fill_price=0.0,
            price=None, timestamp=datetime.now(tz=UTC),
        )
        handler.handle.return_value = HandlerResult(
            success=True, order=order, message="ok",
        )
    else:
        handler.handle.return_value = HandlerResult(
            success=False, order=None, message="rejected",
            rejection_reason=reject_reason,
        )
    return handler


@pytest.fixture
def config() -> WebhookConfig:
    return WebhookConfig(
        shared_secret="test-secret-12345",
        secret_header="X-Webhook-Secret",
        dedupe_size=100,
        dedupe_ttl_seconds=60.0,
    )


@pytest.fixture
def cache() -> DedupeCache:
    """Fresh cache per test — avoids cross-test contamination."""
    return DedupeCache(max_size=100, ttl_seconds=60.0)


@pytest.fixture
def handler() -> MagicMock:
    return _make_handler_returning(success=True)


@pytest.fixture
def client(handler: MagicMock, config: WebhookConfig, cache: DedupeCache) -> FlaskClient:
    app = create_app(handler=handler, config=config, dedupe_cache=cache)
    app.config["TESTING"] = True
    return app.test_client()


def _alert_body(**overrides) -> dict:
    base = {
        "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
        "account": "main", "unit": 1, "size": 0.05,
        "price": 50000.0, "sl": 48500.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client: FlaskClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"

    def test_unknown_path_404(self, client: FlaskClient) -> None:
        resp = client.get("/unknown")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_missing_header_401(self, client: FlaskClient) -> None:
        resp = client.post("/tv-webhook", json=_alert_body())
        assert resp.status_code == 401
        assert resp.get_json()["status"] == "rejected"

    def test_wrong_secret_401(self, client: FlaskClient) -> None:
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "wrong"},
        )
        assert resp.status_code == 401

    def test_correct_secret_passes(self, client: FlaskClient) -> None:
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_json_400(self, client: FlaskClient) -> None:
        resp = client.post(
            "/tv-webhook",
            data=b"not-json{",
            content_type="application/json",
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 400
        assert "bad json" in resp.get_json()["message"].lower()

    def test_missing_required_field_400(self, client: FlaskClient) -> None:
        body = _alert_body()
        del body["price"]  # required
        resp = client.post(
            "/tv-webhook", json=body,
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 400
        assert resp.get_json()["status"] == "rejected"

    def test_invalid_action_400(self, client: FlaskClient) -> None:
        resp = client.post(
            "/tv-webhook", json=_alert_body(action="bogus"),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_entry_alert_accepted(self, client: FlaskClient, handler: MagicMock) -> None:
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "accepted"
        assert body["order_id"] == "ord-1"
        # Handler was called once
        assert handler.handle.call_count == 1

    def test_response_includes_signal_id(self, client: FlaskClient) -> None:
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        body = resp.get_json()
        assert "signal_id" in body
        assert body["signal_id"]


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_identical_alert_deduped_2nd_time(
        self, client: FlaskClient, handler: MagicMock,
    ) -> None:
        body = _alert_body()
        headers = {"X-Webhook-Secret": "test-secret-12345"}

        r1 = client.post("/tv-webhook", json=body, headers=headers)
        r2 = client.post("/tv-webhook", json=body, headers=headers)

        assert r1.status_code == 200
        assert r1.get_json()["status"] == "accepted"
        assert r2.status_code == 200
        assert r2.get_json()["status"] == "duplicate"
        # Handler called only once — 2nd request stopped at dedup
        assert handler.handle.call_count == 1

    def test_different_alert_not_deduped(
        self, client: FlaskClient, handler: MagicMock,
    ) -> None:
        headers = {"X-Webhook-Secret": "test-secret-12345"}
        r1 = client.post("/tv-webhook", json=_alert_body(unit=1), headers=headers)
        r2 = client.post("/tv-webhook", json=_alert_body(unit=2), headers=headers)
        assert r1.get_json()["status"] == "accepted"
        assert r2.get_json()["status"] == "accepted"
        assert handler.handle.call_count == 2


# ---------------------------------------------------------------------------
# Handler errors
# ---------------------------------------------------------------------------


class TestHandlerErrors:
    def test_handler_rejects_returns_200(
        self, config: WebhookConfig, cache: DedupeCache,
    ) -> None:
        # Risk-rejected alerts use 200 + status="rejected" — TV side should
        # see "we got the signal but didn't act on it" not "deliver again"
        h = _make_handler_returning(success=False, reject_reason="daily_limit")
        app = create_app(handler=h, config=config, dedupe_cache=cache)
        client = app.test_client()
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "rejected"

    def test_handler_raises_exchange_error_returns_502(
        self, config: WebhookConfig, cache: DedupeCache,
    ) -> None:
        h = MagicMock(spec=SignalHandler)
        h.handle.side_effect = ExchangeError("OKX down")
        app = create_app(handler=h, config=config, dedupe_cache=cache)
        client = app.test_client()
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 502
        assert resp.get_json()["status"] == "error"

    def test_handler_unexpected_crash_returns_500(
        self, config: WebhookConfig, cache: DedupeCache,
    ) -> None:
        h = MagicMock(spec=SignalHandler)
        h.handle.side_effect = RuntimeError("unexpected boom")
        app = create_app(handler=h, config=config, dedupe_cache=cache)
        client = app.test_client()
        resp = client.post(
            "/tv-webhook", json=_alert_body(),
            headers={"X-Webhook-Secret": "test-secret-12345"},
        )
        assert resp.status_code == 500
        assert resp.get_json()["status"] == "error"


# ---------------------------------------------------------------------------
# WebhookConfig.from_env
# ---------------------------------------------------------------------------


class TestConfigFromEnv:
    def test_missing_secret_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APEXFX_WEBHOOK_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="APEXFX_WEBHOOK_SECRET"):
            WebhookConfig.from_env()

    def test_reads_secret_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APEXFX_WEBHOOK_SECRET", "from-env-456")
        config = WebhookConfig.from_env()
        assert config.shared_secret == "from-env-456"

    def test_reads_optional_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APEXFX_WEBHOOK_SECRET", "s")
        monkeypatch.setenv("APEXFX_DEDUP_SIZE", "500")
        monkeypatch.setenv("APEXFX_DEDUP_TTL", "120")
        config = WebhookConfig.from_env()
        assert config.dedupe_size == 500
        assert config.dedupe_ttl_seconds == 120.0
