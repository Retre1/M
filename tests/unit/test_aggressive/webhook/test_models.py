"""Tests for webhook pydantic models — the contract layer with TradingView."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from apexfx.aggressive.webhook.models import (
    AlertAction,
    AlertSide,
    ExitReason,
    SignalId,
    TradingViewAlert,
    WebhookResponse,
)


class TestTradingViewAlertParsing:
    def test_minimal_entry_alert(self) -> None:
        a = TradingViewAlert.model_validate({
            "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
            "account": "main", "price": 50000.0,
        })
        assert a.action is AlertAction.ENTRY
        assert a.side is AlertSide.LONG
        assert a.symbol == "BTCUSDT.P"

    def test_full_entry_alert(self) -> None:
        a = TradingViewAlert.model_validate({
            "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
            "account": "main", "unit": 1, "size": 0.05,
            "price": 50000.0, "sl": 48500.0,
        })
        assert a.unit == 1
        assert a.size == 0.05
        assert a.sl == 48500.0

    def test_pyramid_alert(self) -> None:
        a = TradingViewAlert.model_validate({
            "action": "pyramid", "symbol": "BTCUSDT.P", "side": "long",
            "account": "main", "unit": 2, "size": 0.05, "price": 51000.0,
        })
        assert a.action is AlertAction.PYRAMID
        assert a.is_entry_or_pyramid()

    def test_exit_alert(self) -> None:
        a = TradingViewAlert.model_validate({
            "action": "exit", "symbol": "BTCUSDT.P", "side": "long",
            "account": "main", "price": 49000.0, "reason": "donchian_exit",
        })
        assert a.is_exit()
        assert a.reason is ExitReason.DONCHIAN_EXIT

    def test_symbol_strips_exchange_prefix(self) -> None:
        a = TradingViewAlert.model_validate({
            "action": "entry", "symbol": "OKX:BTCUSDT.P", "side": "long",
            "account": "main", "price": 50000.0,
        })
        assert a.symbol == "BTCUSDT.P"  # OKX: prefix stripped

    def test_symbol_uppercased(self) -> None:
        a = TradingViewAlert.model_validate({
            "action": "entry", "symbol": "btcusdt.p", "side": "long",
            "account": "main", "price": 50000.0,
        })
        assert a.symbol == "BTCUSDT.P"


class TestTradingViewAlertValidation:
    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "swing_now", "symbol": "BTCUSDT.P", "side": "long",
                "account": "main", "price": 50000.0,
            })

    def test_invalid_side_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "entry", "symbol": "BTCUSDT.P", "side": "north",
                "account": "main", "price": 50000.0,
            })

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
                "account": "main", "price": -50000.0,
            })

    def test_zero_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
                "account": "main", "price": 50000.0, "size": 0.0,
            })

    def test_unit_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
                "account": "main", "price": 50000.0, "unit": 999,
            })

    def test_extra_fields_rejected(self) -> None:
        # extra="forbid" guards against typos becoming silent bugs
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
                "account": "main", "price": 50000.0,
                "actoin_typo": "boom",
            })

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TradingViewAlert.model_validate({
                "action": "entry", "symbol": "", "side": "long",
                "account": "main", "price": 50000.0,
            })


class TestSignalId:
    def _make_alert(self, **overrides) -> TradingViewAlert:
        base = {
            "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
            "account": "main", "unit": 1, "size": 0.05,
            "price": 50000.0,
        }
        base.update(overrides)
        return TradingViewAlert.model_validate(base)

    def test_same_alert_same_minute_same_key(self) -> None:
        a = self._make_alert()
        # Both timestamps fall in the same 60-second bucket
        # (1700000040 // 60 == 1700000099 // 60 == 28333334)
        s1 = SignalId.from_alert(a, 1700000040)
        s2 = SignalId.from_alert(a, 1700000099)
        assert s1.to_key() == s2.to_key()

    def test_different_minute_different_key(self) -> None:
        a = self._make_alert()
        s1 = SignalId.from_alert(a, 1700000040)
        s2 = SignalId.from_alert(a, 1700000100)  # next 60s bucket
        assert s1.to_key() != s2.to_key()

    def test_different_unit_different_key(self) -> None:
        a1 = self._make_alert(unit=1)
        a2 = self._make_alert(unit=2)
        assert SignalId.from_alert(a1, 0).to_key() != SignalId.from_alert(a2, 0).to_key()

    def test_different_account_different_key(self) -> None:
        a1 = self._make_alert(account="acct1")
        a2 = self._make_alert(account="acct2")
        assert SignalId.from_alert(a1, 0).to_key() != SignalId.from_alert(a2, 0).to_key()

    def test_price_micro_diff_same_key(self) -> None:
        # Prices that match to 6 significant digits → same bucket
        a1 = self._make_alert(price=50000.0)
        a2 = self._make_alert(price=50000.001)  # 7th-digit diff, same .6g
        assert SignalId.from_alert(a1, 0).to_key() == SignalId.from_alert(a2, 0).to_key()

    def test_price_meaningful_diff_different_key(self) -> None:
        a1 = self._make_alert(price=50000.0)
        a2 = self._make_alert(price=50001.0)  # 6th digit differs
        assert SignalId.from_alert(a1, 0).to_key() != SignalId.from_alert(a2, 0).to_key()

    def test_price_handles_low_priced_assets(self) -> None:
        # SHIB-style prices need to bucket sensibly too
        a1 = self._make_alert(price=0.00003123)
        a2 = self._make_alert(price=0.00003123)
        a3 = self._make_alert(price=0.00003456)
        assert SignalId.from_alert(a1, 0).to_key() == SignalId.from_alert(a2, 0).to_key()
        assert SignalId.from_alert(a1, 0).to_key() != SignalId.from_alert(a3, 0).to_key()


class TestWebhookResponse:
    def test_serializes_with_optional_fields(self) -> None:
        r = WebhookResponse(
            status="accepted", signal_id="x", order_id="ord-1",
            message="ok",
        )
        d = r.model_dump(exclude_none=True)
        assert "order_id" in d
        assert d["status"] == "accepted"

    def test_serializes_omits_none(self) -> None:
        r = WebhookResponse(status="rejected", message="bad")
        d = r.model_dump(exclude_none=True)
        assert "order_id" not in d
        assert "signal_id" not in d
