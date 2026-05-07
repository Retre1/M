"""Tests for the position sizer — sanity check + Pine Script parity."""

from __future__ import annotations

import pytest

from apexfx.aggressive.exchanges.base import SymbolInfo
from apexfx.aggressive.risk.position_sizer import (
    SizingConfig,
    expected_size,
    size_in_contracts,
    verify_pine_size,
)


@pytest.fixture
def cfg() -> SizingConfig:
    return SizingConfig(
        risk_per_unit_pct=0.015,    # 1.5%
        stop_atr_mult=2.0,
        max_size_drift_pct=0.30,
    )


@pytest.fixture
def btc_info() -> SymbolInfo:
    return SymbolInfo(
        symbol="BTC-USDT-SWAP",
        base_currency="BTC", quote_currency="USDT",
        contract_size=0.01, tick_size=0.1,
        lot_size=1.0, min_quantity=1.0, max_leverage=100.0,
    )


# ---------------------------------------------------------------------------


class TestSizingConfig:
    def test_invalid_risk_pct_rejected(self) -> None:
        with pytest.raises(ValueError):
            SizingConfig(risk_per_unit_pct=0.0)
        with pytest.raises(ValueError):
            SizingConfig(risk_per_unit_pct=0.5)  # 50% per trade — sanity reject

    def test_invalid_stop_mult_rejected(self) -> None:
        with pytest.raises(ValueError):
            SizingConfig(stop_atr_mult=0)
        with pytest.raises(ValueError):
            SizingConfig(stop_atr_mult=-1)

    def test_invalid_drift_pct_rejected(self) -> None:
        with pytest.raises(ValueError):
            SizingConfig(max_size_drift_pct=-0.1)
        with pytest.raises(ValueError):
            SizingConfig(max_size_drift_pct=1.1)


# ---------------------------------------------------------------------------


class TestExpectedSize:
    def test_basic_formula(self, cfg: SizingConfig) -> None:
        # equity 1000, ATR N=500 (e.g. BTC)
        # risk = 1000 × 0.015 = 15
        # stop_distance = 2 × 500 = 1000
        # size = 15 / 1000 = 0.015 (BTC)
        size = expected_size(equity=1000.0, atr_n=500.0, config=cfg)
        assert size == pytest.approx(0.015)

    def test_zero_equity_returns_zero(self, cfg: SizingConfig) -> None:
        assert expected_size(equity=0.0, atr_n=500.0, config=cfg) == 0.0

    def test_zero_atr_returns_zero(self, cfg: SizingConfig) -> None:
        # Without volatility info, can't size — return 0
        assert expected_size(equity=1000.0, atr_n=0.0, config=cfg) == 0.0

    def test_higher_vol_smaller_size(self, cfg: SizingConfig) -> None:
        s1 = expected_size(equity=1000.0, atr_n=500.0, config=cfg)
        s2 = expected_size(equity=1000.0, atr_n=1000.0, config=cfg)
        assert s2 < s1

    def test_higher_equity_proportional_size(self, cfg: SizingConfig) -> None:
        s1 = expected_size(equity=1000.0, atr_n=500.0, config=cfg)
        s2 = expected_size(equity=2000.0, atr_n=500.0, config=cfg)
        assert s2 == pytest.approx(s1 * 2)


# ---------------------------------------------------------------------------


class TestSizeInContracts:
    def test_basic_conversion(self, btc_info: SymbolInfo) -> None:
        # $50 notional at $50,000/BTC = 0.001 BTC = 0.1 contracts
        # But min_quantity is 1, so result floor to 0
        result = size_in_contracts(
            quote_size=50.0, symbol_info=btc_info, last_price=50000.0,
        )
        assert result == 0.0  # below min

    def test_above_min_quantity(self, btc_info: SymbolInfo) -> None:
        # $1000 notional at $50,000/BTC = 0.02 BTC = 2 contracts
        result = size_in_contracts(
            quote_size=1000.0, symbol_info=btc_info, last_price=50000.0,
        )
        assert result == 2.0

    def test_rounded_down_to_lot_size(self, btc_info: SymbolInfo) -> None:
        # $1250 / $50,000 = 0.025 BTC = 2.5 contracts → rounds DOWN to 2
        result = size_in_contracts(
            quote_size=1250.0, symbol_info=btc_info, last_price=50000.0,
        )
        assert result == 2.0

    def test_zero_quote_returns_zero(self, btc_info: SymbolInfo) -> None:
        assert size_in_contracts(0.0, btc_info, last_price=50000.0) == 0.0

    def test_zero_price_returns_zero(self, btc_info: SymbolInfo) -> None:
        assert size_in_contracts(1000.0, btc_info, last_price=0.0) == 0.0


# ---------------------------------------------------------------------------


class TestVerifyPineSize:
    def test_match_passes(self, cfg: SizingConfig) -> None:
        ok, drift = verify_pine_size(pine_size=0.015, expected=0.015, config=cfg)
        assert ok is True
        assert drift == 0.0

    def test_within_tolerance_passes(self, cfg: SizingConfig) -> None:
        # 25% drift — within 30% tolerance
        ok, drift = verify_pine_size(pine_size=0.0125, expected=0.015, config=cfg)
        assert ok is True
        assert pytest.approx(drift, rel=0.01) == 0.1666666

    def test_over_tolerance_fails(self, cfg: SizingConfig) -> None:
        # 50% drift — over 30% tolerance
        ok, drift = verify_pine_size(pine_size=0.0075, expected=0.015, config=cfg)
        assert ok is False
        assert drift > cfg.max_size_drift_pct

    def test_zero_expected_no_verification(self, cfg: SizingConfig) -> None:
        # Can't verify when we have no expectation — defer
        ok, _ = verify_pine_size(pine_size=0.015, expected=0.0, config=cfg)
        assert ok is True

    def test_zero_pine_size_fails(self, cfg: SizingConfig) -> None:
        ok, drift = verify_pine_size(pine_size=0.0, expected=0.015, config=cfg)
        assert ok is False
        assert drift == 1.0
