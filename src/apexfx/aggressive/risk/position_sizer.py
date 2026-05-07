"""Position sizing math — must match Pine Script ``donchian_turtle.pine``.

Why this exists if Pine already computes size
---------------------------------------------
Pine Script computes size on the TV side and emits it in the alert.  We
trust that — but we still **verify** here as a defence-in-depth check:

  * Pine's equity used for sizing is its *strategy.equity* — a backtest
    construct, not the real OKX balance.  A chart configured with
    initial_capital=1000 gives sizes for $1k account regardless of actual
    OKX balance.
  * If sizes mismatch by more than ``max_size_drift_pct`` (default 30%)
    we reject the alert.  This catches the user forgetting to update
    initial_capital when adding funds, or vice versa.

Sizing formula (must match Pine)
--------------------------------
    risk_dollars = equity × risk_per_unit_pct
    stop_distance = stop_atr_mult × N
    size_in_contracts = risk_dollars / stop_distance

For OKX BTC-USDT-SWAP, contract_size = 0.01 BTC, so the size in contracts
needs an extra division by ``contract_size`` if the price-per-contract is
in USD terms.  This module abstracts that.
"""

from __future__ import annotations

from dataclasses import dataclass

from apexfx.aggressive.exchanges.base import SymbolInfo
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SizingConfig:
    """Sizing parameters — must match Pine Script inputs."""

    risk_per_unit_pct: float = 0.015       # 1.5% of equity per unit
    stop_atr_mult: float = 2.0             # Stop distance in N (ATR units)
    max_size_drift_pct: float = 0.30       # Reject if Pine size differs >30%
    min_units: float = 0.0                 # Lower bound (passed through to exchange validation)

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_unit_pct < 0.20:
            raise ValueError(
                f"risk_per_unit_pct must be in (0, 0.20), got {self.risk_per_unit_pct}"
            )
        if self.stop_atr_mult <= 0:
            raise ValueError(f"stop_atr_mult must be positive, got {self.stop_atr_mult}")
        if not 0 <= self.max_size_drift_pct <= 1:
            raise ValueError(
                f"max_size_drift_pct must be in [0, 1], got {self.max_size_drift_pct}"
            )


def expected_size(
    equity: float,
    atr_n: float,
    config: SizingConfig,
) -> float:
    """Compute target position size in contracts.

    Mirrors Pine's ``contracts_per_unit`` formula:
        risk_dollars / (stop_atr_mult × N)

    For OKX SWAP with contract_size = 0.01 BTC at $50k, this returns
    SIZE IN BTC — multiply by 100 to get OKX contracts.  But Pine
    returns size already in contracts, so for parity we leave the
    contract-size unit conversion to the caller.
    """
    if equity <= 0:
        return 0.0
    if atr_n <= 0:
        # No volatility info → can't size — return 0 (caller skips signal)
        return 0.0
    risk_dollars = equity * config.risk_per_unit_pct
    stop_distance = config.stop_atr_mult * atr_n
    return risk_dollars / stop_distance


def size_in_contracts(
    quote_size: float,
    symbol_info: SymbolInfo,
    *,
    last_price: float,
) -> float:
    """Convert a quote-currency notional (e.g. $50 BTC) to OKX contract count.

    OKX SWAP contract = ``contract_size`` units of base asset (e.g.
    0.01 BTC for BTC-USDT-SWAP).  So:
        contracts = (quote_size / last_price) / contract_size

    Result is rounded down to the nearest ``lot_size`` and floored at
    ``min_quantity``.  Returns 0.0 if the rounded size is below the
    minimum — caller should treat as "skip this signal, position too
    small for this leverage".
    """
    if quote_size <= 0 or last_price <= 0:
        return 0.0
    base_qty = quote_size / last_price
    contracts = base_qty / symbol_info.contract_size

    # Round DOWN to nearest lot_size
    if symbol_info.lot_size > 0:
        contracts = (contracts // symbol_info.lot_size) * symbol_info.lot_size

    if contracts < symbol_info.min_quantity:
        logger.warning(
            "Computed size below exchange minimum — skipping",
            symbol=symbol_info.symbol,
            computed=contracts, minimum=symbol_info.min_quantity,
        )
        return 0.0
    return contracts


def verify_pine_size(
    pine_size: float,
    expected: float,
    config: SizingConfig,
) -> tuple[bool, float]:
    """Sanity check Pine's reported size against our recomputed expectation.

    Returns (ok, drift_pct).  ``ok=False`` means the alert should be
    rejected — Pine and Python disagree on sizing by more than the
    configured tolerance, which usually indicates:
      * Pine ``initial_capital`` not synced with real balance
      * Stale ATR (different bar than we expect)
      * Pine running with different ``risk_per_unit_pct`` than this config

    We log the drift either way for monitoring.
    """
    if expected <= 0:
        # Can't verify — defer decision to caller
        return (True, 0.0)
    if pine_size <= 0:
        return (False, 1.0)

    drift = abs(pine_size - expected) / expected
    ok = drift <= config.max_size_drift_pct
    if not ok:
        logger.warning(
            "Pine size differs from expected — rejecting alert",
            pine_size=pine_size, expected=expected, drift=drift,
            max_allowed=config.max_size_drift_pct,
        )
    return (ok, drift)
