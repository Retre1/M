"""Pre-trade liquidity checks: spread, session, and news blackout filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from apexfx.utils.logging import get_logger
from apexfx.utils.time_utils import get_active_sessions, is_forex_market_open

if TYPE_CHECKING:
    from apexfx.config.schema import ExecutionConfig, SymbolConfig
    from apexfx.data.mt5_client import MT5Client

logger = get_logger(__name__)


@dataclass
class LiquidityStatus:
    """Result of liquidity check."""
    tradeable: bool
    current_spread: float
    spread_limit: float
    session_active: bool
    market_open: bool
    reason: str


class LiquidityGuard:
    """
    Pre-trade liquidity filter. Blocks trades when:
    - Spread is too wide (news, low liquidity)
    - Outside allowed trading sessions
    - Market is closed
    - News blackout period active
    """

    def __init__(
        self,
        config: ExecutionConfig,
        symbol_config: SymbolConfig,
        mt5_client: MT5Client | None = None,
    ) -> None:
        self._config = config
        self._symbol_config = symbol_config
        self._mt5 = mt5_client

        # News blackout schedule (can be extended with external calendar)
        self._blackout_times: list[tuple[datetime, datetime]] = []

    def add_news_blackout(self, start: datetime, end: datetime) -> None:
        """Add a news blackout window."""
        self._blackout_times.append((start, end))

    def check(self, symbol: str) -> LiquidityStatus:
        """Perform all liquidity checks for a symbol."""
        now = datetime.now(UTC)
        spread_limit_price = (
            self._symbol_config.spread_limit_pips
            * self._symbol_config.pip_value
        )

        # Check 1: Market open
        if not is_forex_market_open(now):
            return LiquidityStatus(
                tradeable=False,
                current_spread=0,
                spread_limit=spread_limit_price,
                session_active=False,
                market_open=False,
                reason="Forex market closed",
            )

        # Check 2: Session filter
        session_active = True
        if self._config.session_filter.enabled:
            active_sessions = get_active_sessions(now.hour)
            allowed = self._config.session_filter.allowed_sessions
            session_active = any(
                s.value in allowed for s in active_sessions
            )
            if not session_active:
                return LiquidityStatus(
                    tradeable=False,
                    current_spread=0,
                    spread_limit=spread_limit_price,
                    session_active=False,
                    market_open=True,
                    reason=f"Outside allowed sessions: {[s.value for s in active_sessions]}",
                )

        # Check 3: News blackout
        if self._config.news_blackout.enabled:
            for blackout_start, blackout_end in self._blackout_times:
                if blackout_start <= now <= blackout_end:
                    return LiquidityStatus(
                        tradeable=False,
                        current_spread=0,
                        spread_limit=spread_limit_price,
                        session_active=session_active,
                        market_open=True,
                        reason="News blackout active",
                    )

        # Check 4: Spread
        current_spread = 0.0

        if self._mt5 is not None:
            try:
                info = self._mt5.get_symbol_info(symbol)
                current_spread = (info.ask - info.bid)
            except Exception as e:
                logger.warning("Failed to get spread", error=str(e))
                # If we can't check spread, be conservative
                return LiquidityStatus(
                    tradeable=False,
                    current_spread=0,
                    spread_limit=spread_limit_price,
                    session_active=session_active,
                    market_open=True,
                    reason="Cannot verify spread (MT5 error)",
                )

        if current_spread > spread_limit_price:
            return LiquidityStatus(
                tradeable=False,
                current_spread=current_spread,
                spread_limit=spread_limit_price,
                session_active=session_active,
                market_open=True,
                reason=(
                    f"Spread too wide: "
                    f"{current_spread / self._symbol_config.pip_value:.1f} "
                    f"pips (limit: "
                    f"{self._symbol_config.spread_limit_pips} pips)"
                ),
            )

        return LiquidityStatus(
            tradeable=True,
            current_spread=current_spread,
            spread_limit=spread_limit_price,
            session_active=session_active,
            market_open=True,
            reason="All checks passed",
        )
