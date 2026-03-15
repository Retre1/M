"""Risk-aware alert integration — monitors RiskManager events and fires alerts.

This module bridges the RiskManager with the AlertManager, checking for
critical risk events and dispatching alerts without modifying the core
RiskManager class.
"""

from __future__ import annotations

import time

from apexfx.alerts.alert_manager import AlertLevel, AlertManager
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class RiskAlertMonitor:
    """Monitors risk events and dispatches alerts.

    Wraps around a RiskManager to fire Telegram/Webhook alerts on:
    - Kill switch activation
    - Daily loss limit breach
    - Drawdown critical / breached
    - Weekend close signal
    - MT5 disconnect (called externally)
    - Urgent news detected
    - Model rollback triggered
    """

    def __init__(self, alert_manager: AlertManager) -> None:
        self._alerts = alert_manager
        self._last_kill_switch = False
        self._last_daily_loss_triggered = False
        self._last_drawdown_critical = False
        self._last_drawdown_breached = False
        self._last_mt5_connected = True
        self._mt5_disconnect_time: float | None = None

    async def check_risk_state(
        self,
        risk_manager: object,
        portfolio_value: float = 0.0,
        symbol: str = "",
    ) -> None:
        """Check all risk states and fire alerts for new events.

        Call this periodically (e.g., every bar processing cycle).
        """
        # --- Kill switch ---
        ks = getattr(risk_manager, "kill_switch", None)
        if ks and ks.is_active and not self._last_kill_switch:
            self._last_kill_switch = True
            await self._alerts.emergency(
                "KILL SWITCH ACTIVATED",
                f"All trading halted.\nReason: {ks.reason}",
                symbol=symbol,
                portfolio_value=f"${portfolio_value:,.2f}",
            )
        elif ks and not ks.is_active:
            self._last_kill_switch = False

        # --- Daily loss ---
        dlg = getattr(risk_manager, "daily_loss_guard", None)
        if dlg and dlg.is_triggered and not self._last_daily_loss_triggered:
            self._last_daily_loss_triggered = True
            await self._alerts.critical(
                "Daily Loss Limit Breached",
                f"Trading paused until next day.\nLoss: {dlg.daily_loss_pct:.2%}",
                symbol=symbol,
                portfolio_value=f"${portfolio_value:,.2f}",
            )
        elif dlg and not dlg.is_triggered:
            self._last_daily_loss_triggered = False

        # --- Drawdown ---
        dd = getattr(risk_manager, "drawdown", None)
        if dd:
            if dd.is_breached and not self._last_drawdown_breached:
                self._last_drawdown_breached = True
                await self._alerts.critical(
                    "Max Drawdown Breached",
                    f"Drawdown: {dd.current_drawdown:.2%}\nAll new trades blocked.",
                    symbol=symbol,
                    portfolio_value=f"${portfolio_value:,.2f}",
                )
            elif not dd.is_breached:
                self._last_drawdown_breached = False

            if dd.is_critical and not self._last_drawdown_critical:
                self._last_drawdown_critical = True
                await self._alerts.warning(
                    "Drawdown Critical",
                    f"Drawdown: {dd.current_drawdown:.2%}\nPosition sizes reduced 50%.",
                    symbol=symbol,
                )
            elif not dd.is_critical:
                self._last_drawdown_critical = False

    async def on_mt5_disconnect(self) -> None:
        """Call when MT5 connection is lost."""
        if self._last_mt5_connected:
            self._last_mt5_connected = False
            self._mt5_disconnect_time = time.time()
            await self._alerts.critical(
                "MT5 Disconnected",
                "Connection to MetaTrader 5 lost. Attempting reconnect...",
            )

    async def on_mt5_reconnect(self) -> None:
        """Call when MT5 connection is restored."""
        if not self._last_mt5_connected:
            downtime = ""
            if self._mt5_disconnect_time:
                dt = time.time() - self._mt5_disconnect_time
                downtime = f"\nDowntime: {dt:.0f}s"
            self._last_mt5_connected = True
            self._mt5_disconnect_time = None
            await self._alerts.info(
                "MT5 Reconnected",
                f"Connection restored.{downtime}",
            )

    async def on_urgent_news(self, headline: str, source: str) -> None:
        """Call when urgent/breaking news is detected."""
        await self._alerts.warning(
            "Urgent News Detected",
            f"{headline}",
            source=source,
        )

    async def on_model_rollback(self, reason: str) -> None:
        """Call when online learner triggers a model rollback."""
        await self._alerts.warning(
            "Model Rollback",
            f"Online learner reverted to previous checkpoint.\nReason: {reason}",
        )

    async def on_trade_opened(
        self, symbol: str, direction: str, size: float, price: float
    ) -> None:
        """Call when a trade is opened (INFO level)."""
        await self._alerts.info(
            "Trade Opened",
            f"{direction.upper()} {size} {symbol} @ {price:.5f}",
            symbol=symbol,
        )

    async def on_trade_closed(
        self, symbol: str, pnl: float, duration_bars: int
    ) -> None:
        """Call when a trade is closed (INFO level)."""
        emoji = "\u2705" if pnl >= 0 else "\u274c"
        await self._alerts.info(
            f"Trade Closed {emoji}",
            f"{symbol}: PnL ${pnl:+.2f} ({duration_bars} bars)",
            symbol=symbol,
        )
