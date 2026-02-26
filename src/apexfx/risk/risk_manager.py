"""Central Risk Manager — hard-coded wrapper between model output and execution.

This is the NON-NEGOTIABLE safety layer. The model's action CANNOT bypass it.
It is NOT trainable. It operates as a series of gate checks.

Changes from original:
- FIX: VaR scaling applies to position_size, not action (prevents double-scaling)
- ADD: DailyLossGuard — hard daily loss limit
- ADD: KillSwitch — emergency halt with auto-triggers and manual file-based kill
- ADD: VolatilityTargeter — scales exposure to target portfolio vol
- FIX: record_trade accepts trade_return for proper Kelly computation
- FIX: Drawdown critical scales position_size, not action
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from apexfx.risk.cooldown import CooldownManager
from apexfx.risk.drawdown_monitor import DrawdownMonitor
from apexfx.risk.position_sizer import PositionSizer
from apexfx.risk.var_calculator import VaRCalculator
from apexfx.utils.logging import get_logger

if TYPE_CHECKING:
    from apexfx.config.schema import RiskConfig

logger = get_logger(__name__)


@dataclass
class MarketState:
    """Current market state for risk evaluation."""
    current_price: float
    current_spread: float
    current_atr: float | None = None
    historical_atr: float | None = None
    spread_limit: float = 0.0002  # 2 pips default


@dataclass
class RiskDecision:
    """Result of risk evaluation."""
    approved: bool
    adjusted_action: float
    position_size: float  # lots
    reason: str
    checks_passed: list[str]
    checks_failed: list[str]
    var_scale: float = 1.0  # scaling applied to position (not action)


class DailyLossGuard:
    """Hard daily loss limit — stops all trading when daily loss exceeds threshold."""

    def __init__(self, max_daily_loss_pct: float = 0.02) -> None:
        self._max_daily_loss_pct = max_daily_loss_pct
        self._day_start_equity: float | None = None
        self._current_equity: float = 0.0
        self._current_date: date | None = None
        self._triggered = False

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def daily_loss_pct(self) -> float:
        if self._day_start_equity is None or self._day_start_equity <= 0:
            return 0.0
        return max(0.0, (self._day_start_equity - self._current_equity) / self._day_start_equity)

    def update(self, current_equity: float) -> bool:
        """Update with current equity. Returns True if trading is allowed."""
        self._current_equity = current_equity
        today = datetime.now(UTC).date()

        if self._current_date != today:
            self._day_start_equity = current_equity
            self._current_date = today
            self._triggered = False

        if self._day_start_equity is not None and self._day_start_equity > 0:
            daily_loss = (self._day_start_equity - current_equity) / self._day_start_equity
            if daily_loss >= self._max_daily_loss_pct:
                if not self._triggered:
                    logger.error(
                        "DAILY LOSS LIMIT BREACHED",
                        daily_loss=f"{daily_loss:.2%}",
                        limit=f"{self._max_daily_loss_pct:.2%}",
                        day_start=round(self._day_start_equity, 2),
                        current=round(current_equity, 2),
                    )
                self._triggered = True
                return False
        return True


class KillSwitch:
    """Emergency kill switch — immediate halt of all trading.

    Can be triggered:
    - Automatically by risk thresholds
    - Manually via kill file (touch data/KILL_SWITCH)
    """

    KILL_FILE = Path("data/KILL_SWITCH")

    def __init__(
        self,
        max_consecutive_rejections: int = 10,
        equity_floor_pct: float = 0.80,
    ) -> None:
        self._max_rejections = max_consecutive_rejections
        self._equity_floor_pct = equity_floor_pct
        self._consecutive_rejections = 0
        self._initial_balance: float | None = None
        self._active = False
        self._reason = ""

    @property
    def is_active(self) -> bool:
        if self.KILL_FILE.exists():
            self._active = True
            self._reason = "Manual kill switch (file)"
        return self._active

    @property
    def reason(self) -> str:
        return self._reason

    def activate(self, reason: str) -> None:
        self._active = True
        self._reason = reason
        logger.error("KILL SWITCH ACTIVATED", reason=reason)
        try:
            self.KILL_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.KILL_FILE.write_text(
                f"{datetime.now(UTC).isoformat()}: {reason}\n"
            )
        except Exception:
            pass

    def record_rejection(self) -> None:
        self._consecutive_rejections += 1
        if self._consecutive_rejections >= self._max_rejections:
            self.activate(
                f"Too many consecutive rejections ({self._consecutive_rejections})"
            )

    def record_success(self) -> None:
        self._consecutive_rejections = 0

    def check_equity(self, equity: float, initial_balance: float) -> None:
        if self._initial_balance is None:
            self._initial_balance = initial_balance
        floor = self._initial_balance * self._equity_floor_pct
        if equity < floor:
            self.activate(
                f"Equity {equity:.2f} below floor {floor:.2f} "
                f"({self._equity_floor_pct:.0%} of initial {self._initial_balance:.2f})"
            )

    def reset(self) -> None:
        """Manual reset — requires removing kill file too."""
        self._active = False
        self._reason = ""
        self._consecutive_rejections = 0
        if self.KILL_FILE.exists():
            self.KILL_FILE.unlink()
        logger.info("Kill switch manually reset")


class VolatilityTargeter:
    """Targets a specific portfolio volatility level.

    Scales position exposure so realized portfolio vol converges to target.
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        lookback: int = 60,
        annualization: int = 252,
    ) -> None:
        self._target_vol = target_vol
        self._lookback = lookback
        self._ann = annualization
        self._returns: list[float] = []

    def update(self, daily_return: float) -> None:
        self._returns.append(daily_return)
        if len(self._returns) > self._lookback * 2:
            self._returns = self._returns[-self._lookback:]

    def compute_leverage(self) -> float:
        """Compute leverage multiplier to achieve target volatility."""
        if len(self._returns) < max(20, self._lookback // 2):
            return 0.5  # conservative until enough data
        recent = np.array(self._returns[-self._lookback:])
        realized = np.std(recent, ddof=1) * np.sqrt(self._ann)
        if realized < 1e-6:
            return 1.0
        leverage = self._target_vol / realized
        return float(np.clip(leverage, 0.1, 2.0))


class WeekendGapGuard:
    """Reduces or eliminates exposure before weekends and holidays.

    Friday ~21:00 UTC = market close. Holding positions over the weekend
    exposes the portfolio to gap risk from events occurring when markets
    are closed (geopolitical events, natural disasters, etc.).
    """

    def __init__(
        self,
        close_before_hour_utc: int = 20,
        close_on_friday: bool = True,
        reduce_on_friday: float = 0.5,
    ) -> None:
        self._close_hour = close_before_hour_utc
        self._close_friday = close_on_friday
        self._reduce_friday = reduce_on_friday

    def check(self, utc_now: datetime | None = None) -> tuple[bool, float]:
        """Check weekend risk.

        Returns:
            (should_block_new_trades, position_scale_factor)
        """
        if utc_now is None:
            utc_now = datetime.now(UTC)

        weekday = utc_now.weekday()  # Mon=0, Fri=4
        hour = utc_now.hour

        if weekday == 4 and hour >= self._close_hour and self._close_friday:
            return True, 0.0  # Block all new trades, signal close

        if weekday == 4 and hour >= 14:
            return False, self._reduce_friday  # Reduce position on Friday afternoon

        if weekday in (5, 6):
            return True, 0.0  # Weekend — block everything

        return False, 1.0


class RegimeAdaptiveRisk:
    """Dynamically adjusts risk parameters based on detected market regime.

    In volatile regimes: reduces max position, tightens drawdown limits.
    In trending regimes: allows larger positions, wider stops.
    In flat regimes: reduces position (low opportunity), tighter stops.
    """

    # Regime scaling factors: (max_position_scale, var_limit_scale)
    REGIME_PROFILES = {
        "trending": (1.2, 1.0),       # Slightly larger positions in trends
        "mean_reverting": (1.0, 1.0),  # Normal
        "volatile": (0.5, 0.5),        # Half position, half VaR limit
        "flat": (0.7, 0.8),            # Reduced opportunity
    }

    def __init__(self) -> None:
        self._current_regime: str = "flat"

    def set_regime(self, regime: str) -> None:
        self._current_regime = regime

    def get_scales(self) -> tuple[float, float]:
        """Returns (position_scale, var_limit_scale) for current regime."""
        return self.REGIME_PROFILES.get(self._current_regime, (1.0, 1.0))

    @property
    def current_regime(self) -> str:
        return self._current_regime


class RiskManager:
    """
    Hard risk management layer. Wraps all risk checks.
    Called BEFORE every trade execution.

    The risk manager can:
    1. VETO an action entirely (force to neutral)
    2. REDUCE position size (via var_scale on position, not action)
    3. FORCE close existing positions
    4. BLOCK trading for a cooldown period
    5. KILL all trading via kill switch
    6. LIMIT daily losses
    7. TARGET portfolio volatility
    8. BLOCK trading before weekends (gap risk)
    9. ADAPT risk to market regime
    """

    def __init__(
        self,
        config: RiskConfig,
        initial_balance: float = 100_000.0,
    ) -> None:
        self._config = config

        self.var_calc = VaRCalculator(
            confidence=config.var_confidence,
            lookback_days=config.var_lookback_days,
            method=config.var_method,
        )

        self.drawdown = DrawdownMonitor(
            max_drawdown_pct=config.max_drawdown_pct,
            warning_threshold=config.max_drawdown_pct * 0.6,
            critical_threshold=config.max_drawdown_pct * 0.8,
        )

        self.cooldown = CooldownManager(
            after_n_losses=config.cooldown.after_n_losses,
            duration_minutes=config.cooldown.duration_minutes,
            tilt_drawdown_pct=config.cooldown.tilt_drawdown_pct,
            tilt_window_minutes=config.cooldown.tilt_window_minutes,
        )

        self.position_sizer = PositionSizer(
            max_position_pct=config.position_sizing.max_position_pct,
            kelly_fraction=config.position_sizing.kelly_fraction,
            min_trades_for_kelly=config.position_sizing.min_trades_for_kelly,
            vol_lookback_bars=config.position_sizing.vol_lookback_bars,
            min_lot_size=config.position_sizing.min_lot_size,
        )

        self.daily_loss_guard = DailyLossGuard(
            max_daily_loss_pct=config.daily_var_limit,
        )
        self.kill_switch = KillSwitch(
            max_consecutive_rejections=10,
            equity_floor_pct=max(0.5, 1.0 - config.max_drawdown_pct * 3),
        )
        self.vol_targeter = VolatilityTargeter(target_vol=0.10)
        self.weekend_guard = WeekendGapGuard()
        self.regime_risk = RegimeAdaptiveRisk()

        self._portfolio_value = initial_balance
        self._initial_balance = initial_balance

    def update_portfolio(self, portfolio_value: float) -> None:
        """Update portfolio value for all risk components."""
        self._portfolio_value = portfolio_value
        self.drawdown.update(portfolio_value)
        self.cooldown.record_portfolio_value(portfolio_value)
        self.daily_loss_guard.update(portfolio_value)
        self.kill_switch.check_equity(portfolio_value, self._initial_balance)

    def record_daily_return(self, daily_return: float) -> None:
        """Update VaR calculator and vol targeter with daily return."""
        self.var_calc.update(daily_return)
        self.vol_targeter.update(daily_return)

    def record_trade(self, pnl: float, trade_return: float | None = None) -> None:
        """Record a closed trade for cooldown and Kelly tracking.

        Args:
            pnl: Absolute P&L in account currency.
            trade_return: Return on risk (PnL / notional). If None, estimates
                          from PnL and current portfolio value.
        """
        self.cooldown.record_trade(pnl)
        if trade_return is not None:
            self.position_sizer.update_trade_stats(trade_return)
        elif self._portfolio_value > 0:
            self.position_sizer.update_trade_stats(pnl / self._portfolio_value)

    def evaluate_action(
        self,
        action: float,
        market_state: MarketState,
    ) -> RiskDecision:
        """
        Evaluate whether the proposed action passes all risk checks.

        CRITICAL FIX: VaR/drawdown/vol scaling is applied to position_size,
        NOT to action. Action represents model conviction and should not be
        double-scaled (action already scales position in PositionSizer.compute).
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        adjusted_action = action
        var_scale = 1.0

        # --- Check 0: Kill switch ---
        if self.kill_switch.is_active:
            checks_failed.append(f"kill_switch ({self.kill_switch.reason})")
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason=f"Kill switch active: {self.kill_switch.reason}",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("kill_switch_clear")

        # --- Check 0b: Weekend gap risk ---
        weekend_block, weekend_scale = self.weekend_guard.check()
        if weekend_block:
            checks_failed.append("weekend_gap_risk")
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason="Weekend gap risk: no new trades before market close",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        if weekend_scale < 1.0:
            var_scale *= weekend_scale
            checks_passed.append(f"weekend_scaled ({weekend_scale:.2f})")
        else:
            checks_passed.append("weekend_ok")

        # --- Check 0c: Daily loss limit ---
        if self.daily_loss_guard.is_triggered:
            checks_failed.append(
                f"daily_loss_limit ({self.daily_loss_guard.daily_loss_pct:.2%})"
            )
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason=f"Daily loss limit: {self.daily_loss_guard.daily_loss_pct:.2%}",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("daily_loss_ok")

        # --- Check 1: Cooldown ---
        if self.cooldown.is_active:
            remaining = self.cooldown.remaining
            checks_failed.append(f"cooldown_active (remaining: {remaining})")
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason=f"Cooldown active: {remaining}",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("cooldown_clear")

        # --- Check 2: Drawdown ---
        if self.drawdown.is_breached:
            checks_failed.append("max_drawdown_breached")
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason=f"Max drawdown breached: {self.drawdown.current_drawdown:.2%}",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        if self.drawdown.is_critical:
            # Scale position size (NOT action) in critical zone
            var_scale *= 0.5
            checks_passed.append("drawdown_critical_scaled")
        else:
            checks_passed.append("drawdown_ok")

        # --- Check 3: Spread check ---
        if market_state.current_spread > market_state.spread_limit:
            checks_failed.append(
                f"spread_too_wide ({market_state.current_spread:.5f} > "
                f"{market_state.spread_limit:.5f})"
            )
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason="Spread too wide",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("spread_ok")

        # --- Check 4: VaR limit (scales position, NOT action) ---
        if self.var_calc.has_sufficient_data:
            current_var = self.var_calc.compute_var(self._portfolio_value)
            if current_var > self._config.daily_var_limit:
                scale = self._config.daily_var_limit / (current_var + 1e-10)
                var_scale *= min(scale, 1.0)
                checks_passed.append(f"var_scaled (scale={scale:.2f})")
            else:
                checks_passed.append("var_ok")
        else:
            checks_passed.append("var_insufficient_data")

        # --- Check 4b: Volatility targeting ---
        vol_leverage = self.vol_targeter.compute_leverage()
        if vol_leverage < 1.0:
            var_scale *= vol_leverage
            checks_passed.append(f"vol_target_scaled ({vol_leverage:.2f})")
        else:
            checks_passed.append("vol_target_ok")

        # --- Check 4c: Regime-adaptive scaling ---
        pos_regime_scale, var_regime_scale = self.regime_risk.get_scales()
        if pos_regime_scale < 1.0:
            var_scale *= pos_regime_scale
            checks_passed.append(
                f"regime_scaled ({self.regime_risk.current_regime}: {pos_regime_scale:.2f})"
            )
        else:
            checks_passed.append(f"regime_ok ({self.regime_risk.current_regime})")

        # --- Check 5: Position sizing ---
        if abs(adjusted_action) < 0.05:
            return RiskDecision(
                approved=True,
                adjusted_action=0.0,
                position_size=0.0,
                reason="Action too small, staying neutral",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        position_size = self.position_sizer.compute(
            action=adjusted_action,
            portfolio_value=self._portfolio_value,
            current_price=market_state.current_price,
            current_atr=market_state.current_atr,
            historical_atr=market_state.historical_atr,
        )

        # Apply VaR/drawdown/vol scaling to position size (NOT action)
        position_size *= var_scale
        position_size = self.position_sizer._round_to_lot_step(position_size)

        if position_size <= 0:
            checks_failed.append("position_size_zero")
            return RiskDecision(
                approved=False,
                adjusted_action=0.0,
                position_size=0.0,
                reason="Position size computed to zero",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        checks_passed.append(f"position_sized ({position_size:.2f} lots)")

        logger.debug(
            "Risk evaluation passed",
            original_action=round(action, 4),
            adjusted_action=round(adjusted_action, 4),
            position_size=round(position_size, 4),
            var_scale=round(var_scale, 4),
            n_passed=len(checks_passed),
        )

        return RiskDecision(
            approved=True,
            adjusted_action=adjusted_action,
            position_size=position_size,
            reason="All checks passed",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            var_scale=var_scale,
        )

    def set_regime(self, regime: str) -> None:
        """Update the current market regime for adaptive risk."""
        self.regime_risk.set_regime(regime)

    def force_close_all(self) -> bool:
        """Signal that all positions should be force-closed."""
        if self.drawdown.is_breached:
            logger.error("FORCE CLOSE ALL: max drawdown breached")
            return True
        if self.kill_switch.is_active:
            logger.error("FORCE CLOSE ALL: kill switch active")
            return True
        if self.daily_loss_guard.is_triggered:
            logger.error("FORCE CLOSE ALL: daily loss limit")
            return True
        # Weekend close: force close before market close on Friday
        weekend_block, _ = self.weekend_guard.check()
        if weekend_block:
            logger.warning("FORCE CLOSE ALL: weekend gap protection")
            return True
        return False
