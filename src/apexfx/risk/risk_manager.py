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

import numpy as np

from apexfx.config.schema import RiskConfig
from apexfx.env.trade_filter import StrategyFilter
from apexfx.risk.cooldown import CooldownManager
from apexfx.risk.drawdown_monitor import DrawdownMonitor
from apexfx.risk.position_sizer import PositionSizer
from apexfx.risk.stress_testing import StressTester
from apexfx.risk.var_calculator import VaRCalculator
from apexfx.utils.logging import get_logger

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


@dataclass
class DynamicStopConfig:
    """Dynamic stop-loss parameters based on regime and uncertainty."""
    atr_mult: float            # ATR multiplier for stop distance
    trailing: bool             # Use trailing stop (trends only)
    stop_distance: float | None = None  # Absolute stop distance (atr_mult * ATR)


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
        except Exception as e:
            logger.error(
                "CRITICAL: Kill file write failed — kill switch is memory-only",
                error=str(e),
            )

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
    10. APPLY strategy filter rules (news blackout, structure confirmation, etc.)
    """

    def __init__(
        self,
        config: RiskConfig,
        initial_balance: float = 100_000.0,
        uncertainty_weight: float = 0.5,
        uncertainty_min_scale: float = 0.1,
    ) -> None:
        self._config = config
        self._uncertainty_weight = uncertainty_weight
        self._uncertainty_min_scale = uncertainty_min_scale

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

        # Stress tester
        st_cfg = getattr(config, "stress_test", None)
        if st_cfg and st_cfg.enabled:
            self.stress_tester = StressTester(
                var_limit=config.daily_var_limit,
                margin_requirement=0.01,
            )
        else:
            self.stress_tester: StressTester | None = None

        # Strategy filter (rule-based trade rules)
        sf_cfg = getattr(config, "strategy_filter", None)
        if sf_cfg and sf_cfg.enabled:
            self.strategy_filter = StrategyFilter(
                news_blackout_threshold=sf_cfg.news_blackout_threshold,
                time_to_event_threshold=sf_cfg.time_to_event_threshold,
                min_fundamental_bias=sf_cfg.min_fundamental_bias,
                require_structure_confirm=sf_cfg.require_structure_confirm,
                exit_on_conflict=sf_cfg.exit_on_conflict,
                reduce_scale_pre_news=sf_cfg.reduce_scale_pre_news,
                pre_news_time_threshold=sf_cfg.pre_news_time_threshold,
                block_against_bias=sf_cfg.block_against_bias,
                min_bias_for_direction=sf_cfg.min_bias_for_direction,
            )
        else:
            self.strategy_filter: StrategyFilter | None = None

        # Last observation for strategy filter evaluation
        self._last_obs: dict | None = None

        # Portfolio context for multi-symbol trading
        self._portfolio_positions: list | None = None
        self._portfolio_max_total_exposure: float = 0.40
        self._portfolio_max_per_symbol: float = 0.25

        self._portfolio_value = initial_balance
        self._initial_balance = initial_balance

        # Per-symbol return tracking for Portfolio VaR
        self._symbol_returns: dict[str, list[float]] = {}
        self._symbol_returns_lookback: int = 60

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

    def record_symbol_return(self, symbol: str, daily_return: float) -> None:
        """Record a per-symbol daily return for Portfolio VaR calculation."""
        if symbol not in self._symbol_returns:
            self._symbol_returns[symbol] = []
        self._symbol_returns[symbol].append(daily_return)
        # Truncate to lookback
        if len(self._symbol_returns[symbol]) > self._symbol_returns_lookback * 2:
            self._symbol_returns[symbol] = self._symbol_returns[symbol][-self._symbol_returns_lookback:]

    def _compute_symbol_vol(self, symbol: str) -> float:
        """Compute realized daily volatility for a symbol.

        Returns config default if insufficient data.
        """
        import numpy as _np
        returns = self._symbol_returns.get(symbol, [])
        if len(returns) >= 10:
            return float(_np.std(returns, ddof=1))
        # Fallback: use config default
        pvar_cfg = getattr(self._config, "portfolio_var", None)
        return getattr(pvar_cfg, "default_symbol_vol", 0.01) if pvar_cfg else 0.01

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

    def set_observation(self, obs: dict) -> None:
        """Set the current observation for strategy filter evaluation."""
        self._last_obs = obs

    def evaluate_action(
        self,
        action: float,
        market_state: MarketState,
        uncertainty_score: float | None = None,
        current_position: float = 0.0,
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

        # --- Check 0d: Strategy filter (rule-based trade rules) ---
        if self.strategy_filter and self._last_obs:
            filter_decision = self.strategy_filter.check(
                self._last_obs, adjusted_action, current_position
            )
            if filter_decision.force_close:
                adjusted_action = 0.0
                checks_passed.append("strategy_filter_force_close")
                logger.info(
                    "Strategy filter: force close",
                    reason=filter_decision.reason,
                )
            elif not filter_decision.allowed:
                checks_failed.append(f"strategy_filter ({filter_decision.reason})")
                return RiskDecision(
                    approved=False,
                    adjusted_action=0.0,
                    position_size=0.0,
                    reason=f"Strategy filter: {filter_decision.reason}",
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                )
            elif filter_decision.scale < 1.0:
                var_scale *= filter_decision.scale
                checks_passed.append(
                    f"strategy_filter_scaled ({filter_decision.scale:.2f})"
                )
            else:
                checks_passed.append("strategy_filter_ok")

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

        # --- Check 4b: Portfolio VaR (multi-asset) ---
        # Uses variance-covariance method: VaR_p = sqrt(w' × Σ × w) where
        # Σ_ij = VaR_i × VaR_j × ρ_ij, with dynamic correlations from
        # DynamicCorrelationTracker and realized per-symbol volatility.
        pvar_cfg = getattr(self._config, "portfolio_var", None)
        if (
            pvar_cfg
            and pvar_cfg.multi_asset
            and self._portfolio_positions
        ):
            try:
                import numpy as _np
                from scipy import stats as _stats

                positions = self._portfolio_positions
                total_notional = sum(abs(p.notional) for p in positions)

                if total_notional > 0 and len(positions) >= 2:
                    pos_dict = {p.symbol: p.notional for p in positions}
                    symbols = list(pos_dict.keys())
                    n = len(symbols)

                    # 1. Build correlation matrix using DynamicCorrelationTracker
                    corr_matrix = _np.eye(n)
                    try:
                        from apexfx.live.portfolio_manager import get_correlation_tracker
                        tracker = get_correlation_tracker()
                    except ImportError:
                        tracker = None

                    for i in range(n):
                        for j in range(i + 1, n):
                            if tracker is not None:
                                rho = tracker.get_correlation(symbols[i], symbols[j])
                            else:
                                rho = 0.30  # Conservative default
                            corr_matrix[i, j] = rho
                            corr_matrix[j, i] = rho

                    # 2. Individual VaR per position using realized volatility
                    z_alpha = _stats.norm.ppf(self.var_calc._confidence)
                    ind_vars = {}
                    for p in positions:
                        vol = self._compute_symbol_vol(p.symbol)
                        # VaR_i = |notional| × σ_daily × z_α
                        ind_vars[p.symbol] = abs(p.notional) * vol * z_alpha

                    # 3. Compute portfolio VaR via variance-covariance
                    portfolio_var = self.var_calc.compute_portfolio_var(
                        pos_dict, corr_matrix, ind_vars,
                    )

                    pvar_limit = pvar_cfg.daily_limit * self._portfolio_value

                    logger.debug(
                        "Portfolio VaR computed",
                        portfolio_var=f"{portfolio_var:.2f}",
                        pvar_limit=f"{pvar_limit:.2f}",
                        n_positions=n,
                        symbols=symbols,
                    )

                    if portfolio_var > pvar_limit:
                        pvar_scale = pvar_limit / (portfolio_var + 1e-10)
                        var_scale *= min(pvar_scale, 1.0)
                        checks_passed.append(
                            f"portfolio_var_scaled (pVaR={portfolio_var:.0f}, limit={pvar_limit:.0f})"
                        )
                    else:
                        checks_passed.append("portfolio_var_ok")
                else:
                    checks_passed.append("portfolio_var_single_position")
            except Exception as e:
                logger.debug("Portfolio VaR calculation failed", error=str(e))
                checks_passed.append("portfolio_var_fallback")
        elif pvar_cfg and pvar_cfg.multi_asset:
            checks_passed.append("portfolio_var_no_positions")

        # --- Check 4c: Volatility targeting ---
        vol_leverage = self.vol_targeter.compute_leverage()
        if vol_leverage < 1.0:
            var_scale *= vol_leverage
            checks_passed.append(f"vol_target_scaled ({vol_leverage:.2f})")
        else:
            checks_passed.append("vol_target_ok")

        # --- Check 4d: Regime-adaptive scaling ---
        pos_regime_scale, var_regime_scale = self.regime_risk.get_scales()
        if pos_regime_scale < 1.0:
            var_scale *= pos_regime_scale
            checks_passed.append(
                f"regime_scaled ({self.regime_risk.current_regime}: {pos_regime_scale:.2f})"
            )
        else:
            checks_passed.append(f"regime_ok ({self.regime_risk.current_regime})")

        # --- Check 4e: Portfolio concentration (multi-symbol) ---
        if self._portfolio_positions:
            total_exposure = sum(abs(p.notional) for p in self._portfolio_positions)
            exposure_ratio = total_exposure / self._portfolio_value if self._portfolio_value > 0 else 0.0
            if exposure_ratio > self._portfolio_max_total_exposure:
                checks_failed.append(
                    f"portfolio_exposure ({exposure_ratio:.1%} > {self._portfolio_max_total_exposure:.1%})"
                )
                return RiskDecision(
                    approved=False,
                    adjusted_action=0.0,
                    position_size=0.0,
                    reason=f"Portfolio exposure limit: {exposure_ratio:.1%}",
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                )
            checks_passed.append(f"portfolio_exposure_ok ({exposure_ratio:.1%})")
        else:
            checks_passed.append("portfolio_n/a")

        # --- Check 4f: Uncertainty-based scaling ---
        if uncertainty_score is not None and uncertainty_score > 0:
            unc_scale = max(
                self._uncertainty_min_scale,
                1.0 - self._uncertainty_weight * uncertainty_score,
            )
            var_scale *= unc_scale
            checks_passed.append(f"uncertainty_scaled ({unc_scale:.2f}, score={uncertainty_score:.3f})")
        else:
            checks_passed.append("uncertainty_n/a")

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
            uncertainty_score=round(uncertainty_score, 4) if uncertainty_score is not None else None,
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

    def run_startup_stress_test(self, portfolio_value: float) -> None:
        """Run stress tests on startup to assess portfolio resilience."""
        if self.stress_tester is None:
            return
        st_cfg = getattr(self._config, "stress_test", None)
        if st_cfg and not st_cfg.run_on_startup:
            return

        try:
            results = self.stress_tester.run_all_presets(portfolio_value)
            failures = [r for r in results if not r.survival]
            if failures:
                logger.error(
                    "Stress test: portfolio would NOT survive some scenarios",
                    failed_scenarios=[r.scenario.name for r in failures],
                )
            else:
                logger.info("Stress test: portfolio survives all preset scenarios")
        except Exception as e:
            logger.error("Startup stress test failed", error=str(e))

    def set_regime(self, regime: str) -> None:
        """Update the current market regime for adaptive risk."""
        self.regime_risk.set_regime(regime)

    def set_portfolio_context(self, open_positions: list) -> None:
        """Inject cross-pair context for portfolio-aware risk checks.

        Args:
            open_positions: List of PositionInfo from PortfolioManager.
        """
        self._portfolio_positions = open_positions

    def compute_dynamic_stop(
        self,
        uncertainty_score: float,
        regime: str,
        current_atr: float | None,
    ) -> DynamicStopConfig:
        """Compute regime-aware dynamic stop-loss parameters.

        Wider stops in volatile/uncertain conditions, tighter in trends.
        Trailing stops only enabled in trending regimes.
        """
        base_mult = {
            "trending": 2.0,
            "mean_reverting": 3.0,
            "volatile": 4.0,
            "flat": 2.5,
        }
        mult = base_mult.get(regime, 2.5) * (1.0 + 0.5 * uncertainty_score)
        trailing = regime == "trending"
        stop_distance = mult * current_atr if current_atr is not None else None
        return DynamicStopConfig(
            atr_mult=mult,
            trailing=trailing,
            stop_distance=stop_distance,
        )

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
