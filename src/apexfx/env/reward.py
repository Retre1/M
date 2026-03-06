"""Reward functions: Differential Sharpe, Sortino, Calmar-weighted, Z-Score, Hold, and LogReturn.

Changes from original:
- ADD: CalmarWeightedReward — superlinear DD penalty + duration penalty + recovery bonus
- FIX: QuantumZScoreReward.compute() now properly includes z_score when available
         (previously compute_with_zscore was never called from env)
- FIX: QuantumZScoreReward now delegates to CalmarWeightedReward (better base)
- ADD: HoldAwareReward — bonus for holding profitable positions, preventing churn
- ADD: LogReturnReward — simple log-return with asymmetric loss penalty (cleaner gradient signal)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class DifferentialSharpeReward(BaseRewardFunction):
    """
    R_t = ΔPortfolio_t / σ_rolling − λ · Drawdown_t

    Implements the Differential Sharpe Ratio (Moody & Saffell, 1998).
    """

    def __init__(
        self,
        eta: float = 0.01,
        lambda_dd: float = 2.0,
        transaction_cost_penalty: float = 0.0001,
    ) -> None:
        self.eta = eta
        self.lambda_dd = lambda_dd
        self.transaction_cost_penalty = transaction_cost_penalty

        self._A: float = 0.0
        self._B: float = 0.0
        self._peak: float = 0.0
        self._step: int = 0

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0
        self._peak = 0.0
        self._step = 0

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        if prev_portfolio_value <= 0:
            return 0.0

        self._step += 1
        R_t = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

        delta_A = R_t - self._A
        delta_B = R_t**2 - self._B
        self._A += self.eta * delta_A
        self._B += self.eta * delta_B

        sigma_sq = self._B - self._A**2
        if sigma_sq > 1e-10:
            dsr = (delta_A * self._B - 0.5 * self._A * delta_B) / (sigma_sq**1.5 + 1e-10)
        else:
            dsr = 0.0

        self._peak = max(self._peak, portfolio_value)
        drawdown = (self._peak - portfolio_value) / self._peak if self._peak > 0 else 0.0

        reward = dsr - self.lambda_dd * drawdown
        return float(np.clip(reward, -10.0, 10.0))


class SortinoReward(BaseRewardFunction):
    """Sortino-style reward: penalizes only downside deviation."""

    def __init__(
        self,
        window: int = 100,
        lambda_dd: float = 2.0,
        target_return: float = 0.0,
    ) -> None:
        self.window = window
        self.lambda_dd = lambda_dd
        self.target_return = target_return

        self._returns: list[float] = []
        self._peak: float = 0.0

    def reset(self) -> None:
        self._returns = []
        self._peak = 0.0

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        if prev_portfolio_value <= 0:
            return 0.0

        R_t = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self._returns.append(R_t)

        if len(self._returns) > self.window:
            self._returns = self._returns[-self.window :]

        returns = np.array(self._returns)
        excess = returns - self.target_return

        downside = np.minimum(excess, 0)
        downside_std = np.sqrt(np.mean(downside**2) + 1e-10)

        sortino = np.mean(excess) / downside_std if downside_std > 1e-10 else 0.0

        self._peak = max(self._peak, portfolio_value)
        drawdown = (self._peak - portfolio_value) / self._peak if self._peak > 0 else 0.0

        reward = sortino - self.lambda_dd * drawdown
        return float(np.clip(reward, -10.0, 10.0))


class CalmarWeightedReward(BaseRewardFunction):
    """Calmar-weighted reward with superlinear drawdown penalty and recovery bonus.

    Key improvements over DifferentialSharpeReward:
    1. Superlinear DD penalty: 2% DD → 0.04, 5% DD → 0.25 (with exponent=2)
    2. Duration penalty: prolonged drawdowns penalized more than brief ones
    3. Recovery bonus: rewards climbing out of drawdown (prevents learned passivity)
    """

    def __init__(
        self,
        eta: float = 0.01,
        dd_exponent: float = 2.0,
        time_decay: float = 0.001,
        recovery_weight: float = 1.0,
        lambda_dd: float = 2.0,
    ) -> None:
        self.eta = eta
        self.dd_exponent = dd_exponent
        self.time_decay = time_decay
        self.recovery_weight = recovery_weight
        self.lambda_dd = lambda_dd

        self._A: float = 0.0
        self._B: float = 0.0
        self._peak: float = 0.0
        self._step: int = 0
        self._dd_duration: int = 0
        self._prev_drawdown: float = 0.0

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0
        self._peak = 0.0
        self._step = 0
        self._dd_duration = 0
        self._prev_drawdown = 0.0

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        if prev_portfolio_value <= 0:
            return 0.0

        self._step += 1
        R_t = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

        delta_A = R_t - self._A
        delta_B = R_t**2 - self._B
        self._A += self.eta * delta_A
        self._B += self.eta * delta_B

        sigma_sq = self._B - self._A**2
        if sigma_sq > 1e-10:
            dsr = (delta_A * self._B - 0.5 * self._A * delta_B) / (sigma_sq**1.5 + 1e-10)
        else:
            dsr = 0.0

        # Drawdown tracking
        self._peak = max(self._peak, portfolio_value)
        drawdown = (self._peak - portfolio_value) / self._peak if self._peak > 0 else 0.0

        if drawdown > 0:
            self._dd_duration += 1
        else:
            self._dd_duration = 0

        # Superlinear drawdown penalty
        dd_penalty = self.lambda_dd * (drawdown ** self.dd_exponent)

        # Duration penalty
        duration_penalty = self.time_decay * self._dd_duration * drawdown

        # Recovery bonus
        recovery_bonus = 0.0
        if self._prev_drawdown > 0.02 and drawdown < self._prev_drawdown:
            recovery_bonus = (self._prev_drawdown - drawdown) * self.recovery_weight

        self._prev_drawdown = drawdown

        reward = dsr - dd_penalty - duration_penalty + recovery_bonus
        return float(np.clip(reward, -10.0, 10.0))


class QuantumZScoreReward(BaseRewardFunction):
    """Quantum Z-Score reward: adds bonus when agent trades against extreme Z-Score.

    FIX: Now stores z_score state so compute() uses it automatically.
    The env sets z_score via set_zscore() before calling compute().
    Uses CalmarWeightedReward as the base (better than plain DSR).
    """

    def __init__(
        self,
        eta: float = 0.01,
        lambda_dd: float = 2.0,
        z_score_bonus_weight: float = 0.1,
    ) -> None:
        self.base_reward = CalmarWeightedReward(eta=eta, lambda_dd=lambda_dd)
        self.z_score_bonus_weight = z_score_bonus_weight
        self._current_z_score: float = 0.0

    def reset(self) -> None:
        self.base_reward.reset()
        self._current_z_score = 0.0

    def set_zscore(self, z_score: float) -> None:
        """Set current price Z-Score for next reward computation."""
        self._current_z_score = z_score

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        """Compute reward including Z-Score bonus when available."""
        base = self.base_reward.compute(portfolio_value, prev_portfolio_value)

        if prev_portfolio_value <= 0:
            return base

        portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Quantum bonus: reward for trading against extreme Z-Scores
        quantum_bonus = 0.0
        z = self._current_z_score
        if abs(z) > 2.0 and np.sign(portfolio_return) != np.sign(z):
            quantum_bonus = abs(z) * self.z_score_bonus_weight

        return float(np.clip(base + quantum_bonus, -10.0, 10.0))


class HoldAwareReward(BaseRewardFunction):
    """Reward that includes a bonus for holding profitable positions.

    Problem solved: without hold reward, the agent learns to churn —
    opening and closing positions frequently to capture tiny moves,
    paying transaction costs each time. This reward explicitly rewards
    the agent for staying in a winning position.

    Components:
    1. Base Calmar reward (DSR + drawdown)
    2. Hold bonus: small per-step reward while unrealized PnL is positive
    3. Churn penalty: penalizes closing a profitable position too early
    4. Z-Score bonus (optional): mean-reversion signal

    The env must call set_position_info() each step.
    """

    def __init__(
        self,
        eta: float = 0.01,
        lambda_dd: float = 2.0,
        hold_bonus_per_step: float = 0.001,
        churn_penalty: float = 0.05,
        z_score_bonus_weight: float = 0.1,
        min_hold_bars: int = 3,
    ) -> None:
        self.base_reward = CalmarWeightedReward(eta=eta, lambda_dd=lambda_dd)
        self.hold_bonus_per_step = hold_bonus_per_step
        self.churn_penalty = churn_penalty
        self.z_score_bonus_weight = z_score_bonus_weight
        self.min_hold_bars = min_hold_bars

        self._current_z_score: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._time_in_position: int = 0
        self._position_direction: int = 0
        self._prev_position_direction: int = 0
        self._prev_unrealized_pnl: float = 0.0

    def reset(self) -> None:
        self.base_reward.reset()
        self._current_z_score = 0.0
        self._unrealized_pnl = 0.0
        self._time_in_position = 0
        self._position_direction = 0
        self._prev_position_direction = 0
        self._prev_unrealized_pnl = 0.0

    def set_zscore(self, z_score: float) -> None:
        self._current_z_score = z_score

    def set_position_info(
        self,
        direction: int,
        unrealized_pnl: float,
        time_in_position: int,
    ) -> None:
        """Set current position state for hold/churn calculations."""
        self._prev_position_direction = self._position_direction
        self._prev_unrealized_pnl = self._unrealized_pnl
        self._position_direction = direction
        self._unrealized_pnl = unrealized_pnl
        self._time_in_position = time_in_position

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        base = self.base_reward.compute(portfolio_value, prev_portfolio_value)

        if prev_portfolio_value <= 0:
            return base

        portfolio_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Hold bonus: reward for staying in a profitable position
        hold_bonus = 0.0
        if self._position_direction != 0 and self._unrealized_pnl > 0:
            # Scale bonus by how profitable the position is
            pnl_pct = self._unrealized_pnl / prev_portfolio_value
            hold_bonus = self.hold_bonus_per_step * min(pnl_pct * 100, 5.0)

        # Churn penalty: closed a profitable position too early
        churn_penalty = 0.0
        was_in_position = self._prev_position_direction != 0
        now_flat = self._position_direction == 0
        if was_in_position and now_flat:
            if self._prev_unrealized_pnl > 0 and self._time_in_position < self.min_hold_bars:
                churn_penalty = self.churn_penalty

        # Z-Score bonus (same as QuantumZScoreReward)
        quantum_bonus = 0.0
        z = self._current_z_score
        if abs(z) > 2.0 and np.sign(portfolio_return) != np.sign(z):
            quantum_bonus = abs(z) * self.z_score_bonus_weight

        reward = base + hold_bonus - churn_penalty + quantum_bonus
        return float(np.clip(reward, -10.0, 10.0))


class LogReturnReward(BaseRewardFunction):
    """Simple log-return reward with asymmetric loss penalty.

    Why this works better for RL training than CalmarWeightedReward:
    1. **Clean gradient signal**: log(V_t / V_{t-1}) is smooth and well-behaved
    2. **Asymmetric penalty**: losses hurt ``loss_weight``x more than gains help,
       teaching the agent risk aversion without complex drawdown tracking
    3. **No hidden state**: reward depends only on current step's return,
       so the agent can clearly attribute actions to outcomes
    4. **Scale-invariant**: log-returns are comparable across different equity levels

    Reward formula::

        r_t = log(V_t / V_{t-1})
        if r_t < 0:
            r_t *= loss_weight   # e.g. 2.0 → losses penalized 2x

    Optionally scaled by ``reward_scale`` for numerical stability with SAC.
    """

    def __init__(
        self,
        loss_weight: float = 2.0,
        reward_scale: float = 1000.0,
    ) -> None:
        self.loss_weight = loss_weight
        self.reward_scale = reward_scale

    def reset(self) -> None:
        pass  # no state to reset

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        if prev_portfolio_value <= 0 or portfolio_value <= 0:
            return -10.0

        log_ret = np.log(portfolio_value / prev_portfolio_value)

        # Asymmetric: losses hurt more
        if log_ret < 0:
            log_ret *= self.loss_weight

        reward = log_ret * self.reward_scale
        return float(np.clip(reward, -10.0, 10.0))


class TradingReward(BaseRewardFunction):
    """Production-grade reward: vol-adjusted returns + CVaR + costs.

    Key upgrades for hedge-fund-level performance:
    1. **Volatility-adjusted return**: return / rolling_vol normalizes across regimes.
       Same absolute return in high-vol is worth less (riskier).
    2. **CVaR penalty**: penalizes tail losses (worst 5% of returns), teaching
       the agent to avoid catastrophic drawdowns, not just average loss.
    3. **Spread-cost penalty**: deducts estimated cost on every position change.
    4. **Churn penalty**: penalizes closing positions too quickly.
    5. **Drawdown penalty**: superlinear penalty during drawdowns.
    6. **Hold bonus**: rewards maintaining profitable positions.
    7. **Asymmetric log-return base**: clean gradient signal.

    The env should call ``set_trade_info()`` each step with position state.
    Optionally call ``set_atr()`` for volatility-adjusted reward.
    """

    def __init__(
        self,
        loss_weight: float = 2.0,
        reward_scale: float = 1000.0,
        spread_cost_pips: float = 1.5,
        pip_value: float = 0.0001,
        churn_penalty: float = 0.3,
        min_hold_bars: int = 3,
        dd_weight: float = 1.0,
        dd_exponent: float = 2.0,
        hold_bonus: float = 0.05,
        cvar_window: int = 50,
        cvar_alpha: float = 0.05,
        cvar_weight: float = 0.5,
        vol_lookback: int = 20,
        # Phase 3: Professional trading rewards
        hold_winner_bonus: float = 0.1,
        quick_cut_bonus: float = 0.2,
        news_trade_penalty: float = 0.3,
        structure_confirm_bonus: float = 0.15,
    ) -> None:
        self.loss_weight = loss_weight
        self.reward_scale = reward_scale
        self.spread_cost = spread_cost_pips * pip_value
        self.churn_penalty = churn_penalty
        self.min_hold_bars = min_hold_bars
        self.dd_weight = dd_weight
        self.dd_exponent = dd_exponent
        self.hold_bonus = hold_bonus
        self.cvar_window = cvar_window
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.vol_lookback = vol_lookback
        # Phase 3 params
        self.hold_winner_bonus = hold_winner_bonus
        self.quick_cut_bonus = quick_cut_bonus
        self.news_trade_penalty = news_trade_penalty
        self.structure_confirm_bonus = structure_confirm_bonus

        self._peak: float = 0.0
        self._prev_action: float = 0.0
        self._time_in_position: int = 0
        self._position_direction: int = 0
        self._prev_position_direction: int = 0
        self._unrealized_pnl: float = 0.0
        self._returns_history: list[float] = []
        self._current_atr: float | None = None
        # Phase 3: news and structure awareness
        self._news_active: bool = False
        self._structure_aligned: bool = False
        self._was_losing: bool = False

    def reset(self) -> None:
        self._peak = 0.0
        self._prev_action = 0.0
        self._time_in_position = 0
        self._position_direction = 0
        self._prev_position_direction = 0
        self._unrealized_pnl = 0.0
        self._returns_history = []
        self._current_atr = None
        self._news_active = False
        self._structure_aligned = False
        self._was_losing = False

    def set_trade_info(
        self,
        action: float,
        direction: int,
        unrealized_pnl: float,
        time_in_position: int,
        news_active: bool = False,
        structure_aligned: bool = False,
    ) -> None:
        """Set current position state — called by env before compute()."""
        self._prev_action = action
        self._prev_position_direction = self._position_direction
        # Track if position was losing before this step (for quick cut bonus)
        if self._position_direction != 0 and self._unrealized_pnl < 0:
            self._was_losing = True
        self._position_direction = direction
        self._unrealized_pnl = unrealized_pnl
        self._time_in_position = time_in_position
        self._news_active = news_active
        self._structure_aligned = structure_aligned

    def set_atr(self, atr: float | None) -> None:
        """Set current ATR for volatility-adjusted reward."""
        self._current_atr = atr

    def _compute_rolling_vol(self) -> float:
        """Compute rolling volatility from recent returns."""
        if len(self._returns_history) < 5:
            return 1e-8
        recent = self._returns_history[-self.vol_lookback:]
        return float(np.std(recent)) + 1e-8

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        if prev_portfolio_value <= 0 or portfolio_value <= 0:
            return -10.0

        # 1. Base: asymmetric log-return
        log_ret = np.log(portfolio_value / prev_portfolio_value)
        raw_return = log_ret  # save unscaled for CVaR tracking

        if log_ret < 0:
            log_ret *= self.loss_weight

        # Volatility adjustment: normalize by current ATR
        # Same return in high-vol regime scores less (risk was higher)
        vol_scale = 1.0
        if self._current_atr is not None and self._current_atr > 1e-8:
            rolling_vol = self._compute_rolling_vol()
            if rolling_vol > 1e-8:
                vol_scale = min(rolling_vol / self._current_atr, 3.0)

        base_reward = log_ret * self.reward_scale * vol_scale

        # Track returns for CVaR
        self._returns_history.append(raw_return)
        if len(self._returns_history) > self.cvar_window:
            self._returns_history = self._returns_history[-self.cvar_window:]

        # 2. CVaR penalty: penalize tail risk (worst alpha% of returns)
        cvar_penalty = 0.0
        if len(self._returns_history) >= 10:
            sorted_returns = sorted(self._returns_history)
            cutoff = max(1, int(len(sorted_returns) * self.cvar_alpha))
            tail_returns = sorted_returns[:cutoff]
            cvar = abs(float(np.mean(tail_returns)))
            cvar_penalty = self.cvar_weight * cvar * self.reward_scale

        # 3. Spread-cost penalty on position changes
        cost_penalty = 0.0
        direction_changed = self._position_direction != self._prev_position_direction
        if direction_changed and self._position_direction != 0:
            cost_penalty = self.spread_cost * self.reward_scale * 0.5

        # 4. Churn penalty: closed profitable position too early
        churn = 0.0
        was_in = self._prev_position_direction != 0
        now_flat = self._position_direction == 0
        if was_in and now_flat and self._time_in_position < self.min_hold_bars:
            churn = self.churn_penalty

        # 5. Drawdown penalty (superlinear)
        self._peak = max(self._peak, portfolio_value)
        dd = (self._peak - portfolio_value) / self._peak if self._peak > 0 else 0.0
        dd_penalty = self.dd_weight * (dd ** self.dd_exponent) if dd > 0.01 else 0.0

        # 6. Hold bonus: reward staying in profitable positions
        hold = 0.0
        if self._position_direction != 0 and self._unrealized_pnl > 0:
            pnl_pct = self._unrealized_pnl / prev_portfolio_value
            hold = self.hold_bonus * min(pnl_pct * 100, 3.0)

        # --- Phase 3: Professional trading rewards ---

        # 7. Hold winner bonus: extra reward for holding winners per bar
        #    "Let your winners run" — the longer you hold a profitable position, the more bonus
        winner_bonus = 0.0
        if self._position_direction != 0 and self._unrealized_pnl > 0:
            # Bonus grows with time in position (up to 10 bars)
            time_factor = min(self._time_in_position / 10.0, 1.0)
            winner_bonus = self.hold_winner_bonus * time_factor

        # 8. Quick cut bonus: reward for cutting a losing position within 5 bars
        #    "Cut your losers fast"
        quick_cut = 0.0
        was_in = self._prev_position_direction != 0
        now_flat = self._position_direction == 0
        if was_in and now_flat and self._was_losing and self._time_in_position <= 5:
            quick_cut = self.quick_cut_bonus
        if now_flat:
            self._was_losing = False

        # 9. News trade penalty: punish opening positions during active news
        news_penalty = 0.0
        if self._news_active and direction_changed and self._position_direction != 0:
            news_penalty = self.news_trade_penalty

        # 10. Structure confirmation bonus: reward entries that align with structure break
        struct_bonus = 0.0
        if self._structure_aligned and direction_changed and self._position_direction != 0:
            struct_bonus = self.structure_confirm_bonus

        reward = (
            base_reward
            - cost_penalty
            - churn
            - dd_penalty
            + hold
            - cvar_penalty
            + winner_bonus
            + quick_cut
            - news_penalty
            + struct_bonus
        )
        return float(np.clip(reward, -10.0, 10.0))
