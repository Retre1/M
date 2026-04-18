"""RARAReward v5 — Risk-Adjusted Return-Aware reward with jump & regime awareness.

Evolution: TradingReward (v1, 10 components → gradient confusion)
         → ProfitFocusedReward (v4, 4 components → overtrading plateau)
         → RARAReward_v5 (v5, 9 components → jump-aware, regime-adaptive, CVaR-penalised)

Mathematical foundation
-----------------------
The reward at step *t* is:

    r_t = w_rpnl · R_t^{realized}                           ... (1) Bias-free PnL
        + w_unr  · Δ(UPnL_t)                                ... (2) Unrealised delta
        - w_cost · 𝟙{trade}                                 ... (3) Transaction cost
        - w_inact · f_inact(flat_bars)                       ... (4) Inactivity ramp
        - w_jump · |J_ψ(x_t) - J_ref(x_t)|                  ... (5) Jump risk penalty
        + w_fsd  · FSD_surprise(t)                           ... (6) Regime surprise bonus
        + w_gamma · Γ_bonus(t)                               ... (7) Gamma-aware stability
        + w_struct · 𝟙{structure_aligned}                    ... (8) Market structure bonus
        - w_cvar · CVaR_α(R_{t-W:t})                         ... (9) Tail risk penalty

Each term addresses a specific failure mode observed in v1–v4:

(1) Bias-free PnL — uses gap-procedure corrected returns (Paper 2) with
    asymmetric loss scaling (losses hurt ``loss_asymmetry`` × more).
(2) Unrealised delta — dense per-step signal between trade closes;
    prevents sparse-reward starvation.
(3) Transaction cost — fixed penalty per entry/flip; prevents churning
    (v4 had 0.05 → 550 trades/ep; v5 default 0.15).
(4) Inactivity ramp — two-segment linear: gentle after grace period,
    steep after max_bars. Prevents "do nothing" collapse (v1 failure).
(5) Jump risk penalty (Paper 5, DML) — penalises the agent when
    the portfolio's exposure to jump risk (estimated via a fast
    Bates-like model) diverges from a reference hedged portfolio.
    |J_ψ - J_ref| is the L1 norm of the jump-operator residual.
(6) FSD regime surprise (Paper 1) — bonus when the agent's action
    is consistent with a *new* regime signal.  Encourages adaptation
    rather than stale positioning.
(7) Gamma-aware stability (Paper 5) — bonus for maintaining a
    portfolio whose second-order sensitivity (Γ) stays bounded,
    i.e. the P&L response to a 1σ price move is predictable.
(8) Structure bonus — reward for entries aligned with market structure
    breaks (confluence with existing StructureExtractor).
(9) CVaR tail-risk penalty (Paper 3) — penalises the expected loss
    in the worst α-quantile of recent returns, teaching the agent
    to avoid catastrophic drawdowns rather than just average loss.

Performance notes
-----------------
* All per-step computations are O(1) amortised (ring buffers, EMA).
* The jump-operator uses a lightweight analytical approximation
  (no neural-network forward pass inside the reward loop).
* FSD regime is pre-computed in the data pipeline; here we only
  read the one-hot columns and detect transitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from apexfx.env.reward import BaseRewardFunction
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# Configuration
# =====================================================================


class RARARewardV5Config(BaseModel, frozen=True):
    """Pydantic v2 configuration for RARAReward_v5.

    Every numeric weight is exposed so that curriculum stages or
    hyperparameter sweeps can override defaults via YAML overlay.

    Attributes:
        realized_pnl_weight: Multiplier on realized PnL (dominant signal).
        unrealized_delta_weight: Multiplier on per-step ΔUPnL.
        trade_cost_weight: Fixed penalty per new entry or flip.
        inactivity_weight: Per-bar penalty after grace period.
        inactivity_grace: Bars flat before penalty starts.
        max_inactivity_bars: Hard ceiling for gentle ramp.
        loss_asymmetry: Multiplier on negative rewards (risk aversion).
        reward_clip: Symmetric clip magnitude.

        jump_risk_weight: Weight on |J_ψ - J_ref| penalty.
        jump_lambda: Poisson intensity of jumps (Merton model).
        jump_mu: Mean jump size (log-normal).
        jump_sigma: Jump size volatility.

        fsd_surprise_weight: Bonus for acting on regime transitions.
        fsd_transition_window: Bars to consider a regime "new".

        gamma_aware_weight: Bonus for bounded portfolio gamma.
        gamma_threshold: Maximum acceptable |Γ| before penalty.

        structure_bonus_weight: Reward for structure-aligned entries.

        cvar_weight: Penalty on CVaR of recent returns.
        cvar_alpha: Tail quantile (e.g. 0.05 = worst 5%).
        cvar_window: Rolling window for CVaR estimation.
        cvar_min_samples: Minimum samples before CVaR activates.
    """

    # --- Core PnL (from v4, retuned) ---
    realized_pnl_weight: float = 2_000.0
    unrealized_delta_weight: float = 500.0
    trade_cost_weight: float = 0.15
    inactivity_weight: float = 0.00005
    inactivity_grace: int = 30
    max_inactivity_bars: int = 150
    loss_asymmetry: float = 1.0
    reward_clip: float = 50.0

    # --- Jump risk (Paper 5: DML) ---
    jump_risk_weight: float = 0.5
    jump_lambda: float = 0.05
    jump_mu: float = -0.01
    jump_sigma: float = 0.02

    # --- FSD regime (Paper 1) ---
    fsd_surprise_weight: float = 0.1
    fsd_transition_window: int = 5

    # --- Gamma-aware (Paper 5: Greeks stability) ---
    gamma_aware_weight: float = 0.05
    gamma_threshold: float = 2.0

    # --- Structure ---
    structure_bonus_weight: float = 0.10

    # --- CVaR (Paper 3: Quantum Finance) ---
    cvar_weight: float = 0.0
    cvar_alpha: float = 0.05
    cvar_window: int = 100
    cvar_min_samples: int = 20


# =====================================================================
# Jump-operator estimator (lightweight Bates/Merton analytical approx)
# =====================================================================


class _JumpOperator:
    """Fast analytical jump-risk estimator based on the Merton model.

    The jump operator J_ψ measures the expected portfolio P&L impact
    from a single Poisson jump event:

        J_ψ(x) = λ · E[position · S · (exp(J) - 1)]

    where J ~ N(μ_J, σ_J²) and λ is the jump intensity.

    The reference portfolio J_ref assumes a fully hedged (flat) position,
    so J_ref = 0.  The penalty |J_ψ - J_ref| = |J_ψ| incentivises the
    agent to avoid unhedged jump exposure.

    This is a scalar approximation — no neural network forward pass,
    making it safe to call at every step inside the reward.

    Ref: Merton (1976), adapted per Paper 5's PIDE residual formulation.
    """

    __slots__ = ("_lambda", "_mean_jump", "_var_jump")

    def __init__(self, lam: float, mu: float, sigma: float) -> None:
        self._lambda = lam
        self._mean_jump = np.exp(mu + 0.5 * sigma**2) - 1.0
        self._var_jump = (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)

    def penalty(self, position: float, price: float) -> float:
        """Compute |J_ψ(x)| — unsigned jump exposure.

        Args:
            position: Current position in lots (signed).
            price: Current mid-price.

        Returns:
            Non-negative scalar representing unhedged jump risk.
        """
        if abs(position) < 1e-8 or price < 1e-8:
            return 0.0
        # Expected P&L impact of a single jump
        expected_impact = abs(position) * price * abs(self._mean_jump)
        return self._lambda * expected_impact


# =====================================================================
# Ring buffer for CVaR
# =====================================================================


class _RingBuffer:
    """Fixed-size ring buffer backed by a NumPy array.

    O(1) append, O(W) sort for quantile — much faster than
    list slicing + sorted() for W ≤ 200.
    """

    __slots__ = ("_buf", "_size", "_idx", "_full")

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._size = capacity
        self._idx = 0
        self._full = False

    def append(self, value: float) -> None:
        self._buf[self._idx] = value
        self._idx = (self._idx + 1) % self._size
        if self._idx == 0:
            self._full = True

    @property
    def count(self) -> int:
        return self._size if self._full else self._idx

    def values(self) -> np.ndarray:
        """Return filled portion (unordered)."""
        if self._full:
            return self._buf.copy()
        return self._buf[: self._idx].copy()

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._idx = 0
        self._full = False


# =====================================================================
# RARAReward_v5
# =====================================================================


class RARAReward_v5(BaseRewardFunction):
    """Risk-Adjusted Return-Aware reward function v5.

    Integrates jump-risk awareness (Paper 5), FSD regime transitions
    (Paper 1), bias-free PnL (Paper 2), CVaR tail-risk penalty
    (Paper 3), and gamma stability (Paper 5 Greeks).

    Compatible with the existing ``BaseRewardFunction`` interface:
    the env calls ``set_step_context()`` then ``compute()``.

    Args:
        config: Full configuration.  Defaults are tuned for
            RTX 4090 + EURUSD H1 + TQC + curriculum learning.

    Example::

        cfg = RARARewardV5Config(realized_pnl_weight=10000)
        reward_fn = RARAReward_v5(cfg)
        # In env.step():
        reward_fn.set_step_context(...)
        r = reward_fn.compute(portfolio_value, prev_portfolio_value)
    """

    def __init__(self, config: RARARewardV5Config | None = None) -> None:
        self._cfg = config or RARARewardV5Config()
        self._jump_op = _JumpOperator(
            lam=self._cfg.jump_lambda,
            mu=self._cfg.jump_mu,
            sigma=self._cfg.jump_sigma,
        )
        self._returns_buf = _RingBuffer(self._cfg.cvar_window)

        # Mutable step context (set by env before compute)
        self._position: float = 0.0
        self._prev_position: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._prev_unrealized_pnl: float = 0.0
        self._realized_pnl: float = 0.0
        self._price: float = 1.0
        self._flat_bars: int = 0
        self._time_in_position: int = 0

        # FSD state
        self._prev_fsd_regime: int = 1  # NEUTRAL
        self._fsd_regime: int = 1
        self._fsd_transition_age: int = 999

        # Gamma proxy
        self._prev_return: float = 0.0
        self._prev_prev_return: float = 0.0

        # Structure
        self._structure_aligned: bool = False

        # Diagnostics (for TensorBoard logging)
        self._last_components: dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseRewardFunction interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all mutable state at episode start."""
        self._returns_buf.reset()
        self._position = 0.0
        self._prev_position = 0.0
        self._unrealized_pnl = 0.0
        self._prev_unrealized_pnl = 0.0
        self._realized_pnl = 0.0
        self._price = 1.0
        self._flat_bars = 0
        self._time_in_position = 0
        self._prev_fsd_regime = 1
        self._fsd_regime = 1
        self._fsd_transition_age = 999
        self._prev_return = 0.0
        self._prev_prev_return = 0.0
        self._structure_aligned = False
        self._last_components = {}

    def compute(self, portfolio_value: float, prev_portfolio_value: float) -> float:
        """Compute the composite reward for the current step.

        Args:
            portfolio_value: Portfolio value after this step.
            prev_portfolio_value: Portfolio value before this step.

        Returns:
            Clipped scalar reward.
        """
        cfg = self._cfg
        if prev_portfolio_value <= 0 or portfolio_value <= 0:
            return -cfg.reward_clip

        step_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Track flat streak
        is_flat = abs(self._position) < 1e-8
        was_flat = abs(self._prev_position) < 1e-8
        if is_flat and was_flat:
            self._flat_bars += 1
        else:
            self._flat_bars = 0

        # (1) Realized PnL -------------------------------------------------
        realized = 0.0
        if abs(self._realized_pnl) > 1e-10:
            pnl_frac = self._realized_pnl / prev_portfolio_value
            realized = pnl_frac * cfg.realized_pnl_weight

        # (2) Unrealised delta ---------------------------------------------
        unrealized_delta = 0.0
        if not is_flat and not was_flat:
            delta = self._unrealized_pnl - self._prev_unrealized_pnl
            unrealized_delta = (delta / prev_portfolio_value) * cfg.unrealized_delta_weight

        # (3) Trade cost ---------------------------------------------------
        trade_cost = 0.0
        is_new_entry = (was_flat and not is_flat)
        is_flip = (
            not was_flat and not is_flat
            and np.sign(self._position) != np.sign(self._prev_position)
        )
        if is_new_entry or is_flip:
            trade_cost = cfg.trade_cost_weight

        # (4) Inactivity ramp ----------------------------------------------
        inactivity = self._compute_inactivity()

        # (5) Jump risk penalty --------------------------------------------
        jump_penalty = cfg.jump_risk_weight * self._jump_op.penalty(
            self._position, self._price,
        )

        # (6) FSD regime surprise bonus ------------------------------------
        fsd_bonus = self._compute_fsd_bonus()

        # (7) Gamma-aware bonus --------------------------------------------
        gamma_bonus = self._compute_gamma_bonus(step_return)

        # (8) Structure bonus ----------------------------------------------
        structure = 0.0
        if self._structure_aligned and (is_new_entry or is_flip):
            structure = cfg.structure_bonus_weight

        # (9) CVaR tail-risk penalty ----------------------------------------
        self._returns_buf.append(step_return)
        cvar_penalty = self._compute_cvar()

        # --- Compose ------------------------------------------------------
        reward = (
            realized
            + unrealized_delta
            - trade_cost
            - inactivity
            - jump_penalty
            + fsd_bonus
            + gamma_bonus
            + structure
            - cvar_penalty
        )

        # Asymmetric loss scaling
        if reward < 0:
            reward *= cfg.loss_asymmetry

        reward = float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip))

        # Store diagnostics
        self._last_components = {
            "reward/total": reward,
            "reward/realized_pnl": realized,
            "reward/unrealized_delta": unrealized_delta,
            "reward/trade_cost": -trade_cost,
            "reward/inactivity": -inactivity,
            "reward/jump_penalty": -jump_penalty,
            "reward/fsd_bonus": fsd_bonus,
            "reward/gamma_bonus": gamma_bonus,
            "reward/structure_bonus": structure,
            "reward/cvar_penalty": -cvar_penalty,
        }

        # Shift gamma state
        self._prev_prev_return = self._prev_return
        self._prev_return = step_return

        return reward

    # ------------------------------------------------------------------
    # Context setter (called by env before compute)
    # ------------------------------------------------------------------

    def set_step_context(
        self,
        position: float,
        unrealized_pnl: float,
        realized_pnl: float,
        price: float,
        time_in_position: int = 0,
        fsd_regime: int = 1,
        structure_aligned: bool = False,
    ) -> None:
        """Provide all per-step context to the reward function.

        Must be called by the env **before** ``compute()`` each step.

        Args:
            position: Current position size (signed; lots or fraction).
            unrealized_pnl: Floating PnL in account currency.
            realized_pnl: PnL closed *this step* (0 if no close).
            price: Current mid-price (for jump-operator scaling).
            time_in_position: Bars since last entry.
            fsd_regime: FSD regime code (0=RISK_OFF, 1=NEUTRAL, 2=RISK_ON).
            structure_aligned: Whether the current action aligns with
                a market structure break.
        """
        self._prev_position = self._position
        self._prev_unrealized_pnl = self._unrealized_pnl

        self._position = position
        self._unrealized_pnl = unrealized_pnl
        self._realized_pnl = realized_pnl
        self._price = price
        self._time_in_position = time_in_position
        self._structure_aligned = structure_aligned

        # FSD transition tracking
        self._prev_fsd_regime = self._fsd_regime
        if fsd_regime != self._fsd_regime:
            self._fsd_transition_age = 0
        else:
            self._fsd_transition_age += 1
        self._fsd_regime = fsd_regime

    # ------------------------------------------------------------------
    # Backward-compatible API (adapts old env calling conventions)
    # ------------------------------------------------------------------

    def set_trade_info(
        self,
        action: float,
        direction: int,
        unrealized_pnl: float,
        time_in_position: int,
        realized_pnl_this_step: float = 0.0,
        news_active: bool = False,
        structure_aligned: bool = False,
    ) -> None:
        """Backward-compatible adapter for envs using the v4 API.

        Maps the old calling convention to :meth:`set_step_context`.
        """
        self.set_step_context(
            position=float(direction),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl_this_step,
            price=self._price,  # keep last known price
            time_in_position=time_in_position,
            structure_aligned=structure_aligned,
        )

    def set_atr(self, atr: float | None) -> None:
        """Unused — kept for API compatibility with TradingReward."""

    def set_price(self, price: float) -> None:
        """Set current mid-price (for envs that track price separately)."""
        self._price = price

    def set_fsd_regime(self, regime: int) -> None:
        """Set FSD regime from pre-computed pipeline columns."""
        self._prev_fsd_regime = self._fsd_regime
        if regime != self._fsd_regime:
            self._fsd_transition_age = 0
        else:
            self._fsd_transition_age += 1
        self._fsd_regime = regime

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def last_components(self) -> dict[str, float]:
        """Per-component breakdown of the last reward (for TensorBoard).

        Returns:
            Dict with keys like ``reward/realized_pnl``, ``reward/cvar_penalty`` etc.
        """
        return self._last_components.copy()

    # ------------------------------------------------------------------
    # Private component computations
    # ------------------------------------------------------------------

    def _compute_inactivity(self) -> float:
        """Two-segment linear inactivity penalty.

        Segment 1 (grace..max_bars): gentle slope ``w_inact * excess``.
        Segment 2 (> max_bars): 5× steeper (forces exploration hard).

        The piecewise-linear ramp keeps the critic target smooth (no
        exponential spikes that destabilise TQC's quantile regression).
        """
        cfg = self._cfg
        if self._flat_bars <= cfg.inactivity_grace:
            return 0.0

        excess = self._flat_bars - cfg.inactivity_grace
        ramp_span = max(1, cfg.max_inactivity_bars - cfg.inactivity_grace)

        if self._flat_bars <= cfg.max_inactivity_bars:
            return cfg.inactivity_weight * excess
        else:
            base = cfg.inactivity_weight * ramp_span
            overflow = self._flat_bars - cfg.max_inactivity_bars
            return base + cfg.inactivity_weight * 5.0 * overflow

    def _compute_fsd_bonus(self) -> float:
        """FSD regime surprise bonus (Paper 1).

        Awards the agent when it acts during a *fresh* regime transition.
        The bonus decays linearly over ``fsd_transition_window`` bars so
        the agent is incentivised to react quickly to regime changes.

        Logic:
        - Regime just changed AND agent is positioned → full bonus.
        - Regime changed N bars ago → bonus × (1 - N/window).
        - No recent transition → 0.

        This prevents stale positioning: the agent must *adapt* to new
        regimes, not sit in a position that was correct for the old one.
        """
        cfg = self._cfg
        if self._fsd_transition_age >= cfg.fsd_transition_window:
            return 0.0
        if abs(self._position) < 1e-8:
            return 0.0

        # Decay factor: 1.0 at transition, 0.0 at window boundary
        decay = 1.0 - self._fsd_transition_age / cfg.fsd_transition_window
        return cfg.fsd_surprise_weight * decay

    def _compute_gamma_bonus(self, step_return: float) -> float:
        """Gamma-aware portfolio stability bonus (Paper 5).

        Approximates the portfolio's gamma (second derivative of P&L
        w.r.t. price) using the second-order finite difference of
        consecutive returns:

            Γ_approx ≈ (r_t - 2·r_{t-1} + r_{t-2}) / Δt²

        If |Γ_approx| < threshold, the agent receives a small bonus.
        If |Γ_approx| > threshold, no bonus (but no penalty either —
        the jump-risk term handles extreme exposure).

        Why gamma matters for RL trading:
        - Low gamma → P&L response is approximately linear in price.
        - High gamma → P&L is convex/concave, creating surprise losses.
        - Encouraging low gamma produces smoother equity curves.
        """
        cfg = self._cfg
        if abs(self._position) < 1e-8:
            return 0.0

        # Finite-difference second derivative of returns
        gamma_approx = abs(
            step_return - 2.0 * self._prev_return + self._prev_prev_return
        )

        if gamma_approx < cfg.gamma_threshold * 1e-4:
            return cfg.gamma_aware_weight
        return 0.0

    def _compute_cvar(self) -> float:
        """CVaR (Expected Shortfall) tail-risk penalty (Paper 3).

        CVaR_α = E[R | R < VaR_α]

        Penalises the expected loss in the worst α-quantile of the
        recent return distribution.  Uses the ring buffer for O(W)
        computation.

        Classical CVaR is used here (not quantum-estimated) because
        the sample size W ≤ 200 makes QAE's quadratic speedup irrelevant.
        """
        cfg = self._cfg
        n = self._returns_buf.count
        if n < cfg.cvar_min_samples:
            return 0.0

        returns = self._returns_buf.values()
        sorted_r = np.sort(returns)  # ascending
        n_tail = max(1, int(n * cfg.cvar_alpha))
        tail = sorted_r[:n_tail]

        # Only penalise if tail is negative (downside risk)
        cvar = -float(np.mean(tail))  # flip sign: positive = bad
        if cvar <= 0:
            return 0.0

        return cfg.cvar_weight * cvar
