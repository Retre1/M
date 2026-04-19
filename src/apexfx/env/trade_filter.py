"""Rule-based strategy filter for trade decisions.

Enforces non-negotiable trading rules that a professional institutional
trader would NEVER violate. These are hard constraints over the RL agent:

1. **News blackout** — no entries near high-impact events
2. **Conflicting signals** — force exit when fundamentals conflict
3. **Minimum fundamental bias** — require conviction to enter
4. **Structure confirmation** — require BOS for entry
5. **Direction alignment** — don't trade against fundamental bias
6. **Pre-news scaling** — reduce position before events

The filter works at two levels:
- **Training (TradeFilterWrapper)**: teaches the agent that rules exist
- **Live (RiskManager integration)**: hard safety layer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FilterDecision:
    """Result of strategy filter evaluation."""

    allowed: bool
    scale: float  # 0.0 = blocked, 0.5 = reduced, 1.0 = full
    reason: str
    force_close: bool  # True if filter demands closing current position


class StrategyFilter:
    """Rule-based trading filter using fundamental + structure features.

    Evaluates proposed actions against professional trading rules and
    either allows, blocks, scales, or forces position closure.

    Features consumed from observation dict:
        fundamental_features[1] → news_impact_active
        fundamental_features[2] → time_to_next_event
        fundamental_features[3] → fundamental_bias
        fundamental_features[7] → conflicting_signals
        structure_features[2]   → structure_break_bull
        structure_features[3]   → structure_break_bear
    """

    # Feature indices within the fundamental_features array
    _IDX_NEWS_IMPACT_ACTIVE = 1
    _IDX_TIME_TO_NEXT_EVENT = 2
    _IDX_FUNDAMENTAL_BIAS = 3
    _IDX_CONFLICTING_SIGNALS = 7

    # Feature indices within the structure_features array
    _IDX_STRUCTURE_BREAK_BULL = 2
    _IDX_STRUCTURE_BREAK_BEAR = 3

    def __init__(
        self,
        news_blackout_threshold: float = 0.5,
        time_to_event_threshold: float = 0.01,  # ~15 min before event
        min_fundamental_bias: float = 0.3,
        require_structure_confirm: bool = True,
        exit_on_conflict: bool = True,
        reduce_scale_pre_news: float = 0.5,
        pre_news_time_threshold: float = 0.1,  # ~2.4h before event
        block_against_bias: bool = True,
        min_bias_for_direction: float = 0.5,
        enabled: bool = True,
    ) -> None:
        """Initialize strategy filter.

        Args:
            news_blackout_threshold: Block if news_impact_active >= this value.
            time_to_event_threshold: Block if time_to_next_event < this (normalized).
            min_fundamental_bias: Minimum |fundamental_bias| required for new entry.
            require_structure_confirm: Require structure break for entry.
            exit_on_conflict: Force close when conflicting_signals = 1.
            reduce_scale_pre_news: Scale factor when approaching news.
            pre_news_time_threshold: When to start reducing (normalized).
            block_against_bias: Block entries against fundamental direction.
            min_bias_for_direction: Minimum |bias| to enforce direction alignment.
            enabled: Whether the filter is active.
        """
        self._news_blackout_threshold = news_blackout_threshold
        self._time_to_event_threshold = time_to_event_threshold
        self._min_fundamental_bias = min_fundamental_bias
        self._require_structure_confirm = require_structure_confirm
        self._exit_on_conflict = exit_on_conflict
        self._reduce_scale_pre_news = reduce_scale_pre_news
        self._pre_news_time_threshold = pre_news_time_threshold
        self._block_against_bias = block_against_bias
        self._min_bias_for_direction = min_bias_for_direction
        self._enabled = enabled

    def check(
        self,
        obs: dict[str, np.ndarray],
        proposed_action: float,
        current_position: float,
    ) -> FilterDecision:
        """Evaluate proposed action against strategy rules.

        Args:
            obs: Observation dict with "fundamental_features" and
                "structure_features" keys (numpy arrays).
            proposed_action: The RL agent's proposed action [-1, 1].
            current_position: Current position size (negative=short).

        Returns:
            FilterDecision with allowed/scale/reason/force_close.
        """
        if not self._enabled:
            return FilterDecision(allowed=True, scale=1.0, reason="", force_close=False)

        # Extract features safely
        fund = obs.get("fundamental_features")
        struct = obs.get("structure_features")

        if fund is None or struct is None:
            # No fundamental/structure data available — allow all
            return FilterDecision(
                allowed=True, scale=1.0, reason="no_filter_data", force_close=False
            )

        fund = np.asarray(fund, dtype=np.float64)
        struct = np.asarray(struct, dtype=np.float64)

        news_active = float(fund[self._IDX_NEWS_IMPACT_ACTIVE]) if len(fund) > self._IDX_NEWS_IMPACT_ACTIVE else 0.0
        time_to_event = float(fund[self._IDX_TIME_TO_NEXT_EVENT]) if len(fund) > self._IDX_TIME_TO_NEXT_EVENT else 1.0
        fundamental_bias = float(fund[self._IDX_FUNDAMENTAL_BIAS]) if len(fund) > self._IDX_FUNDAMENTAL_BIAS else 0.0
        conflicting = float(fund[self._IDX_CONFLICTING_SIGNALS]) if len(fund) > self._IDX_CONFLICTING_SIGNALS else 0.0

        break_bull = float(struct[self._IDX_STRUCTURE_BREAK_BULL]) if len(struct) > self._IDX_STRUCTURE_BREAK_BULL else 0.0
        break_bear = float(struct[self._IDX_STRUCTURE_BREAK_BEAR]) if len(struct) > self._IDX_STRUCTURE_BREAK_BEAR else 0.0

        has_position = abs(current_position) > 0.01
        is_new_entry = not has_position and abs(proposed_action) > 0.05
        is_adding = has_position and self._same_direction(proposed_action, current_position)

        # --- Rule 1: Conflicting signals → force close ---
        if self._exit_on_conflict and conflicting >= 0.5 and has_position:
            return FilterDecision(
                allowed=True,
                scale=1.0,
                reason="conflicting_signals_exit",
                force_close=True,
            )

        # --- Rule 2: News blackout → block new entries ---
        if news_active >= self._news_blackout_threshold and (is_new_entry or is_adding):
            return FilterDecision(
                allowed=False,
                scale=0.0,
                reason="news_blackout",
                force_close=False,
            )

        # --- Rule 3: Time to event too close → block new entries ---
        if time_to_event < self._time_to_event_threshold and (is_new_entry or is_adding):
            return FilterDecision(
                allowed=False,
                scale=0.0,
                reason="event_imminent",
                force_close=False,
            )

        # --- Rule 4: Minimum fundamental bias for entry ---
        if is_new_entry and abs(fundamental_bias) < self._min_fundamental_bias:
            return FilterDecision(
                allowed=False,
                scale=0.0,
                reason="insufficient_bias",
                force_close=False,
            )

        # --- Rule 5: Structure confirmation for entry ---
        if self._require_structure_confirm and is_new_entry:
            has_structure = break_bull > 0.5 or break_bear > 0.5
            if not has_structure:
                return FilterDecision(
                    allowed=False,
                    scale=0.0,
                    reason="no_structure_confirm",
                    force_close=False,
                )

            # Check alignment: bullish break → long, bearish break → short
            if break_bull > 0.5 and proposed_action < -0.05:
                return FilterDecision(
                    allowed=False,
                    scale=0.0,
                    reason="structure_direction_mismatch",
                    force_close=False,
                )
            if break_bear > 0.5 and proposed_action > 0.05:
                return FilterDecision(
                    allowed=False,
                    scale=0.0,
                    reason="structure_direction_mismatch",
                    force_close=False,
                )

        # --- Rule 6: Direction alignment with fundamental bias ---
        if (
            self._block_against_bias
            and is_new_entry
            and abs(fundamental_bias) >= self._min_bias_for_direction
        ):
            bias_direction = 1 if fundamental_bias > 0 else -1
            action_direction = 1 if proposed_action > 0 else -1
            if bias_direction != action_direction:
                return FilterDecision(
                    allowed=False,
                    scale=0.0,
                    reason="against_fundamental_bias",
                    force_close=False,
                )

        # --- Rule 7: Pre-news position scaling ---
        if time_to_event < self._pre_news_time_threshold and has_position:
            return FilterDecision(
                allowed=True,
                scale=self._reduce_scale_pre_news,
                reason="pre_news_reduction",
                force_close=False,
            )

        # All checks passed
        return FilterDecision(allowed=True, scale=1.0, reason="", force_close=False)

    @staticmethod
    def _same_direction(action: float, position: float) -> bool:
        """Check if action is adding to current position direction."""
        if position > 0:
            return action > 0.05
        elif position < 0:
            return action < -0.05
        return False
