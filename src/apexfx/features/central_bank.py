"""Central bank speech and statement analysis.

Parses FOMC statements, ECB press conferences, and other central bank
communications for hawkish/dovish scoring. Central bank language shifts
are the single most powerful FX signal.

Key signals:
- Hawkish/dovish tone shift (rate expectations)
- Forward guidance changes
- Quantitative easing/tightening signals
- Inflation concern level
- Employment focus vs inflation focus
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Hawkish keywords — signals tighter monetary policy (currency bullish)
_HAWKISH_PHRASES = [
    "raise rates", "rate hike", "tighten", "tightening", "restrictive",
    "inflation concern", "price stability", "overheating", "above target",
    "further increases", "additional firming", "vigilant on inflation",
    "strong labor market", "robust growth", "higher for longer",
    "reduce balance sheet", "quantitative tightening", "elevated inflation",
    "upside risks to inflation", "sufficiently restrictive",
    "data dependent", "measured pace",
]

# Dovish keywords — signals looser monetary policy (currency bearish)
_DOVISH_PHRASES = [
    "cut rates", "rate cut", "ease", "easing", "accommodative",
    "below target", "downside risks", "weakness", "slowdown",
    "support growth", "further easing", "lower rates",
    "unemployment concern", "labor market softening",
    "quantitative easing", "asset purchases",
    "patient", "gradual", "cautious", "well anchored",
    "balanced risks", "uncertain outlook",
]

# Emphasis multipliers for certain phrases
_HIGH_IMPACT_HAWKISH = frozenset({
    "raise rates", "rate hike", "restrictive", "higher for longer",
    "quantitative tightening", "sufficiently restrictive",
})
_HIGH_IMPACT_DOVISH = frozenset({
    "cut rates", "rate cut", "accommodative", "quantitative easing",
    "asset purchases", "further easing",
})


@dataclass
class CentralBankStatement:
    """A parsed central bank communication."""
    text: str
    source: str  # "fomc", "ecb", "boe", "boj", "rba", "rbnz"
    timestamp: datetime
    hawkish_score: float = 0.0
    dovish_score: float = 0.0
    net_score: float = 0.0  # hawkish - dovish, [-1, 1]
    key_phrases: list[str] = field(default_factory=list)


class CentralBankAnalyzer:
    """Analyzes central bank statements for hawkish/dovish tone.

    Maintains a rolling buffer of recent statements and computes
    features that capture the current monetary policy stance.
    """

    def __init__(
        self,
        max_statements: int = 20,
        decay_days: float = 30.0,
    ) -> None:
        self._max_statements = max_statements
        self._decay_days = decay_days
        self._statements: list[CentralBankStatement] = []

    def analyze_text(self, text: str, source: str) -> CentralBankStatement:
        """Score a central bank statement for hawkish/dovish tone."""
        text_lower = text.lower()

        hawkish_count = 0
        dovish_count = 0
        key_phrases: list[str] = []

        for phrase in _HAWKISH_PHRASES:
            matches = len(re.findall(re.escape(phrase), text_lower))
            if matches > 0:
                weight = 2.0 if phrase in _HIGH_IMPACT_HAWKISH else 1.0
                hawkish_count += matches * weight
                if matches > 0:
                    key_phrases.append(f"+{phrase}")

        for phrase in _DOVISH_PHRASES:
            matches = len(re.findall(re.escape(phrase), text_lower))
            if matches > 0:
                weight = 2.0 if phrase in _HIGH_IMPACT_DOVISH else 1.0
                dovish_count += matches * weight
                if matches > 0:
                    key_phrases.append(f"-{phrase}")

        total = hawkish_count + dovish_count
        if total > 0:
            hawkish_score = hawkish_count / total
            dovish_score = dovish_count / total
            net_score = hawkish_score - dovish_score
        else:
            hawkish_score = 0.0
            dovish_score = 0.0
            net_score = 0.0

        stmt = CentralBankStatement(
            text=text[:500],  # Truncate for storage
            source=source,
            timestamp=datetime.now(timezone.utc),
            hawkish_score=hawkish_score,
            dovish_score=dovish_score,
            net_score=net_score,
            key_phrases=key_phrases[:10],
        )

        self._statements.append(stmt)
        if len(self._statements) > self._max_statements:
            self._statements = self._statements[-self._max_statements:]

        return stmt

    def get_current_stance(self, bank: str | None = None) -> dict[str, float]:
        """Get current monetary policy stance with time-decay weighting.

        Returns dict with:
        - stance: net hawkish/dovish score [-1, 1]
        - momentum: change in stance
        - conviction: how many recent statements
        """
        now = datetime.now(timezone.utc)
        filtered = self._statements
        if bank:
            filtered = [s for s in filtered if s.source == bank]

        if not filtered:
            return {"stance": 0.0, "momentum": 0.0, "conviction": 0.0}

        scores = []
        weights = []
        for s in filtered:
            days_ago = (now - s.timestamp).total_seconds() / 86400
            weight = np.exp(-0.693 * days_ago / self._decay_days)
            scores.append(s.net_score)
            weights.append(weight)

        scores_arr = np.array(scores)
        weights_arr = np.array(weights)
        total_weight = np.sum(weights_arr) + 1e-10

        stance = float(np.sum(scores_arr * weights_arr) / total_weight)

        # Momentum: compare last 2 vs previous 2
        if len(scores) >= 4:
            recent = np.mean(scores[-2:])
            previous = np.mean(scores[-4:-2])
            momentum = recent - previous
        elif len(scores) >= 2:
            momentum = scores[-1] - scores[-2]
        else:
            momentum = 0.0

        conviction = min(len(filtered) / 5.0, 1.0)

        return {
            "stance": np.clip(stance, -1, 1),
            "momentum": np.clip(momentum, -1, 1),
            "conviction": conviction,
        }


class CentralBankExtractor(BaseFeatureExtractor):
    """Feature extractor for central bank monetary policy signals.

    6 features:
    - cb_hawkish_score: Current hawkish stance [0, 1]
    - cb_dovish_score: Current dovish stance [0, 1]
    - cb_net_stance: Net hawkish-dovish [-1, 1]
    - cb_stance_momentum: Change in stance
    - cb_conviction: Number of recent statements (normalized)
    - cb_divergence: Base vs quote central bank stance difference
    """

    def __init__(
        self,
        base_bank: str = "ecb",
        quote_bank: str = "fomc",
    ) -> None:
        self._base_bank = base_bank
        self._quote_bank = quote_bank
        self._analyzer = CentralBankAnalyzer()

    @property
    def feature_names(self) -> list[str]:
        return [
            "cb_hawkish_score",
            "cb_dovish_score",
            "cb_net_stance",
            "cb_stance_momentum",
            "cb_conviction",
            "cb_divergence",
        ]

    @property
    def analyzer(self) -> CentralBankAnalyzer:
        return self._analyzer

    def update_statement(self, text: str, source: str) -> CentralBankStatement:
        """Feed a new central bank statement."""
        return self._analyzer.analyze_text(text, source)

    def extract(
        self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Extract central bank features.

        In backtesting, returns zeros (no historical statement data).
        For live, call update_statement() then extract().
        """
        n = len(bars)
        result = pd.DataFrame(index=bars.index)

        # Get current stances
        base_stance = self._analyzer.get_current_stance(self._base_bank)
        quote_stance = self._analyzer.get_current_stance(self._quote_bank)

        # If no data, return zeros
        if base_stance["conviction"] == 0 and quote_stance["conviction"] == 0:
            for name in self.feature_names:
                result[name] = 0.0
            return result

        # Combine: for EUR/USD, hawkish ECB = EUR bullish, hawkish Fed = EUR bearish
        combined_stance = base_stance["stance"] - quote_stance["stance"]

        result["cb_hawkish_score"] = max(0, combined_stance)
        result["cb_dovish_score"] = max(0, -combined_stance)
        result["cb_net_stance"] = np.clip(combined_stance, -1, 1)
        result["cb_stance_momentum"] = base_stance["momentum"] - quote_stance["momentum"]
        result["cb_conviction"] = max(base_stance["conviction"], quote_stance["conviction"])
        result["cb_divergence"] = base_stance["stance"] - quote_stance["stance"]

        return result
