"""NLP-based news sentiment features using FinBERT or lightweight fallback.

News sentiment drives 30-40% of high-impact FX moves. This extractor:
- Uses FinBERT (ProsusAI/finbert) for financial sentiment if available
- Falls back to keyword-based scoring when transformers not installed
- Time-decays old headlines (recent news matters more)
- Tracks sentiment momentum and divergence
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# Keyword-based fallback sentiment
_BULLISH_KEYWORDS = frozenset({
    "surge", "rally", "gain", "rise", "jump", "soar", "boom", "recovery",
    "hawkish", "upgrade", "beat", "strong", "exceed", "growth", "expand",
    "optimism", "bullish", "positive", "upbeat", "robust",
})
_BEARISH_KEYWORDS = frozenset({
    "crash", "plunge", "drop", "fall", "decline", "slump", "crisis", "recession",
    "dovish", "downgrade", "miss", "weak", "disappointing", "contraction", "loss",
    "pessimism", "bearish", "negative", "concern", "risk",
})


class SentimentExtractor(BaseFeatureExtractor):
    """NLP-based news sentiment features.

    6 features capturing market sentiment from news headlines:
    - sentiment_score: aggregated [-1, 1] bearish to bullish
    - sentiment_magnitude: confidence/intensity
    - sentiment_divergence: disagreement between headlines
    - sentiment_momentum: change in sentiment over time
    - headline_count: number of recent headlines (activity level)
    - sentiment_surprise: deviation from rolling average
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_headlines: int = 20,
        decay_hours: float = 4.0,
        use_nlp: bool | None = None,
    ) -> None:
        """Initialize sentiment extractor.

        Args:
            model_name: HuggingFace model for sentiment (only if transformers available)
            max_headlines: Maximum headlines to keep in buffer
            decay_hours: Half-life for time decay of headline scores
            use_nlp: Force NLP on/off. None = auto-detect transformers availability
        """
        self._model_name = model_name
        self._max_headlines = max_headlines
        self._decay_hours = decay_hours
        self._headlines: list[dict] = []  # {"text", "timestamp", "score", "magnitude"}
        self._sentiment_history: list[float] = []
        self._nlp_pipeline = None

        # Auto-detect NLP availability
        if use_nlp is None:
            try:
                import transformers  # noqa: F401
                use_nlp = True
            except ImportError:
                use_nlp = False

        self._use_nlp = use_nlp

        if self._use_nlp:
            try:
                from transformers import pipeline as hf_pipeline
                self._nlp_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT sentiment pipeline loaded", model=model_name)
            except Exception as e:
                logger.warning(
                    "Failed to load FinBERT, falling back to keyword-based",
                    error=str(e),
                )
                self._use_nlp = False

    @property
    def feature_names(self) -> list[str]:
        return [
            "sentiment_score",
            "sentiment_magnitude",
            "sentiment_divergence",
            "sentiment_momentum",
            "headline_count",
            "sentiment_surprise",
        ]

    def update_headlines(self, headlines: list[dict]) -> None:
        """Feed new headlines for sentiment analysis.

        Args:
            headlines: List of {"text": str, "timestamp": datetime | str, "source": str}
        """
        for h in headlines:
            text = h.get("text", "")
            ts = h.get("timestamp", datetime.now(timezone.utc))
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = datetime.now(timezone.utc)

            score, magnitude = self._score_headline(text)

            self._headlines.append({
                "text": text,
                "timestamp": ts,
                "score": score,
                "magnitude": magnitude,
            })

        # Keep only recent headlines
        if len(self._headlines) > self._max_headlines:
            self._headlines = self._headlines[-self._max_headlines:]

    def _score_headline(self, text: str) -> tuple[float, float]:
        """Score a single headline. Returns (score, magnitude).

        score: [-1, 1] where -1 is bearish, +1 is bullish
        magnitude: [0, 1] confidence/intensity
        """
        if self._use_nlp and self._nlp_pipeline is not None:
            return self._score_nlp(text)
        return self._score_keywords(text)

    def _score_nlp(self, text: str) -> tuple[float, float]:
        """Score using FinBERT NLP pipeline."""
        try:
            result = self._nlp_pipeline(text)[0]
            label = result["label"].lower()
            confidence = result["score"]

            if label == "positive":
                return confidence, confidence
            elif label == "negative":
                return -confidence, confidence
            else:  # neutral
                return 0.0, confidence * 0.3
        except Exception:
            return 0.0, 0.0

    def _score_keywords(self, text: str) -> tuple[float, float]:
        """Score using keyword matching (fallback)."""
        words = set(text.lower().split())

        bull_count = len(words & _BULLISH_KEYWORDS)
        bear_count = len(words & _BEARISH_KEYWORDS)
        total = bull_count + bear_count

        if total == 0:
            return 0.0, 0.0

        score = (bull_count - bear_count) / total
        magnitude = min(total / 5.0, 1.0)  # Normalize to [0, 1]

        return score, magnitude

    def _compute_features(self) -> np.ndarray:
        """Compute 6 sentiment features from current headline buffer."""
        if not self._headlines:
            return np.zeros(6)

        now = datetime.now(timezone.utc)

        # Time-decay weighted scores
        scores = []
        magnitudes = []
        weights = []

        for h in self._headlines:
            ts = h["timestamp"]
            if not isinstance(ts, datetime):
                ts = datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            hours_ago = (now - ts).total_seconds() / 3600.0
            weight = math.exp(-0.693 * hours_ago / self._decay_hours)  # Half-life decay

            scores.append(h["score"])
            magnitudes.append(h["magnitude"])
            weights.append(weight)

        scores_arr = np.array(scores)
        mags_arr = np.array(magnitudes)
        weights_arr = np.array(weights)
        total_weight = np.sum(weights_arr) + 1e-10

        # 1. Weighted sentiment score
        sentiment_score = float(np.sum(scores_arr * weights_arr) / total_weight)

        # 2. Weighted magnitude
        sentiment_magnitude = float(np.sum(mags_arr * weights_arr) / total_weight)

        # 3. Divergence (std of individual scores)
        sentiment_divergence = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0

        # 4. Momentum (change from previous computation)
        self._sentiment_history.append(sentiment_score)
        if len(self._sentiment_history) > 100:
            self._sentiment_history = self._sentiment_history[-50:]

        if len(self._sentiment_history) >= 2:
            sentiment_momentum = sentiment_score - self._sentiment_history[-2]
        else:
            sentiment_momentum = 0.0

        # 5. Headline count (activity level, normalized)
        headline_count = min(len(self._headlines) / self._max_headlines, 1.0)

        # 6. Surprise (deviation from rolling average)
        if len(self._sentiment_history) >= 5:
            rolling_avg = np.mean(self._sentiment_history[-5:])
            sentiment_surprise = sentiment_score - rolling_avg
        else:
            sentiment_surprise = 0.0

        return np.array([
            sentiment_score,
            sentiment_magnitude,
            sentiment_divergence,
            sentiment_momentum,
            headline_count,
            sentiment_surprise,
        ], dtype=np.float32)

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """For backtesting: returns neutral features (no historical headlines).

        In backtesting mode, all sentiment features are 0.0 since we don't
        have historical headline data synchronized to bar timestamps.
        For live mode, call update_headlines() then extract_live().
        """
        n = len(data)
        result = pd.DataFrame(index=data.index)

        for name in self.feature_names:
            result[name] = 0.0

        return result

    def extract_live(self) -> np.ndarray:
        """For live trading: compute features from buffered headlines."""
        return self._compute_features()

    def clear_headlines(self) -> None:
        """Clear all buffered headlines."""
        self._headlines.clear()
