"""Score a strategy against the baselines it has to beat.

The audit's finding was not "the model is weak" but "the model loses to buy &
hold, and nobody was checking". A Sharpe reported on its own cannot show that.
This module runs the candidate and every baseline over the same bars, through
the same engine and cost model, and answers one question: did it win?

The gate follows the audit's rule — a strategy has no demonstrated edge unless
it beats the best baseline. Across walk-forward windows the threshold is 80% of
windows, which is the caller's job to apply; here each window is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from apexfx.backtest.baselines import default_baselines
from apexfx.backtest.engine import BacktestConfig, BacktestEngine
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyScore:
    """One strategy's result over one set of bars."""

    name: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    n_trades: int
    avg_exposure_pct: float = 0.0
    annual_volatility_pct: float = 0.0

    @classmethod
    def from_metrics(cls, name: str, metrics: dict[str, float]) -> StrategyScore:
        return cls(
            name=name,
            total_return_pct=float(metrics.get("total_return_pct", 0.0)),
            sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
            max_drawdown_pct=float(metrics.get("max_drawdown_pct", 0.0)),
            profit_factor=float(metrics.get("profit_factor", 0.0)),
            # BacktestResult names this "total_trades"; "n_trades" silently
            # read 0 and made every comparison look like nobody traded.
            n_trades=int(metrics.get("total_trades", 0)),
            # Carried because a comparison between strategies that never took a
            # position is not a comparison. Run 5 traded 165 times at ~$50
            # notional on $100k and its result was indistinguishable from noise.
            avg_exposure_pct=float(metrics.get("avg_exposure_pct", 0.0)),
            annual_volatility_pct=float(metrics.get("annual_volatility_pct", 0.0)),
        )


@dataclass
class ComparisonResult:
    """A candidate scored against the baselines over one window."""

    candidate: StrategyScore
    baselines: list[StrategyScore] = field(default_factory=list)

    @property
    def best_baseline(self) -> StrategyScore | None:
        """Highest-Sharpe baseline, excluding the random calibration probe."""
        contenders = [b for b in self.baselines if b.name != "random"]
        if not contenders:
            return None
        return max(contenders, key=lambda s: s.sharpe_ratio)

    @property
    def beats_best_baseline(self) -> bool:
        """Did the candidate out-Sharpe every baseline?

        Sharpe rather than raw return, so a candidate cannot win by taking on
        more risk for the same money.
        """
        best = self.best_baseline
        if best is None:
            return True
        return self.candidate.sharpe_ratio > best.sharpe_ratio

    @property
    def costs_look_charged(self) -> bool:
        """Sanity check on the cost model, not on the candidate.

        Random trading must lose money. If it comes out flat or positive the
        spread and commission are not being applied, and every other number in
        the comparison is inflated.
        """
        for score in self.baselines:
            if score.name == "random":
                return score.total_return_pct < 0
        return True

    @property
    def exposure_is_meaningful(self) -> bool:
        """Did the candidate take a position large enough to measure?

        A run at negligible size cannot demonstrate edge or the lack of it, so
        the verdict above is not evidence either way. Run 5 is the cautionary
        case: 165 trades, $50 notional on $100k, a -$4.77 result that says
        nothing. 0.5% average exposure is already a very low bar.
        """
        return self.candidate.avg_exposure_pct >= 0.5

    def to_frame(self) -> pd.DataFrame:
        rows = [self.candidate, *self.baselines]
        return pd.DataFrame([
            {
                "strategy": s.name,
                "return_pct": round(s.total_return_pct, 2),
                "sharpe": round(s.sharpe_ratio, 3),
                "max_dd_pct": round(s.max_drawdown_pct, 2),
                "profit_factor": round(s.profit_factor, 3),
                "n_trades": s.n_trades,
                "avg_exposure_pct": round(s.avg_exposure_pct, 3),
                "annual_vol_pct": round(s.annual_volatility_pct, 3),
            }
            for s in rows
        ])

    def summary(self) -> str:
        best = self.best_baseline
        verdict = "BEATS" if self.beats_best_baseline else "LOSES TO"
        against = best.name if best else "nothing to compare"
        lines = [
            f"{self.candidate.name} {verdict} the best baseline ({against})",
            self.to_frame().to_string(index=False),
        ]
        if not self.exposure_is_meaningful:
            lines.append(
                f"WARNING: candidate average exposure is "
                f"{self.candidate.avg_exposure_pct:.3f}% of equity — too small "
                f"for the verdict above to be evidence of anything",
            )
        if not self.costs_look_charged:
            lines.append(
                "WARNING: random trading did not lose money — check the cost "
                "model before trusting any row above",
            )
        return "\n".join(lines)


def compare_against_baselines(
    bars: pd.DataFrame,
    candidate,
    *,
    config: BacktestConfig | None = None,
    candidate_name: str = "model",
    baselines: list | None = None,
    pipeline=None,
    risk_config=None,
) -> ComparisonResult:
    """Run *candidate* and the baselines over the same bars.

    Every strategy gets its own engine instance but the identical bars, config,
    feature pipeline and risk settings — comparing against a baseline scored
    under different costs would prove nothing.

    Args:
        bars: OHLCV frame.
        candidate: Object with ``on_bar(features, bar) -> float``.
        config: Backtest configuration; defaults apply if omitted.
        candidate_name: Label for the candidate in the output.
        baselines: Overrides the default comparison set.
        pipeline: Feature pipeline handed to every engine.
        risk_config: Risk settings handed to every engine.

    Returns:
        The candidate's score alongside each baseline's.
    """
    config = config or BacktestConfig()
    baselines = default_baselines() if baselines is None else baselines

    def _score(strategy, name: str) -> StrategyScore:
        if hasattr(strategy, "reset"):
            strategy.reset()
        engine = BacktestEngine(
            bars=bars,
            strategy=strategy,
            config=config,
            pipeline=pipeline,
            risk_config=risk_config,
        )
        result = engine.run()
        metrics = result.metrics or result.compute_metrics()
        return StrategyScore.from_metrics(name, metrics)

    candidate_score = _score(candidate, candidate_name)
    baseline_scores = [
        _score(b, getattr(b, "name", type(b).__name__)) for b in baselines
    ]

    comparison = ComparisonResult(candidate=candidate_score, baselines=baseline_scores)

    logger.info(
        "Baseline comparison complete",
        candidate=candidate_name,
        candidate_sharpe=round(candidate_score.sharpe_ratio, 3),
        best_baseline=comparison.best_baseline.name if comparison.best_baseline else None,
        beats_baseline=comparison.beats_best_baseline,
        costs_charged=comparison.costs_look_charged,
    )
    return comparison
