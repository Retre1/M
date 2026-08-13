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

import numpy as np
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


@dataclass
class FoldComparison:
    """A candidate scored against the baselines across many market segments.

    One backtest number is a single draw. What decides whether an edge exists is
    the *distribution* across segments and whether it survives the fact that
    many configurations were tried.
    """

    candidate_name: str
    fold_sharpe: pd.DataFrame        # index = fold, columns = strategy
    fold_returns: dict[str, np.ndarray]
    gate_win_rate: float = 0.8       # the plan's threshold for gate 2

    @property
    def baseline_names(self) -> list[str]:
        """Baselines the candidate must clear — random is a cost probe, not a rival."""
        return [c for c in self.fold_sharpe.columns
                if c not in (self.candidate_name, "random")]

    @property
    def win_rate(self) -> float:
        """Share of folds where the candidate out-Sharpes every baseline."""
        # No folds means no evidence, which is not the same as a clean sweep.
        # Every fold can be skipped when the segments come out shorter than the
        # feature warmup, and reporting 1.0 there would pass the gate on a run
        # that never happened.
        if len(self.fold_sharpe) == 0:
            return 0.0
        if not self.baseline_names:
            return 1.0
        candidate = self.fold_sharpe[self.candidate_name]
        best = self.fold_sharpe[self.baseline_names].max(axis=1)
        return float((candidate > best).mean())

    @property
    def passes_gate(self) -> bool:
        return self.win_rate >= self.gate_win_rate

    def probability_of_overfitting(self, n_splits: int = 8) -> float | None:
        """PBO across the strategy set, or None when there is too little data."""
        from apexfx.backtest.validation import probability_of_backtest_overfitting

        columns = list(self.fold_returns)
        # Checked before the length, or min() over an empty sequence raises.
        if len(columns) < 2:
            return None
        length = min(len(self.fold_returns[c]) for c in columns)
        if length < n_splits:
            return None
        matrix = np.column_stack([self.fold_returns[c][:length] for c in columns])
        return probability_of_backtest_overfitting(matrix, n_splits=n_splits)

    def summary(self) -> str:
        stats = self.fold_sharpe.agg(["median", "min", "max"]).T.round(3)
        lines = [
            f"{self.candidate_name} beat every baseline on "
            f"{self.win_rate:.0%} of {len(self.fold_sharpe)} folds "
            f"({'PASSES' if self.passes_gate else 'FAILS'} the "
            f"{self.gate_win_rate:.0%} gate)",
            "",
            "Sharpe by strategy across folds:",
            stats.to_string(),
        ]
        pbo = self.probability_of_overfitting()
        if pbo is not None:
            lines += ["", f"PBO {pbo:.3f} "
                          f"({'acceptable' if pbo < 0.5 else 'SELECTING NOISE'})"]
        return "\n".join(lines)


def compare_across_folds(
    bars: pd.DataFrame,
    candidate,
    *,
    n_folds: int = 9,
    config: BacktestConfig | None = None,
    candidate_name: str = "model",
    baselines: list | None = None,
    pipeline=None,
    risk_config=None,
) -> FoldComparison:
    """Score the candidate and baselines on each of *n_folds* market segments.

    One backtest answers "did it work on this stretch of history". Gate 2 asks
    a different question — "does it beat the alternatives across conditions" —
    and that needs the spread, not a point.

    Each distinct fold is backtested once per strategy. Evaluating every CPCV
    combination directly would repeat the same fold many times: with 9 folds
    and 3 test folds that is 84 splits x 3 x 7 strategies = 1764 runs, against
    63 for the same information. The combinatorial structure matters for
    *recombining* paths, not for how often a segment must be replayed.

    Args:
        bars: Full OHLCV history; split into contiguous segments.
        candidate: Object with ``on_bar(features, bar) -> float``.
        n_folds: Number of segments. More folds means more evidence per
            strategy but less history in each.

    Returns:
        Per-fold Sharpe for every strategy, plus the per-fold return series
        needed for PBO.
    """
    config = config or BacktestConfig()
    baselines = default_baselines() if baselines is None else baselines
    strategies = [(candidate_name, candidate)] + [
        (getattr(b, "name", type(b).__name__), b) for b in baselines
    ]

    folds = np.array_split(np.arange(len(bars)), n_folds)
    sharpe_rows: list[dict[str, float]] = []
    returns: dict[str, list[np.ndarray]] = {name: [] for name, _ in strategies}

    for fold_id, index in enumerate(folds):
        segment = bars.iloc[index].reset_index(drop=True)
        if len(segment) <= config.warmup_bars:
            logger.warning(
                "Fold shorter than the warmup — skipped",
                fold=fold_id, bars=len(segment), warmup=config.warmup_bars,
            )
            continue

        row: dict[str, float] = {}
        for name, strategy in strategies:
            if hasattr(strategy, "reset"):
                strategy.reset()
            engine = BacktestEngine(
                bars=segment, strategy=strategy, config=config,
                pipeline=pipeline, risk_config=risk_config,
            )
            result = engine.run()
            metrics = result.metrics or result.compute_metrics()
            row[name] = float(metrics.get("sharpe_ratio", 0.0))
            returns[name].append(np.asarray(result.returns_series, dtype=np.float64))
        sharpe_rows.append(row)

    # Columns are named even when every fold was skipped, so the frame keeps its
    # shape and callers do not have to special-case the empty result.
    fold_sharpe = pd.DataFrame(sharpe_rows, columns=[name for name, _ in strategies])
    fold_sharpe.index.name = "fold"

    comparison = FoldComparison(
        candidate_name=candidate_name,
        fold_sharpe=fold_sharpe,
        fold_returns={
            name: np.concatenate(chunks) if chunks else np.zeros(0)
            for name, chunks in returns.items()
        },
    )

    logger.info(
        "Fold comparison complete",
        candidate=candidate_name,
        n_folds=len(fold_sharpe),
        win_rate=round(comparison.win_rate, 3),
        passes_gate=comparison.passes_gate,
    )
    return comparison
