"""Score a strategy against the baselines it has to beat.

The audit's finding was not "the model is weak" but "the model loses to buy &
hold, and nobody was checking". A Sharpe reported on its own cannot show that.
This module runs the candidate and every baseline over the same bars, through
the same engine and cost model, and answers one question: did it win?

The gate follows the audit's rule — a strategy has no demonstrated edge unless
it beats the best baseline. Across segments that judgement is made by
``FoldComparison``, on the per-trade expectancy in R plus a consistency check,
rather than on a fixed share of winning windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from apexfx.backtest.baselines import default_baselines
from apexfx.backtest.engine import BacktestConfig, BacktestEngine
from apexfx.backtest.result import trade_r_multiples
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


# A fold with fewer trades than this cannot support a verdict of its own. The
# audit's runs closed 165-174 trades in total, which is ~18 per fold across 9
# folds — far too few to tell a profit factor of 0.95 from 1.15.
MIN_TRADES_PER_FOLD = 30


@dataclass
class FoldComparison:
    """A candidate scored against the baselines across many market segments.

    One backtest number is a single draw. What decides whether an edge exists
    is the distribution across segments, and whether the per-trade expectancy
    is distinguishable from zero at the sample size actually available.

    **Why R and not profit factor.** PF compresses a trade set into a ratio of
    sums, which has no usable sampling distribution and hides the sample size
    completely: PF 1.15 on 18 trades and on 1800 trades read identically. R —
    P&L divided by the money risked at entry — is a per-trade quantity, so the
    mean has a standard error and the question "could this be zero?" becomes
    answerable.

    **What the win rate is and is not.** It stays as a consistency check, with
    an exact binomial p-value instead of a fixed threshold. Treat that p-value
    as optimistic: the folds are contiguous slices of one price series, so a
    market regime spanning two folds makes their outcomes correlated, and the
    effective number of independent observations is smaller than the fold
    count. It is a secondary read; the t-statistic on R is the primary one.
    """

    candidate_name: str
    fold_sharpe: pd.DataFrame        # index = fold, columns = strategy
    fold_returns: dict[str, np.ndarray]
    fold_trade_r: dict[str, list[np.ndarray]] = field(default_factory=dict)
    min_trades_per_fold: int = MIN_TRADES_PER_FOLD
    alpha: float = 0.05

    # -- which folds carry evidence ---------------------------------------

    @property
    def candidate_trades_by_fold(self) -> list[np.ndarray]:
        return self.fold_trade_r.get(self.candidate_name, [])

    @property
    def evaluable_folds(self) -> list[int]:
        """Folds where the candidate traded enough to say anything."""
        return [
            i for i, r in enumerate(self.candidate_trades_by_fold)
            if len(r) >= self.min_trades_per_fold
        ]

    @property
    def underpowered_folds(self) -> list[int]:
        """Folds excluded for too few trades — reported, never scored as losses."""
        return [
            i for i, r in enumerate(self.candidate_trades_by_fold)
            if len(r) < self.min_trades_per_fold
        ]

    @property
    def candidate_r(self) -> np.ndarray:
        """Every candidate trade from the folds that carry evidence."""
        folds = self.evaluable_folds
        if not folds:
            return np.zeros(0)
        return np.concatenate([self.candidate_trades_by_fold[i] for i in folds])

    # -- primary evidence: is mean R above zero ---------------------------

    @property
    def n_trades(self) -> int:
        return int(len(self.candidate_r))

    @property
    def mean_r(self) -> float:
        r = self.candidate_r
        return float(np.mean(r)) if len(r) else 0.0

    @property
    def std_r(self) -> float:
        r = self.candidate_r
        return float(np.std(r, ddof=1)) if len(r) > 1 else 0.0

    @property
    def t_statistic(self) -> float | None:
        """t of mean R against zero. None when the sample cannot support one."""
        n, sd = self.n_trades, self.std_r
        if n < 2 or sd <= 0:
            return None
        return float(self.mean_r / (sd / math.sqrt(n)))

    @property
    def p_value(self) -> float | None:
        """One-sided p for mean R > 0."""
        t = self.t_statistic
        if t is None:
            return None
        from scipy.stats import t as student_t

        return float(student_t.sf(t, df=self.n_trades - 1))

    def minimum_detectable_r(self, power: float = 0.8) -> float | None:
        """Smallest true mean R this sample could detect, at *power*.

        The number that answers "do we even have the evidence to conclude
        anything". If it comes out above the edge being claimed, a null result
        says nothing about the strategy — only about the sample size.
        """
        n, sd = self.n_trades, self.std_r
        if n < 2 or sd <= 0:
            return None
        from scipy.stats import norm

        z = norm.ppf(1.0 - self.alpha) + norm.ppf(power)
        return float(z * sd / math.sqrt(n))

    # -- secondary evidence: consistency against the baselines ------------

    @property
    def baseline_names(self) -> list[str]:
        """Baselines the candidate must clear — random is a cost probe, not a rival."""
        return [c for c in self.fold_sharpe.columns
                if c not in (self.candidate_name, "random")]

    @property
    def wins(self) -> int:
        """Evaluable folds where the candidate out-Sharped every baseline."""
        folds = self.evaluable_folds
        if not folds or not self.baseline_names:
            return 0
        rows = self.fold_sharpe.iloc[folds]
        return int((rows[self.candidate_name] > rows[self.baseline_names].max(axis=1)).sum())

    @property
    def win_rate(self) -> float:
        folds = self.evaluable_folds
        if not folds:
            return 0.0
        if not self.baseline_names:
            return 1.0
        return self.wins / len(folds)

    @property
    def win_rate_p_value(self) -> float | None:
        """Exact binomial p for beating the baselines more often than a coin.

        Replaces the old fixed 80% line, which was asserted rather than
        derived. At 9 folds this bar is 8 wins (p = 0.020); 7 of 9 gives
        p = 0.090 and does not clear alpha — so the old threshold was in fact
        slightly lenient, as well as arbitrary.
        """
        n = len(self.evaluable_folds)
        if n == 0 or not self.baseline_names:
            return None
        from scipy.stats import binomtest

        return float(binomtest(self.wins, n, 0.5, alternative="greater").pvalue)

    # -- verdict -----------------------------------------------------------

    @property
    def passes_gate(self) -> bool:
        """Positive expectancy AND consistency, both at *alpha*.

        Two hurdles rather than one: a strategy can show a significant mean R
        while still losing to buy & hold on most segments, and it can win most
        segments on noise while making no money per trade.
        """
        if not self.evaluable_folds:
            return False
        p = self.p_value
        if p is None or p >= self.alpha:
            return False
        wp = self.win_rate_p_value
        return wp is not None and wp < self.alpha

    def probability_of_overfitting(self, n_splits: int = 8) -> float | None:
        """PBO across the strategy set, or None when there is too little data.

        Read this with care. PBO measures overfitting *by selection*, so it is
        meaningful over the set of configurations a search actually tried. Run
        against a fixed set of pre-specified baselines it answers a narrower
        question — whether the in-sample best holds up out of sample — and it
        is close to vacuous for a single candidate. It becomes the intended
        statistic only when the columns are the trials of a real search.
        """
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
        n_folds = len(self.fold_sharpe)
        lines = [
            f"{self.candidate_name}: "
            f"{'PASSES' if self.passes_gate else 'FAILS'} the gate",
            "",
        ]

        if not self.evaluable_folds:
            lines.append(
                f"No fold reached {self.min_trades_per_fold} trades "
                f"({n_folds} folds run) — there is no evidence here either way.",
            )
            # PBO is computed from bar returns and does not depend on the trade
            # floor, so it survives a run that produced no verdict. Reported
            # here rather than suppressed: it is a diagnostic on the return
            # series, not a claim about the candidate.
            pbo = self.probability_of_overfitting()
            if pbo is not None:
                lines += ["", f"PBO {pbo:.3f} over the baseline set "
                              f"({'acceptable' if pbo < 0.5 else 'SELECTING NOISE'}) "
                              f"— not a search-overfitting number, see docstring"]
            return "\n".join(lines)

        t, p = self.t_statistic, self.p_value
        mde = self.minimum_detectable_r()
        lines += [
            f"Expectancy   mean R {self.mean_r:+.4f} over {self.n_trades} trades "
            f"in {len(self.evaluable_folds)} of {n_folds} folds",
            f"             t = {t:.2f}, one-sided p = {p:.4f}"
            if t is not None else "             t undefined (sample too small)",
        ]
        if mde is not None:
            lines.append(
                f"             smallest detectable mean R at 80% power: {mde:+.4f}",
            )
            if self.mean_r < mde and p is not None and p >= self.alpha:
                lines.append(
                    "             a null result at this size says nothing about "
                    "the strategy",
                )

        wp = self.win_rate_p_value
        lines += [
            "",
            f"Consistency  beat every baseline in {self.wins} of "
            f"{len(self.evaluable_folds)} folds ({self.win_rate:.0%})"
            + (f", binomial p = {wp:.4f}" if wp is not None else ""),
            "             folds overlap in regime, so this p-value is optimistic",
        ]

        if self.underpowered_folds:
            lines.append(
                f"             excluded for under {self.min_trades_per_fold} "
                f"trades: folds {self.underpowered_folds}",
            )

        lines += ["", "Sharpe by strategy across folds:",
                  self.fold_sharpe.agg(["median", "min", "max"]).T.round(3).to_string()]

        pbo = self.probability_of_overfitting()
        if pbo is not None:
            lines += ["", f"PBO {pbo:.3f} over the baseline set "
                          f"({'acceptable' if pbo < 0.5 else 'SELECTING NOISE'}) "
                          f"— not a search-overfitting number, see docstring"]
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
    trade_r: dict[str, list[np.ndarray]] = {name: [] for name, _ in strategies}

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
            trade_r[name].append(trade_r_multiples(result.trades))
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
        fold_trade_r=trade_r,
    )

    logger.info(
        "Fold comparison complete",
        candidate=candidate_name,
        n_folds=len(fold_sharpe),
        evaluable_folds=len(comparison.evaluable_folds),
        n_trades=comparison.n_trades,
        mean_r=round(comparison.mean_r, 4),
        p_value=comparison.p_value,
        passes_gate=comparison.passes_gate,
    )
    return comparison
