"""Is the result real, or the best of many tries?

A single walk-forward number cannot answer that. Run fifty configurations and
report the best, and you have measured your own search, not the market. This
module supplies the two things that separate a finding from a selection effect:

* **Combinatorial Purged CV** — many backtest paths instead of one, with
  purging and embargo so a training fold cannot leak into the test fold that
  follows it. The splitting itself comes from ``skfolio``; it is fiddly and
  already well tested, and reimplementing it would be exactly the mistake this
  project keeps paying for.
* **PBO and the Deflated Sharpe Ratio** — implemented here, because both are
  compact formulas from the Bailey / López de Prado papers and the maintained
  packages that carry them have gone commercial. They are covered by tests
  against their analytic behaviour.

Why it matters here specifically: the runs so far compared configurations on a
single split, and the metric used to rank them was not even a financial one.
Any conclusion drawn that way is unfalsifiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class OverfittingReport:
    """What the search itself contributed to the observed performance."""

    observed_sharpe: float
    n_trials: int
    expected_max_sharpe: float
    deflated_sharpe: float
    pbo: float | None = None

    @property
    def survives_multiple_testing(self) -> bool:
        """Would this Sharpe still be notable after accounting for the search?

        0.95 is the conventional bar: a 95% probability that the true Sharpe is
        above zero once the number of trials is priced in.
        """
        return self.deflated_sharpe >= 0.95

    def summary(self) -> str:
        lines = [
            f"observed Sharpe      {self.observed_sharpe:.3f}",
            f"trials               {self.n_trials}",
            f"expected max by luck {self.expected_max_sharpe:.3f}",
            f"deflated Sharpe      {self.deflated_sharpe:.3f}"
            f"  ({'survives' if self.survives_multiple_testing else 'DOES NOT survive'}"
            f" multiple testing)",
        ]
        if self.pbo is not None:
            verdict = "acceptable" if self.pbo < 0.5 else "OVERFIT"
            lines.append(f"PBO                  {self.pbo:.3f}  ({verdict})")
        return "\n".join(lines)


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """The Sharpe the best of *n_trials* random strategies would show anyway.

    From Bailey & López de Prado (2014). With enough tries something always
    looks good; this is how good, purely from the count of attempts. It is the
    null hypothesis a reported Sharpe has to beat.
    """
    if n_trials < 2 or trial_variance <= 0:
        return 0.0

    sigma = math.sqrt(trial_variance)
    quantile_1 = norm.ppf(1.0 - 1.0 / n_trials)
    quantile_2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - EULER_MASCHERONI) * quantile_1
                          + EULER_MASCHERONI * quantile_2))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    trial_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true Sharpe is positive, after pricing in the search.

    Args:
        observed_sharpe: The Sharpe being claimed, per observation (not annualised).
        n_observations: Length of the return series behind it.
        n_trials: How many configurations were tried before choosing this one.
        trial_variance: Variance of the Sharpes across those trials.
        skew: Skewness of the return series. Negative skew inflates a naive Sharpe.
        kurtosis: Kurtosis of the return series; 3.0 is the normal case.

    Returns:
        A probability in [0, 1]. Below 0.95 the result is not distinguishable
        from the best of a lucky search.
    """
    if n_observations < 2:
        return 0.0

    benchmark = expected_max_sharpe(n_trials, trial_variance)

    # Standard error of the Sharpe estimator under non-normal returns.
    denominator = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    if denominator <= 0:
        # Heavy negative skew can drive this non-positive; the estimate is then
        # meaningless rather than merely uncertain.
        return 0.0

    statistic = ((observed_sharpe - benchmark) * math.sqrt(n_observations - 1)
                 / math.sqrt(denominator))
    return float(norm.cdf(statistic))


def probability_of_backtest_overfitting(
    performance: np.ndarray,
    n_splits: int = 8,
) -> float:
    """Fraction of splits where the in-sample winner underperforms out of sample.

    Combinatorially Symmetric Cross-Validation (Bailey, Borwein, López de Prado
    and Zhu). The observation series is cut into ``n_splits`` blocks; every way
    of choosing half of them as in-sample is enumerated; the configuration that
    won in-sample is looked up out-of-sample. If the winner is regularly below
    median out of sample, the selection procedure is fitting noise.

    Args:
        performance: ``(n_observations, n_configurations)`` of per-period
            returns — one column per configuration that was tried.
        n_splits: Number of blocks. Must be even; C(n, n/2) combinations are
            evaluated, so 8 gives 70 and 10 gives 252.

    Returns:
        PBO in [0, 1]. Above 0.5 the search is selecting noise: the in-sample
        best is worse than a coin flip out of sample.
    """
    performance = np.asarray(performance, dtype=np.float64)
    if performance.ndim != 2:
        raise ValueError(
            f"expected a 2-D (observations, configurations) array, got {performance.shape}",
        )
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")

    n_obs, n_configs = performance.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least two configurations to choose between")
    if n_obs < n_splits:
        raise ValueError(f"need at least {n_splits} observations, got {n_obs}")

    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits = []

    for in_sample_blocks in combinations(range(n_splits), half):
        oos_blocks = [b for b in range(n_splits) if b not in in_sample_blocks]
        is_idx = np.concatenate([blocks[b] for b in in_sample_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_score = _sharpe_per_column(performance[is_idx])
        oos_score = _sharpe_per_column(performance[oos_idx])

        best = int(np.argmax(is_score))
        # Relative rank of the in-sample winner among out-of-sample results.
        rank = float(np.sum(oos_score <= oos_score[best])) / (n_configs + 1)
        rank = min(max(rank, 1e-9), 1 - 1e-9)
        logits.append(math.log(rank / (1.0 - rank)))

    logits_arr = np.asarray(logits)
    pbo = float(np.mean(logits_arr <= 0.0))

    logger.info(
        "PBO computed",
        n_configurations=n_configs,
        n_combinations=len(logits),
        pbo=round(pbo, 4),
    )
    return pbo


def _sharpe_per_column(returns: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe of each column; zero where it is undefined."""
    mean = np.nanmean(returns, axis=0)
    std = np.nanstd(returns, axis=0, ddof=1)
    out = np.zeros_like(mean)
    usable = std > 1e-12
    out[usable] = mean[usable] / std[usable]
    return out


def combinatorial_purged_splits(
    n_observations: int,
    *,
    n_folds: int = 10,
    n_test_folds: int = 2,
    purged_size: int = 0,
    embargo_size: int = 0,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Splits from ``skfolio``'s Combinatorial Purged CV.

    Thin wrapper so callers do not build the placeholder frames skfolio's
    sklearn-style API expects, and so the dependency sits in one place.

    ``purged_size`` drops observations adjacent to the test fold from training,
    ``embargo_size`` drops those immediately after it. Both exist because
    financial features overlap in time: without them a training row can carry
    information about the test period and the split silently leaks.

    Returns:
        A list of ``(train_index, test_folds)``. **``test_folds`` is a list of
        arrays, one per test fold — not a single index array.** That
        distinction is the point of CPCV: each fold belongs to a different
        backtest path, and concatenating them would collapse the many paths
        back into the one that walk-forward already gives. ``len(test_folds)``
        is the fold count, not a bar count; reading it as bars is the obvious
        way to misuse this.
    """
    # Imported lazily: skfolio pulls in sklearn and plotting machinery that the
    # live trading path has no reason to load.
    import pandas as pd
    from skfolio.model_selection import CombinatorialPurgedCV

    cv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purged_size,
        embargo_size=embargo_size,
    )
    placeholder = pd.DataFrame(np.zeros((n_observations, 1)))
    return [
        (np.asarray(train), [np.asarray(fold) for fold in test])
        for train, test in cv.split(placeholder)
    ]


def optimal_split_shape(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
) -> tuple[int, int]:
    """Pick ``(n_folds, n_test_folds)`` for a desired train size and path count.

    Delegates to skfolio's ``optimal_folds_number``. More paths give a better
    estimate of the performance distribution but shrink each training window —
    the trade-off this resolves numerically instead of by guesswork.
    """
    from skfolio.model_selection import optimal_folds_number

    n_folds, n_test_folds = optimal_folds_number(
        n_observations=n_observations,
        target_train_size=target_train_size,
        target_n_test_paths=target_n_test_paths,
    )
    return int(n_folds), int(n_test_folds)
