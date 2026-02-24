"""Walk-Forward Validation with rolling windows for out-of-sample evaluation.

Implements Purged Walk-Forward with embargo gaps to prevent two forms of leakage:

1. **Feature look-ahead (purge):** Features computed near the train/test boundary
   can span across it (e.g. SpectralExtractor uses an FFT window of 256 bars).
   A *purge gap* of ``purge_days * bars_per_day`` bars is inserted between the
   end of the training window and the start of the test window so that no
   training sample's features overlap with the test period.

2. **Label leakage across folds (embargo):** When a fold's test period ends at
   bar T, the bars in [T, T + embargo_bars) carry information that "leaked"
   forward through feature computation. The *embargo gap* prevents these bars
   from entering the training window of any subsequent fold that immediately
   follows, stopping look-ahead from propagating into future retraining cycles.

Fold layout (H1 example, train=252d, test=63d, purge=11d, embargo=5d):

    |<---- train (252d) ---->|< purge 11d >|<-- test (63d) -->|< emb 5d >|
    ^train_start             ^train_end    ^test_start        ^test_end  ^next safe start
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from apexfx.config.schema import AppConfig
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.reward import DifferentialSharpeReward
from apexfx.features.pipeline import FeaturePipeline
from apexfx.training.trainer import Trainer
from apexfx.utils.logging import get_logger
from apexfx.utils.metrics import compute_all_metrics

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    purge_end: int   # first bar of test window (= train_end + purge_bars)
    test_start: int  # alias for purge_end, kept for clarity
    test_end: int
    embargo_end: int  # first bar safe for the next fold's training data to include
    metrics: dict[str, float]
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class WalkForwardResults:
    """Aggregated results across all walk-forward folds."""
    folds: list[FoldResult]
    aggregate_metrics: dict[str, float]
    oos_returns: np.ndarray  # all out-of-sample returns concatenated


class WalkForwardValidator:
    """
    Purged Walk-Forward Validator.

    Train on a rolling window, test on the next window, slide forward.
    Two leakage guards are applied:

    * **purge_bars** — gap *before* each test window (feature look-ahead).
    * **embargo_bars** — gap *after* each test window (label leakage across folds).

    These values are driven by config (``walk_forward.purge_days`` and
    ``walk_forward.embargo_days``) so they can be tuned without touching code.
    """

    def __init__(
        self,
        config: AppConfig,
        data: pd.DataFrame,
        bars_per_day: int = 24,  # H1 bars
    ) -> None:
        self._config = config
        self._data = data
        self._bars_per_day = bars_per_day

        wf = config.training.walk_forward
        self._train_window = wf.train_window_days * bars_per_day
        self._test_window = wf.test_window_days * bars_per_day
        self._step_size = wf.step_size_days * bars_per_day
        self._max_folds = wf.n_folds
        self._retrain = wf.retrain_each_fold

        # --- Leakage guards ---
        # purge_days covers the longest feature look-back window in use.
        # SpectralExtractor's FFT window is 256 H1 bars ≈ 10.7 days → 11d default.
        self._purge_bars = wf.purge_days * bars_per_day
        # embargo_days prevents the bars just after a test window from re-entering
        # the training set of a subsequent fold before their features have "expired".
        self._embargo_bars = wf.embargo_days * bars_per_day

        logger.info(
            "WalkForwardValidator initialised",
            train_window_bars=self._train_window,
            test_window_bars=self._test_window,
            step_size_bars=self._step_size,
            purge_bars=self._purge_bars,
            embargo_bars=self._embargo_bars,
            total_bars=len(data),
        )

    # ------------------------------------------------------------------
    # Monte-Carlo permutation test
    # ------------------------------------------------------------------

    @staticmethod
    def monte_carlo_test(
        oos_returns: np.ndarray,
        n_permutations: int = 10_000,
        metric_fn: callable | None = None,
    ) -> float:
        """Return a p-value for the OOS returns via Monte-Carlo permutation.

        Randomly shuffles the sign of returns ``n_permutations`` times and
        measures how often the permuted metric meets or exceeds the observed
        metric.  The default metric is the mean return (equivalent to a
        randomised sign-test on cumulative PnL).

        Parameters
        ----------
        oos_returns : np.ndarray
            Concatenated out-of-sample step returns.
        n_permutations : int
            Number of random permutations.
        metric_fn : callable | None
            ``f(returns) -> float``.  Defaults to ``np.mean``.

        Returns
        -------
        float
            Proportion of permuted metrics >= observed metric (one-tailed).
        """
        if len(oos_returns) == 0:
            return 1.0

        if metric_fn is None:
            metric_fn = np.mean

        observed = metric_fn(oos_returns)

        rng = np.random.default_rng(seed=42)
        count_ge = 0
        abs_returns = np.abs(oos_returns)
        for _ in range(n_permutations):
            signs = rng.choice([-1, 1], size=len(oos_returns))
            permuted = abs_returns * signs
            if metric_fn(permuted) >= observed:
                count_ge += 1

        return (count_ge + 1) / (n_permutations + 1)  # +1 for continuity correction

    # ------------------------------------------------------------------

    def run(self) -> WalkForwardResults:
        """Execute purged walk-forward validation across all folds.

        Each fold layout:
            [train_start ... train_end) [purge) [test_start ... test_end) [embargo)

        The training data for fold N+1 is capped so that it does NOT include
        bars inside the embargo zone of fold N's test window, preventing
        label leakage from fold N's test period into subsequent retraining.
        """
        n = len(self._data)
        folds: list[FoldResult] = []
        all_oos_returns: list[np.ndarray] = []

        fold_idx = 0
        start = 0
        # Track the embargo boundary from the previous fold so that subsequent
        # training windows can be capped to exclude embargoed bars.
        prev_embargo_end: int = 0

        while True:
            train_start = start
            train_end = train_start + self._train_window

            # --- Embargo enforcement ---
            # If this fold's training window extends into the embargo zone of
            # the *previous* fold's test period, cap train_end to exclude those
            # bars. This removes at most embargo_bars from the tail of training.
            if prev_embargo_end > train_start and prev_embargo_end < train_end:
                effective_train_end = prev_embargo_end
            else:
                effective_train_end = train_end

            # Purge gap: skip bars whose features may span the train/test boundary.
            test_start = effective_train_end + self._purge_bars
            test_end = test_start + self._test_window
            embargo_end = test_end + self._embargo_bars

            if test_end > n:
                break

            if self._max_folds is not None and fold_idx >= self._max_folds:
                break

            embargoed = train_end - effective_train_end  # bars dropped due to embargo
            logger.info(
                "Walk-forward fold",
                fold=fold_idx,
                train=f"[{train_start}, {effective_train_end})",
                purge_bars=self._purge_bars,
                embargo_bars_dropped=embargoed,
                test=f"[{test_start}, {test_end})",
                embargo_zone=f"[{test_end}, {embargo_end})",
            )

            train_data = self._data.iloc[train_start:effective_train_end]
            test_data = self._data.iloc[test_start:test_end]

            if self._retrain or fold_idx == 0:
                trainer = Trainer(self._config, real_data=train_data)
                trainer.train()

            metrics, returns, equity = self._evaluate(trainer, test_data)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=effective_train_end,
                purge_end=test_start,
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
                metrics=metrics,
                equity_curve=equity,
            )
            folds.append(fold_result)
            all_oos_returns.append(returns)

            logger.info(
                "Fold complete",
                fold=fold_idx,
                sharpe=round(metrics.get("sharpe_ratio", 0), 4),
                max_dd=round(metrics.get("max_drawdown", 0), 4),
                total_return=round(metrics.get("total_return", 0), 4),
            )

            prev_embargo_end = embargo_end
            start += self._step_size
            fold_idx += 1

        # Aggregate results
        oos_returns = np.concatenate(all_oos_returns) if all_oos_returns else np.array([])
        aggregate = compute_all_metrics(oos_returns) if len(oos_returns) > 0 else {}

        # Monte-Carlo permutation p-value for OOS statistical significance
        p_value = self.monte_carlo_test(oos_returns)
        aggregate["mc_p_value"] = p_value
        logger.info("Monte-Carlo permutation test", p_value=round(p_value, 4))

        # Average fold metrics
        if folds:
            for key in folds[0].metrics:
                fold_values = [f.metrics.get(key, 0) for f in folds]
                aggregate[f"avg_{key}"] = float(np.mean(fold_values))
                aggregate[f"std_{key}"] = float(np.std(fold_values))

        logger.info("Walk-forward complete", n_folds=len(folds), aggregate=aggregate)

        return WalkForwardResults(
            folds=folds,
            aggregate_metrics=aggregate,
            oos_returns=oos_returns,
        )

    def _evaluate(
        self, trainer: Trainer, test_data: pd.DataFrame
    ) -> tuple[dict[str, float], np.ndarray, list[float]]:
        """Run the trained model on test data and collect metrics."""
        feature_pipeline = FeaturePipeline()
        features = feature_pipeline.compute(test_data)

        env = ForexTradingEnv(
            data=features,
            initial_balance=100_000.0,
            reward_fn=DifferentialSharpeReward(),
            max_drawdown_pct=1.0,  # Don't terminate during evaluation
        )

        model = trainer.model
        obs, info = env.reset()
        done = False
        returns: list[float] = []
        equity: list[float] = [100_000.0]
        prev_value = 100_000.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_value = info.get("portfolio_value", prev_value)
            if prev_value > 0:
                step_return = (current_value - prev_value) / prev_value
                returns.append(step_return)
            equity.append(current_value)
            prev_value = current_value

        returns_arr = np.array(returns)
        metrics = compute_all_metrics(returns_arr)

        return metrics, returns_arr, equity
