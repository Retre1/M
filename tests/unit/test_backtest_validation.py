"""Tests for telling a finding apart from a lucky search.

Each test states the property that makes the statistic worth having: that the
expected best-of-N grows with N, that a Sharpe stops being significant once
enough configurations were tried, and that selecting on noise is detected.
"""

from __future__ import annotations

import numpy as np
import pytest

from apexfx.backtest.validation import (
    OverfittingReport,
    combinatorial_purged_splits,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    optimal_split_shape,
    probability_of_backtest_overfitting,
)


class TestExpectedMaxSharpe:
    def test_a_single_trial_has_nothing_to_beat(self):
        assert expected_max_sharpe(1, 0.01) == 0.0

    def test_more_trials_raise_the_bar(self):
        """The whole point: search inflates the best result you will see."""
        bars = [expected_max_sharpe(n, 0.01) for n in (10, 50, 500)]
        assert bars == sorted(bars)
        assert bars[0] > 0

    def test_wider_spread_across_trials_raises_the_bar(self):
        assert expected_max_sharpe(50, 0.04) > expected_max_sharpe(50, 0.01)

    def test_zero_variance_means_no_luck_available(self):
        assert expected_max_sharpe(500, 0.0) == 0.0


class TestDeflatedSharpe:
    def test_an_unsearched_result_survives(self):
        assert deflated_sharpe_ratio(0.20, 1000, n_trials=1, trial_variance=0.01) > 0.95

    def test_the_same_sharpe_dies_after_a_wide_search(self):
        """0.20 is below what 500 tries produce by luck alone (~0.305)."""
        assert deflated_sharpe_ratio(0.20, 1000, n_trials=500, trial_variance=0.01) < 0.05

    def test_a_larger_sample_makes_the_same_sharpe_more_credible(self):
        """Only above the luck benchmark — see the inversion test below."""
        short = deflated_sharpe_ratio(0.30, 200, n_trials=10, trial_variance=0.01)
        long = deflated_sharpe_ratio(0.30, 5000, n_trials=10, trial_variance=0.01)
        assert long > short

    def test_more_evidence_sharpens_a_verdict_in_both_directions(self):
        """Below the benchmark, more data makes the *rejection* more certain.

        The statistic is ``(SR - benchmark)``; its sign flips when the observed
        Sharpe does not clear what the search produces by luck. More
        observations then push the probability down, not up. Worth pinning:
        the first version of these tests read the effect backwards by choosing
        an SR of 0.10 against a benchmark of 0.157.
        """
        benchmark = expected_max_sharpe(10, 0.01)
        below = 0.5 * benchmark
        short = deflated_sharpe_ratio(below, 200, n_trials=10, trial_variance=0.01)
        long = deflated_sharpe_ratio(below, 5000, n_trials=10, trial_variance=0.01)
        assert long < short

    def test_negative_skew_is_penalised(self):
        """Strategies that sell tails look good until the tail arrives.

        Measured above the luck benchmark, where the statistic is positive and
        a larger standard error genuinely reduces confidence.
        """
        symmetric = deflated_sharpe_ratio(0.30, 1000, 10, 0.01, skew=0.0, kurtosis=3.0)
        skewed = deflated_sharpe_ratio(0.30, 1000, 10, 0.01, skew=-1.5, kurtosis=8.0)
        assert skewed < symmetric

    def test_too_short_a_series_yields_nothing(self):
        assert deflated_sharpe_ratio(2.0, 1, n_trials=1, trial_variance=0.01) == 0.0


class TestProbabilityOfBacktestOverfitting:
    @staticmethod
    def _noise(n_obs: int = 2000, n_configs: int = 20, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).normal(0, 0.01, (n_obs, n_configs))

    def test_selecting_among_noise_is_flagged(self):
        """No configuration is better, so the in-sample winner is arbitrary."""
        assert probability_of_backtest_overfitting(self._noise()) > 0.2

    def test_a_genuine_edge_is_not_flagged(self):
        performance = self._noise()
        performance[:, 7] += 0.0025  # one configuration that really is better
        assert probability_of_backtest_overfitting(performance) < 0.1

    def test_edge_beats_noise_on_the_same_data(self):
        noise = self._noise()
        edged = noise.copy()
        edged[:, 3] += 0.003
        assert (probability_of_backtest_overfitting(edged)
                < probability_of_backtest_overfitting(noise))

    def test_odd_split_count_is_rejected(self):
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(self._noise(), n_splits=7)

    def test_one_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="two configurations"):
            probability_of_backtest_overfitting(self._noise(n_configs=1))

    def test_wrong_dimensionality_is_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            probability_of_backtest_overfitting(np.zeros(100))


class TestCombinatorialSplits:
    N_OBS = 4000

    def test_many_paths_not_one(self):
        """The reason for CPCV: walk-forward gives a single test path."""
        splits = combinatorial_purged_splits(
            self.N_OBS, n_folds=8, n_test_folds=2, purged_size=10, embargo_size=10,
        )
        assert len(splits) == 28  # C(8, 2)

    def test_test_folds_are_separate_arrays(self):
        """len(test_folds) counts folds, not bars — the easy misreading."""
        splits = combinatorial_purged_splits(self.N_OBS, n_folds=8, n_test_folds=3)
        _, test_folds = splits[0]
        assert len(test_folds) == 3
        assert all(len(fold) > 100 for fold in test_folds)

    def test_train_and_test_do_not_overlap(self):
        splits = combinatorial_purged_splits(
            self.N_OBS, n_folds=8, n_test_folds=2, purged_size=20, embargo_size=20,
        )
        for train, test_folds in splits[:5]:
            test_idx = np.concatenate(test_folds)
            assert len(np.intersect1d(train, test_idx)) == 0

    def test_purging_removes_neighbouring_observations(self):
        """Without purging, a training row can carry test-period information."""
        plain = combinatorial_purged_splits(self.N_OBS, n_folds=8, n_test_folds=2)
        purged = combinatorial_purged_splits(
            self.N_OBS, n_folds=8, n_test_folds=2, purged_size=50, embargo_size=50,
        )
        assert len(purged[0][0]) < len(plain[0][0])


class TestOptimalSplitShape:
    def test_returns_a_usable_pair(self):
        n_folds, n_test_folds = optimal_split_shape(
            12_321, target_train_size=8000, target_n_test_paths=30,
        )
        assert n_folds > n_test_folds >= 1
        splits = combinatorial_purged_splits(
            12_321, n_folds=n_folds, n_test_folds=n_test_folds,
        )
        assert len(splits) > 1


class TestReport:
    def test_a_searched_result_is_marked_as_not_surviving(self):
        report = OverfittingReport(
            observed_sharpe=0.20, n_trials=500,
            expected_max_sharpe=0.305, deflated_sharpe=0.0005, pbo=0.62,
        )
        assert not report.survives_multiple_testing
        assert "DOES NOT survive" in report.summary()
        assert "OVERFIT" in report.summary()

    def test_a_clean_result_is_marked_as_surviving(self):
        report = OverfittingReport(
            observed_sharpe=0.35, n_trials=3,
            expected_max_sharpe=0.05, deflated_sharpe=0.99, pbo=0.11,
        )
        assert report.survives_multiple_testing
        assert "DOES NOT" not in report.summary()
