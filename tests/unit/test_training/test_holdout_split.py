"""The holdout must be reserved before training sees anything.

The defect these pin down: the curriculum trained on every bar, and the
post-training backtest then evaluated "the last 30%" of those same bars and
logged it as out-of-sample. Two separate contaminations rode on that — the
agent had trained on the evaluation slice, and ``FeatureSelector`` had ranked
features by ``close[t+1] > close[t]`` labels computed across it.

The two surviving numbers from runs 1-6, OOS PF 0.957 and 0.736, came from
that path and are therefore in-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.training.trainer import (
    HOLDOUT_FRACTION,
    HOLDOUT_PURGE_BARS,
    Trainer,
)


def _bars(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.full(n, 500),
    })


class TestHoldoutIsReserved:
    N = 12_321  # the project's actual bar count

    def test_training_stops_before_the_holdout_begins(self):
        data = _bars(self.N)
        train, holdout_start = Trainer._split_holdout(data)
        assert len(train) < holdout_start, (
            "training data must end before the holdout starts, or the backtest "
            "is scoring bars the agent was trained on"
        )

    def test_the_gap_is_the_purge_width(self):
        """Rolling windows and the selector's next-bar label both reach
        forward, so adjacent slices are not disjoint in information."""
        data = _bars(self.N)
        train, holdout_start = Trainer._split_holdout(data)
        assert holdout_start - len(train) == HOLDOUT_PURGE_BARS

    def test_the_holdout_is_the_configured_share(self):
        data = _bars(self.N)
        _, holdout_start = Trainer._split_holdout(data)
        held = self.N - holdout_start
        assert held == pytest.approx(self.N * HOLDOUT_FRACTION, rel=0.01)

    def test_no_training_bar_appears_in_the_holdout(self):
        data = _bars(self.N)
        train, holdout_start = Trainer._split_holdout(data)
        holdout = data.iloc[holdout_start:]
        assert train["time"].max() < holdout["time"].min()

    def test_a_short_series_refuses_to_pretend(self):
        """Too little history to purge means no honest holdout exists. Saying
        so beats reporting an in-sample number as a test."""
        train, holdout_start = Trainer._split_holdout(_bars(200))
        assert holdout_start is None
        assert len(train) == 200

    def test_no_data_is_handled(self):
        assert Trainer._split_holdout(None) == (None, None)

    def test_an_empty_frame_is_handled(self):
        empty = _bars(0)
        train, holdout_start = Trainer._split_holdout(empty)
        assert holdout_start is None
        assert train.empty


class TestSelectorCannotSeeTheHoldout:
    """FeatureSelector labels a bar by the next bar's direction and is fitted
    inside training. Reserving the holdout first is what keeps those labels
    off the evaluation slice."""

    def test_the_selector_would_have_ranked_on_future_returns(self):
        """Documents why the split matters, by showing what the selector uses."""
        from apexfx.features.selector import FeatureSelector

        assert FeatureSelector().__init__.__defaults__ is not None
        selector = FeatureSelector()
        assert selector._forward_return_bars >= 1, (
            "the label is built from bars after t; fitting it on the "
            "evaluation slice leaks that slice's direction into the "
            "choice of features"
        )

    def test_training_data_ends_before_any_label_reaches_the_holdout(self):
        data = _bars(12_321)
        train, holdout_start = Trainer._split_holdout(data)
        from apexfx.features.selector import FeatureSelector

        forward = FeatureSelector()._forward_return_bars
        # The furthest bar any training label can read.
        assert len(train) - 1 + forward < holdout_start
