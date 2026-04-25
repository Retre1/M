"""Tests for the walk-forward comparison report.

This module is the *honest verdict* layer — its job is to compare
model Sharpe against baseline Sharpe per fold and produce a clear
"edge / no edge" decision.  The tests lock in:

* aggregation logic (beats_best_baseline, n_beats counter)
* threshold semantics (60% default for "edge exists")
* DataFrame export (CSV-ready)
* error handling on input mismatch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.eval.baselines import (
    BuyAndHoldBaseline,
    DonchianBaseline,
    MACrossBaseline,
    RandomBaseline,
    TradingBaseline,
)
from apexfx.eval.walk_forward_report import (
    BaselineComparisonRow,
    compare_to_baselines,
    comparison_rows_to_dataframe,
    format_comparison_table,
)


def _make_fold_data(n_folds: int, n_bars: int = 200, seed: int = 100) -> list[tuple[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    folds: list[tuple[str, pd.DataFrame]] = []
    for k in range(n_folds):
        log_returns = rng.normal(loc=0.0002, scale=0.0005, size=n_bars)
        close = 1.10 * np.exp(np.cumsum(log_returns))
        high = close * (1.0 + np.abs(rng.normal(0, 0.0002, n_bars)))
        low = close * (1.0 - np.abs(rng.normal(0, 0.0002, n_bars)))
        df = pd.DataFrame({"close": close, "high": high, "low": low})
        folds.append((f"fold-{k}-period", df))
    return folds


class TestCompareToBaselines:
    def test_length_mismatch_raises(self) -> None:
        folds = _make_fold_data(3)
        with pytest.raises(ValueError):
            compare_to_baselines(
                folds,
                fold_model_sharpes=[0.5, 0.7],  # only 2, need 3
                baselines=[BuyAndHoldBaseline()],
            )

    def test_returns_one_row_per_fold(self) -> None:
        folds = _make_fold_data(4)
        rows = compare_to_baselines(
            folds,
            fold_model_sharpes=[0.5, 0.7, -0.1, 1.2],
            baselines=[BuyAndHoldBaseline(), MACrossBaseline(5, 20)],
        )
        assert len(rows) == 4
        for k, row in enumerate(rows):
            assert row.fold_idx == k
            assert row.period_label == f"fold-{k}-period"

    def test_each_row_has_one_sharpe_per_baseline(self) -> None:
        folds = _make_fold_data(2)
        baselines: list[TradingBaseline] = [
            BuyAndHoldBaseline(),
            DonchianBaseline(20),
            RandomBaseline(seed=99),
        ]
        rows = compare_to_baselines(
            folds, fold_model_sharpes=[0.5, 0.5], baselines=baselines
        )
        for row in rows:
            assert set(row.baseline_sharpes) == {b.name for b in baselines}

    def test_beats_best_baseline_logic(self) -> None:
        folds = _make_fold_data(1)
        baselines = [BuyAndHoldBaseline()]
        # Get B&H sharpe to construct a passing/failing model
        rows_init = compare_to_baselines(folds, [0.0], baselines)
        bh_sr = rows_init[0].baseline_sharpes["B&H"]

        # Model just above B&H — should beat
        rows_pass = compare_to_baselines(folds, [bh_sr + 0.5], baselines)
        assert rows_pass[0].beats_best_baseline is True

        # Model just below B&H — should not beat
        rows_fail = compare_to_baselines(folds, [bh_sr - 0.5], baselines)
        assert rows_fail[0].beats_best_baseline is False

    def test_best_baseline_name_is_identified(self) -> None:
        folds = _make_fold_data(1)
        baselines = [
            BuyAndHoldBaseline(),
            MACrossBaseline(5, 20),
            DonchianBaseline(20),
            RandomBaseline(seed=99),
        ]
        rows = compare_to_baselines(folds, [-99.0], baselines)
        row = rows[0]
        # best_baseline_sharpe should be the max of all baseline sharpes
        assert row.best_baseline_sharpe == max(row.baseline_sharpes.values())
        # And best_baseline_name should be the corresponding key
        assert row.baseline_sharpes[row.best_baseline_name] == row.best_baseline_sharpe


class TestFormatComparisonTable:
    def test_empty_rows_produce_message(self) -> None:
        out = format_comparison_table([])
        assert "no folds" in out.lower()

    def test_table_contains_verdict(self) -> None:
        folds = _make_fold_data(3)
        rows = compare_to_baselines(folds, [10.0, 10.0, 10.0], [BuyAndHoldBaseline()])
        out = format_comparison_table(rows)
        assert "VERDICT" in out
        assert "EDGE" in out

    def test_no_edge_verdict_when_below_threshold(self) -> None:
        folds = _make_fold_data(5)
        # Model loses on every fold
        rows = compare_to_baselines(
            folds, [-99.0, -99.0, -99.0, -99.0, -99.0], [BuyAndHoldBaseline()]
        )
        out = format_comparison_table(rows, edge_threshold_pct=60.0)
        assert "NO EDGE" in out
        assert "0/5" in out

    def test_edge_verdict_when_above_threshold(self) -> None:
        folds = _make_fold_data(5)
        # Model wins on every fold
        rows = compare_to_baselines(
            folds, [99.0, 99.0, 99.0, 99.0, 99.0], [BuyAndHoldBaseline()]
        )
        out = format_comparison_table(rows, edge_threshold_pct=60.0)
        assert "EDGE EXISTS" in out
        assert "5/5" in out

    def test_threshold_boundary_inclusive(self) -> None:
        # 3/5 = 60% should pass at threshold 60.0
        folds = _make_fold_data(5)
        rows: list[BaselineComparisonRow] = []
        for k in range(5):
            rows.append(
                BaselineComparisonRow(
                    fold_idx=k,
                    period_label=f"f{k}",
                    model_sharpe=1.0 if k < 3 else -1.0,
                    baseline_sharpes={"B&H": 0.0},
                    beats_best_baseline=k < 3,
                    best_baseline_name="B&H",
                    best_baseline_sharpe=0.0,
                )
            )
        out = format_comparison_table(rows, edge_threshold_pct=60.0)
        assert "EDGE EXISTS" in out


class TestComparisonRowsToDataframe:
    def test_empty_returns_empty_dataframe(self) -> None:
        df = comparison_rows_to_dataframe([])
        assert df.empty

    def test_dataframe_columns(self) -> None:
        folds = _make_fold_data(2)
        baselines = [BuyAndHoldBaseline(), MACrossBaseline(5, 20)]
        rows = compare_to_baselines(folds, [0.5, 0.7], baselines)
        df = comparison_rows_to_dataframe(rows)
        assert list(df.columns) == [
            "fold_idx",
            "period_label",
            "model_sharpe",
            "B&H_sharpe",
            "MA(5,20)_sharpe",
            "best_baseline",
            "best_baseline_sharpe",
            "beats_best_baseline",
            "model_minus_best",
        ]
        assert len(df) == 2

    def test_dataframe_can_be_csv_exported(self, tmp_path) -> None:
        folds = _make_fold_data(3)
        rows = compare_to_baselines(folds, [0.5, 0.7, -0.2], [BuyAndHoldBaseline()])
        df = comparison_rows_to_dataframe(rows)
        out = tmp_path / "comparison.csv"
        df.to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert len(loaded) == 3
        assert "model_sharpe" in loaded.columns
