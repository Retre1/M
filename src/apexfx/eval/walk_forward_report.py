"""Walk-forward comparison report: model vs baselines, per fold.

Produces the canonical decision artefact for "is there edge?":

    Fold | Period            | Model SR | B&H SR | MA SR | Donchian SR | Random SR | Beats best
    -----|-------------------|----------|--------|-------|-------------|-----------|------------
    0    | 2024-01..2024-03  |   0.82   |  0.45  | -0.12 |    0.38     |   -1.20   |    YES
    1    | 2024-04..2024-06  |   0.31   |  0.78  |  0.21 |    0.40     |   -1.05   |    NO
    ...
    Aggregate: model beats best baseline on 8/12 folds (67%) — EDGE EXISTS

If the model does not beat the *best* baseline on at least 60% of folds,
do not trade live.  Period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from apexfx.eval.baselines import (
    BaselineEvalResult,
    BaselineExecConfig,
    TradingBaseline,
    evaluate_on_data,
)


@dataclass(frozen=True)
class BaselineComparisonRow:
    """One row of the comparison table — one walk-forward fold."""

    fold_idx: int
    period_label: str
    model_sharpe: float
    baseline_sharpes: dict[str, float]  # baseline_name -> sharpe
    beats_best_baseline: bool
    best_baseline_name: str
    best_baseline_sharpe: float

    @property
    def model_minus_best(self) -> float:
        return self.model_sharpe - self.best_baseline_sharpe


def compare_to_baselines(
    fold_data: list[tuple[str, pd.DataFrame]],
    fold_model_sharpes: list[float],
    baselines: list[TradingBaseline],
    exec_config: BaselineExecConfig | None = None,
    annualisation_periods: int = 252 * 6,
) -> list[BaselineComparisonRow]:
    """Run baselines on each fold's test data, build comparison rows.

    Parameters
    ----------
    fold_data : list[(label, DataFrame)]
        One entry per walk-forward fold.  Label is shown in the report
        ("2024-01..2024-03"); DataFrame is the OOS price slice.
    fold_model_sharpes : list[float]
        Model's Sharpe per fold, in the same order as ``fold_data``.
    baselines : list[TradingBaseline]
        Strategies to compare against.
    exec_config : BaselineExecConfig | None
        Spread / balance / sizing.  Defaults to retail-realistic.
    annualisation_periods : int
        Bars per year — H4 default.

    Returns
    -------
    list[BaselineComparisonRow]
        One row per fold.  Use ``format_comparison_table`` to render.
    """
    if len(fold_data) != len(fold_model_sharpes):
        raise ValueError(
            f"fold_data has {len(fold_data)} entries but fold_model_sharpes "
            f"has {len(fold_model_sharpes)} — must match"
        )

    rows: list[BaselineComparisonRow] = []
    for fold_idx, ((label, df), model_sharpe) in enumerate(
        zip(fold_data, fold_model_sharpes, strict=True)
    ):
        baseline_sharpes: dict[str, float] = {}
        for baseline in baselines:
            result: BaselineEvalResult = evaluate_on_data(
                baseline,
                df,
                config=exec_config,
                annualisation_periods=annualisation_periods,
            )
            baseline_sharpes[baseline.name] = result.sharpe_ratio

        if baseline_sharpes:
            best_name = max(baseline_sharpes, key=baseline_sharpes.__getitem__)
            best_sr = baseline_sharpes[best_name]
        else:
            best_name = ""
            best_sr = 0.0

        rows.append(
            BaselineComparisonRow(
                fold_idx=fold_idx,
                period_label=label,
                model_sharpe=float(model_sharpe),
                baseline_sharpes=baseline_sharpes,
                beats_best_baseline=model_sharpe > best_sr,
                best_baseline_name=best_name,
                best_baseline_sharpe=best_sr,
            )
        )

    return rows


def format_comparison_table(
    rows: list[BaselineComparisonRow],
    *,
    edge_threshold_pct: float = 60.0,
) -> str:
    """Render comparison rows as a plain-text table + verdict.

    The verdict ("EDGE EXISTS / NO EDGE") is determined by whether the
    model beats the best baseline on at least ``edge_threshold_pct`` of
    folds.  60% is the audit-recommended threshold — a model that wins
    only on a coin-flip basis (50%) hasn't demonstrated meaningful edge.
    """
    if not rows:
        return "(no folds — nothing to compare)"

    # Discover baseline names from first row (assume all rows have the same set)
    baseline_names = list(rows[0].baseline_sharpes.keys())

    # Header
    header_cells = ["Fold", "Period", "Model SR"]
    for name in baseline_names:
        header_cells.append(f"{name} SR")
    header_cells.append("Beats best?")
    col_widths = [max(len(c), 8) for c in header_cells]
    # Period column wider
    if len(col_widths) >= 2:
        col_widths[1] = max(col_widths[1], 22)

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True))

    lines: list[str] = []
    lines.append(fmt_row(header_cells))
    lines.append("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

    n_beats = 0
    for row in rows:
        cells = [
            str(row.fold_idx),
            row.period_label,
            f"{row.model_sharpe:+.3f}",
        ]
        for name in baseline_names:
            sr = row.baseline_sharpes.get(name, 0.0)
            cells.append(f"{sr:+.3f}")
        if row.beats_best_baseline:
            cells.append("YES")
            n_beats += 1
        else:
            cells.append("no")
        lines.append(fmt_row(cells))

    # Aggregate
    lines.append("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    n = len(rows)
    pct = 100.0 * n_beats / n
    avg_model_minus_best = sum(r.model_minus_best for r in rows) / n
    verdict = (
        "EDGE EXISTS — proceed to paper trading"
        if pct >= edge_threshold_pct
        else "NO EDGE — do not trade live (model does not consistently beat trivial strategies)"
    )
    lines.append(
        f"Model beats best baseline on {n_beats}/{n} folds ({pct:.0f}%); "
        f"avg ΔSharpe (model − best baseline) = {avg_model_minus_best:+.3f}"
    )
    lines.append(f"VERDICT: {verdict}")
    return "\n".join(lines)


def comparison_rows_to_dataframe(rows: list[BaselineComparisonRow]) -> pd.DataFrame:
    """Convert comparison rows to a flat ``pandas.DataFrame`` for CSV export.

    Columns: ``fold_idx``, ``period_label``, ``model_sharpe``,
    ``<baseline>_sharpe`` for each baseline, ``best_baseline``,
    ``best_baseline_sharpe``, ``beats_best_baseline``, ``model_minus_best``.
    """
    if not rows:
        return pd.DataFrame()

    baseline_names = list(rows[0].baseline_sharpes.keys())
    records: list[dict[str, object]] = []
    for row in rows:
        record: dict[str, object] = {
            "fold_idx": row.fold_idx,
            "period_label": row.period_label,
            "model_sharpe": row.model_sharpe,
        }
        for name in baseline_names:
            record[f"{name}_sharpe"] = row.baseline_sharpes.get(name, 0.0)
        record["best_baseline"] = row.best_baseline_name
        record["best_baseline_sharpe"] = row.best_baseline_sharpe
        record["beats_best_baseline"] = row.beats_best_baseline
        record["model_minus_best"] = row.model_minus_best
        records.append(record)

    return pd.DataFrame.from_records(records)
