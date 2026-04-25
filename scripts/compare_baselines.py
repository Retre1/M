"""Run trivial trading baselines on real EURUSD data and report results.

This is the *honest validation gate*: before spending any compute on RL
training, find out how trivial strategies perform on the same data.  If
Buy & Hold dominates (e.g. clean uptrend), an RL system that even matches
B&H is "good".  If MA crossover loses 5%/year, an RL system that loses
3%/year is *still losing* — no amount of "AI" can rescue that.

Usage::

    # Walk-forward baseline comparison on EURUSD H4 (default)
    python scripts/compare_baselines.py

    # Specific timeframe
    python scripts/compare_baselines.py --timeframe H1

    # Single full-period evaluation (no folds)
    python scripts/compare_baselines.py --no-folds

    # Different cost model (test sensitivity)
    python scripts/compare_baselines.py --spread-pips 3.0

    # Save CSV for later analysis
    python scripts/compare_baselines.py --output baseline_results.csv

The output is the same comparison table that ``walk_forward_report``
produces — it just feeds an empty model column when no model is loaded,
so you see exactly what the bar-to-beat looks like.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apexfx.eval.baselines import (
    BaselineExecConfig,
    BuyAndHoldBaseline,
    DonchianBaseline,
    MACrossBaseline,
    RandomBaseline,
    TradingBaseline,
    evaluate_on_data,
)
from apexfx.eval.walk_forward_report import (
    BaselineComparisonRow,
    comparison_rows_to_dataframe,
    format_comparison_table,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bars(symbol: str, timeframe: str, data_dir: str = "data") -> pd.DataFrame:
    """Read all parquet bars for ``symbol/timeframe`` and return a sorted DataFrame.

    Columns: time (UTC), open, high, low, close, volume.
    """
    pattern = f"{data_dir}/raw/bars/{symbol}/{timeframe}/*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern}")
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    if "time" not in df.columns:
        raise ValueError(f"data missing 'time' column; columns: {list(df.columns)}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").set_index("time")
    return df


def resample_to(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample H1 bars to a higher timeframe (e.g. ``4h`` for H4, ``1D`` for D1)."""
    rule_map = {"H1": "1h", "H4": "4h", "D1": "1D"}
    rule = rule_map.get(target_tf, target_tf.lower())
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna()


# ---------------------------------------------------------------------------
# Walk-forward fold generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    """One walk-forward fold defined by index ranges."""

    fold_idx: int
    label: str
    test_df: pd.DataFrame


def make_folds(
    df: pd.DataFrame,
    train_window_bars: int,
    test_window_bars: int,
    step_bars: int,
    purge_bars: int = 0,
) -> list[FoldSpec]:
    """Slide a (train, purge, test) window over ``df`` and return test slices."""
    folds: list[FoldSpec] = []
    n = len(df)
    fold_idx = 0
    start = 0
    while True:
        train_end = start + train_window_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_window_bars
        if test_end > n:
            break
        slice_ = df.iloc[test_start:test_end]
        label_start = slice_.index[0].strftime("%Y-%m-%d")
        label_end = slice_.index[-1].strftime("%Y-%m-%d")
        folds.append(
            FoldSpec(
                fold_idx=fold_idx,
                label=f"{label_start}..{label_end}",
                test_df=slice_.reset_index(drop=False),
            )
        )
        start += step_bars
        fold_idx += 1
    return folds


def make_single_fold(df: pd.DataFrame) -> list[FoldSpec]:
    """Treat the entire dataset as a single 'fold' for non-WF mode."""
    if df.empty:
        return []
    label_start = df.index[0].strftime("%Y-%m-%d")
    label_end = df.index[-1].strftime("%Y-%m-%d")
    return [
        FoldSpec(
            fold_idx=0,
            label=f"{label_start}..{label_end}",
            test_df=df.reset_index(drop=False),
        )
    ]


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


def run_baselines_on_folds(
    folds: list[FoldSpec],
    baselines: list[TradingBaseline],
    exec_config: BaselineExecConfig,
    annualisation_periods: int,
) -> list[BaselineComparisonRow]:
    """Evaluate every baseline on every fold; build comparison rows.

    Since we don't have a trained model here, ``model_sharpe`` is the
    *best* baseline's Sharpe on each fold — i.e. "the bar to beat".
    Substitute the model when one is available (see TODO in __main__).
    """
    rows: list[BaselineComparisonRow] = []
    for fold in folds:
        baseline_sharpes: dict[str, float] = {}
        for baseline in baselines:
            res = evaluate_on_data(
                baseline,
                fold.test_df,
                config=exec_config,
                annualisation_periods=annualisation_periods,
            )
            baseline_sharpes[baseline.name] = res.sharpe_ratio
        if baseline_sharpes:
            best_name = max(baseline_sharpes, key=baseline_sharpes.__getitem__)
            best_sr = baseline_sharpes[best_name]
        else:
            best_name = ""
            best_sr = 0.0
        rows.append(
            BaselineComparisonRow(
                fold_idx=fold.fold_idx,
                period_label=fold.label,
                model_sharpe=best_sr,  # placeholder — best baseline is the bar
                baseline_sharpes=baseline_sharpes,
                beats_best_baseline=False,  # nothing to beat itself
                best_baseline_name=best_name,
                best_baseline_sharpe=best_sr,
            )
        )
    return rows


def print_per_fold_detail(
    folds: list[FoldSpec],
    baselines: list[TradingBaseline],
    exec_config: BaselineExecConfig,
    annualisation_periods: int,
) -> None:
    """Long-form per-fold breakdown showing total return / DD / trades for every baseline."""
    print("\nPER-FOLD DETAIL")
    print("=" * 100)
    for fold in folds:
        print(f"\nFold {fold.fold_idx}: {fold.label}  (n_bars={len(fold.test_df)})")
        print("-" * 100)
        print(
            f"{'Baseline':<18}  {'Return %':>10}  {'Sharpe':>8}  "
            f"{'Sortino':>8}  {'MaxDD %':>8}  {'WinRate':>8}  {'PF':>6}  {'Trades':>7}"
        )
        for baseline in baselines:
            res = evaluate_on_data(
                baseline,
                fold.test_df,
                config=exec_config,
                annualisation_periods=annualisation_periods,
            )
            m = res.metrics
            pf = m.get("profit_factor", 0.0)
            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            print(
                f"{baseline.name:<18}  {res.total_return_pct:>10.2f}  "
                f"{res.sharpe_ratio:>8.3f}  "
                f"{m.get('sortino_ratio', 0.0):>8.3f}  "
                f"{m.get('max_drawdown', 0.0) * 100:>8.2f}  "
                f"{m.get('win_rate', 0.0):>8.3f}  "
                f"{pf_str:>6}  {res.n_trades:>7d}"
            )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Honest baseline comparison for trading edge validation.")
    p.add_argument("--symbol", default="EURUSD", help="Symbol to load (default: EURUSD)")
    p.add_argument("--timeframe", default="H4",
                   help="Output timeframe — H1/H4/D1 (default: H4). Source bars are H1.")
    p.add_argument("--data-dir", default="data", help="Root data dir (default: data)")
    p.add_argument("--no-folds", action="store_true",
                   help="Single full-period evaluation instead of walk-forward folds")
    p.add_argument("--train-days", type=int, default=180,
                   help="Walk-forward train window in days (default: 180 = 6 months)")
    p.add_argument("--test-days", type=int, default=30,
                   help="Walk-forward test window in days (default: 30 = 1 month)")
    p.add_argument("--step-days", type=int, default=30,
                   help="Walk-forward step size in days (default: 30 = 1 month)")
    p.add_argument("--spread-pips", type=float, default=2.0,
                   help="Transaction cost in pips per direction change (default: 2.0 = retail-realistic)")
    p.add_argument("--initial-balance", type=float, default=100_000.0,
                   help="Starting balance for backtest (default: 100000)")
    p.add_argument("--output", default=None,
                   help="Optional path to save CSV with per-fold metrics")
    p.add_argument("--detail", action="store_true",
                   help="Print per-fold detail table (return / DD / trades for every baseline)")
    args = p.parse_args()

    # Load and (optionally) resample data
    print(f"Loading {args.symbol} bars from {args.data_dir}/raw/bars/{args.symbol}/H1/ ...")
    df_h1 = load_bars(args.symbol, "H1", args.data_dir)
    print(f"  loaded {len(df_h1)} H1 bars from {df_h1.index[0]} to {df_h1.index[-1]}")
    if args.timeframe.upper() != "H1":
        df = resample_to(df_h1, args.timeframe.upper())
        print(f"  resampled to {args.timeframe}: {len(df)} bars")
    else:
        df = df_h1

    # Annualisation
    bars_per_day = {"H1": 24, "H4": 6, "D1": 1}.get(args.timeframe.upper(), 24)
    annualisation_periods = 252 * bars_per_day

    # Build folds
    if args.no_folds:
        folds = make_single_fold(df)
        mode = "single full-period"
    else:
        folds = make_folds(
            df,
            train_window_bars=args.train_days * bars_per_day,
            test_window_bars=args.test_days * bars_per_day,
            step_bars=args.step_days * bars_per_day,
        )
        mode = (
            f"walk-forward ({args.train_days}d train / {args.test_days}d test, "
            f"step {args.step_days}d)"
        )

    if not folds:
        print(
            "ERROR: not enough data for the requested fold geometry. "
            "Try --no-folds or smaller windows."
        )
        return

    print(f"\nBaseline comparison mode: {mode}")
    print(f"Cost model: {args.spread_pips} pips per direction change")
    print(f"Folds: {len(folds)}")

    baselines: list[TradingBaseline] = [
        BuyAndHoldBaseline(),
        MACrossBaseline(20, 50),
        DonchianBaseline(20),
        DonchianBaseline(20, long_only=True),
        DonchianBaseline(55),
        RandomBaseline(seed=42),
    ]

    exec_config = BaselineExecConfig(
        initial_balance=args.initial_balance,
        transaction_cost_pips=args.spread_pips,
    )

    # Per-fold detail (full table for each baseline)
    if args.detail:
        print_per_fold_detail(folds, baselines, exec_config, annualisation_periods)

    # Comparison rows (Sharpe-only summary)
    rows = run_baselines_on_folds(folds, baselines, exec_config, annualisation_periods)

    # Render summary table
    print("\nSHARPE COMPARISON (best-baseline = the bar to beat)")
    print("=" * 100)
    print(format_comparison_table(rows, edge_threshold_pct=60.0))

    # Save CSV
    if args.output:
        out = Path(args.output)
        df_out = comparison_rows_to_dataframe(rows)
        df_out.to_csv(out, index=False)
        print(f"\nCSV saved: {out}")

    print(
        "\nNOTE: model_sharpe column shows the *best baseline* on each fold "
        "(no model loaded). When you have a trained model, re-run via "
        "scripts/backtest.py and feed model Sharpes into compare_to_baselines()."
    )


if __name__ == "__main__":
    main()
