"""Walk-forward evaluation of an already-trained model + baseline comparison.

Unlike ``backtest.py`` (which retrains every fold via ``WalkForwardValidator``)
this script does **inference-only** walk-forward — useful when you want to
compare an existing checkpoint to baselines without paying the GPU cost of
retraining.

Usage::

    # Default: Run 5 best_sharpe vs baselines on EURUSD H1
    python scripts/eval_existing_model.py \\
        --model-path vm_snapshot_20260419_024501/models/v2_checkpoints_run5/best_sharpe/model.zip

    # Different checkpoint, custom spread, save CSV
    python scripts/eval_existing_model.py \\
        --model-path vm_snapshot_20260419_024501/models/v2_checkpoints_run6/best_sharpe/model.zip \\
        --spread-pips 2.5 \\
        --output reports/run6_best_eval.csv

The output is the canonical comparison table — model vs baselines per WF
fold, with the explicit "edge / no edge" verdict.
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC

from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.reward import LogReturnReward
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
from apexfx.features.pipeline import FeaturePipeline


# ---------------------------------------------------------------------------
# Model loader (try TQC → SAC → PPO)
# ---------------------------------------------------------------------------


def load_model(path: str, device: str = "cpu"):
    """Try every supported algorithm; return ``(algo_name, model)``."""
    last_error: Exception | None = None
    for algo_name, cls in [("TQC", TQC), ("SAC", SAC), ("PPO", PPO)]:
        try:
            model = cls.load(path, device=device)
            return algo_name, model
        except Exception as e:  # pragma: no cover
            last_error = e
    raise RuntimeError(f"Could not load {path} as TQC/SAC/PPO; last error: {last_error}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bars(symbol: str, timeframe: str, data_dir: str = "data") -> pd.DataFrame:
    pattern = f"{data_dir}/raw/bars/{symbol}/{timeframe}/*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").set_index("time")
    return df


# ---------------------------------------------------------------------------
# Walk-forward fold spec (mirrors compare_baselines.FoldSpec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    fold_idx: int
    label: str
    test_df: pd.DataFrame  # raw OHLCV; index reset


def make_folds(
    df: pd.DataFrame,
    train_window_bars: int,
    test_window_bars: int,
    step_bars: int,
) -> list[FoldSpec]:
    folds: list[FoldSpec] = []
    n = len(df)
    fold_idx = 0
    start = 0
    while True:
        train_end = start + train_window_bars
        test_start = train_end
        test_end = test_start + test_window_bars
        if test_end > n:
            break
        slice_ = df.iloc[test_start:test_end]
        label = (
            f"{slice_.index[0].strftime('%Y-%m-%d')}"
            f"..{slice_.index[-1].strftime('%Y-%m-%d')}"
        )
        folds.append(FoldSpec(fold_idx=fold_idx, label=label,
                              test_df=slice_.reset_index(drop=False)))
        start += step_bars
        fold_idx += 1
    return folds


# ---------------------------------------------------------------------------
# Run model on a single fold using the env, return fold metrics
# ---------------------------------------------------------------------------


def evaluate_model_on_fold(
    model,
    fold_df: pd.DataFrame,
    feature_pipeline: FeaturePipeline,
    n_market_features: int,
    lookback: int,
    spread_pips: float,
    initial_balance: float,
    annualisation_periods: int,
) -> dict[str, float]:
    """Run the trained model deterministically on ``fold_df``; return metrics."""
    # Compute features on the fold's OHLC slice
    features = feature_pipeline.compute(fold_df)
    if len(features) < lookback + 10:
        # Too short for a meaningful run
        return {"sharpe_ratio": 0.0, "total_return": 0.0, "n_trades": 0, "max_drawdown": 0.0}

    env = ForexTradingEnv(
        data=features,
        initial_balance=initial_balance,
        n_market_features=n_market_features,
        lookback=lookback,
        reward_fn=LogReturnReward(),  # reward unused at inference, but env needs one
        max_drawdown_pct=1.0,  # don't terminate on DD during eval
        transaction_cost_pips=spread_pips,
        episode_length=10_000_000,  # effectively unbounded
    )

    obs, _info = env.reset()
    done = False
    returns: list[float] = []
    n_trades = 0
    prev_position = 0.0
    prev_value = initial_balance
    while not done:
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception as exc:  # observation shape mismatch, etc
            print(f"  predict() failed mid-fold: {exc}", file=sys.stderr)
            break
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cur_value = float(info.get("portfolio_value", prev_value))
        cur_position = float(info.get("position", 0.0))
        if prev_value > 0:
            returns.append((cur_value - prev_value) / prev_value)
        # Count opens (transitions from flat to non-flat)
        if abs(prev_position) < 1e-6 and abs(cur_position) > 1e-6:
            n_trades += 1
        prev_value = cur_value
        prev_position = cur_position

    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return {"sharpe_ratio": 0.0, "total_return": 0.0, "n_trades": n_trades, "max_drawdown": 0.0}

    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1))
    sharpe = (mean_r / std_r * np.sqrt(annualisation_periods)) if std_r > 1e-12 else 0.0

    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))

    return {
        "sharpe_ratio": float(sharpe),
        "total_return": float(equity[-1] - 1.0),
        "n_trades": n_trades,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward eval of an existing model + baselines.")
    p.add_argument("--model-path", required=True, help="Path to model.zip (SAC/TQC/PPO)")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="H1",
                   help="Bar timeframe to load. Match training timeframe (default: H1).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-days", type=int, default=180,
                   help="Walk-forward train window (skipped — model is pre-trained, default: 180)")
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--spread-pips", type=float, default=2.0)
    p.add_argument("--initial-balance", type=float, default=100_000.0)
    p.add_argument("--n-market-features", type=int, default=30,
                   help="Must match what the model was trained on (default: 30)")
    p.add_argument("--lookback", type=int, default=100)
    p.add_argument("--output", default=None)
    p.add_argument("--max-folds", type=int, default=None,
                   help="Cap fold count for fast iteration (default: all)")
    args = p.parse_args()

    # Load model
    print(f"Loading model: {args.model_path}")
    algo_name, model = load_model(args.model_path)
    print(f"  loaded as {algo_name}; obs keys: {list(model.observation_space.spaces.keys())}")

    # Detect market_features dim from model — overrides --n-market-features if mismatch
    market_dim = int(model.observation_space.spaces["market_features"].shape[0])
    inferred_n_features = market_dim // args.lookback
    if inferred_n_features != args.n_market_features:
        print(
            f"  WARNING: model expects market_features={market_dim} (= {inferred_n_features} × "
            f"lookback {args.lookback}); overriding --n-market-features {args.n_market_features} "
            f"→ {inferred_n_features}"
        )
        args.n_market_features = inferred_n_features

    # Load bars
    print(f"Loading {args.symbol} {args.timeframe} bars...")
    df = load_bars(args.symbol, args.timeframe, args.data_dir)
    print(f"  {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Compute folds
    bars_per_day = {"H1": 24, "H4": 6, "D1": 1}.get(args.timeframe.upper(), 24)
    folds = make_folds(
        df,
        train_window_bars=args.train_days * bars_per_day,
        test_window_bars=args.test_days * bars_per_day,
        step_bars=args.step_days * bars_per_day,
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    if not folds:
        print("ERROR: not enough data for the requested fold geometry.")
        return
    print(
        f"\n{len(folds)} walk-forward folds: train={args.train_days}d / "
        f"test={args.test_days}d / step={args.step_days}d  (spread={args.spread_pips} pips)"
    )

    # Set up baselines + feature pipeline (shared)
    baselines: list[TradingBaseline] = [
        BuyAndHoldBaseline(),
        MACrossBaseline(20, 50),
        DonchianBaseline(20),
        DonchianBaseline(20, long_only=True),
        DonchianBaseline(55),
        RandomBaseline(seed=42),
    ]
    annualisation_periods = 252 * bars_per_day
    exec_config = BaselineExecConfig(
        initial_balance=args.initial_balance,
        transaction_cost_pips=args.spread_pips,
    )
    feature_pipeline = FeaturePipeline()

    # Run model + baselines per fold
    rows: list[BaselineComparisonRow] = []
    print()
    for fold in folds:
        print(f"Fold {fold.fold_idx}: {fold.label}  (n_bars={len(fold.test_df)})")

        # Model
        model_metrics = evaluate_model_on_fold(
            model,
            fold.test_df,
            feature_pipeline,
            n_market_features=args.n_market_features,
            lookback=args.lookback,
            spread_pips=args.spread_pips,
            initial_balance=args.initial_balance,
            annualisation_periods=annualisation_periods,
        )
        model_sharpe = model_metrics["sharpe_ratio"]
        print(
            f"  MODEL ({algo_name}):  Sharpe={model_sharpe:+.3f}  "
            f"Return={model_metrics['total_return']*100:+.2f}%  "
            f"Trades={model_metrics['n_trades']}  "
            f"MaxDD={model_metrics['max_drawdown']*100:.2f}%"
        )

        # Baselines
        baseline_sharpes: dict[str, float] = {}
        for baseline in baselines:
            res = evaluate_on_data(
                baseline,
                fold.test_df,
                config=exec_config,
                annualisation_periods=annualisation_periods,
            )
            baseline_sharpes[baseline.name] = res.sharpe_ratio
        best_name = max(baseline_sharpes, key=baseline_sharpes.__getitem__)
        best_sr = baseline_sharpes[best_name]
        print(f"  Best baseline: {best_name} Sharpe={best_sr:+.3f}; model beats? "
              f"{'YES' if model_sharpe > best_sr else 'no'}")

        rows.append(
            BaselineComparisonRow(
                fold_idx=fold.fold_idx,
                period_label=fold.label,
                model_sharpe=model_sharpe,
                baseline_sharpes=baseline_sharpes,
                beats_best_baseline=model_sharpe > best_sr,
                best_baseline_name=best_name,
                best_baseline_sharpe=best_sr,
            )
        )

    # Summary
    print("\n" + "=" * 100)
    print("SHARPE COMPARISON — MODEL vs BASELINES (walk-forward)")
    print("=" * 100)
    print(format_comparison_table(rows, edge_threshold_pct=60.0))

    if args.output:
        out = Path(args.output)
        df_out = comparison_rows_to_dataframe(rows)
        df_out.to_csv(out, index=False)
        print(f"\nCSV saved: {out}")


if __name__ == "__main__":
    main()
