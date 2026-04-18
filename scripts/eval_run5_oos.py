"""OOS evaluation of Run 5 TQC model on last 30% of EURUSD H1 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.reward_v5 import RARAReward_v5
from apexfx.features.pipeline import FeaturePipeline
from apexfx.utils.logging import get_logger, setup_logging
from apexfx.utils.metrics import compute_all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS eval of Run 5 TQC model")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument(
        "--model-path",
        default="artifacts/run5/v2_checkpoints_run5/best_sharpe/model.zip",
    )
    parser.add_argument("--oos-frac", type=float, default=0.30)
    parser.add_argument("--output", default="artifacts/run5/oos_report.json")
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(level=config.base.logging.level, fmt="console")
    logger = get_logger(__name__)

    from sb3_contrib import TQC

    store = DataStore(config.base.paths.data_dir)
    data = store.read_bars(args.symbol, args.timeframe)
    if data.empty:
        logger.error("No data available for backtest")
        return
    logger.info("Raw bars loaded", n_bars=len(data))

    pipeline = FeaturePipeline()
    features = pipeline.compute(data)
    logger.info("Features computed", n_bars=len(features), n_features=pipeline.n_features)

    split_idx = int(len(features) * (1.0 - args.oos_frac))
    test_data = features.iloc[split_idx:].reset_index(drop=True)
    logger.info(
        "OOS split",
        train_end_idx=split_idx,
        test_start_idx=split_idx,
        test_n_bars=len(test_data),
        oos_frac=args.oos_frac,
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    logger.info("Loading TQC model", path=str(model_path))
    model = TQC.load(str(model_path), device="cpu")
    logger.info("Model loaded")

    env = ForexTradingEnv(
        data=test_data,
        initial_balance=100_000.0,
        reward_fn=RARAReward_v5(),
        max_drawdown_pct=1.0,
    )

    obs, info = env.reset()
    done = False
    returns: list[float] = []
    equity: list[float] = [100_000.0]
    trade_pnls: list[float] = []
    prev_value = 100_000.0
    prev_position = 0.0
    entry_value = 100_000.0
    n_trades_opened = 0
    step_count = 0

    logger.info("Running OOS rollout (deterministic=True)...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_value = float(info.get("portfolio_value", prev_value))
        position = float(info.get("position", 0.0))

        if prev_value > 0:
            returns.append((current_value - prev_value) / prev_value)

        if abs(position) > 1e-9 and abs(prev_position) < 1e-9:
            entry_value = prev_value
            n_trades_opened += 1
        elif abs(position) < 1e-9 and abs(prev_position) > 1e-9:
            trade_pnls.append(current_value - entry_value)

        equity.append(current_value)
        prev_value = current_value
        prev_position = position
        step_count += 1

    if abs(prev_position) > 1e-9:
        trade_pnls.append(equity[-1] - entry_value)

    returns_arr = np.array(returns) if returns else np.array([0.0])
    metrics = compute_all_metrics(returns_arr)

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (peak - equity_arr) / peak
    max_dd_pct = float(np.max(drawdowns) * 100.0)

    final_value = float(equity_arr[-1])
    total_return_pct = (final_value - 100_000.0) / 100_000.0 * 100.0

    trade_pnls_arr = np.array(trade_pnls) if trade_pnls else np.array([])
    if len(trade_pnls_arr) > 0:
        wins = trade_pnls_arr[trade_pnls_arr > 0]
        losses = trade_pnls_arr[trade_pnls_arr < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        trade_profit_factor = (
            gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")
        )
        trade_win_rate = float(len(wins) / len(trade_pnls_arr))
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    else:
        trade_profit_factor = 0.0
        trade_win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        gross_profit = 0.0
        gross_loss = 0.0

    print()
    print("=" * 60)
    print("  Run 5 OOS EVAL — best_sharpe checkpoint")
    print("=" * 60)
    print(f"  Test period:          {len(test_data)} bars ({args.oos_frac*100:.0f}% hold-out)")
    print("  Initial balance:      $100,000.00")
    print(f"  Final balance:        ${final_value:,.2f}")
    print(f"  Total return:         {total_return_pct:+.2f}%")
    print(f"  Max drawdown:         {max_dd_pct:.2f}%")
    print("-" * 60)
    print(f"  Trades opened:        {n_trades_opened}")
    print(f"  Trades closed:        {len(trade_pnls_arr)}")
    print(f"  Trade win rate:       {trade_win_rate*100:.2f}%")
    print(f"  Trade profit factor:  {trade_profit_factor:.3f}")
    print(f"  Gross profit:         ${gross_profit:,.2f}")
    print(f"  Gross loss:           ${gross_loss:,.2f}")
    print(f"  Avg win:              ${avg_win:,.2f}")
    print(f"  Avg loss:             ${avg_loss:,.2f}")
    print("-" * 60)
    print(f"  Steps:                {step_count}")
    for key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown"]:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:.6f}")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(model_path),
        "oos_frac": args.oos_frac,
        "test_bars": int(len(test_data)),
        "initial_balance": 100_000.0,
        "final_balance": final_value,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "n_trades_opened": int(n_trades_opened),
        "n_trades_closed": int(len(trade_pnls_arr)),
        "trade_win_rate": trade_win_rate,
        "trade_profit_factor": trade_profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "return_metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
