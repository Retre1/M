"""Quick backtest using an already-trained model (no retraining)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from apexfx.config.registry import init_config
from apexfx.data.data_store import DataStore
from apexfx.data.mtf_synthetic import resample_real_data
from apexfx.env.forex_env import ForexTradingEnv
from apexfx.env.mtf_forex_env import MTFForexTradingEnv
from apexfx.env.reward import DifferentialSharpeReward
from apexfx.features.pipeline import FeaturePipeline
from apexfx.utils.logging import get_logger, setup_logging
from apexfx.utils.metrics import compute_all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick backtest with pre-trained model")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--model-path", default="models/best/final_model",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--output", default="backtest_results.json")
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(level=config.base.logging.level, fmt="console")
    logger = get_logger(__name__)

    # Load data
    store = DataStore(config.base.paths.data_dir)
    data = store.read_bars(args.symbol, args.timeframe)

    if data.empty:
        logger.error("No data available for backtest")
        return

    # Compute features
    pipeline = FeaturePipeline()
    features = pipeline.compute(data)
    logger.info("Data ready", n_bars=len(features))

    # Load pre-trained model
    model_path = args.model_path
    model_path_zip = model_path + ".zip" if not model_path.endswith(".zip") else model_path

    if Path(model_path_zip).exists():
        load_path = model_path_zip
    elif Path(model_path).exists():
        load_path = model_path
    else:
        logger.error(f"Model not found: {model_path}")
        return

    logger.info("Loading model", path=load_path)
    try:
        model = SAC.load(load_path, device="cpu")
    except Exception:
        model = PPO.load(load_path, device="cpu")
    logger.info("Model loaded")

    # Split: first 70% for "seen" data, last 30% for out-of-sample test
    split_idx = int(len(features) * 0.7)
    test_data = features.iloc[split_idx:].reset_index(drop=True)
    logger.info("Test data", n_bars=len(test_data), split="last 30%")

    n_features = min(pipeline.n_features, 30)

    # Check if MTF mode is enabled
    mtf_enabled = config.model.mtf.enabled if hasattr(config.model, "mtf") else False

    if mtf_enabled:
        logger.info("MTF mode enabled — resampling test data to D1/H1/M5")
        d1_test, m5_test = resample_real_data(test_data)
        d1_features = pipeline.compute(d1_test)
        m5_features = pipeline.compute(m5_test)

        mtf_cfg = config.model.mtf
        env = MTFForexTradingEnv(
            h1_data=test_data,
            d1_data=d1_features,
            m5_data=m5_features,
            initial_balance=100_000.0,
            n_market_features=n_features,
            d1_lookback=mtf_cfg.lookback.d1,
            h1_lookback=mtf_cfg.lookback.h1,
            m5_lookback=mtf_cfg.lookback.m5,
            reward_fn=DifferentialSharpeReward(),
            max_drawdown_pct=1.0,
        )
    else:
        env = ForexTradingEnv(
            data=test_data,
            initial_balance=100_000.0,
            n_market_features=n_features,
            lookback=config.data.feature_window,
            reward_fn=DifferentialSharpeReward(),
            max_drawdown_pct=1.0,  # Don't terminate during evaluation
        )

    # Run backtest
    obs, info = env.reset()
    done = False
    returns: list[float] = []
    equity: list[float] = [100_000.0]
    trades: list[dict] = []
    prev_value = 100_000.0
    prev_position = 0.0
    step_count = 0

    logger.info("Running backtest...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_value = info.get("portfolio_value", prev_value)
        position = info.get("position", 0.0)

        if prev_value > 0:
            step_return = (current_value - prev_value) / prev_value
            returns.append(step_return)

        # Track trades
        if abs(position) > 0 and abs(prev_position) == 0:
            trades.append({"step": step_count, "type": "open", "position": position})
        elif abs(position) == 0 and abs(prev_position) > 0:
            trades.append({"step": step_count, "type": "close", "pnl": current_value - prev_value})

        equity.append(current_value)
        prev_value = current_value
        prev_position = position
        step_count += 1

    # Compute metrics
    returns_arr = np.array(returns)
    metrics = compute_all_metrics(returns_arr)

    # Additional stats
    final_value = equity[-1]
    total_return_pct = (final_value - 100_000) / 100_000 * 100
    max_equity = max(equity)
    if max_equity > 0:
        drawdown = max_equity - min(equity[equity.index(max_equity):])
        max_drawdown_pct = drawdown / max_equity * 100
    else:
        max_drawdown_pct = 0
    n_trades = len([t for t in trades if t["type"] == "open"])

    # Print results
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS (Out-of-Sample: last 30%)")
    print("=" * 60)
    print(f"  Test period:        {len(test_data)} bars")
    print("  Initial balance:    $100,000.00")
    print(f"  Final balance:      ${final_value:,.2f}")
    print(f"  Total return:       {total_return_pct:+.2f}%")
    print(f"  Max drawdown:       {max_drawdown_pct:.2f}%")
    print(f"  Total trades:       {n_trades}")
    print(f"  Steps:              {step_count}")
    print("-" * 60)

    for key, value in sorted(metrics.items()):
        print(f"  {key:25s}: {value:.6f}")

    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    results = {
        "test_bars": len(test_data),
        "initial_balance": 100_000,
        "final_balance": float(final_value),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "n_trades": n_trades,
        "n_steps": step_count,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
