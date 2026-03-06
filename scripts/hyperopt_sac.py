#!/usr/bin/env python3
"""SAC hyperparameter optimization.

Runs Optuna-based hyperparameter search for SAC trading agent.
Results are saved to JSON and optionally to an SQLite study database.

Usage:
    python scripts/hyperopt_sac.py --n-trials 40 --timesteps 50000
    python scripts/hyperopt_sac.py --n-trials 20 --timeout 1800 --synthetic-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apexfx.config.loader import load_config
from apexfx.data.synthetic import SyntheticDataGenerator
from apexfx.features.pipeline import FeaturePipeline
from apexfx.features.selector import FeatureSelector
from apexfx.training.hyperopt import SACHyperoptManager
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=40, help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training steps per trial")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds for entire search")
    parser.add_argument("--top-n", type=int, default=15, help="Feature selection top-N")
    parser.add_argument("--synthetic-only", action="store_true", help="Use synthetic data")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite path for study persistence")
    args = parser.parse_args()

    config = load_config()

    # Load data
    raw_dir = Path(config.base.paths.data_dir) / "raw" / "bars"
    parquet_files = list(raw_dir.rglob("*.parquet")) if not args.synthetic_only else []

    if parquet_files:
        import pandas as pd
        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        data = pd.concat(dfs, ignore_index=True)
        logger.info("Loaded %d real bars", len(data))
    else:
        logger.info("Generating synthetic data for hyperopt")
        gen = SyntheticDataGenerator(seed=config.base.seed)
        data = gen.generate_gbm(n_steps=10_000, mu=0.0001, sigma=0.01)

    # Feature pipeline + selection
    pipeline = FeaturePipeline()
    features = pipeline.compute(data)

    selector = FeatureSelector(top_n=args.top_n)
    selected = selector.fit_transform(features)
    n_features = len(selector.selected_features)

    logger.info("Features: %d selected from %d total", n_features, pipeline.n_features)

    # Default DB path
    db_path = args.db_path
    if db_path is None:
        db_path = str(Path(config.base.paths.logs_dir) / "hyperopt_study.db")

    # Run optimization
    manager = SACHyperoptManager(
        base_config=config,
        features=selected,
        n_features=n_features,
        n_trials=args.n_trials,
        train_timesteps=args.timesteps,
        storage_path=db_path,
        device=args.device,
    )

    results = manager.optimize(timeout=args.timeout)

    # Print results
    print("\n" + "=" * 70)
    print("HYPEROPT RESULTS")
    print("=" * 70)
    print(f"  Trials completed: {results['n_completed']}")
    print(f"  Best trial:       #{results['best_trial']}")
    print(f"  Best Sharpe:      {results['best_value']:.4f}")
    print()
    print("  Best parameters:")
    for k, v in sorted(results["best_params"].items()):
        print(f"    {k:<25s} = {v}")
    print("=" * 70)

    # Save results to JSON
    save_path = Path(config.base.paths.logs_dir) / "hyperopt_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove study object (not serializable)
    save_data = {k: v for k, v in results.items() if k != "study"}
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {save_path}")

    # Show top-5 trials
    sorted_trials = sorted(results["trials"], key=lambda t: t["sharpe"], reverse=True)
    print("\n  Top-5 trials:")
    print(f"  {'#':<5}{'Sharpe':<10}{'Return%':<10}{'MaxDD%':<10}{'Trades':<8}{'Time(s)':<8}")
    print("  " + "-" * 55)
    for t in sorted_trials[:5]:
        ua = t.get("user_attrs", {})
        print(
            f"  {t['number']:<5}"
            f"{t['sharpe']:<10.4f}"
            f"{ua.get('total_return_pct', 0):<10.2f}"
            f"{ua.get('max_drawdown_pct', 0):<10.2f}"
            f"{ua.get('n_trades', 0):<8}"
            f"{t.get('duration_s', 0) or 0:<8.0f}"
        )

    # Apply best config
    best_config = manager.apply_best(results)
    config_save = Path(config.base.paths.logs_dir) / "best_hyperparams.json"
    with open(config_save, "w") as f:
        json.dump(results["best_params"], f, indent=2)
    print(f"\n  Best params saved to: {config_save}")
    print("=" * 70)


if __name__ == "__main__":
    main()
