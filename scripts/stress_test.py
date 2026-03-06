#!/usr/bin/env python3
"""Standalone stress testing CLI for ApexFX Quantum.

Usage:
    python scripts/stress_test.py --portfolio-value 100000
    python scripts/stress_test.py --portfolio-value 100000 --position-pct 0.15
    python scripts/stress_test.py --portfolio-value 100000 --monte-carlo --returns-file data/returns.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from apexfx.risk.stress_testing import StressTester


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ApexFX Quantum — Stress Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --portfolio-value 100000
  %(prog)s --portfolio-value 100000 --position-pct 0.15
  %(prog)s --portfolio-value 100000 --monte-carlo --returns-file data/returns.csv
  %(prog)s --portfolio-value 100000 --reverse --target-loss 0.20
        """,
    )

    parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100_000.0,
        help="Current portfolio value (default: 100000)",
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=0.10,
        help="Current position as fraction of portfolio (default: 0.10)",
    )
    parser.add_argument(
        "--var-limit",
        type=float,
        default=0.02,
        help="VaR limit as fraction (default: 0.02)",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo simulation (requires --returns-file)",
    )
    parser.add_argument(
        "--returns-file",
        type=str,
        default=None,
        help="Path to CSV file with daily returns for Monte Carlo",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Monte Carlo horizon in trading days (default: 21)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run reverse stress test",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.20,
        help="Target loss for reverse stress test (default: 0.20)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  APEXFX QUANTUM — STRESS TESTING FRAMEWORK")
    print("=" * 70)
    print(f"  Portfolio Value:  ${args.portfolio_value:,.2f}")
    print(f"  Position Size:    {args.position_pct:.1%}")
    print(f"  VaR Limit:        {args.var_limit:.1%}")
    print("=" * 70)
    print()

    tester = StressTester(
        var_limit=args.var_limit,
        margin_requirement=0.01,
    )

    # --- Preset Scenarios ---
    print("📊 PRESET SCENARIO ANALYSIS")
    print("-" * 70)
    results = tester.run_all_presets(
        portfolio_value=args.portfolio_value,
        current_position_pct=args.position_pct,
    )

    print(f"{'Scenario':<22} {'P&L Impact':>12} {'Max DD':>8} {'VaR Breach':>11} {'Survival':>10}")
    print("-" * 70)
    for r in results:
        survival_str = "✅ YES" if r.survival else "❌ NO"
        var_str = "⚠️  YES" if r.var_breach else "✅ NO"
        print(
            f"{r.scenario.name:<22} "
            f"${r.pnl_impact:>10,.2f} "
            f"{r.max_drawdown_pct:>7.2%} "
            f"{var_str:>11} "
            f"{survival_str:>10}"
        )

    n_survived = sum(1 for r in results if r.survival)
    n_var_breach = sum(1 for r in results if r.var_breach)
    print("-" * 70)
    print(f"Survived: {n_survived}/{len(results)} | VaR Breaches: {n_var_breach}/{len(results)}")
    print()

    # --- Monte Carlo ---
    if args.monte_carlo:
        print("🎲 MONTE CARLO SIMULATION")
        print("-" * 70)

        if args.returns_file:
            import pandas as pd
            df = pd.read_csv(args.returns_file)
            # Expect a column named 'return' or the first numeric column
            if "return" in df.columns:
                returns = df["return"].dropna().values
            else:
                returns = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values
        else:
            # Generate synthetic returns for demo
            print("  (No returns file provided — using synthetic returns)")
            rng = np.random.default_rng(42)
            returns = rng.normal(0.0001, 0.01, size=500)

        mc = tester.monte_carlo_stress(
            returns=returns,
            n_simulations=args.n_sims,
            horizon_days=args.horizon,
            portfolio_value=args.portfolio_value,
        )

        print(f"  Simulations:        {mc.n_simulations:,}")
        print(f"  Horizon:            {args.horizon} trading days")
        print(f"  VaR (95%):          {mc.var_95:.4f}  (${mc.var_95 * args.portfolio_value:,.2f})")
        print(f"  VaR (99%):          {mc.var_99:.4f}  (${mc.var_99 * args.portfolio_value:,.2f})")
        print(f"  CVaR (95%):         {mc.cvar_95:.4f}  (${mc.cvar_95 * args.portfolio_value:,.2f})")
        print(f"  CVaR (99%):         {mc.cvar_99:.4f}  (${mc.cvar_99 * args.portfolio_value:,.2f})")
        print(f"  Median Return:      {mc.median_return:.4f}")
        print(f"  Worst Path Return:  {mc.worst_path_return:.4f}")
        print(f"  Survival Rate:      {mc.survival_rate:.1%}")
        print()

    # --- Reverse Stress Test ---
    if args.reverse:
        print("🔍 REVERSE STRESS TEST")
        print("-" * 70)
        rs = tester.reverse_stress_test(
            portfolio_value=args.portfolio_value,
            current_position_pct=args.position_pct,
            target_loss_pct=args.target_loss,
        )
        print(f"  Target Loss:        {rs.target_loss_pct:.1%}")
        print(f"  Min Shock Required: {rs.min_shock_pct:.4%}")
        if rs.probability_estimate is not None:
            print(f"  Probability Est:    {rs.probability_estimate:.6f}")
        print(f"  Description:        {rs.scenario_description}")
        print()

    print("=" * 70)
    print("  Stress testing complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
