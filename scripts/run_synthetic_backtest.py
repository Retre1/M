"""Run a full backtest on synthetic data with a rule-based strategy.

No MT5 or trained model required — generates realistic synthetic market data,
computes features, runs the BacktestEngine with an SMA crossover strategy,
and produces an HTML report.

Usage:
    python scripts/run_synthetic_backtest.py
    python scripts/run_synthetic_backtest.py --bars 20000 --output report.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from apexfx.backtest.engine import BacktestConfig, BacktestEngine
from apexfx.backtest.report import generate_html_report
from apexfx.config.schema import (
    CooldownConfig,
    PositionSizingConfig,
    RiskConfig,
    StrategyFilterConfig,
    StressTestConfig,
)
from apexfx.data.synthetic import SyntheticDataGenerator
from apexfx.features.pipeline import FeaturePipeline
from apexfx.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def make_backtest_risk_config() -> RiskConfig:
    """Create a RiskConfig calibrated for backtesting on synthetic data.

    Key differences from production defaults:
    - Strategy filter: disabled structure/fundamental gates (synthetic data has no real
      news events or market structure — these features are always zero)
    - Cooldown: relaxed to 5 losses / 5 min (backtest time doesn't advance in real-time,
      so long cooldowns effectively kill all trading after first losing streak)
    - Position sizing: lower min_trades_for_kelly and higher min_lot_size
      to avoid quarter-Kelly producing sub-minimum lot sizes
    - Stress test: disabled (no point running Monte Carlo on synthetic data)
    """
    return RiskConfig(
        daily_var_limit=0.03,       # 3% daily (relaxed from 2%)
        max_drawdown_pct=0.10,      # 10% hard stop (relaxed from 5%)
        var_confidence=0.95,        # 95% (relaxed from 99%)
        var_lookback_days=60,       # Shorter lookback
        var_method="parametric",
        position_sizing=PositionSizingConfig(
            max_position_pct=0.30,      # 30% max notional — higher than production
                                        # because quarter-Kelly + regime scaling
                                        # reduces effective size to ~5-10%
            kelly_fraction=0.5,         # Half-Kelly (standard)
            min_trades_for_kelly=10,    # 10 trades to start Kelly (from 30)
            vol_lookback_bars=20,
            min_lot_size=0.01,
        ),
        cooldown=CooldownConfig(
            after_n_losses=50,          # Effectively disabled for backtest —
                                        # CooldownManager uses real-time (datetime.now),
                                        # not simulated bar time, making it incompatible
                                        # with backtesting where 15K bars pass in seconds.
            duration_minutes=1,         # 1 min (minimal if somehow triggered)
            tilt_drawdown_pct=0.50,     # 50% — effectively disabled
            tilt_window_minutes=1,
        ),
        strategy_filter=StrategyFilterConfig(
            enabled=True,
            news_blackout_threshold=0.8,       # Higher threshold (from 0.5)
            time_to_event_threshold=0.005,     # Lower threshold
            min_fundamental_bias=0.0,          # No bias required (from 0.3)
            require_structure_confirm=False,   # OFF — synthetic has no real structure
            exit_on_conflict=True,
            reduce_scale_pre_news=0.7,         # Less reduction (from 0.5)
            pre_news_time_threshold=0.05,
            block_against_bias=False,          # OFF — synthetic has no real bias
            min_bias_for_direction=0.8,        # Higher threshold (from 0.5)
        ),
        stress_test=StressTestConfig(
            enabled=False,   # Disabled for backtest
        ),
    )


class SMACrossoverStrategy:
    """SMA crossover strategy with RSI confirmation and regime awareness.

    - Long when fast SMA > slow SMA and RSI < 70
    - Short when fast SMA < slow SMA and RSI > 30
    - Reduced exposure during volatile regimes
    - Signal strength from SMA divergence, amplified for trading frequency
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._closes: list[float] = []

    def _rsi(self, period: int = 14) -> float:
        if len(self._closes) < period + 1:
            return 50.0
        deltas = np.diff(self._closes[-(period + 1) :])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / max(avg_loss, 1e-10)
        return 100 - 100 / (1 + rs)

    def on_bar(self, features: pd.Series, bar: pd.Series) -> float:
        self._closes.append(float(bar["close"]))

        if len(self._closes) < self.slow_period + 1:
            return 0.0

        fast_sma = np.mean(self._closes[-self.fast_period :])
        slow_sma = np.mean(self._closes[-self.slow_period :])
        rsi = self._rsi()

        # Regime filter
        regime_scale = 1.0
        if "regime_label" in features.index:
            regime = features["regime_label"]
            if regime == 2:  # volatile
                regime_scale = 0.4
            elif regime == 3:  # flat
                regime_scale = 0.6

        # Signal: binary direction with confidence from divergence magnitude
        divergence = (fast_sma - slow_sma) / slow_sma

        # Dead zone: skip tiny divergences
        if abs(divergence) < 0.0001:
            return 0.0

        # Direction from crossover, confidence = 0.5 base + 0.5 scaled by strength
        # This ensures minimum confidence of 0.5 for any valid signal,
        # which the PositionSizer can convert to a non-zero lot size
        direction = 1.0 if divergence > 0 else -1.0
        strength = min(abs(divergence) * 200, 1.0)  # 0→1 from divergence
        confidence = 0.5 + 0.5 * strength  # 0.5→1.0

        signal = direction * confidence * regime_scale

        # RSI filter: reduce on extreme readings
        if signal > 0 and rsi > 75:
            signal *= 0.5
        elif signal < 0 and rsi < 25:
            signal *= 0.5

        return float(np.clip(signal, -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest on synthetic data")
    parser.add_argument("--bars", type=int, default=15000, help="Number of bars to generate (synthetic mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="backtest_report.html", help="HTML report output path")
    parser.add_argument("--json-output", default="backtest_results.json", help="JSON results output")
    parser.add_argument("--disable-risk", action="store_true", help="Disable risk manager entirely")
    parser.add_argument("--calibrated-risk", action="store_true", default=True,
                        help="Use backtest-calibrated risk config (default)")
    parser.add_argument("--production-risk", action="store_true",
                        help="Use production risk config (strict)")
    parser.add_argument("--fast-sma", type=int, default=10, help="Fast SMA period")
    parser.add_argument("--slow-sma", type=int, default=30, help="Slow SMA period")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real data from data/ directory instead of synthetic")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol (for real data)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (for real data)")
    parser.add_argument("--data-dir", default="data", help="Data directory (for real data)")
    args = parser.parse_args()

    setup_logging(level="INFO", fmt="console")

    # 1. Load data
    if args.real_data:
        from apexfx.data.data_store import DataStore
        store = DataStore(args.data_dir)
        bars = store.read_bars(args.symbol, args.timeframe)
        if bars.empty:
            logger.error("No real data found. Run download_forex_data.py first.",
                         symbol=args.symbol, timeframe=args.timeframe)
            return
        # Add spread column if missing
        if "spread" not in bars.columns:
            gen = SyntheticDataGenerator(seed=args.seed)
            bars["spread"] = gen.generate_spread(len(bars))
        logger.info(
            "Real data loaded",
            symbol=args.symbol,
            bars=len(bars),
            price_range=f"{bars['close'].min():.4f} - {bars['close'].max():.4f}",
            period=f"{str(bars['time'].iloc[0])[:10]} → {str(bars['time'].iloc[-1])[:10]}",
        )
    else:
        logger.info("Generating synthetic data", bars=args.bars, seed=args.seed)
        gen = SyntheticDataGenerator(seed=args.seed)
        bars = gen.generate_regime_switching(
            n_steps=args.bars,
            initial_price=1.1000,
            use_garch=True,
        )
        bars = gen.inject_black_swans(bars, intensity=0.0005, cluster_prob=0.3)
        bars["spread"] = gen.generate_spread(len(bars))
        logger.info(
            "Synthetic data generated",
            bars=len(bars),
            price_range=f"{bars['close'].min():.4f} - {bars['close'].max():.4f}",
        )

    # 2. Configure backtest
    config = BacktestConfig(
        initial_equity=100_000.0,
        commission_per_lot=7.0,
        slippage_pips=0.5,
        warmup_bars=300,
        symbol="EURUSD",
        disable_risk=args.disable_risk,
        default_volume=0.1,
        atr_stop_mult=2.0,
        atr_tp_mult=3.0,
        use_trailing_stop=True,
        min_position_floor=0.01,  # Floor sub-minimum positions to 0.01 lots
    )

    # 3. Create strategy
    strategy = SMACrossoverStrategy(
        fast_period=args.fast_sma,
        slow_period=args.slow_sma,
    )

    # 4. Build risk config
    risk_config = None
    if not args.disable_risk and not args.production_risk:
        risk_config = make_backtest_risk_config()
        logger.info("Using backtest-calibrated risk config")
    elif args.production_risk:
        logger.info("Using production (strict) risk config")
    else:
        logger.info("Risk manager DISABLED")

    # 5. Run backtest
    logger.info("Running backtest...")
    pipeline = FeaturePipeline(normalize=True)

    engine = BacktestEngine(
        bars=bars,
        strategy=strategy,
        config=config,
        pipeline=pipeline,
        risk_config=risk_config,
    )

    # Calibrate risk manager internals for backtesting:
    # 1. Warm up VolatilityTargeter with synthetic daily returns (otherwise it
    #    returns 0.5 scale for first 20+ bars, killing most position sizes)
    # 2. Warm up VaR calculator with daily returns
    if risk_config is not None and not args.disable_risk:
        closes = bars["close"].values
        daily_closes = closes[::24]  # H1 → daily
        if len(daily_closes) > 1:
            daily_returns = np.diff(daily_closes) / daily_closes[:-1]
            for ret in daily_returns[:60]:
                engine._risk_manager.record_daily_return(float(ret))
        # Override vol targeter with warmup data
        engine._risk_manager.vol_targeter._returns = list(daily_returns[:60])


        logger.info(
            "Risk manager warmed up",
            daily_returns=len(daily_returns[:60]),
            vol_leverage=f"{engine._risk_manager.vol_targeter.compute_leverage():.2f}",
        )

    result = engine.run()

    # 6. Print summary
    print("\n" + result.summary())

    # 6b. Print rejection reasons breakdown
    if result.risk_rejection_reasons:
        print("\n  --- Risk Rejection Breakdown ---")
        for reason, count in sorted(
            result.risk_rejection_reasons.items(), key=lambda x: -x[1]
        ):
            print(f"  {reason:50s}: {count:5d}")
        print()

    # 7. Generate HTML report
    output_path = Path(args.output)
    generate_html_report(result, output_path=output_path)
    logger.info("HTML report saved", path=str(output_path))

    # 8. Save JSON results
    import json

    json_path = Path(args.json_output)
    metrics = result.metrics
    with open(json_path, "w") as f:
        json.dump(
            {
                "strategy": "SMA Crossover",
                "fast_sma": args.fast_sma,
                "slow_sma": args.slow_sma,
                "bars": args.bars,
                "data_type": "synthetic (regime-switching + GARCH + black swans)",
                "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("JSON results saved", path=str(json_path))

    print(f"\nReport: {output_path.resolve()}")
    print(f"Results: {json_path.resolve()}")


if __name__ == "__main__":
    main()
