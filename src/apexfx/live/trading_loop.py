"""Main live trading event loop — the final assembly point."""

from __future__ import annotations

import asyncio
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from apexfx.data.bar_aggregator import BarAggregator
from apexfx.data.data_store import DataStore
from apexfx.data.mt5_client import MT5Client
from apexfx.data.mtf_aligner import MTFDataAligner
from apexfx.data.tick_collector import TickCollector
from apexfx.env.mtf_obs_builder import MTFObservationBuilder
from apexfx.execution.executor import Executor
from apexfx.features.pipeline import FeaturePipeline
from apexfx.live.health_check import HealthCheck
from apexfx.live.signal_generator import MTFSignalGenerator, SignalGenerator
from apexfx.live.state_manager import StateManager
from apexfx.risk.risk_manager import MarketState, RiskManager
from apexfx.utils.logging import get_logger
from apexfx.utils.math_utils import atr

if TYPE_CHECKING:
    from apexfx.config.schema import AppConfig

logger = get_logger(__name__)


class LiveTradingLoop:
    """
    Main async event loop for live trading.

    Flow on each new bar:
    1. Feature pipeline -> update features
    2. Signal generator -> model inference
    3. Risk manager -> approve/veto/scale
    4. Executor -> MT5 order execution
    5. State manager -> update portfolio

    Resilient: MT5 disconnects -> reconnect, inference failures -> do nothing.
    """

    def __init__(
        self,
        config: AppConfig,
        symbol: str = "EURUSD",
        model_path: str | None = None,
    ) -> None:
        self._config = config
        self._symbol = symbol
        self._running = False

        # Resolve symbol config
        sym_cfg = config.symbols.symbols.get(symbol)
        if sym_cfg is None:
            raise ValueError(f"Symbol {symbol} not found in config")
        self._symbol_config = sym_cfg

        # Initialize components
        self._mt5 = MT5Client()
        self._store = DataStore(config.base.paths.data_dir)
        self._tick_collector = TickCollector(
            self._mt5, self._store, symbol,
            poll_interval_ms=config.data.collection.poll_interval_ms,
        )
        self._bar_aggregator = BarAggregator(config.data.timeframes)
        self._feature_pipeline = FeaturePipeline()

        # Model path
        model_file = model_path or str(
            Path(config.base.paths.models_dir) / "best" / "final_model"
        )

        # Risk & Execution
        self._risk_manager = RiskManager(config.risk)
        self._executor = Executor(config.execution, sym_cfg, self._mt5)
        self._executor.set_symbol(symbol)

        # State & Health
        self._state = StateManager()
        self._health = HealthCheck(self._mt5)

        # Data buffers
        self._feature_history = pd.DataFrame()
        self._bars_buffer: list[dict] = []

        # MTF buffers (D1 and M5 bar history for multi-timeframe mode)
        self._mtf_enabled = config.model.mtf.enabled if hasattr(config.model, "mtf") else False
        self._d1_bars_buffer: list[dict] = []
        self._m5_bars_buffer: list[dict] = []

        if self._mtf_enabled:
            self._mtf_obs_builder = MTFObservationBuilder(
                d1_lookback=config.model.mtf.lookback.d1,
                h1_lookback=config.model.mtf.lookback.h1,
                m5_lookback=config.model.mtf.lookback.m5,
            )
            self._signal_gen = MTFSignalGenerator(model_file, device="cpu")
        else:
            self._signal_gen = SignalGenerator(model_file, device="cpu")

        # State persistence interval
        self._persist_interval_s = 60

        # Lock to serialize bar processing (prevent race condition
        # where two H1 bars are processed in parallel)
        self._processing_lock = asyncio.Lock()

    async def run(self) -> None:
        """Main trading loop."""
        self._running = True
        logger.info("=" * 60)
        logger.info("APEXFX QUANTUM — LIVE TRADING STARTED")
        logger.info("=" * 60, symbol=self._symbol)

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        try:
            # Connect to MT5
            self._mt5.connect()

            # Register bar callback
            self._bar_aggregator.on_bar(self._on_bar_finalized)

            # Register tick callback for bar aggregation
            self._tick_collector.on_tick(self._on_new_ticks)

            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self._tick_collector.start()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._state_persist_loop()),
            ]

            # Wait for shutdown
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error("Trading loop error", error=str(e))
        finally:
            await self._cleanup()

    def _shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received")
        self._running = False
        self._tick_collector.stop()

    def _on_new_ticks(self, ticks: pd.DataFrame) -> None:
        """Process incoming ticks through bar aggregator."""
        self._bar_aggregator.process_ticks(ticks)
        self._health.update_tick_time(datetime.now(UTC))

    def _on_bar_finalized(self, bar) -> None:
        """Handle a finalized bar — trigger the trading pipeline."""
        # Capture D1 and M5 bars for MTF mode
        if self._mtf_enabled:
            bar_dict = {
                "time": bar.time,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "tick_count": bar.tick_count,
            }
            if bar.timeframe == "D1":
                self._d1_bars_buffer.append(bar_dict)
                if len(self._d1_bars_buffer) > 100:
                    self._d1_bars_buffer = self._d1_bars_buffer[-50:]
                return
            elif bar.timeframe == "M5":
                self._m5_bars_buffer.append(bar_dict)
                if len(self._m5_bars_buffer) > 5000:
                    self._m5_bars_buffer = self._m5_bars_buffer[-2500:]
                return

        if bar.timeframe != "H1":  # Primary trading timeframe
            return

        # Use lock-protected wrapper to prevent parallel bar processing
        asyncio.get_event_loop().create_task(self._process_bar_locked(bar))

    async def _process_bar_locked(self, bar) -> None:
        """Acquire processing lock before running the bar pipeline."""
        async with self._processing_lock:
            await self._process_bar(bar)

    async def _process_bar(self, bar) -> None:
        """Full trading pipeline for a finalized bar."""
        try:
            # --- Kill switch check: skip all processing if active ---
            if self._risk_manager.kill_switch.is_active:
                logger.error(
                    "Kill switch active — skipping bar processing",
                    reason=self._risk_manager.kill_switch.reason,
                )
                # Force close if we still have a position
                await self._force_close_if_needed(bar)
                return

            # --- Force close check: risk manager signals close-all ---
            if self._risk_manager.force_close_all():
                logger.warning("Risk manager signals force close all")
                await self._force_close_if_needed(bar)
                return

            # 1. Add bar to history
            bar_dict = {
                "time": bar.time,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "tick_count": bar.tick_count,
            }
            self._bars_buffer.append(bar_dict)

            # Keep buffer manageable
            if len(self._bars_buffer) > 1000:
                self._bars_buffer = self._bars_buffer[-500:]

            # Build DataFrame for feature computation
            bars_df = pd.DataFrame(self._bars_buffer)

            # 2. Feature pipeline
            try:
                features = self._feature_pipeline.compute(bars_df)
            except Exception as e:
                logger.error("Feature computation failed", error=str(e))
                return

            # 3. Build observation and generate signal
            if self._mtf_enabled and self._d1_bars_buffer and self._m5_bars_buffer:
                # MTF mode: build observation from D1 + H1 + M5
                d1_df = pd.DataFrame(self._d1_bars_buffer)
                m5_df = pd.DataFrame(self._m5_bars_buffer)

                d1_features = self._feature_pipeline.compute(d1_df)
                m5_features = self._feature_pipeline.compute(m5_df)

                aligner = MTFDataAligner(
                    d1_data=d1_features,
                    h1_data=features,
                    m5_data=m5_features,
                )
                mtf_slice = aligner.get_slice(len(features) - 1)

                observation = self._mtf_obs_builder.build(
                    mtf_slice=mtf_slice,
                    h1_features=features,
                    h1_idx=len(features) - 1,
                    position=float(
                        self._state.state.current_position_volume
                        * self._state.state.current_position_direction
                    ),
                    unrealized_pnl=self._state.state.unrealized_pnl,
                    time_in_position=float(self._state.state.time_in_position),
                    portfolio_value=self._state.state.equity,
                )
            else:
                # Single-TF mode
                obs_builder = self._signal_gen._obs_builder

                observation = obs_builder.build(
                    features=features,
                    current_idx=len(features) - 1,
                    position=float(
                        self._state.state.current_position_volume
                        * self._state.state.current_position_direction
                    ),
                    unrealized_pnl=self._state.state.unrealized_pnl,
                    time_in_position=float(self._state.state.time_in_position),
                    portfolio_value=self._state.state.equity,
                )

            signal = self._signal_gen.generate(observation)
            self._health.update_inference_latency(signal.inference_time_ms)

            # 4. Risk evaluation
            info = self._mt5.get_symbol_info(self._symbol)
            current_spread = info.ask - info.bid

            # Compute ATR for position sizing
            if len(bars_df) >= 14:
                atr_values = atr(
                    bars_df["high"].values,
                    bars_df["low"].values,
                    bars_df["close"].values,
                    period=14,
                )
                current_atr = float(atr_values[-1]) if not pd.isna(atr_values[-1]) else None
            else:
                current_atr = None

            market_state = MarketState(
                current_price=bar.close,
                current_spread=current_spread,
                current_atr=current_atr,
                spread_limit=self._symbol_config.spread_limit_pips * self._symbol_config.pip_value,
            )

            self._risk_manager.update_portfolio(self._state.state.equity)
            risk_decision = self._risk_manager.evaluate_action(signal.action, market_state)

            # 5. Sync MT5 positions before execution to ensure state consistency
            self._sync_mt5_positions()

            # 6. Execute
            if risk_decision.approved or risk_decision.adjusted_action == 0.0:
                result = await self._executor.execute(
                    risk_decision,
                    current_direction=self._state.state.current_position_direction,
                    current_volume=self._state.state.current_position_volume,
                )

                if result.success and result.action_taken == "opened":
                    direction = 1 if signal.action > 0 else -1
                    self._state.open_position(
                        self._symbol, direction,
                        result.volume, result.fill_price or bar.close,
                    )
                elif result.success and result.action_taken == "closed":
                    pnl = self._calculate_pnl(result.fill_price or bar.close)
                    equity = self._state.state.equity
                    trade_return = pnl / equity if equity > 0 else 0.0
                    self._state.close_position(result.fill_price or bar.close, pnl)
                    self._risk_manager.record_trade(pnl, trade_return=trade_return)

            # 7. Update state
            self._update_portfolio_value(bar.close)

            logger.info(
                "Bar processed",
                time=str(bar.time),
                action=round(signal.action, 4),
                risk_approved=risk_decision.approved,
                equity=round(self._state.state.equity, 2),
            )

        except Exception as e:
            logger.error("Bar processing error", error=str(e), exc_info=True)

    def _sync_mt5_positions(self) -> None:
        """Sync internal state with actual MT5 positions.

        Detects drift between our state and MT5 reality (e.g. manual close,
        SL/TP triggered, connection drops) and reconciles.
        """
        try:
            mt5_positions = self._mt5.get_positions(symbol=self._symbol)
            state = self._state.state

            has_internal_position = state.current_position_direction != 0
            has_mt5_position = len(mt5_positions) > 0

            if has_internal_position and not has_mt5_position:
                # Position was closed externally (SL/TP hit, manual close)
                logger.warning(
                    "Position sync: MT5 position gone — recording close",
                    symbol=self._symbol,
                )
                info = self._mt5.get_symbol_info(self._symbol)
                close_price = info.bid if state.current_position_direction > 0 else info.ask
                pnl = self._calculate_pnl(close_price)
                trade_return = pnl / state.equity if state.equity > 0 else 0.0
                self._state.close_position(close_price, pnl)
                self._risk_manager.record_trade(pnl, trade_return=trade_return)

                # Also reset executor internal state
                self._executor._current_position_direction = 0
                self._executor._current_position_volume = 0.0
                self._executor._current_position_ticket = None

            elif not has_internal_position and has_mt5_position:
                # Position exists in MT5 but not tracked — adopt it
                pos = mt5_positions[0]
                direction = 1 if pos.type == 0 else -1
                logger.warning(
                    "Position sync: MT5 position found but not tracked — adopting",
                    symbol=self._symbol,
                    direction=direction,
                    volume=pos.volume,
                )
                self._state.open_position(
                    self._symbol, direction, pos.volume, pos.price_open,
                )
                self._executor._current_position_direction = direction
                self._executor._current_position_volume = pos.volume
                self._executor._current_position_ticket = pos.ticket

            elif has_internal_position and has_mt5_position:
                # Both exist — sync volume/direction from MT5 (source of truth)
                pos = mt5_positions[0]
                mt5_direction = 1 if pos.type == 0 else -1
                if (
                    mt5_direction != state.current_position_direction
                    or abs(pos.volume - state.current_position_volume) > 1e-6
                ):
                    logger.warning(
                        "Position sync: volume/direction mismatch — correcting",
                        internal_dir=state.current_position_direction,
                        mt5_dir=mt5_direction,
                        internal_vol=state.current_position_volume,
                        mt5_vol=pos.volume,
                    )
                    state.current_position_direction = mt5_direction
                    state.current_position_volume = pos.volume
                    self._executor._current_position_direction = mt5_direction
                    self._executor._current_position_volume = pos.volume
                    self._executor._current_position_ticket = pos.ticket

        except Exception as e:
            logger.error("MT5 position sync failed", error=str(e))

    async def _force_close_if_needed(self, bar) -> None:
        """Force close open position if one exists."""
        state = self._state.state
        if state.current_position_direction == 0:
            return

        logger.warning("Force-closing position", symbol=self._symbol)
        try:
            result = await self._executor._close_position()
            if result.success:
                info = self._mt5.get_symbol_info(self._symbol)
                close_price = result.fill_price or (
                    info.bid if state.current_position_direction > 0 else info.ask
                )
                pnl = self._calculate_pnl(close_price)
                trade_return = pnl / state.equity if state.equity > 0 else 0.0
                self._state.close_position(close_price, pnl)
                self._risk_manager.record_trade(pnl, trade_return=trade_return)
                self._update_portfolio_value(bar.close)
                logger.info("Force close completed", pnl=round(pnl, 2))
            else:
                logger.error("Force close failed", message=result.message)
        except Exception as e:
            logger.error("Force close error", error=str(e))

    def _calculate_pnl(self, exit_price: float) -> float:
        """Calculate P&L for closing the current position."""
        state = self._state.state
        price_diff = exit_price - state.current_position_entry_price
        pnl = (
            price_diff * state.current_position_volume
            * self._symbol_config.contract_size
            * state.current_position_direction
        )
        return pnl

    def _update_portfolio_value(self, current_price: float) -> None:
        """Update portfolio equity with unrealized P&L."""
        state = self._state.state
        unrealized = 0.0
        if state.current_position_direction != 0:
            price_diff = current_price - state.current_position_entry_price
            unrealized = (
                price_diff * state.current_position_volume
                * self._symbol_config.contract_size
                * state.current_position_direction
            )
            self._state.increment_time_in_position()

        equity = state.balance + unrealized
        self._state.update_equity(equity, unrealized)

    async def _health_check_loop(self) -> None:
        """Periodic health checks."""
        while self._running:
            health = self._health.check()
            if not health.overall_healthy:
                logger.warning("System unhealthy", issues=health.issues)

                # Auto-reconnect MT5 if disconnected
                if not health.mt5_connected:
                    try:
                        self._mt5.connect()
                        logger.info("MT5 reconnected")
                    except Exception as e:
                        logger.error("MT5 reconnect failed", error=str(e))

                # Safe mode: when unhealthy and has open position, close it
                state = self._state.state
                if state.current_position_direction != 0:
                    logger.warning(
                        "Safe mode: closing position due to unhealthy system",
                        issues=health.issues,
                    )
                    try:
                        result = await self._executor._close_position()
                        if result.success:
                            info = self._mt5.get_symbol_info(self._symbol)
                            close_price = result.fill_price or (
                                info.bid if state.current_position_direction > 0 else info.ask
                            )
                            pnl = self._calculate_pnl(close_price)
                            trade_return = pnl / state.equity if state.equity > 0 else 0.0
                            self._state.close_position(close_price, pnl)
                            self._risk_manager.record_trade(pnl, trade_return=trade_return)
                            logger.info("Safe mode close completed", pnl=round(pnl, 2))
                        else:
                            logger.error("Safe mode close failed", message=result.message)
                    except Exception as e:
                        logger.error("Safe mode close error", error=str(e))

            await asyncio.sleep(30)

    async def _state_persist_loop(self) -> None:
        """Periodic state persistence."""
        while self._running:
            self._state.persist()
            await asyncio.sleep(self._persist_interval_s)

    async def _cleanup(self) -> None:
        """Graceful cleanup on shutdown."""
        logger.info("Cleaning up...")

        # Close any open positions
        state = self._state.state
        if state.current_position_direction != 0:
            logger.warning("Force-closing open position on shutdown")
            try:
                info = self._mt5.get_symbol_info(self._symbol)
                price = info.bid if state.current_position_direction > 0 else info.ask
                pnl = self._calculate_pnl(price)
                self._state.close_position(price, pnl)
            except Exception as e:
                logger.error("Failed to close position on shutdown", error=str(e))

        # Persist final state
        self._state.persist()

        # Disconnect MT5
        self._mt5.disconnect()

        logger.info("=" * 60)
        logger.info("APEXFX QUANTUM — LIVE TRADING STOPPED")
        logger.info("=" * 60)
