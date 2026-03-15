"""Main live trading event loop — the final assembly point."""

from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from apexfx.config.schema import AppConfig, SymbolConfig
from apexfx.data.bar_aggregator import BarAggregator
from apexfx.data.calendar_fetcher import CalendarFetcher
from apexfx.data.data_store import DataStore
from apexfx.data.mt5_client import MT5Client
from apexfx.data.tick_collector import TickCollector
from apexfx.execution.executor import Executor
from apexfx.features.fundamental import FundamentalExtractor
from apexfx.features.pipeline import FeaturePipeline
from apexfx.live.health_check import HealthCheck
from apexfx.live.shadow_trader import ShadowTrader, GradualRollout
from apexfx.data.mtf_aligner import MTFDataAligner, MTFSlice
from apexfx.env.mtf_obs_builder import MTFObservationBuilder
from apexfx.live.signal_generator import MTFSignalGenerator, SignalGenerator
from apexfx.live.state_manager import StateManager
from apexfx.risk.risk_manager import MarketState, RiskManager
from apexfx.utils.logging import get_logger
from apexfx.utils.math_utils import atr

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

        # Calendar fetcher for live economic calendar updates
        self._calendar_fetcher: CalendarFetcher | None = None
        self._fundamental_extractor: FundamentalExtractor | None = None
        cal_cfg = config.data.calendar
        if cal_cfg.enabled and cal_cfg.auto_fetch:
            self._calendar_fetcher = CalendarFetcher()
            # Find FundamentalExtractor in the pipeline
            for ext in self._feature_pipeline._extractors:
                if isinstance(ext, FundamentalExtractor):
                    self._fundamental_extractor = ext
                    break

        # Calendar refresh interval
        self._calendar_interval_s = cal_cfg.fetch_interval_hours * 3600

        # State persistence interval
        self._persist_interval_s = 60

        # --- Phase 4C: Shadow Trading / A/B Testing ---
        shadow_cfg = config.training.shadow_trading
        if shadow_cfg.enabled:
            self._shadow_trader = ShadowTrader(
                evaluation_bars=shadow_cfg.evaluation_bars,
                promotion_threshold=shadow_cfg.promotion_sharpe_delta,
            )
            self._gradual_rollout = GradualRollout(
                ramp_bars=shadow_cfg.gradual_rollout_bars,
            )
        else:
            self._shadow_trader: ShadowTrader | None = None
            self._gradual_rollout: GradualRollout | None = None

        # --- Phase 4C: Online Learning ---
        self._online_learner = None
        ol_cfg = config.training.online_learning
        if ol_cfg.enabled:
            try:
                from apexfx.training.online_learner import OnlineLearner
                self._online_learner = OnlineLearner(
                    model_path=model_file,
                    config=config,
                    mode=ol_cfg.mode,
                    retrain_window_days=ol_cfg.retrain_window_days,
                    retrain_steps=ol_cfg.retrain_steps,
                    retrain_lr=ol_cfg.retrain_lr,
                    min_new_bars=ol_cfg.min_new_bars,
                    validation_sharpe_min=ol_cfg.validation_sharpe_min,
                )
            except Exception as e:
                logger.warning("Online learner init failed", error=str(e))

        self._online_learning_interval_s = ol_cfg.check_interval_hours * 3600
        self._new_bars_count = 0

        # --- TIER 1: Live Online Learner (micro-updates) ---
        self._live_learner = None
        if ol_cfg.enabled and ol_cfg.update_every_n_trades > 0:
            try:
                from apexfx.live.online_learner import LiveOnlineLearner
                self._live_learner = LiveOnlineLearner(
                    model=self._signal_gen._model,
                    config=ol_cfg,
                )
            except Exception as e:
                logger.warning("Live online learner init failed", error=str(e))

        # --- TIER 1: Regime transition tracking ---
        self._prev_regime: str | None = None
        self._regime_transition_cooldown: int = 0

        # --- Phase 4C: News Sentiment ---
        self._sentiment_extractor = None
        try:
            from apexfx.features.sentiment import SentimentExtractor
            for ext in self._feature_pipeline._extractors:
                if isinstance(ext, SentimentExtractor):
                    self._sentiment_extractor = ext
                    break
        except ImportError:
            pass

        # --- Intermarket data (DXY, Gold, US10Y, SPX) ---
        self._intermarket_provider = None
        intermarket_symbols = config.symbols.intermarket
        if intermarket_symbols:
            try:
                from apexfx.data.intermarket import IntermarketDataProvider
                self._intermarket_provider = IntermarketDataProvider(self._mt5)
                self._intermarket_symbols = intermarket_symbols
                logger.info(
                    "Intermarket provider initialized",
                    instruments=intermarket_symbols,
                )
            except Exception as e:
                logger.warning("Intermarket provider init failed", error=str(e))

        # --- Alerting system (Telegram, Webhook) ---
        self._alert_manager = None
        self._risk_alert_monitor = None
        alert_cfg = getattr(config, "alerts", None)
        if alert_cfg and alert_cfg.enabled:
            try:
                from apexfx.alerts.alert_manager import AlertLevel, AlertManager
                from apexfx.alerts.risk_alerts import RiskAlertMonitor

                level_map = {
                    "INFO": AlertLevel.INFO,
                    "WARNING": AlertLevel.WARNING,
                    "CRITICAL": AlertLevel.CRITICAL,
                    "EMERGENCY": AlertLevel.EMERGENCY,
                }
                min_level = level_map.get(alert_cfg.min_level.upper(), AlertLevel.WARNING)
                self._alert_manager = AlertManager(
                    cooldown_s=alert_cfg.cooldown_s,
                    min_level=min_level,
                )

                if alert_cfg.telegram_enabled:
                    import os
                    from apexfx.alerts.telegram_bot import TelegramAlerter
                    token = alert_cfg.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
                    chat_id = alert_cfg.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
                    if token and chat_id:
                        self._alert_manager.add_channel(TelegramAlerter(token, chat_id))

                if alert_cfg.webhook_enabled:
                    import os
                    from apexfx.alerts.webhook import WebhookAlerter
                    url = alert_cfg.webhook_url or os.getenv("ALERT_WEBHOOK_URL", "")
                    if url:
                        self._alert_manager.add_channel(
                            WebhookAlerter(url, fmt=alert_cfg.webhook_format)
                        )

                self._risk_alert_monitor = RiskAlertMonitor(self._alert_manager)
                logger.info("Alert system initialized")
            except Exception as e:
                logger.warning("Alert system init failed", error=str(e))

        # --- Real-time news stream (Finnhub WS + Fast RSS) ---
        self._news_stream = None
        news_cfg = config.data.news
        if news_cfg.enabled and self._sentiment_extractor is not None:
            try:
                from apexfx.data.realtime_news import RealtimeNewsStream
                self._news_stream = RealtimeNewsStream(news_cfg)
            except Exception as e:
                logger.warning("RealtimeNewsStream init failed", error=str(e))

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

            # Run startup stress test
            self._risk_manager.run_startup_stress_test(self._state.state.equity)

            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self._tick_collector.start()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._state_persist_loop()),
                asyncio.create_task(self._calendar_update_loop()),
                asyncio.create_task(self._online_learning_loop()),
            ]

            # Real-time news: prefer WebSocket stream, fallback to polling
            if self._news_stream is not None:
                tasks.append(asyncio.create_task(self._news_stream.start()))
                tasks.append(asyncio.create_task(self._news_consumer_loop()))
            else:
                tasks.append(asyncio.create_task(self._news_update_loop()))

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
        bars = self._bar_aggregator.process_ticks(ticks)
        self._health.update_tick_time(datetime.now(timezone.utc))

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

            # --- Alert monitor: check risk states and fire alerts ---
            if self._risk_alert_monitor:
                await self._risk_alert_monitor.check_risk_state(
                    self._risk_manager,
                    portfolio_value=self._state.state.equity,
                    symbol=self._symbol,
                )

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

            # 1b. Merge intermarket data (DXY, Gold, US10Y, SPX)
            if self._intermarket_provider is not None:
                try:
                    intermarket_df = self._intermarket_provider.get_all_intermarket(
                        self._intermarket_symbols,
                        timeframe="H1",
                        count=len(bars_df),
                    )
                    if not intermarket_df.empty and "time" in intermarket_df.columns:
                        bars_df = bars_df.merge(
                            intermarket_df, on="time", how="left",
                        )
                        bars_df = bars_df.ffill()
                except Exception as e:
                    logger.debug("Intermarket merge failed", error=str(e))

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
            self._new_bars_count += 1

            # --- TIER 1: Regime transition handling ---
            self._risk_manager.set_regime(signal.regime)

            if signal.regime != self._prev_regime and self._prev_regime is not None:
                if self._is_major_transition(self._prev_regime, signal.regime):
                    logger.warning(
                        "Major regime transition — flattening position",
                        from_regime=self._prev_regime,
                        to_regime=signal.regime,
                    )
                    await self._force_close_if_needed(bar)
                    self._regime_transition_cooldown = 3
                else:
                    logger.info(
                        "Minor regime transition — reducing position",
                        from_regime=self._prev_regime,
                        to_regime=signal.regime,
                    )
                    if self._state.state.current_position_volume > 0:
                        await self._executor.reduce_position(0.5)
            self._prev_regime = signal.regime

            # Skip new entries during cooldown
            if self._regime_transition_cooldown > 0:
                self._regime_transition_cooldown -= 1
                logger.debug(
                    "Regime cooldown active",
                    remaining=self._regime_transition_cooldown,
                )
                return

            # Shadow trading: record live signal for comparison
            if self._shadow_trader is not None:
                self._shadow_trader.on_bar(
                    live_action=signal.action,
                    shadow_actions={},  # Shadow models registered externally
                    actual_price=bar.close,
                    live_return=0.0,
                )

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
            # Pass uncertainty score to risk evaluation for position scaling
            risk_decision = self._risk_manager.evaluate_action(
                signal.action,
                market_state,
                uncertainty_score=signal.uncertainty_score,
            )

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
                    trade_return = pnl / self._state.state.equity if self._state.state.equity > 0 else 0.0
                    self._state.close_position(result.fill_price or bar.close, pnl)
                    self._risk_manager.record_trade(pnl, trade_return=trade_return)

                    # --- TIER 1: Live learner — record trade result ---
                    if self._live_learner is not None:
                        self._live_learner.record_trade_result(trade_return)
                        self._live_learner.maybe_update()

            # --- TIER 1: Live learner — record transition ---
            if self._live_learner is not None:
                self._live_learner.record_transition(
                    obs=observation,
                    action=signal.action,
                    reward=0.0,  # Will be overwritten by actual reward later
                    next_obs=observation,  # Approximation — true next_obs unavailable
                    done=False,
                )

            # 7. Update state
            self._update_portfolio_value(bar.close)

            logger.info(
                "Bar processed",
                time=str(bar.time),
                action=round(signal.action, 4),
                regime=signal.regime,
                uncertainty=round(signal.uncertainty_score, 4),
                risk_approved=risk_decision.approved,
                equity=round(self._state.state.equity, 2),
            )

        except Exception as e:
            logger.error("Bar processing error", error=str(e), exc_info=True)

    @staticmethod
    def _is_major_transition(from_regime: str, to_regime: str) -> bool:
        """Determine if a regime transition is major (requires flattening).

        Major transitions involve a fundamental change in market character:
        - trending ↔ mean_reverting: strategy needs to reverse
        - any ↔ volatile: risk profile changes drastically

        Minor transitions are between adjacent regimes:
        - trending ↔ flat, mean_reverting ↔ flat
        """
        major_pairs = {
            frozenset({"trending", "mean_reverting"}),
            frozenset({"trending", "volatile"}),
            frozenset({"mean_reverting", "volatile"}),
        }
        return frozenset({from_regime, to_regime}) in major_pairs

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
        """Periodic health checks with exponential backoff reconnect."""
        reconnect_attempts = 0
        max_reconnect_delay = 120  # Max 2 minutes between retries

        while self._running:
            health = self._health.check()
            if not health.overall_healthy:
                logger.warning("System unhealthy", issues=health.issues)

                # Auto-reconnect MT5 with exponential backoff
                if not health.mt5_connected:
                    # Alert on first disconnect
                    if self._risk_alert_monitor and reconnect_attempts == 0:
                        await self._risk_alert_monitor.on_mt5_disconnect()

                    delay = min(2 ** reconnect_attempts, max_reconnect_delay)
                    reconnect_attempts += 1
                    logger.info(
                        "MT5 reconnect attempt",
                        attempt=reconnect_attempts,
                        delay_s=delay,
                    )
                    try:
                        self._mt5.connect()
                        logger.info("MT5 reconnected", attempts=reconnect_attempts)
                        if self._risk_alert_monitor:
                            await self._risk_alert_monitor.on_mt5_reconnect()
                        reconnect_attempts = 0  # Reset on success
                    except Exception as e:
                        logger.error(
                            "MT5 reconnect failed",
                            error=str(e),
                            next_retry_s=delay,
                        )
                else:
                    reconnect_attempts = 0  # Reset if connected

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

    async def _calendar_update_loop(self) -> None:
        """Periodically fetch latest economic calendar from Forex Factory."""
        if not self._calendar_fetcher or not self._fundamental_extractor:
            return

        while self._running:
            try:
                events = self._calendar_fetcher.fetch_current_week()
                if events:
                    self._fundamental_extractor.set_events(events)
                    logger.info(
                        "Calendar updated",
                        n_events=len(events),
                        next_high_impact=next(
                            (
                                f"{e.name} ({e.currency}) at {e.time_utc:%H:%M UTC}"
                                for e in events
                                if e.impact == "high"
                                and e.time_utc > datetime.now(timezone.utc)
                            ),
                            "none",
                        ),
                    )
            except Exception as e:
                logger.error("Calendar update failed", error=str(e))

            await asyncio.sleep(self._calendar_interval_s)

    async def _online_learning_loop(self) -> None:
        """Periodically check if model should be retrained on recent data."""
        if self._online_learner is None:
            return

        while self._running:
            try:
                if self._online_learner.should_retrain(self._new_bars_count):
                    logger.info(
                        "Online learning: retraining triggered",
                        new_bars=self._new_bars_count,
                    )
                    # Build recent data from bars buffer
                    if len(self._bars_buffer) > 24:
                        recent_data = pd.DataFrame(self._bars_buffer)
                        result = self._online_learner.retrain(recent_data)
                        if result.promoted:
                            # Reload model in signal generator
                            try:
                                self._signal_gen = SignalGenerator(
                                    result.model_path, device="cpu"
                                )
                                logger.info(
                                    "Model updated via online learning",
                                    sharpe_improvement=round(result.sharpe_delta, 4),
                                )
                            except Exception as e:
                                logger.error("Model reload failed", error=str(e))
                        self._new_bars_count = 0
            except Exception as e:
                logger.error("Online learning error", error=str(e))

            await asyncio.sleep(self._online_learning_interval_s)

    async def _news_consumer_loop(self) -> None:
        """Consume headlines from RealtimeNewsStream queue in real-time.

        This replaces the old 15-minute polling with instant consumption
        from WebSocket + fast RSS sources.
        """
        if self._news_stream is None or self._sentiment_extractor is None:
            return

        queue = self._news_stream.headline_queue
        batch: list[dict] = []
        batch_interval = 1.0  # Flush batch every 1 second

        while self._running:
            try:
                # Collect headlines with timeout (batch for efficiency)
                try:
                    headline = await asyncio.wait_for(
                        queue.get(), timeout=batch_interval,
                    )
                    batch.append(headline.to_dict())

                    # Drain queue (non-blocking) to collect burst of news
                    while not queue.empty():
                        try:
                            h = queue.get_nowait()
                            batch.append(h.to_dict())
                        except asyncio.QueueEmpty:
                            break

                except asyncio.TimeoutError:
                    pass  # No new headlines — flush whatever we have

                # Flush batch to sentiment extractor
                if batch:
                    self._sentiment_extractor.update_headlines(batch)
                    urgent_count = sum(
                        1 for h in batch if h.get("is_urgent", False)
                    )
                    logger.debug(
                        "Real-time news batch processed",
                        n_headlines=len(batch),
                        urgent=urgent_count,
                    )
                    batch.clear()

            except Exception as e:
                logger.error("News consumer error", error=str(e))
                await asyncio.sleep(1.0)

    async def _news_update_loop(self) -> None:
        """Fallback: periodically fetch news headlines for sentiment features."""
        if self._sentiment_extractor is None:
            return

        while self._running:
            try:
                from apexfx.data.news_fetcher import NewsFetcher
                fetcher = NewsFetcher()
                headlines = fetcher.fetch_latest(max_items=20)
                if headlines:
                    self._sentiment_extractor.update_headlines(headlines)
                    logger.debug(
                        "Sentiment headlines updated",
                        n_headlines=len(headlines),
                    )
            except ImportError:
                logger.debug("NewsFetcher not available")
                return  # Exit loop if module not available
            except Exception as e:
                logger.error("News update failed", error=str(e))

            await asyncio.sleep(900)  # Every 15 minutes

    async def _cleanup(self) -> None:
        """Graceful cleanup on shutdown."""
        logger.info("Cleaning up...")

        # Stop real-time news stream
        if self._news_stream is not None:
            self._news_stream.stop()

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
