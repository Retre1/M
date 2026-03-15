"""TIER 1 (P0): 5 Hedge-Fund Critical Features — comprehensive test suite.

Tests cover:
- PER Config extraction (schema + trainer integration)
- Advanced Data Augmentation Pipeline
- Multi-Symbol Portfolio Trading (PortfolioManager + MultiSymbolLoop)
- Live Online Incremental Learning
- Regime-Conditional Execution (signal generator, risk manager, trading loop)
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dict_obs(lookback: int = 20, n_features: int = 5) -> dict[str, np.ndarray]:
    """Create a minimal Dict observation matching ForexTradingEnv space."""
    rng = np.random.default_rng(42)
    return {
        "market_features": rng.standard_normal(lookback * n_features).astype(np.float32),
        "time_features": rng.standard_normal(lookback * 4).astype(np.float32),
        "trend_features": rng.standard_normal(8).astype(np.float32),
        "reversion_features": rng.standard_normal(6).astype(np.float32),
        "regime_features": rng.standard_normal(6).astype(np.float32),
        "position_state": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


# =====================================================================
# 1. PER Config
# =====================================================================


class TestPERConfig:
    """Tests for Prioritized Experience Replay configuration."""

    def test_per_config_defaults(self):
        from apexfx.config.schema import PERConfig
        cfg = PERConfig()
        assert cfg.enabled is True
        assert cfg.alpha == 0.6
        assert cfg.beta_start == 0.4
        assert cfg.beta_end == 1.0
        assert cfg.beta_annealing_steps == 0
        assert cfg.epsilon == 1e-6
        assert cfg.update_freq == 100

    def test_per_config_in_training_config(self):
        from apexfx.config.schema import TrainingConfig
        tc = TrainingConfig()
        assert hasattr(tc, "per")
        assert tc.per.enabled is True

    def test_per_config_disabled(self):
        from apexfx.config.schema import PERConfig
        cfg = PERConfig(enabled=False)
        assert cfg.enabled is False


# =====================================================================
# 2. Advanced Data Augmentation
# =====================================================================


class TestAugmentation:
    """Tests for AugmentedObsWrapper and augmentation methods."""

    def test_augmentation_config_defaults(self):
        from apexfx.config.schema import AugmentationConfig
        cfg = AugmentationConfig()
        assert cfg.enabled is True
        assert cfg.time_warp_prob == 0.3
        assert cfg.magnitude_warp_prob == 0.3
        assert cfg.window_slice_prob == 0.2
        assert cfg.mixup_prob == 0.2

    def test_augmentation_config_in_training(self):
        from apexfx.config.schema import TrainingConfig
        tc = TrainingConfig()
        assert hasattr(tc, "augmentation")
        assert tc.augmentation.enabled is True

    def test_time_warp_preserves_shape(self):
        from apexfx.training.augmentation import AugmentedObsWrapper
        wrapper = AugmentedObsWrapper.__new__(AugmentedObsWrapper)
        wrapper._rng = np.random.default_rng(42)
        wrapper._lookback = 20
        arr = np.random.randn(100).astype(np.float32)  # 20 * 5 features
        result = wrapper._time_warp(arr, sigma=0.2)
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype

    def test_magnitude_warp_changes_values(self):
        from apexfx.training.augmentation import AugmentedObsWrapper
        wrapper = AugmentedObsWrapper.__new__(AugmentedObsWrapper)
        wrapper._rng = np.random.default_rng(42)
        wrapper._lookback = 20
        arr = np.ones(100, dtype=np.float32)  # 20 * 5
        result = wrapper._magnitude_warp(arr, sigma=0.1)
        assert result.shape == arr.shape
        # Should be different from all-ones (warped by smooth curve)
        assert not np.allclose(result, arr)

    def test_window_slice_zeros_front(self):
        from apexfx.training.augmentation import AugmentedObsWrapper
        wrapper = AugmentedObsWrapper.__new__(AugmentedObsWrapper)
        wrapper._rng = np.random.default_rng(42)
        wrapper._lookback = 20
        arr = np.ones(100, dtype=np.float32)
        result = wrapper._window_slice(arr, ratio=0.8)
        assert result.shape == arr.shape
        # Front should have zeros (from padding)
        reshaped = result.reshape(20, 5)
        assert np.any(reshaped[:4] == 0.0)  # First ~20% should be padded

    def test_mixup_blending(self):
        from apexfx.training.augmentation import AugmentedObsWrapper
        wrapper = AugmentedObsWrapper.__new__(AugmentedObsWrapper)
        wrapper._rng = np.random.default_rng(42)
        obs1 = {"market_features": np.ones(10, dtype=np.float32), "position_state": np.array([1.0])}
        obs2 = {"market_features": np.zeros(10, dtype=np.float32), "position_state": np.array([2.0])}
        result = wrapper._mixup(obs1, obs2, alpha=0.2)
        # market_features should be blended
        assert np.all(result["market_features"] >= 0.0)
        assert np.all(result["market_features"] <= 1.0)
        # position_state MUST remain unchanged
        np.testing.assert_array_equal(result["position_state"], np.array([1.0]))

    def test_position_state_never_augmented(self):
        """position_state must NEVER be modified by augmentation."""
        from apexfx.training.augmentation import AugmentedObsWrapper
        from apexfx.config.schema import AugmentationConfig

        cfg = AugmentationConfig(
            time_warp_prob=1.0,
            magnitude_warp_prob=1.0,
            window_slice_prob=1.0,
            mixup_prob=0.0,  # Disable mixup to isolate other methods
        )
        wrapper = AugmentedObsWrapper.__new__(AugmentedObsWrapper)
        wrapper._cfg = cfg
        wrapper._rng = np.random.default_rng(42)
        wrapper._lookback = 20
        wrapper._prev_obs = None

        obs = _make_dict_obs()
        original_pos = obs["position_state"].copy()

        result = wrapper._augment(obs)
        np.testing.assert_array_equal(result["position_state"], original_pos)


# =====================================================================
# 3. Multi-Symbol Portfolio Trading
# =====================================================================


class TestPortfolioManager:
    """Tests for PortfolioManager cross-pair risk."""

    def test_total_exposure_limit(self):
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager(max_total_exposure=0.40)
        pm.set_equity(100_000)

        # First trade: 30% exposure → allowed
        result = pm.check_new_trade("EURUSD", 1, 30_000)
        assert result.approved

        pm.update_position("EURUSD", 1, 0.3, 1.1000)

        # Second trade: 20% more → would exceed 40% limit
        result = pm.check_new_trade("GBPUSD", 1, 20_000)
        # Should be scaled down or partially approved
        assert result.scale_factor < 1.0 or not result.approved

    def test_per_symbol_limit(self):
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager(max_per_symbol=0.25)
        pm.set_equity(100_000)

        # First trade: 20% → allowed
        result = pm.check_new_trade("EURUSD", 1, 20_000)
        assert result.approved

        pm.update_position("EURUSD", 1, 0.2, 1.1000)

        # Second trade same symbol: 10% more → 30% > 25% limit
        result = pm.check_new_trade("EURUSD", 1, 10_000)
        assert result.scale_factor < 1.0 or not result.approved

    def test_correlation_check(self):
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager(correlation_limit=0.70)
        pm.set_equity(100_000)

        # Open EURUSD long
        pm.update_position("EURUSD", 1, 0.1, 1.1000)

        # GBPUSD long is highly correlated (~0.85) with EURUSD long
        result = pm.check_new_trade("GBPUSD", 1, 10_000)
        assert not result.approved
        assert "Correlation" in result.reason

    def test_close_position(self):
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        pm.set_equity(100_000)
        pm.update_position("EURUSD", 1, 0.1, 1.1000)
        assert len(pm.get_open_positions()) == 1
        pm.close_position("EURUSD", 1.1050, 50.0)
        assert len(pm.get_open_positions()) == 0

    def test_portfolio_metrics(self):
        from apexfx.live.portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        pm.set_equity(100_000)
        pm.update_position("EURUSD", 1, 0.1, 1.1000)
        metrics = pm.get_portfolio_metrics()
        assert metrics.n_positions == 1
        assert metrics.total_exposure > 0
        assert "EURUSD" in metrics.per_symbol_exposure


class TestMultiSymbolLoop:
    """Tests for MultiSymbolTradingLoop."""

    def test_init_with_symbols(self):
        from apexfx.live.multi_symbol_loop import MultiSymbolTradingLoop
        from apexfx.config.schema import AppConfig
        config = AppConfig()
        loop = MultiSymbolTradingLoop(config, ["EURUSD", "GBPUSD"])
        assert loop.symbols == ["EURUSD", "GBPUSD"]
        assert loop.portfolio is not None

    def test_shared_portfolio(self):
        from apexfx.live.multi_symbol_loop import MultiSymbolTradingLoop
        from apexfx.config.schema import AppConfig
        config = AppConfig()
        loop = MultiSymbolTradingLoop(config, ["EURUSD", "GBPUSD"])
        # Portfolio should be shared across symbols
        loop.portfolio.set_equity(200_000)
        assert loop.portfolio.get_aggregate_equity() == 200_000


# =====================================================================
# 4. Live Online Incremental Learning
# =====================================================================


class TestLiveOnlineLearner:
    """Tests for LiveOnlineLearner micro-update system."""

    def _make_mock_model(self):
        model = MagicMock()
        model.learning_rate = 3e-4
        model.policy = MagicMock()
        model.policy.features_extractor = MagicMock()
        # Make named_parameters return empty to avoid complex mocking
        model.policy.features_extractor.named_parameters = MagicMock(return_value=[])
        model.policy.features_extractor.training = False
        model.policy.state_dict = MagicMock(return_value={})
        model.policy.load_state_dict = MagicMock()
        return model

    def test_buffer_recording(self):
        from apexfx.live.online_learner import LiveOnlineLearner
        from apexfx.config.schema import OnlineLearningConfig
        cfg = OnlineLearningConfig(enabled=True, mini_buffer_size=100)
        model = self._make_mock_model()
        learner = LiveOnlineLearner(model, cfg)

        obs = _make_dict_obs()
        learner.record_transition(obs, 0.5, 1.0, obs, False)
        assert len(learner._mini_buffer) == 1

    def test_trade_result_tracking(self):
        from apexfx.live.online_learner import LiveOnlineLearner
        from apexfx.config.schema import OnlineLearningConfig
        cfg = OnlineLearningConfig(enabled=True)
        model = self._make_mock_model()
        learner = LiveOnlineLearner(model, cfg)

        for i in range(5):
            learner.record_trade_result(0.01)
        assert learner.trade_count == 5
        assert len(learner._rolling_returns) == 5

    def test_drift_detection_negative_sharpe(self):
        from apexfx.live.online_learner import LiveOnlineLearner
        from apexfx.config.schema import OnlineLearningConfig
        cfg = OnlineLearningConfig(
            enabled=True,
            drift_detection_window=30,
            drift_threshold_sharpe=-0.5,
        )
        model = self._make_mock_model()
        learner = LiveOnlineLearner(model, cfg)

        # Record negative returns with some variance → drift
        rng = np.random.default_rng(42)
        for _ in range(30):
            learner.record_trade_result(-0.02 + rng.normal(0, 0.005))

        assert learner._detect_drift() is True

    def test_no_drift_with_positive_returns(self):
        from apexfx.live.online_learner import LiveOnlineLearner
        from apexfx.config.schema import OnlineLearningConfig
        cfg = OnlineLearningConfig(
            enabled=True,
            drift_detection_window=30,
            drift_threshold_sharpe=-0.5,
        )
        model = self._make_mock_model()
        learner = LiveOnlineLearner(model, cfg)

        for _ in range(30):
            learner.record_trade_result(0.01)

        assert learner._detect_drift() is False

    def test_rollback(self):
        from apexfx.live.online_learner import LiveOnlineLearner
        from apexfx.config.schema import OnlineLearningConfig
        cfg = OnlineLearningConfig(enabled=True, max_rollback_checkpoints=3)
        model = self._make_mock_model()
        learner = LiveOnlineLearner(model, cfg)

        # Save checkpoint
        learner._save_checkpoint()
        assert len(learner._checkpoints) == 1

        # Rollback should succeed
        assert learner.rollback() is True
        assert len(learner._checkpoints) == 0

        # Second rollback should fail (no checkpoints)
        assert learner.rollback() is False


# =====================================================================
# 5. Regime-Conditional Execution
# =====================================================================


class TestRegimeExecution:
    """Tests for regime-conditional execution wiring."""

    def test_trading_signal_has_uncertainty_fields(self):
        from apexfx.live.signal_generator import TradingSignal
        signal = TradingSignal(
            action=0.5,
            confidence=0.5,
            trend_agent_action=0.3,
            reversion_agent_action=0.1,
            breakout_agent_action=0.1,
            gating_weights=(0.5, 0.3, 0.2),
            regime="trending",
            timestamp=datetime.now(timezone.utc),
            inference_time_ms=1.0,
        )
        assert hasattr(signal, "uncertainty_score")
        assert hasattr(signal, "position_scale")
        assert hasattr(signal, "recommended_stop_atr_mult")
        assert signal.uncertainty_score == 0.0
        assert signal.position_scale == 1.0
        assert signal.recommended_stop_atr_mult == 2.5

    def test_dynamic_stop_config(self):
        from apexfx.risk.risk_manager import DynamicStopConfig
        stop = DynamicStopConfig(atr_mult=2.5, trailing=True, stop_distance=0.005)
        assert stop.atr_mult == 2.5
        assert stop.trailing is True
        assert stop.stop_distance == 0.005

    def test_compute_dynamic_stop_trending(self):
        from apexfx.config.schema import RiskConfig
        from apexfx.risk.risk_manager import RiskManager
        rm = RiskManager(RiskConfig())
        stop = rm.compute_dynamic_stop(0.0, "trending", 0.001)
        # Trending base = 2.0, uncertainty=0 → mult=2.0
        assert stop.atr_mult == pytest.approx(2.0)
        assert stop.trailing is True
        assert stop.stop_distance == pytest.approx(0.002)

    def test_compute_dynamic_stop_volatile_high_uncertainty(self):
        from apexfx.config.schema import RiskConfig
        from apexfx.risk.risk_manager import RiskManager
        rm = RiskManager(RiskConfig())
        stop = rm.compute_dynamic_stop(0.8, "volatile", 0.002)
        # Volatile base = 4.0, uncertainty=0.8 → mult = 4.0 * 1.4 = 5.6
        assert stop.atr_mult == pytest.approx(5.6)
        assert stop.trailing is False

    def test_regime_transition_major(self):
        """trending → mean_reverting should be a major transition."""
        from apexfx.live.trading_loop import LiveTradingLoop
        assert LiveTradingLoop._is_major_transition("trending", "mean_reverting") is True
        assert LiveTradingLoop._is_major_transition("trending", "volatile") is True
        assert LiveTradingLoop._is_major_transition("mean_reverting", "volatile") is True

    def test_regime_transition_minor(self):
        """trending → flat should be a minor transition."""
        from apexfx.live.trading_loop import LiveTradingLoop
        assert LiveTradingLoop._is_major_transition("trending", "flat") is False
        assert LiveTradingLoop._is_major_transition("mean_reverting", "flat") is False

    def test_uncertainty_passed_to_risk_manager(self):
        """Risk manager should accept and use uncertainty_score."""
        from apexfx.config.schema import RiskConfig
        from apexfx.risk.risk_manager import MarketState, RiskManager
        rm = RiskManager(RiskConfig(), initial_balance=100_000)
        rm.update_portfolio(100_000)

        market_state = MarketState(
            current_price=1.1000,
            current_spread=0.0001,
            current_atr=0.001,
        )

        # With high uncertainty, position should be scaled down
        decision_high = rm.evaluate_action(0.5, market_state, uncertainty_score=0.9)
        decision_low = rm.evaluate_action(0.5, market_state, uncertainty_score=0.1)

        # High uncertainty should result in smaller position
        if decision_high.approved and decision_low.approved:
            assert decision_high.position_size <= decision_low.position_size

    def test_set_portfolio_context(self):
        from apexfx.config.schema import RiskConfig
        from apexfx.risk.risk_manager import RiskManager
        rm = RiskManager(RiskConfig())
        # Should not raise
        rm.set_portfolio_context([])
        assert rm._portfolio_positions == []

    def test_executor_close_all(self):
        """Executor should have public close_all method."""
        from apexfx.execution.executor import Executor
        assert hasattr(Executor, "close_all")

    def test_executor_reduce_position(self):
        """Executor should have public reduce_position method."""
        from apexfx.execution.executor import Executor
        assert hasattr(Executor, "reduce_position")

    def test_stop_mult_computation(self):
        from apexfx.live.signal_generator import SignalGenerator
        # Low uncertainty + trending → tight stop
        mult_trend = SignalGenerator._compute_stop_mult(0.0, "trending")
        assert mult_trend == pytest.approx(1.5)

        # High uncertainty + volatile → very wide stop
        mult_vol = SignalGenerator._compute_stop_mult(1.0, "volatile")
        assert mult_vol == pytest.approx(6.0)  # 4.0 * 1.5

        # Medium uncertainty + flat
        mult_flat = SignalGenerator._compute_stop_mult(0.5, "flat")
        assert mult_flat == pytest.approx(3.125)  # 2.5 * 1.25
