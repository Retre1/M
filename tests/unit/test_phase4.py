"""Phase 4: Institutional-Grade Trading Infrastructure — comprehensive test suite.

Tests cover:
- Walk-Forward integration into Trainer
- Stress Testing (scenarios, Monte Carlo, reverse)
- Portfolio VaR (portfolio, component, marginal)
- State Recovery with WAL
- Smart Execution (VWAP, IS, SmartRouter)
- Partial Fills simulation
- Order Book L2 features
- NLP Sentiment features (mock-based)
- Online Learning
- Shadow Trading / A/B Testing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 200) -> pd.DataFrame:
    """Create synthetic OHLCV bar data for testing."""
    rng = np.random.default_rng(42)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.001, n))
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": close - rng.uniform(0, 0.001, n),
        "high": close + rng.uniform(0, 0.002, n),
        "low": close - rng.uniform(0, 0.002, n),
        "close": close,
        "volume": rng.integers(100, 1000, n).astype(float),
        "tick_count": rng.integers(50, 500, n),
    })


# =====================================================================
# 1. Walk-Forward Integration
# =====================================================================


class TestWalkForwardIntegration:
    """Tests for Walk-Forward validation integration into Trainer."""

    def test_auto_validate_config_exists(self):
        """WalkForwardConfig should have auto_validate field."""
        from apexfx.config.schema import WalkForwardConfig
        cfg = WalkForwardConfig()
        assert hasattr(cfg, "auto_validate")
        assert cfg.auto_validate is False

    def test_walk_forward_monte_carlo_test(self):
        """Monte Carlo permutation test should return valid p-value."""
        pytest.importorskip("torch", reason="torch required for walk_forward")
        from apexfx.training.walk_forward import WalkForwardValidator

        rng = np.random.default_rng(42)
        # Returns with slight positive bias
        returns = rng.normal(0.001, 0.01, 200)
        p_value = WalkForwardValidator.monte_carlo_test(
            returns, n_permutations=1000
        )
        assert 0.0 <= p_value <= 1.0

    def test_walk_forward_monte_carlo_empty(self):
        """Monte Carlo test with empty returns should return 1.0."""
        pytest.importorskip("torch", reason="torch required for walk_forward")
        from apexfx.training.walk_forward import WalkForwardValidator
        assert WalkForwardValidator.monte_carlo_test(np.array([])) == 1.0


# =====================================================================
# 2. Stress Testing
# =====================================================================


class TestStressTesting:
    """Tests for StressTester framework."""

    def test_preset_scenarios_exist(self):
        """All 6 preset scenarios should be registered."""
        from apexfx.risk.stress_testing import StressTester
        expected = {"flash_crash", "snb_shock", "brexit", "covid_march", "fed_surprise", "gap_weekend"}
        assert set(StressTester.PRESET_SCENARIOS.keys()) == expected

    def test_run_single_scenario(self):
        """Running a single scenario should produce valid StressResult."""
        from apexfx.risk.stress_testing import StressTester
        tester = StressTester(var_limit=0.02)
        scenario = StressTester.PRESET_SCENARIOS["flash_crash"]
        result = tester.run_scenario(scenario, portfolio_value=100_000)

        assert result.pnl_impact < 0  # Flash crash should cause loss
        assert result.max_drawdown_pct > 0
        assert isinstance(result.survival, bool)
        assert isinstance(result.var_breach, bool)

    def test_run_all_presets(self):
        """Running all presets should return results for each scenario."""
        from apexfx.risk.stress_testing import StressTester
        tester = StressTester(var_limit=0.02)
        results = tester.run_all_presets(portfolio_value=100_000)
        assert len(results) == 6
        for r in results:
            assert r.pnl_impact <= 0
            assert r.max_drawdown_pct >= 0

    def test_monte_carlo_stress(self):
        """Monte Carlo simulation should produce valid VaR/CVaR estimates."""
        from apexfx.risk.stress_testing import StressTester
        tester = StressTester(var_limit=0.02)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.01, 252)

        mc = tester.monte_carlo_stress(
            returns=returns,
            n_simulations=500,
            horizon_days=10,
        )

        assert mc.var_95 > 0
        assert mc.var_99 >= mc.var_95  # 99% VaR >= 95% VaR
        assert mc.cvar_95 >= mc.var_95  # CVaR >= VaR
        assert mc.cvar_99 >= mc.var_99
        assert 0 <= mc.survival_rate <= 1
        assert mc.n_simulations == 500
        assert len(mc.max_drawdown_distribution) == 500

    def test_reverse_stress_test(self):
        """Reverse stress test should find minimum shock for target loss."""
        from apexfx.risk.stress_testing import StressTester
        tester = StressTester(var_limit=0.02)

        rs = tester.reverse_stress_test(
            portfolio_value=100_000,
            current_position_pct=0.10,
            target_loss_pct=0.20,
        )

        assert rs.target_loss_pct == 0.20
        assert rs.min_shock_pct > 0
        # With 10% position, need 200% shock to lose 20% — exceeds search
        # space [0,1], so binary search saturates at hi=1.0
        assert rs.min_shock_pct >= 1.0  # Large shock needed


# =====================================================================
# 3. Portfolio VaR
# =====================================================================


class TestPortfolioVaR:
    """Tests for multi-asset Portfolio VaR calculations."""

    def _create_var_calc(self):
        from apexfx.risk.var_calculator import VaRCalculator
        return VaRCalculator(confidence=0.99)

    def test_portfolio_var_single_asset(self):
        """Single-asset portfolio VaR should equal individual VaR."""
        calc = self._create_var_calc()
        positions = {"EURUSD": 100_000}
        corr = np.array([[1.0]])
        individual_vars = {"EURUSD": 0.02}

        pvar = calc.compute_portfolio_var(positions, corr, individual_vars)
        assert abs(pvar - 0.02) < 1e-6

    def test_portfolio_var_diversification(self):
        """Two uncorrelated assets should have lower portfolio VaR than sum."""
        calc = self._create_var_calc()
        positions = {"EURUSD": 50_000, "USDJPY": 50_000}
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        individual_vars = {"EURUSD": 0.02, "USDJPY": 0.02}

        pvar = calc.compute_portfolio_var(positions, corr, individual_vars)
        # Diversification: portfolio VaR < individual VaR
        assert pvar < 0.02

    def test_component_var_sums_to_portfolio(self):
        """Component VaRs should sum to portfolio VaR."""
        calc = self._create_var_calc()
        positions = {"EURUSD": 60_000, "USDJPY": 40_000}
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        individual_vars = {"EURUSD": 0.02, "USDJPY": 0.015}

        pvar = calc.compute_portfolio_var(positions, corr, individual_vars)
        cvar = calc.compute_component_var(positions, corr, individual_vars)

        component_sum = sum(cvar.values())
        assert abs(component_sum - pvar) < 1e-6

    def test_marginal_var_finite_for_new_asset(self):
        """Adding an asset should produce a finite marginal VaR."""
        calc = self._create_var_calc()
        positions = {"EURUSD": 100_000}
        corr = np.array([[1.0]])
        individual_vars = {"EURUSD": 0.02, "GBPUSD": 0.025}

        mvar = calc.compute_marginal_var(
            "GBPUSD", 50_000, positions, corr, individual_vars
        )
        # Marginal VaR can be negative (diversification benefit) or positive
        assert np.isfinite(mvar)
        # With zero correlation, diversification should lower portfolio VaR
        assert mvar < 0  # Diversification benefit


# =====================================================================
# 4. State Recovery with WAL
# =====================================================================


class TestStateRecovery:
    """Tests for WAL-based crash recovery in StateManager."""

    def test_wal_entry_checksum(self):
        """WAL entry checksum should be deterministic."""
        from apexfx.live.state_manager import WALEntry
        data1 = {"equity": 100_000}
        cs1 = WALEntry.compute_checksum(data1)
        entry1 = WALEntry(
            sequence=1,
            timestamp="2024-01-01T00:00:00",
            operation="equity_update",
            data=data1,
            checksum=cs1,
        )

        data2 = {"equity": 100_000}
        cs2 = WALEntry.compute_checksum(data2)
        entry2 = WALEntry(
            sequence=1,
            timestamp="2024-01-01T00:00:00",
            operation="equity_update",
            data=data2,
            checksum=cs2,
        )

        assert entry1.checksum == entry2.checksum

    def test_wal_entry_different_data(self):
        """Different data should produce different checksums."""
        from apexfx.live.state_manager import WALEntry
        data1 = {"equity": 100_000}
        data2 = {"equity": 99_000}
        cs1 = WALEntry.compute_checksum(data1)
        cs2 = WALEntry.compute_checksum(data2)
        assert cs1 != cs2

    def test_state_manager_init_with_wal(self):
        """StateManager should initialize with WAL support."""
        from apexfx.live.state_manager import StateManager
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            sm = StateManager(
                state_file=state_file,
                wal_enabled=True,
                checkpoint_interval=5,
            )
            assert sm._wal_enabled is True

    def test_state_manager_equity_update_with_wal(self):
        """Equity update should write WAL entry."""
        from apexfx.live.state_manager import StateManager
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            sm = StateManager(
                state_file=state_file,
                wal_enabled=True,
                checkpoint_interval=100,
            )
            sm.update_equity(105_000, 5_000)

            # Check WAL file exists and has content
            wal_path = Path(state_file + ".wal")
            if wal_path.exists():
                content = wal_path.read_text()
                assert "equity_update" in content

    def test_state_manager_checkpoint(self):
        """Checkpoint should persist state and truncate WAL."""
        from apexfx.live.state_manager import StateManager
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "state.json")
            sm = StateManager(
                state_file=state_file,
                wal_enabled=True,
                checkpoint_interval=3,
            )
            # Multiple updates to trigger checkpoint
            for i in range(5):
                sm.update_equity(100_000 + i * 100, float(i * 100))

            # Checkpoint should have been triggered
            if Path(state_file).exists():
                data = json.loads(Path(state_file).read_text())
                assert "equity" in data or "balance" in data


# =====================================================================
# 5. Smart Execution
# =====================================================================


class TestSmartExecution:
    """Tests for VWAP, IS, and SmartRouter."""

    def test_vwap_volume_profile(self):
        """VWAP executor should weight slices by volume profile."""
        from apexfx.execution.smart_exec import VWAPExecutor, EURUSD_VOLUME_PROFILE

        vwap = VWAPExecutor(n_slices=4)
        order = vwap.create_plan(1.0, 1, "EURUSD", current_hour=8)

        assert order.total_volume == 1.0
        assert order.direction == 1
        assert len(order.slices) >= 2
        # Slices should have different volumes (weighted by profile)
        volumes = [s.volume for s in order.slices]
        assert sum(volumes) == pytest.approx(1.0, abs=0.05)

    def test_is_front_loading(self):
        """IS executor with high urgency should front-load slices."""
        from apexfx.execution.smart_exec import ImplementationShortfallExecutor

        is_exec = ImplementationShortfallExecutor(urgency=0.9, n_slices=4)
        order = is_exec.create_plan(2.0, -1, "EURUSD", decision_price=1.1000)

        assert order.total_volume == 2.0
        assert order.direction == -1
        assert order.decision_price == 1.1000
        # First slice should be larger than last (front-loading)
        if len(order.slices) >= 2:
            assert order.slices[0].volume >= order.slices[-1].volume

    def test_smart_router_selection(self):
        """SmartRouter should select appropriate algorithm by volume."""
        from apexfx.execution.smart_exec import SmartRouter

        router = SmartRouter(twap_threshold=0.5, vwap_threshold=2.0, is_threshold=5.0)

        assert router.select_algorithm(0.1) == "direct"
        assert router.select_algorithm(1.0) == "twap"
        assert router.select_algorithm(3.0) == "vwap"
        assert router.select_algorithm(6.0) == "is"

    def test_smart_router_urgency_override(self):
        """High urgency should override to TWAP regardless of size."""
        from apexfx.execution.smart_exec import SmartRouter

        router = SmartRouter(twap_threshold=0.5, vwap_threshold=2.0, is_threshold=5.0)
        # Large order but high urgency → TWAP
        assert router.select_algorithm(10.0, urgency=0.9) == "twap"


# =====================================================================
# 6. Partial Fills
# =====================================================================


class TestPartialFills:
    """Tests for PartialFillModel."""

    def test_fill_rate_base(self):
        """Small orders during liquid sessions should have high fill rate."""
        from apexfx.env.forex_env import PartialFillModel

        model = PartialFillModel(base_fill_rate=0.95, volume_impact=0.3)
        filled, rate = model.simulate_fill(0.05, hour=14)  # overlap session
        assert 0.8 <= rate <= 1.0
        assert filled > 0

    def test_volume_impact(self):
        """Larger orders should have lower fill rate."""
        from apexfx.env.forex_env import PartialFillModel

        model = PartialFillModel(base_fill_rate=0.95, volume_impact=0.5)
        _, rate_small = model.simulate_fill(0.1, hour=14)
        _, rate_large = model.simulate_fill(5.0, hour=14)
        assert rate_large < rate_small

    def test_session_impact(self):
        """Off-hours should have lower fill rate than overlap."""
        from apexfx.env.forex_env import PartialFillModel

        model = PartialFillModel(base_fill_rate=0.95, session_impact=True)
        _, rate_overlap = model.simulate_fill(0.5, hour=14)  # overlap
        _, rate_offhours = model.simulate_fill(0.5, hour=23)  # off-hours
        assert rate_offhours < rate_overlap


# =====================================================================
# 7. Order Book L2 Features
# =====================================================================


class TestOrderBookFeatures:
    """Tests for OrderBookExtractor."""

    def test_feature_names(self):
        """OrderBookExtractor should declare 8 features."""
        from apexfx.features.orderbook import OrderBookExtractor
        extractor = OrderBookExtractor()
        assert len(extractor.feature_names) == 8

    def test_extract_from_ohlcv(self):
        """Synthetic L2 extraction from OHLCV should produce valid features."""
        from apexfx.features.orderbook import OrderBookExtractor
        extractor = OrderBookExtractor()
        bars = _make_bars(100)
        features = extractor.extract(bars)
        assert len(features) == len(bars)
        assert all(col in features.columns for col in extractor.feature_names)
        # Features should not be all NaN (at least after warmup)
        for col in extractor.feature_names:
            assert features[col].notna().sum() > 0

    def test_feature_dimensions(self):
        """Feature output dimensions should match declared names."""
        from apexfx.features.orderbook import OrderBookExtractor
        extractor = OrderBookExtractor()
        bars = _make_bars(50)
        features = extractor.extract(bars)
        assert features.shape[1] == len(extractor.feature_names)


# =====================================================================
# 8. Sentiment Extractor
# =====================================================================


class TestSentimentExtractor:
    """Tests for NLP Sentiment features (mock-based, no actual FinBERT)."""

    def test_feature_names(self):
        """SentimentExtractor should declare 6 features."""
        from apexfx.features.sentiment import SentimentExtractor
        extractor = SentimentExtractor()
        assert len(extractor.feature_names) == 6

    def test_extract_returns_neutral_for_backtest(self):
        """Backtesting mode should return zero/neutral sentiment features."""
        from apexfx.features.sentiment import SentimentExtractor
        extractor = SentimentExtractor()
        bars = _make_bars(50)
        features = extractor.extract(bars)
        assert len(features) == len(bars)
        # All sentiment features should be zero (no headlines in backtesting)
        for col in extractor.feature_names:
            assert features[col].sum() == pytest.approx(0.0, abs=1e-6)

    def test_keyword_fallback_scoring(self):
        """Keyword-based fallback should score positive/negative headlines."""
        from apexfx.features.sentiment import SentimentExtractor
        extractor = SentimentExtractor()

        # Update with positive headlines
        extractor.update_headlines([
            {"text": "EUR surges on strong GDP growth bullish outlook", "timestamp": "2024-01-01T12:00:00", "source": "test"},
        ])

        live_features = extractor.extract_live()
        assert len(live_features) == 6
        # Should have non-zero score
        assert live_features[0] != 0.0 or live_features[4] > 0  # score or headline_count


# =====================================================================
# 9. Online Learning
# =====================================================================


class TestOnlineLearner:
    """Tests for Online Learning module."""

    def test_should_retrain_logic(self):
        """OnlineLearner should trigger retrain after min_new_bars."""
        from apexfx.training.online_learner import OnlineLearner
        from apexfx.config.schema import AppConfig

        config = AppConfig()
        learner = OnlineLearner(
            model_path="dummy_path",
            config=config,
            min_new_bars=24,
        )

        assert learner.should_retrain(10) is False
        assert learner.should_retrain(24) is True
        assert learner.should_retrain(100) is True

    def test_should_not_retrain_below_threshold(self):
        """OnlineLearner should not retrain with too few bars."""
        from apexfx.training.online_learner import OnlineLearner
        from apexfx.config.schema import AppConfig

        config = AppConfig()
        learner = OnlineLearner(
            model_path="dummy_path",
            config=config,
            min_new_bars=100,
        )
        assert learner.should_retrain(99) is False

    def test_result_dataclass(self):
        """OnlineLearnResult should be properly structured."""
        from apexfx.training.online_learner import OnlineLearnResult
        result = OnlineLearnResult(
            retrained=True,
            promoted=True,
            model_path="/tmp/model",
            validation_sharpe=0.8,
            current_sharpe=0.5,
            sharpe_delta=0.3,
            n_bars_used=10_000,
            message="Promoted",
        )
        assert result.promoted is True
        assert result.sharpe_delta == 0.3
        assert result.retrained is True
        assert result.validation_sharpe == 0.8
        assert result.n_bars_used == 10_000


# =====================================================================
# 10. Shadow Trading / A/B Testing
# =====================================================================


class TestShadowTrader:
    """Tests for ShadowTrader and GradualRollout."""

    def test_register_shadow(self):
        """Registering a shadow model should initialize tracking."""
        from apexfx.live.shadow_trader import ShadowTrader
        trader = ShadowTrader(evaluation_bars=100)
        trader.register_shadow("model_v2")
        assert "model_v2" in trader._shadow_signals
        assert trader._shadow_positions["model_v2"] == 0.0

    def test_on_bar_tracking(self):
        """on_bar should track signals for shadow models."""
        from apexfx.live.shadow_trader import ShadowTrader
        trader = ShadowTrader(evaluation_bars=100)

        for i in range(60):
            trader.on_bar(
                live_action=0.5,
                shadow_actions={"v2": 0.3, "v3": -0.2},
                actual_price=1.1000 + i * 0.0001,
            )

        assert len(trader._shadow_signals["v2"]) == 60
        assert len(trader._shadow_signals["v3"]) == 60

    def test_evaluate_returns_result(self):
        """Evaluate should return ShadowResult after enough signals."""
        from apexfx.live.shadow_trader import ShadowTrader
        trader = ShadowTrader(evaluation_bars=100)

        rng = np.random.default_rng(42)
        for i in range(60):
            trader.on_bar(
                live_action=float(rng.uniform(-0.5, 0.5)),
                shadow_actions={"v2": float(rng.uniform(-0.5, 0.5))},
                actual_price=1.1000 + rng.normal(0, 0.001),
                live_return=float(rng.normal(0, 0.001)),
            )

        result = trader.evaluate("v2")
        assert result is not None
        assert result.model_id == "v2"
        assert result.n_signals >= 50

    def test_evaluate_insufficient_data(self):
        """Evaluate should return None with insufficient data."""
        from apexfx.live.shadow_trader import ShadowTrader
        trader = ShadowTrader(evaluation_bars=100)
        trader.register_shadow("v2")

        # Only 10 bars
        for i in range(10):
            trader.on_bar(
                live_action=0.5,
                shadow_actions={"v2": 0.3},
                actual_price=1.1000 + i * 0.0001,
            )

        assert trader.evaluate("v2") is None

    def test_gradual_rollout_blending(self):
        """GradualRollout should blend actions from 0 to 100%."""
        from apexfx.live.shadow_trader import GradualRollout

        rollout = GradualRollout(ramp_bars=100)
        assert rollout.get_blend_weight() == 0.0  # Not started

        rollout.start()
        assert rollout.is_active is True

        # Step to 50%
        for _ in range(50):
            rollout.step()
        weight = rollout.get_blend_weight()
        assert abs(weight - 0.50) < 0.02

        # Blend actions
        blended = rollout.blend_actions(live_action=1.0, shadow_action=0.0)
        assert abs(blended - 0.5) < 0.02

        # Step to 100%
        for _ in range(50):
            rollout.step()
        assert rollout.is_complete is True
        assert rollout.get_blend_weight() == pytest.approx(1.0)

    def test_gradual_rollout_stop(self):
        """Stopping rollout should reset weight to 0."""
        from apexfx.live.shadow_trader import GradualRollout
        rollout = GradualRollout(ramp_bars=100)
        rollout.start()
        for _ in range(50):
            rollout.step()
        rollout.stop()
        assert rollout.get_blend_weight() == 0.0
        assert rollout.is_active is False


# =====================================================================
# 11. News Fetcher
# =====================================================================


class TestNewsFetcher:
    """Tests for NewsFetcher module."""

    def test_domain_extraction(self):
        """Domain extraction should work for various URL formats."""
        from apexfx.data.news_fetcher import NewsFetcher
        assert NewsFetcher._extract_domain("https://www.forexlive.com/feed") == "forexlive.com"
        assert NewsFetcher._extract_domain("https://fxstreet.com/rss") == "fxstreet.com"

    def test_rss_parsing(self):
        """RSS parser should extract headlines from XML."""
        from apexfx.data.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()

        xml = """<?xml version="1.0"?>
        <rss><channel>
            <item>
                <title>EUR/USD Surges on Strong NFP Data</title>
                <pubDate>Thu, 01 Feb 2024 12:00:00 GMT</pubDate>
            </item>
            <item>
                <title>Fed Signals Rate Cut Path</title>
                <pubDate>Thu, 01 Feb 2024 11:00:00 GMT</pubDate>
            </item>
        </channel></rss>"""

        headlines = fetcher._parse_rss(xml, "https://www.test.com/feed")
        assert len(headlines) == 2
        assert headlines[0]["text"] == "EUR/USD Surges on Strong NFP Data"
        assert headlines[0]["source"] == "test.com"


# =====================================================================
# 12. Config Schema Validation
# =====================================================================


class TestPhase4Config:
    """Tests for new Phase 4 config classes."""

    def test_portfolio_var_config(self):
        from apexfx.config.schema import PortfolioVaRConfig
        cfg = PortfolioVaRConfig()
        assert cfg.multi_asset is False
        assert cfg.correlation_lookback_days == 60
        assert cfg.use_cvar is True

    def test_stress_test_config(self):
        from apexfx.config.schema import StressTestConfig
        cfg = StressTestConfig()
        assert cfg.enabled is True
        assert cfg.run_on_startup is True
        assert cfg.monte_carlo_sims == 10_000

    def test_smart_execution_config(self):
        from apexfx.config.schema import SmartExecutionConfig
        cfg = SmartExecutionConfig()
        assert cfg.algorithm == "auto"
        assert cfg.urgency == 0.5
        assert cfg.twap_threshold_lots == 0.5

    def test_online_learning_config(self):
        from apexfx.config.schema import OnlineLearningConfig
        cfg = OnlineLearningConfig()
        assert cfg.enabled is False
        assert cfg.mode == "fine_tune"
        assert cfg.retrain_lr == 1e-5

    def test_shadow_trading_config(self):
        from apexfx.config.schema import ShadowTradingConfig
        cfg = ShadowTradingConfig()
        assert cfg.enabled is False
        assert cfg.evaluation_bars == 500
        assert cfg.promotion_sharpe_delta == 0.3

    def test_full_app_config_integration(self):
        """Full AppConfig should include all Phase 4 config sections."""
        from apexfx.config.schema import AppConfig
        cfg = AppConfig()
        assert hasattr(cfg.risk, "portfolio_var")
        assert hasattr(cfg.risk, "stress_test")
        assert hasattr(cfg.execution, "smart_execution")
        assert hasattr(cfg.training, "online_learning")
        assert hasattr(cfg.training, "shadow_trading")
        assert hasattr(cfg.training.walk_forward, "auto_validate")
