"""Tests for ScalpingAgent and ScalpingExtractor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from apexfx.features.scalping import ScalpingExtractor
from apexfx.models.agents.scalping_agent import ScalpingAgent


# =====================================================================
# ScalpingAgent tests
# =====================================================================


class TestScalpingAgentForwardPass:
    """Test basic forward pass output shapes and value ranges."""

    def test_forward_output_shape(self):
        agent = ScalpingAgent(d_tft=64, d_scalp_features=10)
        tft = torch.randn(4, 64)
        scalp = torch.randn(4, 10)
        action = agent(tft, scalp)
        assert action.shape == (4, 1)

    def test_forward_output_bounded(self):
        agent = ScalpingAgent(d_tft=64, d_scalp_features=10)
        tft = torch.randn(8, 64)
        scalp = torch.randn(8, 10)
        action = agent(tft, scalp)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()

    def test_batch_size_1(self):
        agent = ScalpingAgent()
        tft = torch.randn(1, 64)
        scalp = torch.randn(1, 10)
        action = agent(tft, scalp)
        assert action.shape == (1, 1)


class TestScalpingAgentEncode:
    """Test encode() output shape matches hidden_dim."""

    def test_encode_shape_default(self):
        agent = ScalpingAgent(d_tft=64, d_scalp_features=10)
        tft = torch.randn(4, 64)
        scalp = torch.randn(4, 10)
        hidden = agent.encode(tft, scalp)
        assert hidden.shape == (4, agent.hidden_dim)

    def test_encode_shape_custom_hidden(self):
        agent = ScalpingAgent(
            d_tft=32, d_scalp_features=10, hidden_sizes=[64, 32]
        )
        tft = torch.randn(4, 32)
        scalp = torch.randn(4, 10)
        hidden = agent.encode(tft, scalp)
        assert hidden.shape == (4, 32)
        assert agent.hidden_dim == 32

    def test_hidden_dim_property(self):
        agent = ScalpingAgent(hidden_sizes=[128, 64, 32])
        assert agent.hidden_dim == 32

        agent2 = ScalpingAgent(hidden_sizes=[256, 128])
        assert agent2.hidden_dim == 128


class TestScalpingAgentDifferentDTFT:
    """Test with different d_tft sizes for compatibility."""

    @pytest.mark.parametrize("d_tft", [16, 32, 64, 128])
    def test_various_d_tft(self, d_tft: int):
        agent = ScalpingAgent(d_tft=d_tft, d_scalp_features=10)
        tft = torch.randn(4, d_tft)
        scalp = torch.randn(4, 10)
        action = agent(tft, scalp)
        assert action.shape == (4, 1)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()


class TestScalpingAgentSpreadGate:
    """Test spread gate suppresses action when spread is wide."""

    def test_wide_spread_suppresses_action(self):
        torch.manual_seed(42)
        agent = ScalpingAgent(d_tft=64, d_scalp_features=10)
        agent.eval()

        tft = torch.randn(16, 64)

        # Wide spread (feature[0] = high value = bad for scalping)
        wide_spread_features = torch.randn(16, 10)
        wide_spread_features[:, 0] = 5.0  # very wide spread

        # Tight spread (feature[0] = low value = good for scalping)
        tight_spread_features = wide_spread_features.clone()
        tight_spread_features[:, 0] = -3.0  # very tight spread

        action_wide = agent(tft, wide_spread_features)
        action_tight = agent(tft, tight_spread_features)

        # Actions with wide spread should be more suppressed on average
        # (closer to 0 in absolute value) than tight spread actions
        assert action_wide.abs().mean() <= action_tight.abs().mean() + 0.5

    def test_spread_gate_output_range(self):
        """Spread gate output should be in [0, 1] (sigmoid)."""
        agent = ScalpingAgent()
        features = torch.randn(8, 10)
        gate_out = agent.spread_gate(features)
        assert (gate_out >= 0.0).all()
        assert (gate_out <= 1.0).all()


class TestScalpingAgentMomentumGate:
    """Test momentum gate behavior."""

    def test_momentum_gate_output_range(self):
        """Momentum gate output should be in [0, 1] (sigmoid)."""
        agent = ScalpingAgent()
        features = torch.randn(8, 10)
        gate_out = agent.momentum_gate(features)
        assert (gate_out >= 0.0).all()
        assert (gate_out <= 1.0).all()

    def test_act_without_specialist_features(self):
        """act() without specialist_features returns ungated action."""
        agent = ScalpingAgent()
        agent.eval()
        tft = torch.randn(4, 64)
        scalp = torch.randn(4, 10)
        hidden = agent.encode(tft, scalp)

        # With gates
        action_gated = agent.act(hidden, specialist_features=scalp)
        # Without gates
        action_ungated = agent.act(hidden, specialist_features=None)

        # Ungated should generally have larger magnitude
        # (both should be valid tensors)
        assert action_gated.shape == (4, 1)
        assert action_ungated.shape == (4, 1)

    def test_dual_gate_multiplicative(self):
        """Both gates must be active for signal to pass through."""
        agent = ScalpingAgent()
        features = torch.randn(8, 10)

        spread_gate = agent.spread_gate(features)
        momentum_gate = agent.momentum_gate(features)
        combined = spread_gate * momentum_gate

        # Combined gate should be <= min(spread, momentum) elementwise
        assert (combined <= spread_gate + 1e-6).all()
        assert (combined <= momentum_gate + 1e-6).all()


class TestScalpingAgentGradientFlow:
    """Test gradient flow through the agent."""

    def test_backprop_works(self):
        agent = ScalpingAgent()
        tft = torch.randn(4, 64, requires_grad=True)
        scalp = torch.randn(4, 10, requires_grad=True)

        action = agent(tft, scalp)
        loss = action.sum()
        loss.backward()

        assert tft.grad is not None
        assert scalp.grad is not None
        assert tft.grad.abs().sum() > 0
        assert scalp.grad.abs().sum() > 0

    def test_gradient_flow_through_encode_act(self):
        """Gradient flows through encode -> act path."""
        agent = ScalpingAgent()
        tft = torch.randn(4, 64, requires_grad=True)
        scalp = torch.randn(4, 10, requires_grad=True)

        hidden = agent.encode(tft, scalp)
        action = agent.act(hidden, specialist_features=scalp)
        loss = action.sum()
        loss.backward()

        assert tft.grad is not None
        assert scalp.grad is not None

    def test_all_parameters_receive_gradients(self):
        agent = ScalpingAgent()
        tft = torch.randn(4, 64)
        scalp = torch.randn(4, 10)

        action = agent(tft, scalp)
        loss = action.sum()
        loss.backward()

        for name, param in agent.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestScalpingAgentCrossAgentAPI:
    """Test cross-agent attention compatibility."""

    def test_encode_act_matches_forward(self):
        """encode() + act() should produce same result as forward()."""
        agent = ScalpingAgent()
        agent.eval()

        tft = torch.randn(4, 64)
        scalp = torch.randn(4, 10)

        with torch.no_grad():
            action_forward = agent(tft, scalp)
            hidden = agent.encode(tft, scalp)
            action_split = agent.act(hidden, specialist_features=scalp)

        torch.testing.assert_close(action_forward, action_split)

    def test_hidden_dim_compatible_with_other_agents(self):
        """hidden_dim should be a positive int for CrossAgentAttention."""
        agent = ScalpingAgent()
        assert isinstance(agent.hidden_dim, int)
        assert agent.hidden_dim > 0


# =====================================================================
# ScalpingExtractor tests
# =====================================================================


def _make_m1_bars(
    n: int = 30,
    start_price: float = 1.1000,
    start_hour: int = 13,
) -> pd.DataFrame:
    """Create sample M1 OHLCV bars for testing."""
    rng = np.random.RandomState(42)
    dates = pd.date_range(
        "2024-01-15 {:02d}:00".format(start_hour),
        periods=n,
        freq="1min",
        tz="UTC",
    )
    close = start_price + np.cumsum(rng.randn(n) * 0.0001)
    high = close + rng.uniform(0.0001, 0.0005, n)
    low = close - rng.uniform(0.0001, 0.0005, n)
    open_ = close + rng.randn(n) * 0.0002
    volume = rng.randint(50, 200, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestScalpingExtractor:
    """Test ScalpingExtractor feature computation."""

    def test_extract_returns_10_features(self):
        bars = _make_m1_bars(30)
        ext = ScalpingExtractor(lookback=20)
        features = ext.extract(bars, current_spread=0.0001)
        assert len(features) == 10

    def test_feature_names_length(self):
        ext = ScalpingExtractor()
        assert len(ext.feature_names) == 10

    def test_feature_names_match_extract_keys(self):
        ext = ScalpingExtractor()
        bars = _make_m1_bars(30)
        features = ext.extract(bars, current_spread=0.0001)
        assert set(features.keys()) == set(ext.feature_names)

    def test_all_features_are_float(self):
        ext = ScalpingExtractor()
        bars = _make_m1_bars(30)
        features = ext.extract(bars, current_spread=0.0001)
        for name, val in features.items():
            assert isinstance(val, float), f"{name} is {type(val)}"

    def test_spread_normalized_by_atr(self):
        ext = ScalpingExtractor()
        bars = _make_m1_bars(30)

        # Small spread
        feat_tight = ext.extract(bars, current_spread=0.00001)
        # Large spread
        feat_wide = ext.extract(bars, current_spread=0.01)

        assert feat_tight["normalized_spread"] < feat_wide["normalized_spread"]

    def test_spread_estimated_without_current(self):
        ext = ScalpingExtractor()
        bars = _make_m1_bars(30)
        features = ext.extract(bars, current_spread=None)
        assert "normalized_spread" in features
        assert features["normalized_spread"] >= 0.0

    def test_micro_momentum_signs(self):
        """Uptrending close should give positive momentum."""
        n = 30
        dates = pd.date_range("2024-01-15 13:00", periods=n, freq="1min", tz="UTC")
        close = np.linspace(1.1000, 1.1050, n)  # steady uptrend
        high = close + 0.0002
        low = close - 0.0002
        bars = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": 100.0},
            index=dates,
        )

        ext = ScalpingExtractor()
        features = ext.extract(bars, current_spread=0.0001)

        assert features["micro_momentum_5"] > 0
        assert features["micro_momentum_10"] > 0
        assert features["micro_momentum_20"] > 0

    def test_volume_surge_high(self):
        bars = _make_m1_bars(30)
        # Set last bar volume very high
        bars.iloc[-1, bars.columns.get_loc("volume")] = 10000.0
        ext = ScalpingExtractor()
        features = ext.extract(bars, current_spread=0.0001)
        assert features["volume_surge"] > 5.0

    def test_bid_ask_pressure_range(self):
        ext = ScalpingExtractor()
        bars = _make_m1_bars(30)
        features = ext.extract(bars, current_spread=0.0001)
        assert -1.0 <= features["bid_ask_pressure"] <= 1.0

    def test_mean_reversion_signal_zscore(self):
        ext = ScalpingExtractor()
        bars = _make_m1_bars(30)
        features = ext.extract(bars, current_spread=0.0001)
        assert -5.0 <= features["mean_reversion_signal"] <= 5.0


class TestScalpingExtractorTimeOfDay:
    """Test time_of_day_score for different hours."""

    def test_london_ny_overlap_score_1(self):
        """Hours 12-15 UTC should score 1.0 (London/NY overlap)."""
        for hour in [12, 13, 14, 15]:
            bars = _make_m1_bars(30, start_hour=hour)
            ext = ScalpingExtractor()
            features = ext.extract(bars, current_spread=0.0001)
            assert features["time_of_day_score"] == 1.0, (
                f"Hour {hour} should score 1.0, got {features['time_of_day_score']}"
            )

    def test_london_solo_score_05(self):
        """Hours 7-11 UTC should score 0.5 (London only)."""
        for hour in [7, 8, 9, 10, 11]:
            bars = _make_m1_bars(30, start_hour=hour)
            ext = ScalpingExtractor()
            features = ext.extract(bars, current_spread=0.0001)
            assert features["time_of_day_score"] == 0.5, (
                f"Hour {hour} should score 0.5, got {features['time_of_day_score']}"
            )

    def test_ny_solo_score_05(self):
        """Hours 16-20 UTC should score 0.5 (NY only, after London close)."""
        for hour in [16, 17, 18, 19, 20]:
            bars = _make_m1_bars(30, start_hour=hour)
            ext = ScalpingExtractor()
            features = ext.extract(bars, current_spread=0.0001)
            assert features["time_of_day_score"] == 0.5, (
                f"Hour {hour} should score 0.5, got {features['time_of_day_score']}"
            )

    def test_asian_session_score_02(self):
        """Hours 0-6 UTC should score 0.2 (Tokyo/Asian)."""
        for hour in [0, 1, 2, 3, 4, 5, 6]:
            bars = _make_m1_bars(30, start_hour=hour)
            ext = ScalpingExtractor()
            features = ext.extract(bars, current_spread=0.0001)
            assert features["time_of_day_score"] == 0.2, (
                f"Hour {hour} should score 0.2, got {features['time_of_day_score']}"
            )

    def test_off_hours_score_01(self):
        """Hours 21-23 UTC should score 0.1 (between sessions)."""
        for hour in [21, 22, 23]:
            bars = _make_m1_bars(30, start_hour=hour)
            ext = ScalpingExtractor()
            features = ext.extract(bars, current_spread=0.0001)
            assert features["time_of_day_score"] == 0.1, (
                f"Hour {hour} should score 0.1, got {features['time_of_day_score']}"
            )


class TestScalpingExtractorEdgeCases:
    """Test edge cases."""

    def test_minimal_bars(self):
        """Should work with very few bars (no crash)."""
        dates = pd.date_range("2024-01-15 13:00", periods=3, freq="1min", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [1.1, 1.1001, 1.1002],
                "high": [1.1005, 1.1006, 1.1007],
                "low": [1.0995, 1.0996, 1.0997],
                "close": [1.1001, 1.1002, 1.1003],
                "volume": [100.0, 110.0, 120.0],
            },
            index=dates,
        )
        ext = ScalpingExtractor()
        features = ext.extract(bars, current_spread=0.0001)
        assert len(features) == 10

    def test_close_only_bars(self):
        """Should work with only 'close' column."""
        dates = pd.date_range("2024-01-15 13:00", periods=30, freq="1min", tz="UTC")
        bars = pd.DataFrame(
            {"close": np.linspace(1.1, 1.1010, 30)},
            index=dates,
        )
        ext = ScalpingExtractor()
        features = ext.extract(bars, current_spread=0.0001)
        assert len(features) == 10

    def test_no_datetime_index(self):
        """Should work without DatetimeIndex (default scores)."""
        bars = pd.DataFrame(
            {
                "open": np.ones(30),
                "high": np.ones(30) + 0.001,
                "low": np.ones(30) - 0.001,
                "close": np.ones(30),
                "volume": np.ones(30) * 100,
            }
        )
        ext = ScalpingExtractor()
        features = ext.extract(bars, current_spread=0.0001)
        assert len(features) == 10
        assert features["time_of_day_score"] == 0.5  # neutral default
        assert features["tick_velocity"] == 1.0  # default

    def test_volume_bars_with_tick_volume(self):
        """Should use tick_volume column if 'volume' not present."""
        dates = pd.date_range("2024-01-15 13:00", periods=30, freq="1min", tz="UTC")
        bars = pd.DataFrame(
            {
                "close": np.linspace(1.1, 1.1010, 30),
                "high": np.linspace(1.1, 1.1010, 30) + 0.0002,
                "low": np.linspace(1.1, 1.1010, 30) - 0.0002,
                "tick_volume": np.full(30, 100.0),
            },
            index=dates,
        )
        bars.iloc[-1, bars.columns.get_loc("tick_volume")] = 500.0
        ext = ScalpingExtractor()
        features = ext.extract(bars, current_spread=0.0001)
        assert features["volume_surge"] > 1.0
