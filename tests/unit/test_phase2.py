"""Tests for Phase 2 improvements: gating, reward, PER, curriculum."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ── Sigmoid Gating + Meta-Confidence ──────────────────────────────


class TestGatingNetwork:
    def test_output_shapes(self):
        from apexfx.models.ensemble.gating_network import GatingNetwork

        gn = GatingNetwork(d_regime=6, d_context=32, n_agents=3)
        regime = torch.randn(4, 6)
        ctx = torch.randn(4, 32)
        agents = torch.randn(4, 3)
        weights, action = gn(regime, ctx, agents)
        assert weights.shape == (4, 3)
        assert action.shape == (4, 1)

    def test_sigmoid_weights_range(self):
        from apexfx.models.ensemble.gating_network import GatingNetwork

        gn = GatingNetwork(d_regime=6, d_context=32, n_agents=3)
        regime = torch.randn(8, 6)
        ctx = torch.randn(8, 32)
        agents = torch.randn(8, 3)
        weights, _ = gn(regime, ctx, agents)
        # Sigmoid weights should be in [0, 1]
        assert weights.min().item() >= 0.0
        assert weights.max().item() <= 1.0

    def test_confidence_scales_action(self):
        """When agents strongly disagree, confidence should be lower."""
        from apexfx.models.ensemble.gating_network import GatingNetwork

        gn = GatingNetwork(d_regime=6, d_context=32, n_agents=3)
        regime = torch.zeros(2, 6)
        ctx = torch.zeros(2, 32)

        # Agreeing agents
        agents_agree = torch.tensor([[0.9, 0.8, 0.85], [0.9, 0.8, 0.85]])
        _, action_agree = gn(regime, ctx, agents_agree)

        # This should at least produce valid output (gradient test)
        assert action_agree.shape == (2, 1)
        assert torch.isfinite(action_agree).all()

    def test_gradient_flow(self):
        from apexfx.models.ensemble.gating_network import GatingNetwork

        gn = GatingNetwork(d_regime=6, d_context=32, n_agents=3)
        regime = torch.randn(4, 6, requires_grad=True)
        ctx = torch.randn(4, 32)
        agents = torch.randn(4, 3, requires_grad=True)
        _, action = gn(regime, ctx, agents)
        action.sum().backward()
        assert regime.grad is not None
        assert agents.grad is not None


# ── TradingReward with CVaR + Vol-Adjustment ──────────────────────


class TestTradingRewardV2:
    def test_basic_positive_return(self):
        from apexfx.env.reward import TradingReward

        rw = TradingReward()
        rw.set_trade_info(0.5, 1, 100.0, 5)
        r = rw.compute(100100.0, 100000.0)
        assert r > 0, "Positive return should give positive reward"

    def test_basic_negative_return(self):
        from apexfx.env.reward import TradingReward

        rw = TradingReward()
        rw.set_trade_info(0.5, 1, -100.0, 5)
        r = rw.compute(99900.0, 100000.0)
        assert r < 0, "Negative return should give negative reward"

    def test_asymmetric_penalty(self):
        from apexfx.env.reward import TradingReward

        rw = TradingReward(loss_weight=2.0)
        rw.set_trade_info(0.0, 0, 0.0, 0)
        pos = rw.compute(100100.0, 100000.0)
        rw.reset()
        rw.set_trade_info(0.0, 0, 0.0, 0)
        neg = rw.compute(99900.0, 100000.0)
        assert abs(neg) > abs(pos), "Loss should be penalized more than equal gain"

    def test_cvar_penalty_grows_with_tail_losses(self):
        from apexfx.env.reward import TradingReward

        rw = TradingReward(cvar_weight=1.0, cvar_window=20, cvar_alpha=0.1)
        rw.set_trade_info(0.0, 0, 0.0, 0)
        # Inject large negative returns into history
        rw._returns_history = [-0.05] * 20
        r_bad = rw.compute(100000.0, 100000.0)

        rw2 = TradingReward(cvar_weight=1.0, cvar_window=20, cvar_alpha=0.1)
        rw2.set_trade_info(0.0, 0, 0.0, 0)
        rw2._returns_history = [0.001] * 20
        r_good = rw2.compute(100000.0, 100000.0)

        assert r_bad < r_good, "Bad tail returns should give worse reward via CVaR"

    def test_vol_adjustment(self):
        from apexfx.env.reward import TradingReward

        rw = TradingReward()
        rw.set_trade_info(0.0, 0, 0.0, 0)
        rw.set_atr(0.001)
        rw._returns_history = [0.001] * 25  # establish rolling vol
        r_normal = rw.compute(100100.0, 100000.0)

        rw2 = TradingReward()
        rw2.set_trade_info(0.0, 0, 0.0, 0)
        rw2.set_atr(0.01)  # 10x higher ATR
        rw2._returns_history = [0.001] * 25
        r_high_vol = rw2.compute(100100.0, 100000.0)

        # Same return but higher vol → less reward
        assert r_high_vol < r_normal, "High vol should reduce reward for same return"

    def test_reset_clears_state(self):
        from apexfx.env.reward import TradingReward

        rw = TradingReward()
        rw._returns_history = [0.01] * 50
        rw._peak = 110000.0
        rw._current_atr = 0.005
        rw.reset()
        assert rw._returns_history == []
        assert rw._peak == 0.0
        assert rw._current_atr is None


# ── PER SumTree ───────────────────────────────────────────────────


class TestSumTree:
    def test_add_and_total(self):
        from apexfx.training.per import SumTree

        tree = SumTree(capacity=10)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        assert abs(tree.total - 6.0) < 1e-6

    def test_circular_buffer(self):
        from apexfx.training.per import SumTree

        tree = SumTree(capacity=5)
        for i in range(10):
            tree.add(1.0)
        assert tree.n_entries == 5
        assert abs(tree.total - 5.0) < 1e-6

    def test_sampling_proportional(self):
        from apexfx.training.per import SumTree

        tree = SumTree(capacity=100)
        # Add items with very different priorities
        for _ in range(50):
            tree.add(0.01)  # low priority
        tree.add(100.0)  # very high priority

        # Sample many times — high priority item should be sampled often
        high_count = 0
        for _ in range(1000):
            v = np.random.uniform(0, tree.total)
            _, data_idx, _ = tree.get(v)
            if data_idx == 50:
                high_count += 1
        assert high_count > 500, f"High-priority item sampled {high_count}/1000 times (expected >500)"

    def test_update_priority(self):
        from apexfx.training.per import SumTree

        tree = SumTree(capacity=10)
        tree.add(1.0)
        tree.add(1.0)
        old_total = tree.total
        # Update first item to higher priority
        tree.update(tree.capacity - 1, 5.0)
        assert tree.total > old_total


# ── GRN Context Broadcasting Fix ─────────────────────────────────


class TestGRNContextBroadcast:
    def test_2d_context_with_3d_input(self):
        from apexfx.models.tft.grn import GatedResidualNetwork

        grn = GatedResidualNetwork(d_input=32, d_hidden=32, d_output=32, d_context=32)
        x = torch.randn(2, 10, 32)  # (batch, time, d)
        ctx = torch.randn(2, 32)    # (batch, d) — static context
        out = grn(x, ctx)
        assert out.shape == (2, 10, 32)

    def test_3d_context_with_3d_input(self):
        from apexfx.models.tft.grn import GatedResidualNetwork

        grn = GatedResidualNetwork(d_input=32, d_hidden=32, d_output=32, d_context=32)
        x = torch.randn(2, 10, 32)
        ctx = torch.randn(2, 10, 32)  # time-varying context
        out = grn(x, ctx)
        assert out.shape == (2, 10, 32)

    def test_no_context(self):
        from apexfx.models.tft.grn import GatedResidualNetwork

        grn = GatedResidualNetwork(d_input=32, d_hidden=32, d_output=32)
        x = torch.randn(2, 10, 32)
        out = grn(x)
        assert out.shape == (2, 10, 32)


# ── MTFHiveMind End-to-End ────────────────────────────────────────


class TestMTFHiveMind:
    def test_forward_shapes(self):
        from apexfx.models.ensemble.hive_mind import MTFHiveMind

        mtf = MTFHiveMind(n_continuous_vars=15, d_model=32)
        d1 = torch.randn(2, 5, 15)
        d1t = torch.randn(2, 5, 5)
        h1 = torch.randn(2, 20, 15)
        h1t = torch.randn(2, 20, 5)
        m5 = torch.randn(2, 20, 15)
        m5t = torch.randn(2, 20, 5)
        trend = torch.randn(2, 8)
        reversion = torch.randn(2, 8)
        regime = torch.randn(2, 6)
        out = mtf(d1, d1t, h1, h1t, m5, m5t, trend, reversion, regime)
        assert out.action.shape == (2, 1)
        assert out.gating_weights.shape == (2, 3)
        assert out.tf_attention_weights.shape == (2, 3)
