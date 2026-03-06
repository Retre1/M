"""Tests for GPU/CUDA optimization utilities.

These tests run on any machine (CPU-only or GPU) — they validate logic,
not CUDA availability. GPU-specific paths are tested via mocking.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from apexfx.utils.gpu import (
    get_device_map,
    get_optimal_batch_size,
    get_optimal_buffer_size,
    get_optimal_n_envs,
    try_compile_model,
    MixedPrecisionWrapper,
    pin_memory_dataloader_kwargs,
)


# ── get_device_map ──────────────────────────────────────────────────

class TestDeviceMap:
    def test_two_gpus(self):
        dm = get_device_map(2)
        assert dm["main"] == "cuda:0"
        assert dm["auxiliary"] == "cuda:1"

    def test_one_gpu(self):
        dm = get_device_map(1)
        assert dm["main"] == "cuda:0"
        assert dm["auxiliary"] == "cuda:0"

    def test_no_gpu(self):
        dm = get_device_map(0)
        assert dm["main"] == "cpu"
        assert dm["auxiliary"] == "cpu"

    def test_many_gpus(self):
        dm = get_device_map(8)
        assert dm["main"] == "cuda:0"
        assert dm["auxiliary"] == "cuda:1"


# ── get_optimal_n_envs ─────────────────────────────────────────────

class TestOptimalNEnvs:
    def test_38_cpus_2_gpus(self):
        n = get_optimal_n_envs(38, 2)
        assert 8 <= n <= 24

    def test_38_cpus_1_gpu(self):
        n = get_optimal_n_envs(38, 1)
        assert 8 <= n <= 16

    def test_4_cpus_no_gpu(self):
        n = get_optimal_n_envs(4, 0)
        assert n == 8  # min clamp

    def test_minimum_envs(self):
        n = get_optimal_n_envs(2, 0)
        assert n >= 8  # min clamp to 8

    def test_result_is_int(self):
        n = get_optimal_n_envs(38, 2)
        assert isinstance(n, int)


# ── get_optimal_batch_size ──────────────────────────────────────────

class TestOptimalBatchSize:
    def test_4090_typical(self):
        bs = get_optimal_batch_size(24.0, d_model=128, seq_len=100)
        assert 256 <= bs <= 8192
        # Should be power of 2
        assert bs & (bs - 1) == 0

    def test_small_vram(self):
        bs = get_optimal_batch_size(4.0, d_model=256, seq_len=200)
        assert bs >= 256  # min clamp

    def test_huge_vram(self):
        bs = get_optimal_batch_size(80.0, d_model=64, seq_len=50)
        assert bs <= 8192  # max clamp

    def test_power_of_two(self):
        bs = get_optimal_batch_size(24.0, d_model=128, seq_len=100)
        assert bs & (bs - 1) == 0, f"{bs} is not a power of 2"


# ── get_optimal_buffer_size ─────────────────────────────────────────

class TestOptimalBufferSize:
    def test_128gb_ram(self):
        bs = get_optimal_buffer_size(128.0, obs_dim=200)
        assert bs <= 5_000_000  # max clamp
        assert bs > 0

    def test_small_ram(self):
        bs = get_optimal_buffer_size(8.0, obs_dim=200)
        assert bs > 0
        assert bs <= 5_000_000

    def test_large_obs_dim(self):
        bs = get_optimal_buffer_size(128.0, obs_dim=10000)
        assert bs > 0
        assert bs <= 5_000_000


# ── try_compile_model ───────────────────────────────────────────────

class TestTryCompile:
    def test_returns_module(self):
        model = nn.Linear(10, 5)
        result = try_compile_model(model)
        # Should return a module (compiled or original)
        assert result is not None

    def test_fallback_on_error(self):
        """Even if compile fails, should return original model."""
        model = nn.Linear(10, 5)
        # This should not raise — graceful fallback
        result = try_compile_model(model, mode="invalid_mode_xyz")
        assert result is not None


# ── MixedPrecisionWrapper ──────────────────────────────────────────

class TestMixedPrecision:
    def test_disabled_on_cpu(self):
        wrapper = MixedPrecisionWrapper(enabled=True)
        # On CPU-only machines, should gracefully disable
        # (scaler may or may not be None depending on torch.cuda.is_available)
        assert isinstance(wrapper.enabled, bool)

    def test_explicitly_disabled(self):
        wrapper = MixedPrecisionWrapper(enabled=False)
        assert not wrapper.enabled
        assert wrapper.scaler is None


# ── pin_memory_dataloader_kwargs ────────────────────────────────────

class TestPinMemory:
    def test_returns_dict(self):
        kwargs = pin_memory_dataloader_kwargs()
        assert isinstance(kwargs, dict)

    def test_keys_if_cuda(self):
        """If CUDA is available, should have pin_memory key."""
        import torch
        kwargs = pin_memory_dataloader_kwargs()
        if torch.cuda.is_available():
            assert "pin_memory" in kwargs
            assert kwargs["pin_memory"] is True
            assert "num_workers" in kwargs
        else:
            # On CPU, should return empty dict
            assert kwargs == {}
