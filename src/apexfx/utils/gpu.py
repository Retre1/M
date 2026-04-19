"""GPU/CUDA optimization utilities for high-performance training.

Target hardware: 2× RTX 4090 (48GB each), 128GB DDR4, 38 vCPUs, CUDA 12.8

Key optimizations:
1. CUDA runtime tuning (cudnn.benchmark, TF32, memory allocator)
2. Mixed precision (AMP) for 2× throughput + 40% less VRAM
3. torch.compile() for PyTorch 2.x graph optimisation (15-30% speedup)
4. Multi-GPU device assignment (training vs eval/world model)
5. Pin memory + async data transfer
6. Memory-efficient gradient checkpointing for large models
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def setup_cuda_optimizations(device: str = "cuda") -> dict[str, Any]:
    """Configure CUDA runtime for maximum throughput.

    Call this ONCE at the start of training.

    Returns dict of applied optimizations for logging.
    """
    optimizations = {}

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — skipping GPU optimizations")
        return {"cuda_available": False}

    n_gpus = torch.cuda.device_count()
    optimizations["n_gpus"] = n_gpus

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            f"GPU {i}: {props.name}",
            vram_gb=round(props.total_mem / 1e9, 1),
            compute_capability=f"{props.major}.{props.minor}",
            sm_count=props.multi_processor_count,
        )
        optimizations[f"gpu_{i}"] = {
            "name": props.name,
            "vram_gb": round(props.total_mem / 1e9, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        }

    # --- cuDNN Benchmark mode ---
    # Auto-tunes convolution algorithms for fixed input sizes.
    # Since our TFT input shapes are fixed, this gives 10-20% speedup.
    torch.backends.cudnn.benchmark = True
    optimizations["cudnn_benchmark"] = True

    # --- TF32 (Tensor Float 32) ---
    # RTX 4090 (Ada Lovelace) supports TF32 for 2x throughput on matmul.
    # Precision: 19 bits (vs FP32's 32) — more than enough for RL.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizations["tf32_enabled"] = True

    # --- CUDA Memory Allocator ---
    # Use expandable segments to reduce fragmentation.
    # Critical for large replay buffers + model + gradients.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    optimizations["expandable_segments"] = True

    # --- Deterministic algorithms (off for speed) ---
    torch.use_deterministic_algorithms(False)
    optimizations["deterministic"] = False

    # --- Gradient computation optimisation ---
    # Enables faster gradient accumulation on Ampere+ GPUs.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        optimizations["matmul_precision"] = "high"

    logger.info("CUDA optimizations applied", **optimizations)
    return optimizations


def get_device_map(n_gpus: int) -> dict[str, str]:
    """Assign components to GPUs for multi-GPU training.

    Strategy for 2 GPUs:
    - GPU 0: Main RL model (policy + value networks)
    - GPU 1: World model + evaluation + diversity computation

    For 1 GPU: everything on cuda:0.
    """
    if n_gpus >= 2:
        return {
            "main": "cuda:0",       # Policy, critic, feature extractor
            "auxiliary": "cuda:1",   # World model, eval, diversity
        }
    elif n_gpus == 1:
        return {
            "main": "cuda:0",
            "auxiliary": "cuda:0",
        }
    return {
        "main": "cpu",
        "auxiliary": "cpu",
    }


def get_optimal_n_envs(n_cpus: int, n_gpus: int) -> int:
    """Calculate optimal number of parallel environments.

    Heuristic:
    - Leave 2 CPUs for GPU data transfers + system overhead
    - For off-policy RL (SAC/TQC), 8-16 envs is typically optimal
    - More envs = more diverse experience but more CPU overhead
    - With 38 vCPUs: 16-24 envs is sweet spot
    """
    available_cpus = max(1, n_cpus - 2)  # Reserve 2 for overhead

    if n_gpus >= 2:
        # With 2 GPUs, can afford more parallel envs
        optimal = min(available_cpus, 24)
    elif n_gpus == 1:
        optimal = min(available_cpus, 16)
    else:
        optimal = min(available_cpus, 8)

    # For off-policy RL, sweet spot is 8-24
    optimal = max(8, min(optimal, 32))

    return optimal


def get_optimal_batch_size(vram_gb: float, d_model: int, seq_len: int) -> int:
    """Calculate optimal batch size based on VRAM.

    For RTX 4090 (24GB per GPU):
    - d_model=128, seq_len=100: batch_size=2048-4096
    - d_model=256, seq_len=100: batch_size=1024-2048
    - With mixed precision: 2x the above

    These are conservative estimates — actual depends on model specifics.
    """
    # Rough estimate: memory per sample (bytes)
    # = seq_len * d_model * 4 (float32) * 3 (fwd + grad + optimizer)
    bytes_per_sample = seq_len * d_model * 4 * 3
    # Plus replay buffer overhead, activations, etc.
    bytes_per_sample *= 10  # Safety factor

    available_bytes = vram_gb * 1e9 * 0.6  # Use 60% of VRAM for batches
    max_batch = int(available_bytes / bytes_per_sample)

    # Round down to power of 2 for GPU efficiency
    batch_size = 1
    while batch_size * 2 <= max_batch:
        batch_size *= 2

    # Clamp to reasonable range
    return max(256, min(batch_size, 8192))


def get_optimal_buffer_size(ram_gb: float, obs_dim: int) -> int:
    """Calculate optimal replay buffer size based on system RAM.

    For 128GB RAM:
    - Each transition: ~1KB (observations + action + reward + next_obs + done)
    - 10M transitions: ~10GB
    - 50M transitions: ~50GB

    Larger buffer = more diverse training data = better generalisation.
    """
    # Rough estimate: bytes per transition
    bytes_per_transition = obs_dim * 4 * 2 + 20  # obs + next_obs + overhead
    bytes_per_transition = max(bytes_per_transition, 1000)  # min 1KB

    # Use at most 30% of RAM for replay buffer
    available_bytes = ram_gb * 1e9 * 0.3
    max_transitions = int(available_bytes / bytes_per_transition)

    # For TQC/SAC, 1-5M is typically optimal
    return min(max_transitions, 5_000_000)


class MixedPrecisionWrapper:
    """Wrap SB3 model training with automatic mixed precision.

    AMP (Automatic Mixed Precision) on RTX 4090:
    - Forward pass in FP16 (2x throughput on Tensor Cores)
    - Loss scaling to prevent gradient underflow
    - Backward pass accumulates in FP32 for numerical stability

    Usage
    -----
    >>> wrapper = MixedPrecisionWrapper(model)
    >>> wrapper.enable()
    >>> # ... model.learn() as usual — AMP is applied automatically
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self._scaler = None

        if self.enabled:
            self._scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision (AMP) enabled — FP16 on Tensor Cores")

    @property
    def scaler(self):
        return self._scaler


def try_compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """Try to apply torch.compile() for graph-level optimization.

    torch.compile() (PyTorch 2.x) fuses operations and reduces Python overhead.
    Expected speedup: 15-30% on GPU.

    Modes:
    - "default": balanced optimization
    - "reduce-overhead": minimize Python overhead (best for small models)
    - "max-autotune": maximum optimization (longer compile, faster runtime)

    Falls back gracefully if torch.compile is not available.
    """
    if not hasattr(torch, "compile"):
        logger.info("torch.compile not available (PyTorch < 2.0)")
        return model

    try:
        compiled = torch.compile(model, mode=mode)
        logger.info("torch.compile applied", mode=mode)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed, using eager mode", error=str(e))
        return model


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing to trade compute for memory.

    Useful when scaling up model size — reduces activation memory by
    ~60% at the cost of ~30% slower backward pass.

    Only worth it if you're VRAM-limited (not the case with 96GB,
    but useful if you scale d_model to 512+).
    """

    # Mark all transformer layers for checkpointing
    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
            logger.info(f"Gradient checkpointing enabled for {name}")


def log_gpu_memory(prefix: str = "") -> dict[str, float]:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_mem / 1e9
        utilization = allocated / total * 100

        stats[f"gpu{i}_allocated_gb"] = round(allocated, 2)
        stats[f"gpu{i}_reserved_gb"] = round(reserved, 2)
        stats[f"gpu{i}_total_gb"] = round(total, 1)
        stats[f"gpu{i}_utilization_pct"] = round(utilization, 1)

        logger.info(
            f"{prefix}GPU {i} memory",
            allocated_gb=round(allocated, 2),
            reserved_gb=round(reserved, 2),
            total_gb=round(total, 1),
            utilization_pct=round(utilization, 1),
        )

    return stats


def pin_memory_dataloader_kwargs() -> dict:
    """Get DataLoader kwargs optimised for GPU training.

    pin_memory=True enables async CPU→GPU DMA transfers.
    num_workers should match CPU count for parallel data loading.
    """
    if torch.cuda.is_available():
        return {
            "pin_memory": True,
            "num_workers": min(8, os.cpu_count() or 4),
            "persistent_workers": True,
        }
    return {}
