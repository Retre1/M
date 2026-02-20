"""Model checkpoint save/load utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    normalizer_state: dict[str, Any] | None,
    save_dir: str | Path,
    name: str = "checkpoint",
) -> Path:
    """Save a complete model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{name}_step{step}_{timestamp}"
    checkpoint_dir = save_dir / checkpoint_name
    checkpoint_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    meta = {
        "step": step,
        "timestamp": timestamp,
        "config": config,
    }
    with open(checkpoint_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if normalizer_state is not None:
        torch.save(normalizer_state, checkpoint_dir / "normalizer.pt")

    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into the model and optimizer."""
    checkpoint_dir = Path(checkpoint_dir)

    model.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True)
    )

    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location=device, weights_only=True)
        )

    with open(checkpoint_dir / "meta.json") as f:
        meta = json.load(f)

    normalizer_state = None
    if (checkpoint_dir / "normalizer.pt").exists():
        normalizer_state = torch.load(
            checkpoint_dir / "normalizer.pt", map_location=device, weights_only=True
        )

    return {
        "step": meta["step"],
        "config": meta.get("config", {}),
        "normalizer_state": normalizer_state,
    }


def find_best_checkpoint(models_dir: str | Path) -> Path | None:
    """Find the most recent checkpoint in the best/ directory."""
    best_dir = Path(models_dir) / "best"
    if not best_dir.exists():
        return None
    checkpoints = sorted(best_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None
