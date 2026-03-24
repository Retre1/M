"""Atomic checkpoint manager for resumable training across sessions.

Saves a complete training state bundle:
- SB3 model (policy + optimizer + replay buffer)
- EWC regularizer state (Fisher + optimal params)
- Feature selector state (selected features)
- Curriculum progress (current stage, timesteps completed)
- Training metadata (timestamp, config hash, metrics)

Designed for interrupted training on free-tier platforms (Colab 12h,
Kaggle 12h) where sessions can be killed without warning.

Usage
-----
>>> manager = CheckpointManager("models/checkpoints")
>>> manager.save(model, stage_idx=1, ewc_state=ewc.state_dict(), ...)
>>> state = CheckpointManager.find_latest("models/checkpoints")
>>> if state: trainer.resume_from(state)
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Sentinel file written last — if it exists, the checkpoint is complete
_COMPLETE_MARKER = "_COMPLETE"


@dataclass
class CheckpointState:
    """Everything needed to resume training from a checkpoint."""

    checkpoint_dir: Path
    stage_idx: int
    total_timesteps_done: int
    remaining_timesteps: int
    model_path: Path
    ewc_state: dict | None = None
    feature_selector_state: dict | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return (self.checkpoint_dir / _COMPLETE_MARKER).exists()


class CheckpointManager:
    """Manages atomic save/load of full training state bundles.

    Checkpoint directory layout::

        models/checkpoints/
        ├── resume_stage_0_step_200000/
        │   ├── model.zip              # SB3 model
        │   ├── ewc_state.pt           # EWC Fisher + optimal params
        │   ├── feature_selector.json  # Selected feature names
        │   ├── metadata.json          # Stage, timesteps, timestamp, metrics
        │   └── _COMPLETE              # Atomicity marker
        ├── resume_stage_1_step_500000/
        │   └── ...
        └── resume_latest -> resume_stage_1_step_500000  (symlink)
    """

    def __init__(self, base_dir: str | Path, keep_n: int = 3) -> None:
        self._base_dir = Path(base_dir)
        self._keep_n = keep_n

    def save(
        self,
        model,
        *,
        stage_idx: int,
        total_timesteps_done: int,
        remaining_timesteps: int,
        stage_name: str = "",
        ewc_state: dict | None = None,
        feature_selector_state: dict | None = None,
        metrics: dict | None = None,
    ) -> Path:
        """Save a complete, atomic checkpoint bundle.

        The checkpoint is only considered valid once the _COMPLETE marker
        is written — if the process dies mid-save, the incomplete directory
        is ignored on resume.
        """
        dirname = f"resume_stage_{stage_idx}_step_{total_timesteps_done}"
        ckpt_dir = self._base_dir / dirname
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save SB3 model (policy + optimizer + replay buffer)
        model_path = ckpt_dir / "model"
        model.save(str(model_path))
        logger.info("Checkpoint: model saved", path=str(model_path))

        # 2. Save replay buffer separately for off-policy algorithms
        if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
            try:
                buf_path = ckpt_dir / "replay_buffer"
                model.save_replay_buffer(str(buf_path))
                logger.info("Checkpoint: replay buffer saved", size=model.replay_buffer.size())
            except Exception as e:
                logger.warning("Checkpoint: replay buffer save failed", error=str(e))

        # 3. Save EWC state
        if ewc_state is not None:
            ewc_path = ckpt_dir / "ewc_state.pt"
            torch.save(ewc_state, str(ewc_path))
            logger.info("Checkpoint: EWC state saved")

        # 4. Save feature selector state
        if feature_selector_state is not None:
            fs_path = ckpt_dir / "feature_selector.json"
            with open(fs_path, "w") as f:
                json.dump(feature_selector_state, f, indent=2)
            logger.info("Checkpoint: feature selector saved")

        # 5. Write metadata
        meta = {
            "stage_idx": stage_idx,
            "stage_name": stage_name,
            "total_timesteps_done": total_timesteps_done,
            "remaining_timesteps": remaining_timesteps,
            "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics or {},
        }
        meta_path = ckpt_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 6. Mark checkpoint as complete (atomic — last step)
        (ckpt_dir / _COMPLETE_MARKER).touch()

        # 7. Update symlink to latest
        latest_link = self._base_dir / "resume_latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(dirname)

        logger.info(
            "Checkpoint saved (complete)",
            dir=dirname,
            stage=stage_idx,
            timesteps=total_timesteps_done,
        )

        # 8. Cleanup old checkpoints
        self._cleanup_old()

        return ckpt_dir

    def _cleanup_old(self) -> None:
        """Keep only the N most recent complete checkpoints."""
        complete = sorted(
            [
                d
                for d in self._base_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("resume_")
                and (d / _COMPLETE_MARKER).exists()
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for old_dir in complete[self._keep_n :]:
            logger.info("Removing old checkpoint", dir=old_dir.name)
            shutil.rmtree(old_dir)

    @classmethod
    def find_latest(cls, base_dir: str | Path) -> CheckpointState | None:
        """Find the most recent complete checkpoint in base_dir.

        Returns None if no valid checkpoint exists.
        """
        base = Path(base_dir)
        if not base.exists():
            return None

        # Try symlink first
        latest_link = base / "resume_latest"
        if latest_link.is_symlink():
            target = base / latest_link.resolve().name
            if target.is_dir():
                state = cls._load_checkpoint(target)
                if state is not None:
                    return state

        # Fallback: scan for newest complete checkpoint
        candidates = sorted(
            [
                d
                for d in base.iterdir()
                if d.is_dir()
                and d.name.startswith("resume_")
                and (d / _COMPLETE_MARKER).exists()
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        for candidate in candidates:
            state = cls._load_checkpoint(candidate)
            if state is not None:
                return state

        return None

    @classmethod
    def _load_checkpoint(cls, ckpt_dir: Path) -> CheckpointState | None:
        """Load checkpoint state from a directory."""
        meta_path = ckpt_dir / "metadata.json"
        model_path = ckpt_dir / "model.zip"

        if not meta_path.exists() or not model_path.exists():
            logger.warning("Incomplete checkpoint (missing files)", dir=ckpt_dir.name)
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        # Load EWC state if available
        ewc_state = None
        ewc_path = ckpt_dir / "ewc_state.pt"
        if ewc_path.exists():
            ewc_state = torch.load(str(ewc_path), map_location="cpu", weights_only=True)

        # Load feature selector state if available
        fs_state = None
        fs_path = ckpt_dir / "feature_selector.json"
        if fs_path.exists():
            with open(fs_path) as f:
                fs_state = json.load(f)

        return CheckpointState(
            checkpoint_dir=ckpt_dir,
            stage_idx=meta["stage_idx"],
            total_timesteps_done=meta["total_timesteps_done"],
            remaining_timesteps=meta["remaining_timesteps"],
            model_path=model_path,
            ewc_state=ewc_state,
            feature_selector_state=fs_state,
            metadata=meta,
        )
