"""Sync checkpoints between platforms (Colab <-> Kaggle <-> Local).

Supports three sync modes:
1. export: Pack latest checkpoint into a portable .tar.gz archive
2. import: Unpack archive into local checkpoint directory
3. status: Show current checkpoint state

Usage:
    # On Colab/Kaggle — export checkpoint to Drive/portable file
    python scripts/sync_checkpoints.py export --checkpoint-dir /content/drive/MyDrive/apexfx/models/checkpoints

    # On new Colab account — import and resume
    python scripts/sync_checkpoints.py import --archive /content/drive/MyDrive/apexfx_checkpoint.tar.gz

    # Check status
    python scripts/sync_checkpoints.py status --checkpoint-dir models/checkpoints
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def find_latest_checkpoint(base_dir: Path) -> Path | None:
    """Find the most recent complete checkpoint."""
    if not base_dir.exists():
        return None

    # Try symlink
    latest_link = base_dir / "resume_latest"
    if latest_link.is_symlink():
        target = latest_link.resolve()
        if target.is_dir() and (target / "_COMPLETE").exists():
            return target

    # Fallback: scan
    candidates = sorted(
        [
            d for d in base_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("resume_")
            and (d / "_COMPLETE").exists()
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def cmd_export(args: argparse.Namespace) -> None:
    """Export latest checkpoint to a portable archive."""
    base_dir = Path(args.checkpoint_dir)
    ckpt = find_latest_checkpoint(base_dir)

    if ckpt is None:
        print("[ERROR] No complete checkpoint found in", base_dir)
        return

    # Read metadata
    meta_path = ckpt / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print("[INFO] Exporting checkpoint:")
        print(f"  Stage: {meta.get('stage_idx', '?')} ({meta.get('stage_name', '?')})")
        print(f"  Timesteps done: {meta.get('total_timesteps_done', '?'):,}")
        print(f"  Remaining: {meta.get('remaining_timesteps', '?'):,}")
        print(f"  Saved at: {meta.get('timestamp_human', '?')}")

    # Create archive
    output = Path(args.output) if args.output else base_dir.parent / "apexfx_checkpoint.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(output), "w:gz") as tar:
        # Add checkpoint directory
        tar.add(str(ckpt), arcname=ckpt.name)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"[OK] Exported to: {output} ({size_mb:.1f} MB)")
    print("[TIP] Copy this file to your next Colab/Kaggle account's Drive")


def cmd_import(args: argparse.Namespace) -> None:
    """Import checkpoint from archive."""
    archive = Path(args.archive)
    if not archive.exists():
        print(f"[ERROR] Archive not found: {archive}")
        return

    target = Path(args.checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Importing from: {archive}")

    with tarfile.open(str(archive), "r:gz") as tar:
        tar.extractall(str(target))

    # Find the imported checkpoint and create symlink
    ckpt = find_latest_checkpoint(target)
    if ckpt:
        latest_link = target / "resume_latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(ckpt.name)

        meta_path = ckpt / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print("[OK] Imported checkpoint:")
            print(f"  Stage: {meta.get('stage_idx', '?')}")
            print(f"  Timesteps: {meta.get('total_timesteps_done', '?'):,}")
            print("  Ready to resume with: python scripts/train.py --config-dir configs/colab --resume")
    else:
        print("[WARN] Imported but no valid checkpoint found")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current checkpoint status."""
    base_dir = Path(args.checkpoint_dir)

    if not base_dir.exists():
        print(f"[INFO] No checkpoint directory: {base_dir}")
        return

    checkpoints = sorted(
        [
            d for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith("resume_")
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        print("[INFO] No checkpoints found")
        return

    print(f"Found {len(checkpoints)} checkpoint(s):\n")

    for ckpt in checkpoints:
        complete = (ckpt / "_COMPLETE").exists()
        meta_path = ckpt / "metadata.json"

        status = "COMPLETE" if complete else "INCOMPLETE"

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            stage = meta.get("stage_idx", "?")
            name = meta.get("stage_name", "?")
            done = meta.get("total_timesteps_done", 0)
            remaining = meta.get("remaining_timesteps", 0)
            saved = meta.get("timestamp_human", "?")

            # Estimate progress
            total = done + remaining
            pct = (done / total * 100) if total > 0 else 0

            print(f"  [{status}] {ckpt.name}")
            print(f"    Stage {stage} ({name})")
            print(f"    Progress: {done:,} / {total:,} ({pct:.1f}%)")
            print(f"    Saved: {saved}")

            # Check files
            files = list(ckpt.iterdir())
            sizes = {f.name: f.stat().st_size / (1024*1024) for f in files if f.is_file()}
            total_mb = sum(sizes.values())
            print(f"    Size: {total_mb:.1f} MB ({len(files)} files)")
        else:
            print(f"  [{status}] {ckpt.name} (no metadata)")

        print()

    # Show latest
    latest = find_latest_checkpoint(base_dir)
    if latest:
        print(f"Latest valid: {latest.name}")
        print("Resume with:  python scripts/train.py --config-dir configs/colab --resume")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ApexFX checkpoints between platforms")
    sub = parser.add_subparsers(dest="command", required=True)

    # export
    p_export = sub.add_parser("export", help="Export latest checkpoint to archive")
    p_export.add_argument("--checkpoint-dir", default="models/checkpoints")
    p_export.add_argument("--output", default=None, help="Output archive path")

    # import
    p_import = sub.add_parser("import", help="Import checkpoint from archive")
    p_import.add_argument("--archive", required=True, help="Path to .tar.gz archive")
    p_import.add_argument("--checkpoint-dir", default="models/checkpoints")

    # status
    p_status = sub.add_parser("status", help="Show checkpoint status")
    p_status.add_argument("--checkpoint-dir", default="models/checkpoints")

    args = parser.parse_args()

    if args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
