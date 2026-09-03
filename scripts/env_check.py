"""Pre-flight check for server training.

Verifies:
- Python 3.11+
- Required packages importable
- CUDA / MPS availability
- Project package importable
- Real-data parquet files exist
- Optional: features cache present
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "torch", "gymnasium", "stable_baselines3", "sb3_contrib",
    "pandas", "numpy", "scipy", "sklearn", "pyarrow", "pydantic",
    "structlog", "tensorboard",
]


def check_python() -> bool:
    ok = sys.version_info >= (3, 11)
    print(f"[{'OK' if ok else 'FAIL'}] Python {platform.python_version()}  (need 3.11+)")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"[OK]   {pkg:<22} {ver}")
        except ImportError as e:
            print(f"[FAIL] {pkg:<22} ImportError: {e}")
            all_ok = False
    return all_ok


def check_devices() -> bool:
    try:
        import torch
    except ImportError:
        print("[FAIL] torch not importable")
        return False
    print(f"[INFO] torch              {torch.__version__}")
    cuda = torch.cuda.is_available()
    mps = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if cuda:
        print(f"[OK]   CUDA available     devices={torch.cuda.device_count()}  "
              f"name={torch.cuda.get_device_name(0)}")
    elif mps:
        print("[OK]   MPS available (macOS) — slower than CUDA")
    else:
        print("[WARN] No GPU detected — will train on CPU (very slow)")
    return True


def check_apexfx() -> bool:
    try:
        from apexfx.training.config import CurriculumV2Config
        from apexfx.training.trainer_v2 import TrainerV2  # noqa: F401
        from apexfx.data.bar_loader import load_bars  # noqa: F401
        cfg = CurriculumV2Config()
        print(f"[OK]   apexfx package importable, n_stages={cfg.n_stages}, "
              f"total_timesteps={cfg.total_timesteps:,}")
        return True
    except Exception as e:
        print(f"[FAIL] apexfx import: {e}")
        return False


def check_data(symbol: str, timeframe: str, data_root: Path,
               cache_root: Path) -> bool:
    bar_dir = data_root / symbol / timeframe
    files = sorted(bar_dir.glob("*.parquet"))
    if not files:
        print(f"[FAIL] No bar files at {bar_dir}")
        return False
    print(f"[OK]   {symbol} {timeframe}: {len(files)} parquet files at {bar_dir}")

    cache_path = cache_root / symbol / timeframe / "features.parquet"
    if cache_path.exists():
        print(f"[OK]   features cache present: {cache_path}")
    else:
        print(f"[WARN] features cache missing: {cache_path}\n"
              f"       Run: python scripts/cache_features_v2.py "
              f"--symbol {symbol} --timeframe {timeframe}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/bars"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/cache/features"))
    args = parser.parse_args()

    checks = [
        check_python(),
        check_packages(),
        check_devices(),
        check_apexfx(),
        check_data(args.symbol, args.timeframe, args.data_root, args.cache_root),
    ]
    failed = sum(1 for c in checks if not c)
    print()
    if failed == 0:
        print("All checks passed — ready to train.")
        return 0
    print(f"{failed} check(s) failed — fix before launching training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
