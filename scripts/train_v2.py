"""Launch Curriculum v2 training on real EURUSD H1 data.

Usage::

    # Smoke test (~4k timesteps, ~2 stages)
    python scripts/train_v2.py --smoke

    # Full 4-stage curriculum from cached features
    python scripts/train_v2.py --features-cache data/cache/features

    # Override total timesteps (e.g. to first verify on a smaller budget)
    python scripts/train_v2.py --total-timesteps 200000

    # TensorBoard logging
    python scripts/train_v2.py --tb-log-dir runs/v2_eurusd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apexfx.data.bar_loader import load_bars, load_features_cache
from apexfx.training.config import (
    CurriculumV2Config,
    StageConfig,
)
from apexfx.training.trainer_v2 import TrainerV2
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def build_smoke_config() -> CurriculumV2Config:
    """Reduced-timestep config for end-to-end smoke testing."""
    return CurriculumV2Config(
        stages=[
            StageConfig(
                name="smoke_warmup", total_timesteps=2_000,
                real_ratio=1.0, sbbts_ratio=0.0,
                filter_quantile=0.95, warm_start=False, lr_override=3e-4,
            ),
            StageConfig(
                name="smoke_full", total_timesteps=2_000,
                real_ratio=0.7, sbbts_ratio=0.3, warm_start=True,
            ),
        ],
        deterministic=False,
        world_model_enabled=False,  # disable to keep smoke fast
    )


def scale_config(cfg: CurriculumV2Config, total: int) -> CurriculumV2Config:
    """Scale per-stage timesteps so the sum equals `total`."""
    current = cfg.total_timesteps
    if current == 0:
        return cfg
    factor = total / current
    new_stages = [
        s.model_copy(update={"total_timesteps": max(1, int(s.total_timesteps * factor))})
        for s in cfg.stages
    ]
    return cfg.model_copy(update={"stages": new_stages})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 curriculum")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--data-root", default="data/raw/bars")
    parser.add_argument("--features-cache", default="data/cache/features",
                        help="Directory of pre-computed feature parquet files")
    parser.add_argument("--checkpoint-dir", default="models/v2_checkpoints")
    parser.add_argument("--tb-log-dir", default=None,
                        help="TensorBoard log directory (optional)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a short smoke test instead of full training")
    parser.add_argument("--max-bars", type=int, default=None,
                        help="Truncate real data to N most recent bars")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override sum of stage timesteps")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip features cache (recompute on the fly)")
    args = parser.parse_args()

    # 1. Load data — prefer cached features for speed
    real_data = None
    if not args.no_cache:
        real_data = load_features_cache(args.symbol, args.timeframe,
                                        args.features_cache)
    if real_data is None:
        logger.info("No cache — loading raw bars",
                    symbol=args.symbol, timeframe=args.timeframe)
        real_data = load_bars(args.symbol, args.timeframe, args.data_root)
        logger.warning(
            "Features will be computed at stage build (slow). "
            "Run scripts/cache_features_v2.py beforehand on the server.",
        )

    # 2. Optional truncation
    if args.max_bars is not None and len(real_data) > args.max_bars:
        real_data = real_data.tail(args.max_bars).reset_index(drop=True)
        logger.info("Truncated bars", n_bars=len(real_data))
    elif args.smoke and len(real_data) > 2500:
        real_data = real_data.tail(2500).reset_index(drop=True)
        logger.info("Smoke truncation", n_bars=len(real_data))
    logger.info("Data ready", n_bars=len(real_data))

    # 3. Config
    cfg = build_smoke_config() if args.smoke else CurriculumV2Config()
    if args.total_timesteps is not None:
        cfg = scale_config(cfg, args.total_timesteps)
        logger.info("Scaled total timesteps",
                    total=args.total_timesteps,
                    per_stage=[s.total_timesteps for s in cfg.stages])

    # 4. Train
    trainer = TrainerV2(
        curriculum_config=cfg,
        real_data=real_data,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    # Wire TB log dir if provided (set via env so SB3 picks it up)
    if args.tb_log_dir:
        import os
        os.environ["SB3_TB_LOG_DIR"] = args.tb_log_dir
        Path(args.tb_log_dir).mkdir(parents=True, exist_ok=True)
        logger.info("TensorBoard logging", dir=args.tb_log_dir)

    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — partial summary may be available")
        sys.exit(130)
    logger.info("Training finished", **summary)


if __name__ == "__main__":
    main()
