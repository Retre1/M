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

    # Use 16 parallel envs on a multi-core server
    python scripts/train_v2.py --n-envs 16 --vec-kind subproc

    # Multi-symbol training across four FX majors
    python scripts/train_v2.py \\
        --symbols EURUSD GBPUSD USDJPY AUDUSD \\
        --n-envs 16 --vec-kind subproc

    # Larger replay buffer (requires RAM — ~30 KB per transition)
    python scripts/train_v2.py --buffer-size 5000000 --batch-size 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apexfx.data.bar_loader import (
    load_bars,
    load_features_cache,
    load_multi_symbol_features,
)
from apexfx.training.config import (
    CurriculumV2Config,
    StageConfig,
    load_curriculum_v2_config,
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


def apply_overrides(
    cfg: CurriculumV2Config,
    *,
    n_envs: int | None,
    vec_kind: str | None,
    buffer_size: int | None,
    batch_size: int | None,
    learning_starts: int | None,
    symbols: tuple[str, ...] | None,
    timeframe: str | None,
) -> CurriculumV2Config:
    """Apply CLI overrides onto an immutable Pydantic config."""
    update: dict = {}

    if n_envs is not None or vec_kind is not None:
        vec = cfg.vec_env
        vec_update: dict = {}
        if n_envs is not None:
            vec_update["n_envs"] = n_envs
        if vec_kind is not None:
            vec_update["kind"] = vec_kind
        update["vec_env"] = vec.model_copy(update=vec_update)

    if any(v is not None for v in (buffer_size, batch_size, learning_starts)):
        rb = cfg.replay_buffer
        rb_update: dict = {}
        if buffer_size is not None:
            rb_update["buffer_size"] = buffer_size
        if batch_size is not None:
            rb_update["batch_size"] = batch_size
        if learning_starts is not None:
            rb_update["learning_starts"] = learning_starts
        update["replay_buffer"] = rb.model_copy(update=rb_update)

    if symbols is not None or timeframe is not None:
        ms = cfg.multi_symbol
        ms_update: dict = {}
        if symbols is not None:
            ms_update["symbols"] = tuple(symbols)
        if timeframe is not None:
            ms_update["timeframe"] = timeframe
        update["multi_symbol"] = ms.model_copy(update=ms_update)

    return cfg.model_copy(update=update) if update else cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 curriculum")
    parser.add_argument("--symbol", default="EURUSD",
                        help="Single-symbol shortcut (ignored if --symbols given)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Multiple symbols for parallel multi-symbol training")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--data-root", default="data/raw/bars")
    parser.add_argument("--features-cache", default="data/cache/features",
                        help="Directory of pre-computed feature parquet files")
    parser.add_argument("--checkpoint-dir", default="models/v2_checkpoints")
    parser.add_argument("--config-dir", default="configs",
                        help="Directory holding training.yaml (ignored with --smoke)")
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
    # Vec-env + replay buffer sizing
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--vec-kind", choices=["dummy", "subproc"], default=None,
                        help="Vec-env backend (subproc scales with CPU cores)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer capacity in transitions")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="SGD batch size for gradient updates")
    parser.add_argument("--learning-starts", type=int, default=None,
                        help="Delay gradient updates until buffer has N samples")
    args = parser.parse_args()

    symbols = tuple(args.symbols) if args.symbols else (args.symbol,)

    # 1. Load data — multi- or single-symbol
    real_data = None
    multi_symbol_data: dict = {}
    if len(symbols) > 1:
        multi_symbol_data = load_multi_symbol_features(
            symbols=symbols,
            timeframe=args.timeframe,
            cache_root=args.features_cache,
            data_root=args.data_root,
        )
        # primary symbol's frame feeds curriculum blending
        real_data = multi_symbol_data[symbols[0]]
    else:
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
        if multi_symbol_data:
            multi_symbol_data = {
                s: d.tail(args.max_bars).reset_index(drop=True)
                for s, d in multi_symbol_data.items()
            }
    elif args.smoke and len(real_data) > 2500:
        real_data = real_data.tail(2500).reset_index(drop=True)
        logger.info("Smoke truncation", n_bars=len(real_data))
        if multi_symbol_data:
            multi_symbol_data = {
                s: d.tail(2500).reset_index(drop=True)
                for s, d in multi_symbol_data.items()
            }
    logger.info("Data ready",
                n_bars=len(real_data),
                n_symbols=len(multi_symbol_data) or 1)

    # 3. Config
    # Smoke runs keep their own tiny built-in curriculum; a real run reads
    # configs/training.yaml. Before this, every run silently used the
    # hardcoded CurriculumV2Config defaults and ignored the YAML entirely.
    cfg = build_smoke_config() if args.smoke else load_curriculum_v2_config(args.config_dir)
    if args.total_timesteps is not None:
        cfg = scale_config(cfg, args.total_timesteps)
        logger.info("Scaled total timesteps",
                    total=args.total_timesteps,
                    per_stage=[s.total_timesteps for s in cfg.stages])
    cfg = apply_overrides(
        cfg,
        n_envs=args.n_envs,
        vec_kind=args.vec_kind,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        symbols=symbols if len(symbols) > 1 else None,
        timeframe=args.timeframe,
    )

    logger.info(
        "Effective runtime config",
        n_envs=cfg.vec_env.n_envs,
        vec_kind=cfg.vec_env.kind,
        buffer_size=cfg.replay_buffer.buffer_size,
        batch_size=cfg.replay_buffer.batch_size,
        learning_starts=cfg.replay_buffer.learning_starts,
        symbols=cfg.multi_symbol.symbols,
    )

    # 4. Train
    trainer = TrainerV2(
        curriculum_config=cfg,
        real_data=real_data,
        checkpoint_dir=Path(args.checkpoint_dir),
        multi_symbol_data=multi_symbol_data or None,
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
