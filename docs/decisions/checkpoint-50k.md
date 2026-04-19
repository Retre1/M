# Decision: Checkpoint every 50K steps

**Date:** April 2026
**Status:** Active

## Context

Run #2 — model at 200K steps was the only usable checkpoint. Training went to 300K but plateau.

## Problem

Config had `save_freq: 100000` which meant 0 checkpoints after 80K steps of training. We only had auto-save, not the 50K interval we wanted.

## Fix

`configs/training.yaml:56` changed `save_freq: 100000` → `save_freq: 50000`.

## Caveat

The callback reads config once at startup. Running containers need restart to pick up the change.

## Rationale

- 50K = ~2 hours training on RTX 4090
- Gives 10 checkpoints per 500K stage
- Can pick best intermediate model if training diverges later
- Disk cost: ~40MB per checkpoint (model.zip + ewc_state.pt + metadata)

#decision #checkpointing
