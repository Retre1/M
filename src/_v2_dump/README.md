# v2.0-quantum-hybrid code dump

Snapshot of 6 key files from branch `v2.0-quantum-hybrid` at commit `a7b611e`
(2026-04-19), copied to `main` so external tools that can't read the branch
directly have a stable URL to fetch from.

## Files
| File | Original path | Purpose |
|---|---|---|
| `trainer_v2.py` | `src/apexfx/training/trainer_v2.py` | Curriculum v2 trainer orchestration |
| `config.py` | `src/apexfx/training/config.py` | **Hardcoded** `CurriculumV2Config` defaults (yaml bypassed) |
| `reward_v5.py` | `src/apexfx/env/reward_v5.py` | RARAReward_v5 with 9-component reward |
| `forex_env.py` | `src/apexfx/env/forex_env.py` | Gymnasium env + `set_step_context` isinstance dispatch |
| `training.yaml` | `configs/training.yaml` | Run 6 training config (note: ignored by trainer_v2) |
| `model.yaml` | `configs/model.yaml` | Model config incl. `uncertainty` section (Run 6 knobs) |

**This is a read-only snapshot.** Edit the originals on `v2.0-quantum-hybrid`,
not these copies. Refresh this folder after major changes.
