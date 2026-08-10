# ApexFX Quantum — Knowledge Base

> TFT Encoder + Multi-Agent RL (TQC) + MTF Environment + Curriculum Learning

## Quick Links

- [[worklog|Рабочий журнал]] — решения, ловушки репозитория, разобранные ошибки
- [[architecture/system-overview|System Overview]]
- [[architecture/tz-architecture|ТЗ — Архитектура программного комплекса]]
- [[research/papers-index|Papers Index]]
- [[runs/runs-index|Training Runs]] — ⚠️ метрики прогонов 1–6 недействительны, см. [[analysis/metrics-invalidation|разбор]]
- [[decisions/decisions-index|Decision Log]]

## Current Status

- **Version:** v2.0-quantum-hybrid (in progress)
- **Best result:** Run #1 baseline, -1.22% return (model undertrained)
- **Next step:** Phase 1 — Bias-Free Data Pipeline (FSD + SBBTS + LIB/LAB correction)

## Architecture Stack

| Layer | Component | Status |
|-------|-----------|--------|
| Data | MTF (D1+H1+M5) + FeaturePipeline (99 features) | Working |
| Encoder | TFT (d=64, 4 heads) | Working |
| Agents | Trend / Reversion / Breakout (TQC) | Working |
| Gating | Gating Network (128→64) | Working |
| Reward | ProfitFocusedReward v4 | Working, needs retune |
| Curriculum | 4 stages, 5M total steps | Tested to 300K |
| v2.0 FSD | Regime detection via skewness | Scaffolded |
| v2.0 SBBTS | Synthetic data generation | Scaffolded |
| v2.0 DML | Differential ML + jump-diffusion | Scaffolded |
| v2.0 Quantum | QAOA/QAE/CVaR kernels | Scaffolded |
