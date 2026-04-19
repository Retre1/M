# Decision: v2.0 Quantum-Hybrid Architecture

**Date:** April 2026
**Status:** In progress

## Context

v1 (TQC + MTF + ProfitFocusedReward) показал что базовая архитектура работает, но нужна:
- Лучшая data quality (bias-free pipeline)
- Больше training data (synthetic augmentation)
- Smarter risk (CVaR-aware reward)
- Regime adaptation (FSD)

## Decision

Интегрировать 5 papers поэтапно, backward-compatible:

## Phased Roadmap

| Phase | Module | Priority | Risk |
|-------|--------|----------|------|
| 1 | Bias-Free Pipeline (LIB/LAB) | **Critical** | Low |
| 1 | FSD Regime Detection | **Critical** | Low |
| 2 | SBBTS Synthetic Data | High | Medium |
| 3 | Reward v5 (CVaR + FSD) | High | Medium |
| 4 | DML World Model | Medium | High |
| 5 | Quantum Kernels | Low | Low (fallback exists) |

## Key Principle

Каждый модуль `enabled: false` по умолчанию. Включаем по одному, валидируем, двигаем дальше. Нет big-bang refactoring.

## Branch

`v2.0-quantum-hybrid` — все v2 work here.

#decision #architecture #v2 #in-progress
