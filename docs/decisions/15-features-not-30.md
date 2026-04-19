# Decision: 15 Features (not 30)

**Date:** April 2026
**Status:** Active

## Context

Feature pipeline генерирует 99 features. Нужно выбрать subset для observation space.

## Decision

15 features через `feature_selector.json` (auto-selected during training).

## Rationale

- MTF obs shape: `(d1_lookback + h1_lookback + m5_lookback) × n_features`
- С 30 features: `(15+40+30) × 30 = 2550` → слишком большой obs для TQC critic
- С 15 features: `(15+40+30) × 15 = 1275` → manageable
- Curse of dimensionality: больше features ≠ лучше при < 1M samples
- Feature selection убирает redundant/noisy features

## Obs Shape Gotcha

Первый баг в visualization: model ожидает obs(225), script давал obs(450) из-за `min(n_features, 30) = 30` вместо `len(selected) = 15`.

**Fix:** Загружать `feature_selector.json` из checkpoint и фильтровать data.

#decision #features #observation-space
