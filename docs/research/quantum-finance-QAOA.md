# Quantum Computing for Finance

**Papers:** QAOA / QAE / CVaR quantum algorithms for portfolio optimization

## Algorithms

### QAOA (Quantum Approximate Optimization)
Комбинаторная оптимизация портфеля:
```
max sum(mu_i * x_i) - gamma * sum(sigma_ij * x_i * x_j)
s.t. sum(x_i) = K
```
Квантовое преимущество при > 50 assets.

### QAE (Quantum Amplitude Estimation)
Оценка хвостовых рисков с квадратичным ускорением:
- Классика: O(1/epsilon^2) сэмплов
- QAE: O(1/epsilon)

### CVaR (Conditional Value at Risk)
```
CVaR_alpha = E[L | L > VaR_alpha]
```
Quantum-ускоренная оценка через QAE.

## Применение в ApexFX

- **Пока:** classical fallback (PennyLane simulation, ~8 qubits)
- **Будущее:** quantum hardware через AWS Braket / IBM Quantum
- **Реально полезно:** CVaR для risk estimation в reward v5

## Реализация

- Файл: `src/apexfx/models/quantum_kernel.py`
- Классы: `QuantumFeatureMap`, `QAOAPortfolioOptimizer`, `QuantumCVaR`
- Phase 5 (lowest priority)

## Honest Assessment

На текущем этапе (8 qubits, simulation) квантовое преимущество = 0.
Но архитектура готова для масштабирования когда hardware дозреет.
Classical fallback работает identically.

#research #quantum #portfolio-optimization #future
