# P0 Implementation Plan — 5 Hedge-Fund Critical Features

## 1. Prioritized Experience Replay (PER)

**New file:** `src/apexfx/training/per.py`

SB3 SAC/TQC поддерживают `replay_buffer_class` + `replay_buffer_kwargs`. ApexFX использует Dict obs space → нужно расширять `DictReplayBuffer`.

**Реализация:**
- `PrioritizedDictReplayBuffer(DictReplayBuffer)` — PER для Dict observation spaces
- Sum-tree для O(log N) пропорционального сэмплинга по приоритетам
- `SumTree` — массивный бинарное дерево: `add(priority, data_idx)`, `sample(batch)`, `update(idx, priority)`
- Priority = |TD-error|^α + ε (α=0.6, ε=1e-6)
- Importance Sampling weights β (аннилинг 0.4→1.0) для корректировки bias
- Override `add()` — назначает max_priority новым переходам
- Override `sample()` → `PrioritizedDictReplayBufferSamples` с полем `weights` для IS
- `update_priorities(indices, td_errors)` — вызывается после gradient step
- `PERCallback(BaseCallback)` — после каждого gradient step извлекает TD-ошибки из critic и вызывает update_priorities

**Интеграция в trainer.py:**
- `_build_model()` и `_build_mtf_model()`: передать `replay_buffer_class=PrioritizedDictReplayBuffer`, `replay_buffer_kwargs={"alpha": 0.6, "beta_start": 0.4, ...}`
- Добавить `PERCallback` в `_build_callbacks()`
- PPO не использует replay buffer → PER только для SAC/TQC

**Config в schema.py:**
```python
class PERConfig(BaseModel):
    enabled: bool = True
    alpha: float = 0.6          # prioritization exponent
    beta_start: float = 0.4     # IS weight start
    beta_end: float = 1.0       # IS weight end
    beta_annealing_steps: int = 0  # 0 = auto (= total_timesteps)
    epsilon: float = 1e-6       # small constant for priority
```

---

## 2. Advanced Data Augmentation Pipeline

**New file:** `src/apexfx/training/augmentation.py`

4 метода аугментации временных рядов, реализованных как gym.Wrapper.

**Методы:**
1. **TimeWarp** — растяжение/сжатие сегментов через cubic spline. Для ключей `market_features` и `time_features` (flattened → reshape → warp → flatten).
2. **MagnitudeWarp** — мультипликативная деформация: feature *= smooth_random_curve. Через cubic spline с σ=0.1.
3. **WindowSlice** — случайный crop 80-90% точек, zero-pad до оригинального размера.
4. **Mixup** — хранит предыдущий observation; new_obs = λ*obs + (1-λ)*prev_obs, λ~Beta(α, α).

**Реализация:**
- `AugmentedObsWrapper(gym.Wrapper)` — применяет все 4 метода с независимыми вероятностями
- Dict observation: аугментирует только `market_features` и `time_features` (последовательные данные)
- `trend_features`, `reversion_features`, `regime_features` — аугментируются только magnitude warp (они точечные, не временные)
- `position_state` — НИКОГДА не аугментируется
- Каждый метод работает с flattened arrays, reshaping по (lookback, n_features)

**Интеграция в trainer.py:**
- Вставить `AugmentedObsWrapper` в `_build_env()`/`_build_mtf_env()` ПОСЛЕ adversarial wrapper
- Аугментация дополняет adversarial noise (разные цели: noise = робастность, augmentation = генерализация)

**Config:**
```python
class AugmentationConfig(BaseModel):
    enabled: bool = True
    time_warp_prob: float = 0.3
    time_warp_sigma: float = 0.2
    magnitude_warp_prob: float = 0.3
    magnitude_warp_sigma: float = 0.1
    window_slice_prob: float = 0.2
    window_slice_ratio: float = 0.9
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.2
```

---

## 3. Multi-Symbol Portfolio Trading

### 3a. Portfolio Manager
**New file:** `src/apexfx/live/portfolio_manager.py`

```python
class PortfolioManager:
    """Coordinates multi-symbol trading with shared equity."""

    def __init__(self, config, symbols, correlation_risk_manager):
        self._states: dict[str, StateManager] = {}  # per-symbol
        self._corr_risk = correlation_risk_manager
        self._open_positions: dict[str, PositionInfo] = {}

    def check_new_trade(self, symbol, direction, notional) -> PortfolioRiskResult:
        """Check if new trade is allowed given portfolio constraints."""
        # 1. Total exposure check (max_total_exposure)
        # 2. Per-symbol concentration check (max_per_symbol)
        # 3. Correlation check via existing CorrelationRiskManager
        # Returns: approved, scale_factor, reason

    def update_position(self, symbol, direction, volume, entry_price): ...
    def close_position(self, symbol, exit_price, pnl): ...
    def get_portfolio_metrics(self) -> PortfolioMetrics: ...
    def get_aggregate_equity(self) -> float: ...
```

### 3b. Multi-Symbol Live Loop
**New file:** `src/apexfx/live/multi_symbol_loop.py`

```python
class MultiSymbolTradingLoop:
    """Orchestrates parallel trading across multiple symbols."""

    def __init__(self, config, symbols: list[str]):
        self._portfolio = PortfolioManager(...)
        self._loops: dict[str, LiveTradingLoop] = {}  # per-symbol
        self._execution_lock = asyncio.Lock()  # serialize execution decisions

    async def run(self):
        # Spawn per-symbol loops with shared portfolio manager
        tasks = [loop.run() for loop in self._loops.values()]
        await asyncio.gather(*tasks)
```

### 3c. Интеграция с RiskManager
**Modify:** `src/apexfx/risk/risk_manager.py`

Добавить метод `set_portfolio_context()`:
```python
def set_portfolio_context(self, open_positions: list[PositionInfo]) -> None:
    """Inject cross-pair context for portfolio-aware risk checks."""
    self._portfolio_positions = open_positions
```

Новый check в `evaluate_action()` после Check 4d (uncertainty):
```python
# --- Check 4e: Portfolio concentration ---
if self._portfolio_positions:
    result = self._corr_risk.check_new_position(...)
    if not result.approved:
        return RiskDecision(approved=False, ...)
    var_scale *= result.scale_factor
```

### 3d. Config
```python
class PortfolioConfig(BaseModel):
    enabled: bool = False
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    max_total_exposure: float = 0.40
    max_per_symbol: float = 0.25
    correlation_limit: float = 0.70
    rebalance_freq_bars: int = 24
```

---

## 4. Online Incremental Learning

**New file:** `src/apexfx/live/online_learner.py`

```python
class OnlineLearner:
    """Micro-updates on live data with EWC protection."""

    def __init__(self, model, ewc_state_path, config):
        self._model = model
        self._ewc = EWCRegularizer(...)
        self._ewc.load_state_dict(torch.load(ewc_state_path))  # From training
        self._mini_buffer: deque = deque(maxlen=config.mini_buffer_size)
        self._trade_count = 0
        self._rolling_returns: deque = deque(maxlen=config.drift_detection_window)
        self._base_lr = model.learning_rate
        self._checkpoints: deque = deque(maxlen=config.max_rollback_checkpoints)

    def record_transition(self, obs, action, reward, next_obs, done):
        """Store live transition in mini-buffer."""
        self._mini_buffer.append((obs, action, reward, next_obs, done))

    def record_trade_result(self, pnl_return: float):
        """Track trade for drift detection."""
        self._rolling_returns.append(pnl_return)
        self._trade_count += 1

    def maybe_update(self) -> bool:
        """Run micro-update if enough trades accumulated."""
        if self._trade_count % config.update_every_n_trades != 0:
            return False
        if len(self._mini_buffer) < config.gradient_steps * model.batch_size:
            return False

        # Detect drift → adjust LR
        drift = self._detect_drift()
        lr = self._base_lr * (config.adaptation_lr_scale if drift else config.lr_scale)
        n_steps = config.adaptation_gradient_steps if drift else config.gradient_steps

        # Checkpoint before update
        self._save_checkpoint()

        # Mini-training loop with EWC penalty
        self._run_gradient_steps(n_steps, lr)
        return True

    def _detect_drift(self) -> bool:
        """Rolling Sharpe < threshold → regime changed."""
        if len(self._rolling_returns) < 20:
            return False
        returns = np.array(self._rolling_returns)
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        return sharpe < config.drift_threshold_sharpe

    def _run_gradient_steps(self, n_steps, lr): ...
    def _save_checkpoint(self): ...
    def rollback(self): ...
```

**Интеграция в LiveTradingLoop:**
- `_process_bar()`: после каждого бара → `online_learner.record_transition()`
- После закрытия позиции → `online_learner.record_trade_result(pnl)`
- После record_trade → `online_learner.maybe_update()`

**Config:**
```python
class OnlineLearningConfig(BaseModel):
    enabled: bool = True
    mini_buffer_size: int = 500
    update_every_n_trades: int = 10
    gradient_steps: int = 3
    lr_scale: float = 0.1
    ewc_enabled: bool = True
    drift_detection_window: int = 50
    drift_threshold_sharpe: float = -0.5
    adaptation_gradient_steps: int = 10
    adaptation_lr_scale: float = 0.5
    checkpoint_every_n_updates: int = 10
    max_rollback_checkpoints: int = 3
```

---

## 5. Regime-Conditional Execution

### 5a. Uncertainty в Signal Generator
**Modify:** `src/apexfx/live/signal_generator.py`

- Добавить `UncertaintyEstimator` в `SignalGenerator.__init__()`
- Добавить поля в `TradingSignal`: `uncertainty_score: float = 0.0`, `position_scale: float = 1.0`, `recommended_stop_atr_mult: float = 2.5`
- В `generate()`: после predict → вызвать `uncertainty_estimator.estimate_from_extractor()`
- Вычислить `recommended_stop_atr_mult`:
  - uncertainty < 0.3 → 1.5 (тугой стоп)
  - uncertainty 0.3-0.6 → 2.5 (нормальный)
  - uncertainty > 0.6 → 4.0 (широкий) или skip

### 5b. Dynamic Stop-Loss
**Modify:** `src/apexfx/risk/risk_manager.py`

Новый метод:
```python
def compute_dynamic_stop(self, uncertainty_score, regime, current_atr) -> DynamicStopConfig:
    # Regime базовые множители:
    base_mult = {"trending": 2.0, "mean_reverting": 3.0, "volatile": 4.0, "flat": 2.5}
    # Uncertainty корректировка: high uncertainty → wider stop
    mult = base_mult[regime] * (1.0 + 0.5 * uncertainty_score)
    trailing = regime == "trending"  # Trail only in trends
    return DynamicStopConfig(atr_mult=mult, trailing=trailing)
```

### 5c. Regime Transition в Live Loop
**Modify:** `src/apexfx/live/trading_loop.py`

```python
# В __init__:
self._prev_regime: str | None = None
self._regime_transition_cooldown: int = 0

# В _process_bar():
if signal.regime != self._prev_regime and self._prev_regime is not None:
    if self._is_major_transition(self._prev_regime, signal.regime):
        # Flatten position on major regime change
        await self._executor.close_all()
        self._regime_transition_cooldown = 3  # bars
    else:
        # Reduce by 50% on minor transition
        await self._executor.reduce_position(0.5)

self._prev_regime = signal.regime

# Skip new entries during cooldown
if self._regime_transition_cooldown > 0:
    self._regime_transition_cooldown -= 1
    return
```

### 5d. Передача uncertainty в RiskManager
**Modify:** `src/apexfx/live/trading_loop.py`

```python
# Текущий код (line 310):
risk_decision = self._risk_manager.evaluate_action(signal.action, market_state)

# Заменить на:
risk_decision = self._risk_manager.evaluate_action(
    signal.action,
    market_state,
    uncertainty_score=signal.uncertainty_score,  # ← NEW
)
```

RiskManager уже поддерживает `uncertainty_score` параметр (Check 4d) — нужно только передать.

---

## Порядок реализации

1. **PER** — 1 новый файл + trainer.py + schema.py (standalone)
2. **Data Augmentation** — 1 новый файл + trainer.py + schema.py (standalone)
3. **Regime-Conditional Execution** — modify 3 существующих файла (signal_generator, risk_manager, trading_loop)
4. **Online Learning** — 1 новый файл + trading_loop.py + schema.py
5. **Multi-Symbol Portfolio** — 2 новых файла + risk_manager + schema.py

## Тесты

- PER: SumTree корректность, пропорциональное сэмплирование, IS weights, интеграция с DictReplayBuffer
- Augmentation: сохранение формы, вероятности, position_state неприкосновенность
- Online Learning: буфер, drift detection, gradient steps, rollback
- Portfolio Manager: лимиты экспозиции, корреляционные проверки
- Regime Execution: transition detection, cooldown, dynamic stops
- Все 61 существующих теста должны проходить
