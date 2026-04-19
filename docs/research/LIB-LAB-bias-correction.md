# LIB/LAB — Look-ahead & Look-back Bias Correction

**Paper:** "The Corporate Bond Factor Replication Crisis"

## Core Problem

Два типа bias убивают реальную P&L моделей, обученных на исторических данных:

### Look-ahead Bias (LAB)
Используем информацию из будущего при расчёте фичей:
- Winsorization/clipping по всей выборке (включая будущие данные)
- Z-score нормализация с mean/std по всему датасету
- Feature selection на основе корреляции с будущими returns

### Look-back Bias (LIB)
Сигнал рассчитан на конце бара t, но return берётся с начала бара t (а не t+1):
- Signal time: close of bar t
- Return должен быть: open(t+1) → close(t+1), NOT close(t) → close(t+1)

## Gap Procedure (Fix для LIB)

```
signal[t] = computed at close of bar t
entry_price = open[t+1]           # реальная точка входа
return[t] = (close[t+1] - open[t+1]) / open[t+1]
```

## Ex-ante Filtering (Fix для LAB)

Все трансформации только по **прошлым** данным:

```python
# WRONG
df['feature_clipped'] = df['feature'].clip(q01, q99)  # q01/q99 по всему df

# RIGHT (expanding window)
for t in range(len(df)):
    history = df.iloc[:t+1]
    q01 = history['feature'].quantile(0.01)
    q99 = history['feature'].quantile(0.99)
    df.loc[t, 'feature_clipped'] = np.clip(df.loc[t, 'feature'], q01, q99)
```

## Адаптация для Forex

- Gap-procedure критична: forex имеет gaps на выходных
- Winsorization должна быть expanding-window
- Z-score: rolling mean/std (не global!)
- Feature pipeline должен пересчитываться по expanding window

## Реализация

- Основной класс: `BiasFreePipeline` в `src/apexfx/data/pipeline.py`
- Этап 1 текущей разработки

## Impact

По paper, коррекция LIB/LAB снижает Sharpe factor strategies на 30-50%.
Это значит наши бэктесты без коррекции **переоценены**.

#research #bias #data-quality #critical
