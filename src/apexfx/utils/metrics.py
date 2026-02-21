"""Performance metric calculations for portfolio analysis."""

from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods
    if len(excess) < 2:
        return 0.0
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Annualized Sortino ratio (only penalizes downside deviation)."""
    excess = returns - risk_free_rate / periods
    if len(excess) < 2:
        return 0.0
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return float("inf")
    return float(np.mean(excess) / downside_std * np.sqrt(periods))


def calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    ann_return = np.mean(returns) * periods
    dd = max_drawdown(returns)
    if dd < 1e-10:
        return float("inf")
    return float(ann_return / dd)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from peak as a fraction."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def max_drawdown_duration(returns: np.ndarray) -> int:
    """Maximum drawdown duration in periods."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    in_drawdown = equity < peak

    max_duration = 0
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def win_rate(returns: np.ndarray) -> float:
    """Fraction of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0:
        return float("inf")
    gross_loss = np.sum(np.abs(losses))
    if gross_loss < 1e-10:
        return float("inf")
    return float(np.sum(gains) / gross_loss)


def expectancy(returns: np.ndarray) -> float:
    """Average expected return per trade."""
    if len(returns) == 0:
        return 0.0
    wr = win_rate(returns)
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
    avg_loss = np.mean(np.abs(returns[returns < 0])) if np.any(returns < 0) else 0.0
    return float(wr * avg_win - (1 - wr) * avg_loss)


def annualized_return(returns: np.ndarray, periods: int = 252) -> float:
    """Annualized return."""
    if len(returns) == 0:
        return 0.0
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / periods
    if n_years < 1e-10:
        return 0.0
    return float((1 + total_return) ** (1 / n_years) - 1)


def annualized_volatility(returns: np.ndarray, periods: int = 252) -> float:
    """Annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods))


def compute_all_metrics(returns: np.ndarray, periods: int = 252) -> dict[str, float]:
    """Compute all performance metrics at once."""
    return {
        "sharpe_ratio": sharpe_ratio(returns, periods=periods),
        "sortino_ratio": sortino_ratio(returns, periods=periods),
        "calmar_ratio": calmar_ratio(returns, periods=periods),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_duration": float(max_drawdown_duration(returns)),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "annualized_return": annualized_return(returns, periods=periods),
        "annualized_volatility": annualized_volatility(returns, periods=periods),
        "total_return": float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0,
        "n_trades": len(returns),
    }
