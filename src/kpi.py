"""
KPI Avanzati - Metriche Professionali
====================================
Calcola Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, etc.

Metriche usate da hedge fund e CTA professionali.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def max_drawdown(equity: pd.Series) -> float:
    """
    Calcola il maximum drawdown.
    
    Args:
        equity: Serie temporale dell'equity
        
    Returns:
        Max drawdown (negativo, es: -0.25 = -25%)
    """
    if len(equity) == 0:
        return 0.0
    
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()


def max_drawdown_duration(equity: pd.Series) -> int:
    """
    Calcola la durata massima del drawdown in giorni.
    
    Args:
        equity: Serie temporale dell'equity
        
    Returns:
        Durata massima in periodi
    """
    if len(equity) == 0:
        return 0
    
    peak = equity.cummax()
    in_drawdown = equity < peak
    
    # Trova i periodi in drawdown
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Calcola lo Sharpe Ratio.
    
    Args:
        returns: Serie dei rendimenti
        risk_free_rate: Tasso risk-free annuale
        annualization_factor: Fattore di annualizzazione (252 per daily)
        
    Returns:
        Sharpe Ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    return excess_returns.mean() / returns.std() * np.sqrt(annualization_factor)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Calcola lo Sortino Ratio (usa solo downside deviation).
    
    Args:
        returns: Serie dei rendimenti
        risk_free_rate: Tasso risk-free annuale
        annualization_factor: Fattore di annualizzazione
        
    Returns:
        Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)


def calmar_ratio(
    equity: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """
    Calcola il Calmar Ratio (CAGR / Max Drawdown).
    
    Args:
        equity: Serie dell'equity
        annualization_factor: Fattore di annualizzazione
        
    Returns:
        Calmar Ratio
    """
    if len(equity) < 2:
        return 0.0
    
    # CAGR
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_periods = len(equity)
    years = n_periods / annualization_factor
    cagr = (total_return ** (1 / years)) - 1 if years > 0 else 0
    
    # Max Drawdown
    mdd = abs(max_drawdown(equity))
    
    if mdd == 0:
        return 0.0
    
    return cagr / mdd


def cagr(
    equity: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """
    Calcola il Compound Annual Growth Rate.
    
    Args:
        equity: Serie dell'equity
        annualization_factor: Fattore di annualizzazione
        
    Returns:
        CAGR (annual return)
    """
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_periods = len(equity)
    years = n_periods / annualization_factor
    
    return (total_return ** (1 / years)) - 1 if years > 0 else 0


def volatility(
    returns: pd.Series,
    annualization_factor: int = 252,
) -> float:
    """
    Calcola la volatilità annualizzata.
    
    Args:
        returns: Serie dei rendimenti
        annualization_factor: Fattore di annualizzazione
        
    Returns:
        Volatilità annualizzata
    """
    if len(returns) == 0:
        return 0.0
    
    return returns.std() * np.sqrt(annualization_factor)


def win_rate(trades: pd.DataFrame) -> float:
    """
    Calcola il win rate.
    
    Args:
        trades: DataFrame con colonne 'pnl'
        
    Returns:
        Win rate (0-1)
    """
    if len(trades) == 0:
        return 0.0
    
    wins = trades[trades["pnl"] > 0]
    return len(wins) / len(trades)


def profit_factor(trades: pd.DataFrame) -> float:
    """
    Calcola il Profit Factor.
    
    Args:
        trades: DataFrame con colonne 'pnl'
        
    Returns:
        Profit Factor (gross profits / gross losses)
    """
    if len(trades) == 0:
        return 0.0
    
    gross_profits = trades[trades["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(trades[trades["pnl"] < 0]["pnl"].sum())
    
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    
    return gross_profits / gross_losses


def expectancy(trades: pd.DataFrame) -> float:
    """
    Calcola l'expectancy per trade.
    
    Args:
        trades: DataFrame con colonne 'pnl'
        
    Returns:
        Expectancy media per trade
    """
    if len(trades) == 0:
        return 0.0
    
    return trades["pnl"].mean()


def average_win(trades: pd.DataFrame) -> float:
    """Media delle vincite."""
    if len(trades) == 0:
        return 0.0
    wins = trades[trades["pnl"] > 0]
    return wins["pnl"].mean() if len(wins) > 0 else 0.0


def average_loss(trades: pd.DataFrame) -> float:
    """Media delle perdite (negativo)."""
    if len(trades) == 0:
        return 0.0
    losses = trades[trades["pnl"] < 0]
    return losses["pnl"].mean() if len(losses) > 0 else 0.0


def risk_reward_ratio(trades: pd.DataFrame) -> float:
    """Rapporto risk/reward medio."""
    avg_win = average_win(trades)
    avg_loss = abs(average_loss(trades))
    
    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0
    
    return avg_win / avg_loss


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calcola il Value at Risk (VaR).
    
    Args:
        returns: Serie dei rendimenti
        confidence: Livello di confidenza (0.95 = 95%)
        
    Returns:
        VaR (negativo)
    """
    if len(returns) == 0:
        return 0.0
    
    return returns.quantile(1 - confidence)


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calcola il Conditional VaR (Expected Shortfall).
    
    Args:
        returns: Serie dei rendimenti
        confidence: Livello di confidenza
        
    Returns:
        CVaR (negativo)
    """
    if len(returns) == 0:
        return 0.0
    
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def compute_all_kpi(
    equity: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> Dict[str, float]:
    """
    Calcola tutti i KPI principali.
    
    Args:
        equity: Serie dell'equity
        trades: DataFrame dei trades (deve avere colonna 'pnl')
        risk_free_rate: Tasso risk-free annuale
        annualization_factor: Fattore di annualizzazione
        
    Returns:
        Dizionario con tutti i KPI
    """
    # Prepara dati
    if len(equity) < 2:
        return {
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Calmar Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Max Drawdown Duration": 0,
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Expectancy": 0.0,
            "Risk/Reward": 0.0,
            "VaR (95%)": 0.0,
            "CVaR (95%)": 0.0,
        }
    
    returns = equity.pct_change().dropna()
    
    # Calcola KPI
    kpi = {
        "CAGR": cagr(equity, annualization_factor),
        "Volatility": volatility(returns, annualization_factor),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate, annualization_factor),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate, annualization_factor),
        "Calmar Ratio": calmar_ratio(equity, annualization_factor),
        "Max Drawdown": max_drawdown(equity),
        "Max Drawdown Duration": max_drawdown_duration(equity),
        "VaR (95%)": value_at_risk(returns, 0.95),
        "CVaR (95%)": conditional_var(returns, 0.95),
    }
    
    # KPI basati sui trades
    if len(trades) > 0:
        kpi.update({
            "Win Rate": win_rate(trades),
            "Profit Factor": profit_factor(trades),
            "Expectancy": expectancy(trades),
            "Average Win": average_win(trades),
            "Average Loss": average_loss(trades),
            "Risk/Reward": risk_reward_ratio(trades),
        })
    else:
        kpi.update({
            "Win Rate": 0.0,
            "Profit Factor": 0.0,
            "Expectancy": 0.0,
            "Average Win": 0.0,
            "Average Loss": 0.0,
            "Risk/Reward": 0.0,
        })
    
    return kpi


def format_kpi(kpi: Dict[str, float]) -> Dict[str, str]:
    """
    Formatta i KPI per visualizzazione.
    
    Args:
        kpi: Dizionario KPI
        
    Returns:
        Dizionario con KPI formattati come stringhe
    """
    formatted = {}
    
    for key, value in kpi.items():
        if "Drawdown" in key and "Duration" not in key:
            formatted[key] = f"{value * 100:.2f}%"
        elif "Duration" in key:
            formatted[key] = f"{int(value)} periods"
        elif "Ratio" in key or "Rate" in key:
            formatted[key] = f"{value:.2f}"
        elif "Factor" in key:
            if value == float('inf'):
                formatted[key] = "∞"
            else:
                formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = f"{value * 100:.2f}%" if abs(value) < 10 else f"{value:.2f}"
    
    return formatted


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    import pandas as pd
    
    # Simula equity curve
    equity = pd.Series([10000, 10200, 10500, 10300, 10800, 11000, 10700, 11500])
    
    # Simula trades
    trades = pd.DataFrame({
        "pnl": [100, -50, 200, -30, 150, -80, 250, 100]
    })
    
    # Calcola KPI
    kpi = compute_all_kpi(equity, trades)
    
    print("📊 KPI Calcolati:")
    for key, value in kpi.items():
        print(f"  {key}: {value:.4f}")
