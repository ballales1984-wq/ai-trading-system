"""
Risk Metrics Module for Crypto Trading System
==============================================
Professional risk analysis including:
- Sharpe Ratio
- Max Drawdown
- Sortino Ratio
- Calmar Ratio
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Information Ratio

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio - measures risk-adjusted returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods per year (252 for daily)
    
    Returns:
    --------
    float : Sharpe Ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio - uses downside deviation instead of total std.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float : Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate Maximum Drawdown - largest peak-to-trough decline.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of equity values
    
    Returns:
    --------
    Tuple[float, int, int] : (max_drawdown, peak_index, trough_index)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0
    
    # Calculate running maximum (peak)
    peak = equity_curve.expanding(min_periods=1).max()
    
    # Calculate drawdown
    drawdown = (equity_curve - peak) / peak
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find corresponding peak (last index before trough where equity = peak)
    peak_at_trough = peak.loc[trough_idx]
    peak_idx = equity_curve[:trough_idx].idxmax()
    
    return max_dd, peak_idx, trough_idx


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio - return / max drawdown (annualized).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    equity_curve : pd.Series
        Series of equity values
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float : Calmar Ratio
    """
    max_dd, _, _ = max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    # Annualized return
    annual_return = returns.mean() * periods_per_year
    
    return annual_return / abs(max_dd)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) - maximum expected loss at given confidence.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    confidence : float
        Confidence level (0.95 = 95%)
    
    Returns:
    --------
    float : VaR (positive value representing loss)
    """
    if len(returns) == 0:
        return 0.0
    
    return -np.percentile(returns, (1 - confidence) * 100)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    confidence : float
        Confidence level
    
    Returns:
    --------
    float : CVaR (average loss beyond VaR)
    """
    if len(returns) == 0:
        return 0.0
    
    var = value_at_risk(returns, confidence)
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -tail_returns.mean()


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio - active return / tracking error.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float : Information Ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")
    
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    return (active_returns.mean() * periods_per_year) / tracking_error


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate Beta - systematic risk relative to benchmark.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
    
    Returns:
    --------
    float : Beta coefficient
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")
    
    covariance = returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    
    if benchmark_variance == 0:
        return 0.0
    
    return covariance / benchmark_variance


def alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Alpha - excess return beyond benchmark.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
    risk_free : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float : Alpha (annualized)
    """
    b = beta(returns, benchmark_returns)
    
    strategy_return = returns.mean() * periods_per_year
    benchmark_return = benchmark_returns.mean() * periods_per_year
    
    return strategy_return - (risk_free + b * (benchmark_return - risk_free))


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    float : Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_all_risk_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate all risk metrics at once.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    equity_curve : pd.Series
        Equity curve series
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    risk_free : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods per year
    
    Returns:
    --------
    Dict[str, float] : All calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0
    metrics['annual_return'] = returns.mean() * periods_per_year
    metrics['volatility'] = volatility(returns, periods_per_year)
    
    # Risk-adjusted ratios
    metrics['sharpe_ratio'] = sharpe_ratio(returns, risk_free, periods_per_year)
    metrics['sortino_ratio'] = sortino_ratio(returns, risk_free, periods_per_year)
    metrics['calmar_ratio'] = calmar_ratio(returns, equity_curve, periods_per_year)
    
    # Drawdown metrics
    max_dd, peak_idx, trough_idx = max_drawdown(equity_curve)
    metrics['max_drawdown'] = max_dd
    metrics['max_drawdown_pct'] = max_dd * 100
    
    # VaR metrics
    metrics['var_95'] = value_at_risk(returns, 0.95)
    metrics['cvar_95'] = conditional_var(returns, 0.95)
    metrics['var_99'] = value_at_risk(returns, 0.99)
    metrics['cvar_99'] = conditional_var(returns, 0.99)
    
    # Benchmark-relative metrics
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        metrics['beta'] = beta(returns, benchmark_returns)
        metrics['alpha'] = alpha(returns, benchmark_returns, risk_free, periods_per_year)
        metrics['information_ratio'] = information_ratio(returns, benchmark_returns, periods_per_year)
    
    # Win rate
    metrics['win_rate'] = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return metrics


def rolling_sharpe(returns: pd.Series, window: int = 30, periods_per_year: int = 252) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    window : int
        Rolling window size
    periods_per_year : int
        Periods per year for annualization
    
    Returns:
    --------
    pd.Series : Rolling Sharpe ratio
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    return np.sqrt(periods_per_year) * rolling_mean / rolling_std


def rolling_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate rolling drawdown.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of equity values
    
    Returns:
    --------
    pd.Series : Rolling drawdown
    """
    peak = equity_curve.expanding(min_periods=1).max()
    return (equity_curve - peak) / peak
