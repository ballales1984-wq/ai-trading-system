import pandas as pd
import numpy as np

def calculate_risk_adjusted_returns(returns, risk_free_rate=0.02):
    """
    Calculate risk-adjusted returns (Sharpe ratio).
    """
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

def generate_performance_report(returns, equity_curve):
    """
    Generate performance report.
    """
    ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    sharpe = calculate_risk_adjusted_returns(returns)
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    return {
        'annual_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

def calculate_all_performance_metrics(returns, equity_curve, benchmark_returns=None, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate all professional performance metrics.
    """
    # Basic metrics
    ann_ret = annual_return(returns, periods_per_year)
    ann_vol = annual_volatility(returns, periods_per_year)
    
    # Placeholder for other metrics
    sharpe = calculate_risk_adjusted_returns(returns, risk_free_rate)
    
    return {
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sharpe,  # placeholder
        'calmar_ratio': sharpe,   # placeholder
        'max_drawdown': abs(max_dd)
    }

def annual_return(returns, periods_per_year=252):
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

def annual_volatility(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)
