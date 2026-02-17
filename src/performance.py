"""
Performance Metrics for Quantitative Investment
==============================================
Professional investment metrics in language of hedge funds:
- Annual Return
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio
- Alpha/Beta
- Value at Risk
- Maximum Drawdown

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Professional performance metrics container"""
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    exposure_pct: float


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Periods per year (252 for daily)
    
    Returns:
    --------
    float : Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / periods_per_year
    
    if n_years <= 0:
        return 0.0
    
    return (1 + total_return) ** (1 / n_years) - 1


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Periods per year
    
    Returns:
    --------
    float : Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (downside risk-adjusted return).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    target_return : float
        Target/minimum acceptable return
    periods_per_year : int
        Periods per year
    
    Returns:
    --------
    float : Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess = returns - target_return / periods_per_year
    downside = returns[returns < target_return]
    
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    
    downside_std = downside.std() * np.sqrt(periods_per_year)
    
    ann_return = annual_return(returns, periods_per_year)
    
    return ann_return / downside_std


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio (return / max drawdown).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    equity_curve : pd.Series
        Equity curve
    periods_per_year : int
        Periods per year
    
    Returns:
    --------
    float : Calmar Ratio
    """
    ann_return = annual_return(returns, periods_per_year)
    
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    if max_dd == 0:
        return 0.0
    
    return ann_return / abs(max_dd)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio (active return / tracking error).
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int
        Periods per year
    
    Returns:
    --------
    float : Information Ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns must have same length as benchmark")
    
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    active_return = annual_return(active_returns, periods_per_year)
    
    return active_return / tracking_error


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio (probability-weighted ratio of gains vs losses).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    threshold : float
        Threshold return
    
    Returns:
    --------
    float : Omega Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    return gains / losses


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """
    Calculate Gain to Pain Ratio (sum of returns / sum of absolute losses).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    
    Returns:
    --------
    float : Gain to Pain Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    total_return = returns.sum()
    pain = abs(returns[returns < 0]).sum()
    
    if pain == 0:
        return float('inf') if total_return > 0 else 0.0
    
    return total_return / pain


def skewness(returns: pd.Series) -> float:
    """
    Calculate skewness of returns distribution.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    
    Returns:
    --------
    float : Skewness
    """
    if len(returns) < 3:
        return 0.0
    
    return returns.skew()


def kurtosis(returns: pd.Series) -> float:
    """
    Calculate kurtosis of returns distribution.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    
    Returns:
    --------
    float : Excess kurtosis
    """
    if len(returns) < 4:
        return 0.0
    
    return returns.kurtosis()


def tail_ratio(returns: pd.Series) -> float:
    """
    Calculate Tail Ratio (95th percentile / 5th percentile).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    
    Returns:
    --------
    float : Tail Ratio
    """
    if len(returns) == 0:
        return 0.0
    
    percentile_95 = np.percentile(returns, 95)
    percentile_5 = np.percentile(returns, 5)
    
    if abs(percentile_5) == 0:
        return 0.0
    
    return abs(percentile_95 / percentile_5)


def calculate_all_performance_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate all professional performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    equity_curve : pd.Series
        Equity curve
    benchmark_returns : pd.Series, optional
        Benchmark returns
    risk_free_rate : float
        Risk-free rate (annualized)
    periods_per_year : int
        Periods per year
    
    Returns:
    --------
    PerformanceMetrics : All metrics
    """
    # Basic metrics
    ann_ret = annual_return(returns, periods_per_year)
    ann_vol = annual_volatility(returns, periods_per_year)
    
    # Risk metrics
    from src.risk import max_drawdown as mdd
    max_dd, _, _ = mdd(equity_curve)
    
    # Calculate downside deviation for Sortino
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    
    # Sharpe Ratio
    if ann_vol > 0:
        sharpe = (ann_ret - risk_free_rate) / ann_vol
    else:
        sharpe = 0.0
    
    # Sortino Ratio
    if downside_std > 0:
        sortino = ann_ret / downside_std
    else:
        sortino = 0.0
    
    # Calmar Ratio
    if max_dd != 0:
        calmar = ann_ret / abs(max_dd)
    else:
        calmar = 0.0
    
    # VaR and CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # Trade statistics
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    
    win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else float('inf')
    avg_trade = returns.mean()
    
    # Exposure (percentage of time in market)
    exposure = (returns != 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return PerformanceMetrics(
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd * 100,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        num_trades=len(returns),
        exposure_pct=exposure
    )


def generate_performance_report(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_name: str = "Strategy"
) -> str:
    """
    Generate formatted performance report.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    equity_curve : pd.Series
        Equity curve
    benchmark_returns : pd.Series, optional
        Benchmark returns
    strategy_name : str
        Strategy name
    
    Returns:
    --------
    str : Formatted report
    """
    metrics = calculate_all_performance_metrics(returns, equity_curve, benchmark_returns)
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              {strategy_name:^50}        ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURN METRICS                                                   ║
║  ─────────────────────────────────────────────────────────────    ║
║  Annual Return:          {metrics.annual_return*100:+10.2f}%                            ║
║  Annual Volatility:      {metrics.annual_volatility*100:10.2f}%                            ║
║  Total Trades:           {metrics.num_trades:10d}                            ║
║  Win Rate:               {metrics.win_rate*100:10.2f}%                            ║
║                                                                      ║
║  RISK-ADJUSTED METRICS                                              ║
║  ─────────────────────────────────────────────────────────────    ║
║  Sharpe Ratio:           {metrics.sharpe_ratio:10.2f}                            ║
║  Sortino Ratio:         {metrics.sortino_ratio:10.2f}                            ║
║  Calmar Ratio:          {metrics.calmar_ratio:10.2f}                            ║
║                                                                      ║
║  RISK METRICS                                                       ║
║  ─────────────────────────────────────────────────────────────    ║
║  Max Drawdown:          {metrics.max_drawdown_pct:+10.2f}%                            ║
║  VaR (95%):             {metrics.var_95*100:+10.2f}%                            ║
║  CVaR (95%):            {metrics.cvar_95*100:+10.2f}%                            ║
║  Exposure:               {metrics.exposure_pct*100:10.2f}%                            ║
║                                                                      ║
║  TRADE STATISTICS                                                   ║
║  ─────────────────────────────────────────────────────────────    ║
║  Profit Factor:         {metrics.profit_factor:10.2f}                            ║
║  Average Trade:         {metrics.avg_trade*100:+10.4f}%                            ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return report


def compare_strategies(
    results: Dict[str, Tuple[pd.Series, pd.Series]]
) -> pd.DataFrame:
    """
    Compare multiple strategies.
    
    Parameters:
    -----------
    results : Dict[str, Tuple[pd.Series, pd.Series]]
        Strategy name -> (returns, equity_curve)
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    comparison = []
    
    for name, (returns, equity) in results.items():
        metrics = calculate_all_performance_metrics(returns, equity)
        
        comparison.append({
            'Strategy': name,
            'Ann. Return': f"{metrics.annual_return*100:.2f}%",
            'Ann. Volatility': f"{metrics.annual_volatility*100:.2f}%",
            'Sharpe': f"{metrics.sharpe_ratio:.2f}",
            'Sortino': f"{metrics.sortino_ratio:.2f}",
            'Calmar': f"{metrics.calmar_ratio:.2f}",
            'Max DD': f"{metrics.max_drawdown_pct:.2f}%",
            'Win Rate': f"{metrics.win_rate*100:.2f}%"
        })
    
    return pd.DataFrame(comparison)
