"""
Backtesting Engine for Crypto Trading System
=============================================
Professional backtesting with realistic simulation including:
- Position management (long/short/flat)
- Transaction costs simulation
- Equity curve calculation
- Performance metrics

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtest results"""
    equity_curve: pd.Series
    strategy_returns: pd.Series
    trades: List[Dict]
    metrics: Dict[str, float]
    positions: pd.Series


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 1.0
) -> BacktestResult:
    """
    Run comprehensive backtest on price data with trading signals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data (must contain 'close' column)
    signals : pd.Series
        Series of trading signals (BUY/SELL/HOLD)
    initial_capital : float
        Starting capital for backtest
    commission : float
        Commission rate per trade (0.001 = 0.1%)
    slippage : float
        Slippage rate per trade
    position_size : float
        Position size as fraction of capital (0.0 to 1.0)
    
    Returns:
    --------
    BacktestResult : Object containing all backtest results
    """
    
    # Validate inputs
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    if len(df) != len(signals):
        raise ValueError("DataFrame and signals must have same length")
    
    # Ensure signals is a Series with proper index
    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=df.index)
    
    # Calculate returns
    returns = df['close'].pct_change().fillna(0)
    
    # Initialize tracking variables
    capital = initial_capital
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0
    trades = []
    positions_list = []
    strategy_returns = []
    equity_values = [initial_capital]
    
    for i in range(1, len(signals)):
        current_price = df['close'].iloc[i]
        signal = signals.iloc[i]
        prev_signal = signals.iloc[i-1] if i > 0 else 'HOLD'
        
        # Track position changes for trade logging
        position_changed = False
        
        # Entry logic
        if signal == "BUY" and position == 0:
            # Open long position
            position = 1
            entry_price = current_price * (1 + slippage)
            capital -= entry_price * position_size * commission
            position_changed = True
            
        elif signal == "SELL" and position == 0:
            # Open short position
            position = -1
            entry_price = current_price * (1 - slippage)
            position_changed = True
            
        # Exit logic
        elif signal == "SELL" and position == 1:
            # Close long position
            exit_price = current_price * (1 - slippage)
            pnl = (exit_price - entry_price) * position_size
            capital += exit_price * position_size * (1 - commission)
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'LONG',
                'pnl': pnl,
                'return': pnl / (entry_price * position_size)
            })
            position = 0
            position_changed = True
            
        elif signal == "BUY" and position == -1:
            # Close short position
            exit_price = current_price * (1 + slippage)
            pnl = (entry_price - exit_price) * position_size
            capital += exit_price * position_size * (1 - commission)
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'SHORT',
                'pnl': pnl,
                'return': pnl / (entry_price * position_size)
            })
            position = 0
            position_changed = True
        
        # Calculate strategy return for this period
        if position != 0:
            period_return = returns.iloc[i] * position
        else:
            period_return = 0
        
        strategy_returns.append(period_return)
        
        # Update equity
        if position == 1:
            current_equity = capital + current_price * position_size
        elif position == -1:
            current_equity = capital + (entry_price - current_price) * position_size
        else:
            current_equity = capital
        
        equity_values.append(current_equity)
        positions_list.append(position)
    
    # Create output series
    equity_curve = pd.Series(equity_values, index=df.index[:len(equity_values)])
    strategy_returns = pd.Series(strategy_returns, index=df.index[1:len(strategy_returns)+1])
    positions = pd.Series(positions_list, index=df.index[1:len(positions_list)+1])
    
    # Calculate metrics
    total_return = (equity_values[-1] - initial_capital) / initial_capital
    num_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'final_capital': equity_values[-1],
        'num_trades': num_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'total_pnl': sum([t['pnl'] for t in trades])
    }
    
    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        trades=trades,
        metrics=metrics,
        positions=positions
    )


def run_multi_strategy_backtest(
    df: pd.DataFrame,
    strategies: Dict[str, pd.Series],
    initial_capital: float = 10000.0
) -> Dict[str, BacktestResult]:
    """
    Run backtest for multiple strategies and compare results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    strategies : Dict[str, pd.Series]
        Dictionary of strategy name -> signals
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    Dict[str, BacktestResult] : Results for each strategy
    """
    results = {}
    
    for name, signals in strategies.items():
        results[name] = run_backtest(df, signals, initial_capital)
    
    return results


def optimize_parameters(
    df: pd.DataFrame,
    param_grid: Dict[str, List],
    base_signal_func,
    metric: str = 'total_return'
) -> Dict:
    """
    Simple grid search for parameter optimization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    param_grid : Dict[str, List]
        Parameter combinations to test
    base_signal_func : callable
        Function that generates signals with given parameters
    metric : str
        Metric to optimize
    
    Returns:
    --------
    Dict : Best parameters and results
    """
    from itertools import product
    
    best_metric = float('-inf')
    best_params = None
    all_results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        
        # Generate signals with these parameters
        signals = base_signal_func(df, **params)
        
        # Run backtest
        result = run_backtest(df, signals)
        
        current_metric = result.metrics.get(metric, 0)
        all_results.append({
            'params': params,
            'metric': current_metric,
            'metrics': result.metrics
        })
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_params = params
    
    return {
        'best_params': best_params,
        'best_metric': best_metric,
        'all_results': all_results
    }
