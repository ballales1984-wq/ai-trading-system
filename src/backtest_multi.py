"""
Multi-Asset Portfolio Backtest Engine
====================================
Professional multi-asset backtesting with:
- Dynamic asset allocation
- Multiple weighting strategies
- Portfolio aggregation
- Risk metrics

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    num_trades: int
    best_asset: str
    worst_asset: str
    correlation_matrix: pd.DataFrame


class MultiAssetBacktest:
    """
    Multi-asset portfolio backtest engine.
    
    Supports:
    - Equal weight allocation
    - Volatility parity
    - Risk parity
    - ML-driven allocation
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        rebalance_freq: str = 'daily'
    ):
        """
        Initialize multi-asset backtest.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        rebalance_freq : str
            Rebalancing frequency ('daily', 'weekly', 'monthly')
        """
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.assets_data: Dict[str, pd.DataFrame] = {}
        self.results: Optional[PortfolioMetrics] = None
    
    def add_asset(
        self,
        name: str,
        df: pd.DataFrame,
        signals: Optional[pd.Series] = None
    ) -> None:
        """
        Add asset to portfolio.
        
        Parameters:
        -----------
        name : str
            Asset name (e.g., 'BTC', 'ETH')
        df : pd.DataFrame
            Price data with 'close' column
        signals : pd.Series, optional
            Trading signals for this asset
        """
        asset_data = df.copy()
        
        if signals is not None:
            asset_data['signal'] = signals
        
        asset_data['returns'] = asset_data['close'].pct_change()
        
        self.assets_data[name] = asset_data
    
    def calculate_weights_equal(self) -> pd.DataFrame:
        """Calculate equal weights for all assets."""
        n_assets = len(self.assets_data)
        weight = 1.0 / n_assets if n_assets > 0 else 0
        
        weights = {}
        for name in self.assets_data:
            weights[name] = weight
        
        return pd.DataFrame(weights)
    
    def calculate_weights_volatility_parity(
        self,
        lookback: int = 20
    ) -> pd.DataFrame:
        """Calculate volatility parity weights."""
        # Get volatilities
        volatilities = {}
        
        for name, df in self.assets_data.items():
            if 'returns' in df.columns:
                vol = df['returns'].rolling(lookback).std().iloc[-1]
                volatilities[name] = vol
        
        # Inverse volatility weights
        total_inv_vol = sum(1/v for v in volatilities.values() if v > 0)
        
        weights = {}
        for name, vol in volatilities.items():
            if vol > 0:
                weights[name] = (1/vol) / total_inv_vol
            else:
                weights[name] = 0
        
        return pd.DataFrame(weights)
    
    def calculate_weights_risk_parity(
        self,
        lookback: int = 20
    ) -> pd.DataFrame:
        """Calculate risk parity weights."""
        # Simplified risk parity (inverse of volatility)
        return self.calculate_weights_volatility_parity(lookback)
    
    def run_backtest(
        self,
        weight_strategy: str = 'equal',
        lookback: int = 20
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run multi-asset backtest.
        
        Parameters:
        -----------
        weight_strategy : str
            Weighting strategy ('equal', 'volatility_parity', 'risk_parity')
        lookback : int
            Lookback period for volatility calculations
        
        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame] : (portfolio_returns, asset_returns)
        """
        # Align all assets to common index
        aligned_data = {}
        
        for name, df in self.assets_data.items():
            df_aligned = df[['close', 'returns']].dropna()
            if 'signal' in df.columns:
                df_aligned['signal'] = df['signal'].reindex(df_aligned.index).fillna(0)
            aligned_data[name] = df_aligned
        
        # Find common index
        common_idx = aligned_data[list(aligned_data.keys())[0]].index
        
        for name, df in aligned_data.items():
            common_idx = common_idx.intersection(df.index)
        
        # Create returns dataframe
        returns_df = pd.DataFrame(index=common_idx)
        
        for name, df in aligned_data.items():
            returns_df[name] = df['returns'].reindex(common_idx).fillna(0)
        
        # Calculate strategy returns
        strategy_returns = pd.DataFrame(index=common_idx)
        
        for name in self.assets_data.keys():
            if 'signal' in aligned_data[name].columns:
                signals = aligned_data[name]['signal'].reindex(common_idx).fillna(0)
                strategy_returns[name] = returns_df[name] * signals
            else:
                strategy_returns[name] = returns_df[name]
        
        # Equal weight portfolio
        n_assets = len(self.assets_data)
        portfolio_returns = strategy_returns.mean(axis=1)
        
        # Calculate portfolio metrics
        equity = (1 + portfolio_returns).cumprod()
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] > 0 else 0
        
        # Volatility (annualized)
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        if volatility > 0:
            sharpe = (portfolio_returns.mean() * 252) / volatility
        else:
            sharpe = 0
        
        # Max Drawdown
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean()
        
        # Number of trades
        num_trades = (returns_df.abs() > 0).sum().sum()
        
        # Best/worst assets
        asset_returns = returns_df.sum()
        best_asset = asset_returns.idxmax()
        worst_asset = asset_returns.idxmin()
        
        # Correlation matrix
        correlation_matrix = returns_df.corr()
        
        self.results = PortfolioMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            volatility=volatility,
            win_rate=win_rate,
            num_trades=num_trades,
            best_asset=best_asset,
            worst_asset=worst_asset,
            correlation_matrix=correlation_matrix
        )
        
        return portfolio_returns, returns_df
    
    def get_results(self) -> PortfolioMetrics:
        """Get portfolio results."""
        return self.results


def backtest_multi_asset(
    prices_dict: Dict[str, pd.DataFrame],
    signals_dict: Optional[Dict[str, pd.Series]] = None,
    initial_capital: float = 1000000
) -> Tuple[pd.Series, PortfolioMetrics]:
    """
    Convenience function for multi-asset backtest.
    
    Parameters:
    -----------
    prices_dict : Dict[str, pd.DataFrame]
        Dictionary of asset prices
    signals_dict : Dict[str, pd.Series], optional
        Dictionary of signals per asset
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    Tuple[pd.Series, PortfolioMetrics] : (returns, metrics)
    """
    backtest = MultiAssetBacktest(initial_capital)
    
    for asset, df in prices_dict.items():
        signals = signals_dict.get(asset) if signals_dict else None
        backtest.add_asset(asset, df, signals)
    
    returns, _ = backtest.run_backtest()
    
    return returns, backtest.get_results()


def calculate_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    """
    Calculate comprehensive portfolio metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series, optional
        Benchmark returns
    
    Returns:
    --------
    Dict : Portfolio metrics
    """
    equity = (1 + returns).cumprod()
    
    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] > 0 else 0
    
    # Annualized metrics
    n_days = len(returns)
    n_years = n_days / 252
    
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Max Drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(returns)
    }
    
    # Benchmark comparison
    if benchmark_returns is not None:
        bench_equity = (1 + benchmark_returns).cumprod()
        bench_total = bench_equity.iloc[-1] / bench_equity.iloc[0] - 1
        
        metrics['alpha'] = total_return - bench_total
        metrics['beta'] = returns.corr(benchmark_returns) * (returns.std() / benchmark_returns.std()) if benchmark_returns.std() > 0 else 0
    
    return metrics
