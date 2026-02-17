"""
AI Trading System - Professional Crypto Trading Platform
=========================================================

Modular structure:
- data_loader: Data loading from various sources
- indicators: Technical analysis indicators
- signal_engine: Trading signal generation
- backtest: Backtesting engine
- risk: Risk metrics and analysis
- utils: Utilities and helpers

Author: AI Trading System
Version: 1.0.0
"""

from .data_loader import DataLoader, load_crypto_data
from .indicators import (
    rsi, ema, sma, macd, bollinger_bands, vwap, atr,
    stochastic, adx, obv, calculate_all_indicators
)
from .signal_engine import (
    generate_signal, generate_composite_signal,
    generate_ema_crossover_signal, detect_trend
)
from .backtest import run_backtest, run_multi_strategy_backtest
from .risk import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    value_at_risk, calculate_all_risk_metrics
)
from .utils import (
    walk_forward_split, rolling_window_split, normalize_features,
    add_lag_features, PerformanceTracker
)

__version__ = '1.0.0'
__all__ = [
    'DataLoader',
    'load_crypto_data',
    'rsi', 'ema', 'sma', 'macd', 'bollinger_bands', 'vwap', 'atr',
    'stochastic', 'adx', 'obv', 'calculate_all_indicators',
    'generate_signal', 'generate_composite_signal',
    'generate_ema_crossover_signal', 'detect_trend',
    'run_backtest', 'run_multi_strategy_backtest',
    'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio',
    'value_at_risk', 'calculate_all_risk_metrics',
    'walk_forward_split', 'rolling_window_split', 'normalize_features',
    'add_lag_features', 'PerformanceTracker'
]
