"""
Pytest-benchmark tests for AI Trading System performance.

Usage:
    pytest tests/test_benchmark.py --benchmark-only
    pytest tests/test_benchmark.py --benchmark-autosave
    pytest tests/test_benchmark.py --benchmark-compare
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import time


# ============================================================================
# Risk Engine Benchmarks
# ============================================================================

def risk_engine_position_sizing(benchmark, capital=100000):
    """Benchmark risk-based position sizing."""
    from app.risk.risk_engine import RiskEngine
    
    engine = RiskEngine()
    
    # Benchmark the calculation
    result = benchmark(
        engine.calculate_position_size,
        action='buy',
        price=50000.0,
        confidence=0.8,
        capital=capital
    )
    return result


def risk_engine_var_calculation(benchmark, portfolio_value=100000):
    """Benchmark VaR calculation."""
    from app.risk.risk_engine import RiskEngine
    
    engine = RiskEngine()
    
    # Mock portfolio data
    positions = {
        'BTCUSDT': {'value': 30000, 'weight': 0.3},
        'ETHUSDT': {'value': 25000, 'weight': 0.25},
        'BNBUSDT': {'value': 20000, 'weight': 0.2},
        'SOLUSDT': {'value': 15000, 'weight': 0.15},
        'USDT': {'value': 10000, 'weight': 0.1},
    }
    
    result = benchmark(
        engine._calculate_var_95,
        positions=positions,
        portfolio_value=portfolio_value
    )
    return result


# ============================================================================
# Strategy Benchmarks
# ============================================================================

def mean_reversion_signal_generation(benchmark, symbol="BTCUSDT"):
    """Benchmark mean reversion signal generation."""
    from app.strategies.mean_reversion import MeanReversionStrategy
    
    strategy = MeanReversionStrategy()
    
    # Generate mock price data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    candles = [
        {'open': prices[i], 'high': prices[i] + 50, 'low': prices[i] - 50, 'close': prices[i], 'volume': 1000}
        for i in range(len(prices))
    ]
    
    result = benchmark(
        strategy.generate_signal,
        symbol=symbol,
        context={'candles': candles}
    )
    return result


def momentum_signal_generation(benchmark, symbol="ETHUSDT"):
    """Benchmark momentum strategy signal generation."""
    from app.strategies.momentum import MomentumStrategy
    
    strategy = MomentumStrategy()
    
    # Generate mock price data
    np.random.seed(42)
    prices = 3000 + np.cumsum(np.random.randn(100) * 20)
    candles = [
        {'open': prices[i], 'high': prices[i] + 10, 'low': prices[i] - 10, 'close': prices[i], 'volume': 5000}
        for i in range(len(prices))
    ]
    
    result = benchmark(
        strategy.generate_signal,
        symbol=symbol,
        context={'candles': candles}
    )
    return result


# ============================================================================
# Backtest Benchmarks
# ============================================================================

def backtest_simulation_simple(benchmark):
    """Benchmark simple backtest simulation."""
    from app.backtest import BacktestEngine, BacktestConfig
    
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        max_position_size=0.2,
        min_order_size=0.001
    )
    
    engine = BacktestEngine(config)
    
    # Simple mock data - minimal for benchmark
    np.random.seed(42)
    n_candles = 100
    
    historical_data = {
        'BTCUSDT': [
            {
                'timestamp': datetime.now() - timedelta(hours=n_candles-i),
                'open': 50000 + i * 10 + np.random.randn() * 50,
                'high': 50100 + i * 10 + np.random.randn() * 50,
                'low': 49900 + i * 10 + np.random.randn() * 50,
                'close': 50000 + i * 10 + np.random.randn() * 50,
                'volume': 1000 + np.random.randn() * 100
            }
            for i in range(n_candles)
        ]
    }
    
    result = benchmark(
        engine.run,
        strategy=None,
        symbols=['BTCUSDT'],
        historical_data=historical_data
    )
    return result


# ============================================================================
# Technical Analysis Benchmarks
# ============================================================================

def calculate_rsi_benchmark(benchmark):
    """Benchmark RSI calculation."""
    # Generate mock price data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    
    result = benchmark(
        _simple_rsi,
        prices=prices,
        period=14
    )
    return result


def _simple_rsi(prices, period=14):
    """Simple RSI calculation for benchmarking."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_benchmark(benchmark):
    """Benchmark Bollinger Bands calculation."""
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    
    result = benchmark(
        _simple_bollinger,
        prices=prices,
        period=20
    )
    return result


def _simple_bollinger(prices, period=20):
    """Simple Bollinger Bands for benchmarking."""
    sma = np.mean(prices[:period])
    std = np.std(prices[:period])
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return sma, upper, lower


# ============================================================================
# Math Utility Benchmarks
# ============================================================================

def calculate_sharpe_ratio_benchmark(benchmark):
    """Benchmark Sharpe ratio calculation."""
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.02
    
    result = benchmark(
        _simple_sharpe,
        returns=returns,
        risk_free_rate=0.02
    )
    return result


def _simple_sharpe(returns, risk_free_rate=0.02):
    """Simple Sharpe ratio for benchmarking."""
    excess_returns = returns - (risk_free_rate / 252)
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
