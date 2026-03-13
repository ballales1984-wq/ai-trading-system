oimport pytest
import pytest_benchmark
from app.backtest import BacktestEngine
from app.risk.risk_engine import RiskEngine
from app.strategies.momentum import MomentumStrategy
from app.strategies.mean_reversion import MeanReversionStrategy

@pytest.fixture
def sample_data():
    """Sample market data for benchmarks."""
    import pandas as pd
    import numpy as np
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 1000),
        'high': np.random.normal(102, 5, 1000),
        'low': np.random.normal(98, 5, 1000),
        'close': np.random.normal(100, 5, 1000),
        'volume': np.random.normal(1000, 200, 1000)
    }, index=dates)
    return data

def test_backtest_benchmark(benchmark, sample_data):
    """Benchmark backtest engine."""
    engine = BacktestEngine()
    result = benchmark(engine.run_backtest, sample_data, symbols=['BTCUSDT'])

def test_risk_var_calculation(benchmark, sample_data):
    """Benchmark VaR calculation."""
    engine = RiskEngine()
    returns = sample_data['close'].pct_change().dropna()
    result = benchmark(engine.calculate_var, returns, confidence=0.95)

def test_momentum_signals(benchmark, sample_data):
    """Benchmark momentum strategy signals."""
    strategy = MomentumStrategy()
    signals = benchmark(strategy.generate_signals, sample_data)

def test_mean_reversion_signals(benchmark, sample_data):
    """Benchmark mean reversion strategy."""
    strategy = MeanReversionStrategy()
    signals = benchmark(strategy.generate_signals, sample_data)

@pytest.mark.parametrize('strategy_class', [MomentumStrategy, MeanReversionStrategy])
def test_strategy_benchmark(benchmark, sample_data, strategy_class):
    """Parametrized strategy benchmarks."""
    strategy = strategy_class()
    signals = benchmark(strategy.generate_signals, sample_data)
