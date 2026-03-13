import pytest
from datetime import datetime, timedelta
import numpy as np
from app.backtest import BacktestEngine, BacktestConfig, OHLCV, Position, OrderSide, PositionSide

@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing"""
    return [OHLCV(
        timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
        open=100 + i * 0.1,
        high=101 + i * 0.1,
        low=99 + i * 0.1,
        close=100.5 + i * 0.1,
        volume=1000 + i * 10
    ) for i in range(100)]

@pytest.fixture
def engine():
    config = BacktestConfig(initial_capital=100000, commission_rate=0.001)
    return BacktestEngine(config=config)

def test_backtest_engine_initialization(engine):
    assert engine.config.initial_capital == 100000
    assert engine.capital == 100000
    assert len(engine.positions) == 0
    assert len(engine.trades) == 0

def test_position_update(engine, sample_data):
    # Create position
    engine.positions['BTCUSDT'] = Position(
        symbol='BTCUSDT',
        side=PositionSide.LONG,
        quantity=10,
        entry_price=50000
    )
    
    # Update position
    engine.positions['BTCUSDT'].update(51000)
    
    assert engine.positions['BTCUSDT'].current_price == 51000
    assert engine.positions['BTCUSDT'].unrealized_pnl == 1000

def test_calculate_results(engine, sample_data):
    # Run minimal simulation
    engine.equity_curve = [
        {'timestamp': datetime.now(), 'equity': 100000},
        {'timestamp': datetime.now(), 'equity': 105000},
        {'timestamp': datetime.now(), 'equity': 102000}
    ]
    
    engine.start_time = datetime.now()
    engine.start_date = datetime.now()
    engine.end_date = datetime.now()
    result = engine._calculate_results()
    
    assert result.total_return == 2000
    assert result.total_return_pct == 2.0
    assert result.max_drawdown_pct > 0
    assert len(result.equity_curve) == 3

def test_slippage_application(engine):
    # Test BUY side slippage
    slippage_price = engine._apply_slippage(10000, OrderSide.BUY)
    assert slippage_price > 10000
    
    # Test SELL side slippage
    slippage_price = engine._apply_slippage(10000, OrderSide.SELL)
    assert slippage_price < 10000

@pytest.mark.asyncio
async def test_run_simulation(engine, sample_data):
    class DummyStrategy:
        def generate_signal(self, symbol, context):
            if len(context['prices']) > 10:
                return type('Signal', (), {'action': 'buy'})()
            return None
    
    # Mock historical data
    historical_data = {'BTCUSDT': sample_data}
    
    await engine._run_simulation(DummyStrategy(), historical_data, ['BTCUSDT'])
    
    assert len(engine.equity_curve) > 0

def test_calculate_equity(engine):
    # Add test positions
    engine.positions['BTC'] = Position(
        symbol='BTC',
        side=PositionSide.LONG,
        quantity=1,
        entry_price=50000,
        current_price=51000
    )
    
    current_prices = {'BTC': 51000}
    equity = engine._calculate_equity(current_prices)
    
    assert equity == 100000 + 1000  # capital + position value change
