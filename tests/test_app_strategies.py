"""
Tests for App Strategies
========================
Tests for trading strategies in the app module.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.strategies.base_strategy import (
    BaseStrategy, StrategyConfig, TradingSignal, SignalDirection
)
from app.strategies.momentum import MomentumStrategy
from app.strategies.mean_reversion import MeanReversionStrategy


# ==================== FIXTURES ====================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    
    prices = 45000 + np.cumsum(np.random.randn(200) * 500)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(200) * 100,
        'high': prices + np.abs(np.random.randn(200) * 200) + 100,
        'low': prices - np.abs(np.random.randn(200) * 200) - 100,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    return df


@pytest.fixture
def oversold_data():
    """Generate data with oversold RSI conditions."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    
    # Create downtrend to generate oversold RSI
    prices = 45000 - np.cumsum(np.abs(np.random.randn(200) * 300))
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(200) * 50,
        'high': prices + np.abs(np.random.randn(200) * 100) + 50,
        'low': prices - np.abs(np.random.randn(200) * 100) - 50,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    return df


@pytest.fixture
def overbought_data():
    """Generate data with overbought RSI conditions."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    
    # Create uptrend to generate overbought RSI
    prices = 45000 + np.cumsum(np.abs(np.random.randn(200) * 300))
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(200) * 50,
        'high': prices + np.abs(np.random.randn(200) * 100) + 50,
        'low': prices - np.abs(np.random.randn(200) * 100) - 50,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    return df


# ==================== TESTS ====================

class TestTradingSignal:
    """Test TradingSignal dataclass."""
    
    def test_initialization(self):
        """Test TradingSignal initialization."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=0.8
        )
        
        assert signal.symbol == "BTCUSDT"
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.8
        assert isinstance(signal.timestamp, datetime)
    
    def test_to_dict(self):
        """Test converting signal to dictionary."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=0.8,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000
        )
        
        signal_dict = signal.to_dict()
        
        assert signal_dict['symbol'] == "BTCUSDT"
        assert signal_dict['direction'] == "LONG"
        assert signal_dict['confidence'] == 0.8
        assert signal_dict['entry_price'] == 50000
        assert signal_dict['stop_loss'] == 49000
        assert signal_dict['take_profit'] == 52000
    
    def test_signal_directions(self):
        """Test all signal directions."""
        assert SignalDirection.LONG == "LONG"
        assert SignalDirection.SHORT == "SHORT"
        assert SignalDirection.FLAT == "FLAT"


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""
    
    def test_initialization(self):
        """Test StrategyConfig initialization."""
        config = StrategyConfig(
            name="test_strategy",
            enabled=True,
            parameters={"lookback": 20}
        )
        
        assert config.name == "test_strategy"
        assert config.enabled is True
        assert config.parameters == {"lookback": 20}
    
    def test_default_values(self):
        """Test StrategyConfig default values."""
        config = StrategyConfig(name="test_strategy")
        
        assert config.enabled is True
        assert config.parameters == {}


class TestMomentumStrategy:
    """Test MomentumStrategy class."""
    
    def setup_method(self):
        """Setup test data."""
        config = StrategyConfig(
            name="momentum",
            parameters={
                "lookback_period": 20,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05
            }
        )
        self.strategy = MomentumStrategy(config)
    
    def test_initialization(self):
        """Test MomentumStrategy initialization."""
        assert self.strategy is not None
        assert self.strategy.name == "momentum"
        assert self.strategy.lookback_period == 20
        assert self.strategy.rsi_oversold == 30
        assert self.strategy.rsi_overbought == 70
    
    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Create data with fewer rows than lookback period
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
        df = pd.DataFrame({
            'open': [45000] * 10,
            'high': [46000] * 10,
            'low': [44000] * 10,
            'close': [45500] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        signal = self.strategy.generate_signal(df)
        
        assert signal is None
    
    def test_generate_signal_normal_data(self, sample_ohlcv_data):
        """Test signal generation with normal data."""
        signal = self.strategy.generate_signal(sample_ohlcv_data)
        
        # Signal can be None or a TradingSignal
        if signal is not None:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == ""  # Default symbol
            assert signal.direction in [SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT]
            assert 0 <= signal.confidence <= 1
    
    def test_strategy_enabled(self):
        """Test that strategy is enabled by default."""
        assert self.strategy.enabled is True
    
    def test_strategy_parameters(self):
        """Test strategy parameters."""
        assert self.strategy.lookback_period == 20
        assert self.strategy.rsi_oversold == 30
        assert self.strategy.rsi_overbought == 70
        assert self.strategy.stop_loss_pct == 0.02
        assert self.strategy.take_profit_pct == 0.05


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy class."""
    
    def setup_method(self):
        """Setup test data."""
        config = StrategyConfig(
            name="mean_reversion",
            parameters={
                "lookback_period": 50,
                "z_score_threshold": 2.0,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            }
        )
        self.strategy = MeanReversionStrategy(config)
    
    def test_initialization(self):
        """Test MeanReversionStrategy initialization."""
        assert self.strategy is not None
        assert self.strategy.name == "mean_reversion"
    
    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Create data with fewer rows than lookback period
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
        df = pd.DataFrame({
            'open': [45000] * 10,
            'high': [46000] * 10,
            'low': [44000] * 10,
            'close': [45500] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        signal = self.strategy.generate_signal(df)
        
        assert signal is None
    
    def test_generate_signal_normal_data(self, sample_ohlcv_data):
        """Test signal generation with normal data."""
        signal = self.strategy.generate_signal(sample_ohlcv_data)
        
        # Signal can be None or a TradingSignal
        if signal is not None:
            assert isinstance(signal, TradingSignal)
            assert signal.direction in [SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT]
            assert 0 <= signal.confidence <= 1
    
    def test_strategy_enabled(self):
        """Test that strategy is enabled by default."""
        assert self.strategy.enabled is True
