"""
Test Suite for Strategies Module
================================
Comprehensive tests for trading strategies.
"""

import pytest
from datetime import datetime
from app.strategies.base_strategy import (
    SignalDirection,
    TradingSignal,
    StrategyConfig,
    BaseStrategy,
)


class TestSignalDirection:
    """Tests for SignalDirection enum."""
    
    def test_signal_direction_values(self):
        """Test signal direction values."""
        assert SignalDirection.LONG.value == "LONG"
        assert SignalDirection.SHORT.value == "SHORT"
        assert SignalDirection.FLAT.value == "FLAT"


class TestTradingSignal:
    """Tests for TradingSignal dataclass."""
    
    def test_trading_signal_creation(self):
        """Test trading signal creation."""
        signal = TradingSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.75,
        )
        assert signal.symbol == "BTC/USDT"
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.75
    
    def test_trading_signal_to_dict(self):
        """Test trading signal to dictionary."""
        signal = TradingSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.75,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
        )
        result = signal.to_dict()
        assert result["symbol"] == "BTC/USDT"
        assert result["direction"] == "LONG"
        assert result["confidence"] == 0.75
        assert result["entry_price"] == 50000.0
        assert result["stop_loss"] == 48000.0
        assert result["take_profit"] == 55000.0
    
    def test_trading_signal_defaults(self):
        """Test trading signal default values."""
        signal = TradingSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.FLAT,
            confidence=0.5,
        )
        assert signal.entry_price is None
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert signal.strategy_name == ""
        assert signal.metadata == {}


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""
    
    def test_strategy_config_creation(self):
        """Test strategy config creation."""
        config = StrategyConfig(
            name="momentum",
            enabled=True,
            parameters={"period": 20},
        )
        assert config.name == "momentum"
        assert config.enabled is True
        assert config.parameters["period"] == 20
    
    def test_strategy_config_defaults(self):
        """Test strategy config default values."""
        config = StrategyConfig(name="test")
        assert config.name == "test"
        assert config.enabled is True
        assert config.parameters == {}


class TestBaseStrategy:
    """Tests for BaseStrategy abstract class."""
    
    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseStrategy(config=StrategyConfig(name="test"))
    
    def test_concrete_strategy_creation(self):
        """Test creating a concrete strategy."""
        from app.strategies.momentum import MomentumStrategy
        
        config = StrategyConfig(
            name="momentum",
            enabled=True,
            parameters={"period": 20},
        )
        strategy = MomentumStrategy(config)
        assert strategy.name == "momentum"
        assert strategy.enabled is True
        assert strategy.period == 20
