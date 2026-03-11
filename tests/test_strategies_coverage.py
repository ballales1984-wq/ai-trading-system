"""
Test Coverage for Strategies Module
=================================
Comprehensive tests to improve coverage for app/strategies/* modules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStrategiesBaseStrategy:
    """Test app.strategies.base_strategy module."""
    
    def test_base_strategy_module_import(self):
        """Test base_strategy module can be imported."""
        from app.strategies import base_strategy
        assert base_strategy is not None
    
    def test_signal_direction_enum(self):
        """Test SignalDirection enum exists."""
        from app.strategies.base_strategy import SignalDirection
        assert SignalDirection is not None
        assert hasattr(SignalDirection, 'LONG')
        assert hasattr(SignalDirection, 'SHORT')
        assert hasattr(SignalDirection, 'FLAT')
    
    def test_trading_signal_class(self):
        """Test TradingSignal class exists."""
        from app.strategies.base_strategy import TradingSignal
        assert TradingSignal is not None
    
    def test_strategy_config_class(self):
        """Test StrategyConfig class exists."""
        from app.strategies.base_strategy import StrategyConfig
        assert StrategyConfig is not None
    
    def test_base_strategy_class(self):
        """Test BaseStrategy class exists."""
        from app.strategies.base_strategy import BaseStrategy
        assert BaseStrategy is not None
    
    def test_trading_signal_creation(self):
        """Test TradingSignal creation."""
        from app.strategies.base_strategy import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=0.8
        )
        assert signal.symbol == "BTCUSDT"
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.8
    
    def test_strategy_config_creation(self):
        """Test StrategyConfig creation."""
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="momentum",
            parameters={"period": 14, "threshold": 0.02}
        )
        assert config.name == "momentum"
    
    def test_trading_signal_to_dict(self):
        """Test TradingSignal to_dict method."""
        from app.strategies.base_strategy import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0
        )
        
        result = signal.to_dict()
        assert isinstance(result, dict)
        assert result["symbol"] == "BTCUSDT"


class TestStrategiesMomentum:
    """Test app.strategies.momentum module."""
    
    def test_momentum_module_import(self):
        """Test momentum module can be imported."""
        from app.strategies import momentum
        assert momentum is not None
    
    def test_momentum_strategy_class(self):
        """Test MomentumStrategy class exists."""
        from app.strategies.momentum import MomentumStrategy
        assert MomentumStrategy is not None
    
    def test_momentum_strategy_creation(self):
        """Test MomentumStrategy creation."""
        from app.strategies.momentum import MomentumStrategy
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="momentum",
            parameters={"period": 14}
        )
        
        strategy = MomentumStrategy(config)
        assert strategy is not None
    
    def test_momentum_strategy_parameters(self):
        """Test MomentumStrategy parameters."""
        from app.strategies.momentum import MomentumStrategy
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="momentum",
            parameters={
                "period": 14,
                "threshold": 0.02,
                "stop_loss": 0.05
            }
        )
        
        strategy = MomentumStrategy(config)
        assert strategy.config.name == "momentum"


class TestStrategiesMeanReversion:
    """Test app.strategies.mean_reversion module."""
    
    def test_mean_reversion_module_import(self):
        """Test mean_reversion module can be imported."""
        from app.strategies import mean_reversion
        assert mean_reversion is not None
    
    def test_mean_reversion_strategy_class(self):
        """Test MeanReversionStrategy class exists."""
        from app.strategies.mean_reversion import MeanReversionStrategy
        assert MeanReversionStrategy is not None
    
    def test_mean_reversion_strategy_creation(self):
        """Test MeanReversionStrategy creation."""
        from app.strategies.mean_reversion import MeanReversionStrategy
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="mean_reversion",
            parameters={"period": 20, "std_dev": 2.0}
        )
        
        strategy = MeanReversionStrategy(config)
        assert strategy is not None
    
    def test_mean_reversion_parameters(self):
        """Test MeanReversionStrategy parameters."""
        from app.strategies.mean_reversion import MeanReversionStrategy
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="mean_reversion",
            parameters={
                "period": 20,
                "std_dev": 2.0,
                "entry_threshold": 0.05
            }
        )
        
        strategy = MeanReversionStrategy(config)
        assert strategy.config.name == "mean_reversion"


class TestStrategiesMultiStrategy:
    """Test app.strategies.multi_strategy module."""
    
    def test_multi_strategy_module_import(self):
        """Test multi_strategy module can be imported."""
        from app.strategies import multi_strategy
        assert multi_strategy is not None
    
    def test_strategy_signal_class(self):
        """Test StrategySignal class exists."""
        from app.strategies.multi_strategy import StrategySignal
        assert StrategySignal is not None
    
    def test_multi_strategy_manager_class(self):
        """Test MultiStrategyManager class exists."""
        from app.strategies.multi_strategy import MultiStrategyManager
        assert MultiStrategyManager is not None
    
    def test_strategy_signal_creation(self):
        """Test StrategySignal creation."""
        from app.strategies.multi_strategy import StrategySignal
        from app.strategies.base_strategy import TradingSignal, SignalDirection
        
        # StrategySignal wraps a TradingSignal
        trading_signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=0.75
        )
        
        signal = StrategySignal(
            strategy_name="momentum",
            signal=trading_signal,
            weight=1.0
        )
        assert signal.strategy_name == "momentum"
        assert signal.signal.symbol == "BTCUSDT"
    
    def test_multi_strategy_manager_creation(self):
        """Test MultiStrategyManager creation."""
        from app.strategies.multi_strategy import MultiStrategyManager
        
        manager = MultiStrategyManager()
        assert manager is not None


class TestStrategiesIntegration:
    """Integration tests for strategies module."""
    
    def test_strategy_config_validation(self):
        """Test strategy configuration validation."""
        from app.strategies.base_strategy import StrategyConfig
        
        valid_config = StrategyConfig(
            name="test_strategy",
            parameters={"param1": 10, "param2": "value"}
        )
        
        assert valid_config.name is not None
    
    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking."""
        from app.strategies.momentum import MomentumStrategy
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(name="test", parameters={"period": 14})
        strategy = MomentumStrategy(config)
        
        strategy.signals_generated = 10
        performance = strategy.get_performance()
        
        assert performance["signals_generated"] == 10
    
    def test_multiple_symbol_strategies(self):
        """Test strategies across multiple symbols."""
        from app.strategies.base_strategy import StrategyConfig, TradingSignal, SignalDirection
        
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        
        signals = []
        for symbol in symbols:
            signal = TradingSignal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                confidence=0.7,
                timestamp=datetime.now()
            )
            signals.append(signal)
        
        assert len(signals) == len(symbols)
        assert all(s.direction == SignalDirection.LONG for s in signals)

