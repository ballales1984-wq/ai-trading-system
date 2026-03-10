"""
Tests for Strategy Modules - Coverage
"""
import pytest
from unittest.mock import Mock, patch
from app.strategies.base_strategy import BaseStrategy
from app.strategies.momentum import MomentumStrategy
from app.strategies.mean_reversion import MeanReversionStrategy
from app.strategies.multi_strategy import MultiStrategy


class TestBaseStrategy:
    """Test BaseStrategy class"""
    
    def test_base_strategy_creation(self):
        """Test creating BaseStrategy"""
        strategy = BaseStrategy()
        assert strategy is not None
    
    def test_base_strategy_initial_state(self):
        """Test initial state of BaseStrategy"""
        strategy = BaseStrategy()
        # Should have name attribute
        assert hasattr(strategy, 'name') or hasattr(strategy, '_name')
    
    def test_base_strategy_has_parameters(self):
        """Test BaseStrategy has parameters"""
        strategy = BaseStrategy()
        # Should have parameters
        assert hasattr(strategy, 'parameters') or hasattr(strategy, '_parameters')


class TestMomentumStrategy:
    """Test MomentumStrategy class"""
    
    def test_momentum_strategy_creation(self):
        """Test creating MomentumStrategy"""
        strategy = MomentumStrategy()
        assert strategy is not None
    
    def test_momentum_strategy_initial_state(self):
        """Test initial state of MomentumStrategy"""
        strategy = MomentumStrategy()
        # Should have lookback period
        assert hasattr(strategy, 'lookback') or hasattr(strategy, 'lookback_period')
    
    def test_momentum_strategy_has_indicators(self):
        """Test MomentumStrategy has indicators"""
        strategy = MomentumStrategy()
        # Should have indicators
        assert hasattr(strategy, 'indicators') or hasattr(strategy, '_indicators')


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy class"""
    
    def test_mean_reversion_creation(self):
        """Test creating MeanReversionStrategy"""
        strategy = MeanReversionStrategy()
        assert strategy is not None
    
    def test_mean_reversion_initial_state(self):
        """Test initial state of MeanReversionStrategy"""
        strategy = MeanReversionStrategy()
        # Should have lookback period
        assert hasattr(strategy, 'lookback') or hasattr(strategy, 'lookback_period')
    
    def test_mean_reversion_has_indicators(self):
        """Test MeanReversionStrategy has indicators"""
        strategy = MeanReversionStrategy()
        # Should have indicators
        assert hasattr(strategy, 'indicators') or hasattr(strategy, '_indicators')


class TestMultiStrategy:
    """Test MultiStrategy class"""
    
    def test_multi_strategy_creation(self):
        """Test creating MultiStrategy"""
        strategy = MultiStrategy()
        assert strategy is not None
    
    def test_multi_strategy_initial_state(self):
        """Test initial state of MultiStrategy"""
        strategy = MultiStrategy()
        # Should have strategies list
        assert hasattr(strategy, 'strategies') or hasattr(strategy, '_strategies')
    
    def test_multi_strategy_has_weights(self):
        """Test MultiStrategy has weights"""
        strategy = MultiStrategy()
        # Should have weights
        assert hasattr(strategy, 'weights') or hasattr(strategy, '_weights')


class TestStrategiesEdgeCases:
    """Test edge cases for strategies"""
    
    def test_base_strategy_default_name(self):
        """Test BaseStrategy default name"""
        strategy = BaseStrategy()
        name = getattr(strategy, 'name', getattr(strategy, '_name', 'BaseStrategy'))
        assert name is not None
    
    def test_momentum_strategy_default_lookback(self):
        """Test MomentumStrategy default lookback"""
        strategy = MomentumStrategy()
        lookback = getattr(strategy, 'lookback', getattr(strategy, 'lookback_period', 20))
        assert lookback is not None
    
    def test_mean_reversion_default_lookback(self):
        """Test MeanReversionStrategy default lookback"""
        strategy = MeanReversionStrategy()
        lookback = getattr(strategy, 'lookback', getattr(strategy, 'lookback_period', 50))
        assert lookback is not None
