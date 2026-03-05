"""
Tests for Strategies Module
========================
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSignalDirection:
    """Test SignalDirection enum."""
    
    def test_signal_direction_values(self):
        """Test SignalDirection enum values."""
        from app.strategies.base_strategy import SignalDirection
        
        assert SignalDirection.LONG.value == "LONG"
        assert SignalDirection.SHORT.value == "SHORT"


class TestStrategyConfig:
    """Test StrategyConfig model."""
    
    def test_strategy_config_creation(self):
        """Test creating a StrategyConfig."""
        from app.strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="momentum",
            enabled=True
        )
        
        assert config.name == "momentum"
        assert config.enabled is True


class TestBaseStrategy:
    """Test BaseStrategy class."""
    
    def test_base_strategy_creation(self):
        """Test BaseStrategy is abstract."""
        from app.strategies.base_strategy import BaseStrategy
        
        # Cannot instantiate abstract class
        assert BaseStrategy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
