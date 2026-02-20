# tests/test_strategies.py
"""
Test Suite for Strategy Modules
===============================
Comprehensive tests for all strategy implementations.
"""

import pytest
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import (
    BaseStrategy,
    TradingSignal as Signal,  # Alias for backwards compatibility
    SignalType,
    SignalStrength,
    StrategyContext,
)
from src.strategy.momentum import MomentumStrategy
from src.strategy.mean_reversion import MeanReversionStrategy


# Fixtures
@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    # Generate trending prices
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def sample_volumes():
    """Generate sample volume data."""
    np.random.seed(42)
    return np.random.uniform(1000, 5000, 100)


@pytest.fixture
def basic_context(sample_prices, sample_volumes):
    """Create basic strategy context."""
    return StrategyContext(
        symbol="BTCUSDT",
        prices=sample_prices,
        volumes=sample_volumes,
        timestamps=[datetime.now()] * len(sample_prices),
        indicators={},
    )


# Test TradingSignal
class TestTradingSignal:
    """Tests for TradingSignal dataclass."""
    
    def test_signal_creation(self):
        """Test signal creation."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            price=42000.0,
            timestamp=datetime.now(),
            strategy="test",
        )
        
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.85
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.SELL,
            strength=SignalStrength.MODERATE,
            confidence=0.65,
            price=42000.0,
            timestamp=datetime.now(),
            strategy="test",
        )
        
        data = signal.to_dict()
        
        assert data["symbol"] == "BTCUSDT"
        assert data["signal_type"] == "SELL"
        assert data["strength"] == "moderate"


# Test StrategyContext
class TestStrategyContext:
    """Tests for StrategyContext."""
    
    def test_context_creation(self, sample_prices, sample_volumes):
        """Test context creation."""
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=sample_prices,
            volumes=sample_volumes,
            timestamps=[datetime.now()] * len(sample_prices),
        )
        
        assert context.symbol == "BTCUSDT"
        assert len(context.prices) == 100
        assert context.mc_probability_up is None
    
    def test_context_with_indicators(self, sample_prices, sample_volumes):
        """Test context with indicators."""
        indicators = {
            "rsi": np.random.uniform(30, 70, 100),
            "adx": np.random.uniform(20, 40, 100),
        }
        
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=sample_prices,
            volumes=sample_volumes,
            timestamps=[datetime.now()] * len(sample_prices),
            indicators=indicators,
        )
        
        assert "rsi" in context.indicators
        assert "adx" in context.indicators


# Test BaseStrategy
class TestBaseStrategy:
    """Tests for BaseStrategy."""
    
    def test_strategy_creation(self):
        """Test strategy creation."""
        strategy = MomentumStrategy(
            name="test_momentum",
            config={"params": {"momentum_period": 10}}
        )
        
        assert strategy.name == "test_momentum"
        assert strategy.enabled
    
    def test_strategy_enable_disable(self):
        """Test strategy enable/disable."""
        strategy = MomentumStrategy(name="test")
        
        assert strategy.enabled
        
        strategy.disable()
        assert not strategy.enabled
        
        strategy.enable()
        assert strategy.enabled
    
    def test_signal_strength_classification(self):
        """Test signal strength classification."""
        strategy = MomentumStrategy(name="test")
        
        assert strategy.classify_strength(0.9) == SignalStrength.STRONG
        assert strategy.classify_strength(0.7) == SignalStrength.MODERATE
        assert strategy.classify_strength(0.4) == SignalStrength.WEAK
    
    def test_create_signal(self):
        """Test signal creation."""
        strategy = MomentumStrategy(name="test")
        
        signal = strategy.create_signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=42000.0,
        )
        
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.strategy == "test"
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        strategy = MomentumStrategy(name="test")
        
        # Create some signals
        strategy.create_signal("BTCUSDT", SignalType.BUY, 0.8, 42000.0)
        strategy.create_signal("ETHUSDT", SignalType.SELL, 0.7, 3000.0)
        
        perf = strategy.get_performance()
        
        assert perf["signals_generated"] == 2
        assert perf["enabled"]


# Test MomentumStrategy
class TestMomentumStrategy:
    """Tests for MomentumStrategy."""
    
    def test_strategy_creation(self):
        """Test momentum strategy creation."""
        strategy = MomentumStrategy(
            name="momentum",
            config={
                "params": {
                    "momentum_period": 10,
                    "momentum_threshold": 0.02,
                }
            }
        )
        
        assert strategy.momentum_period == 10
        assert strategy.momentum_threshold == 0.02
    
    def test_momentum_calculation(self, basic_context):
        """Test momentum calculation."""
        strategy = MomentumStrategy(name="test")
        
        momentum = strategy._calculate_momentum(basic_context.prices)
        
        assert momentum is not None
        assert isinstance(momentum, float)
    
    def test_signal_generation_uptrend(self):
        """Test signal generation in uptrend."""
        # Create uptrending prices
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.01, 100)))
        volumes = np.random.uniform(1000, 5000, 100)
        
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=prices,
            volumes=volumes,
            timestamps=[datetime.now()] * len(prices),
        )
        
        strategy = MomentumStrategy(
            name="test",
            config={"params": {"momentum_threshold": 0.01}}
        )
        
        signal = strategy.generate_signal(context)
        
        # Should generate BUY in uptrend
        if signal:
            assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
    
    def test_signal_generation_downtrend(self):
        """Test signal generation in downtrend."""
        # Create downtrending prices
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(-0.005, 0.01, 100)))
        volumes = np.random.uniform(1000, 5000, 100)
        
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=prices,
            volumes=volumes,
            timestamps=[datetime.now()] * len(prices),
        )
        
        strategy = MomentumStrategy(
            name="test",
            config={"params": {"momentum_threshold": 0.01}}
        )
        
        signal = strategy.generate_signal(context)
        
        # Should generate SELL in downtrend
        if signal:
            assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]
    
    def test_disabled_strategy(self, basic_context):
        """Test disabled strategy returns no signal."""
        strategy = MomentumStrategy(name="test")
        strategy.disable()
        
        signal = strategy.generate_signal(basic_context)
        
        assert signal is None
    
    def test_volume_confirmation(self, basic_context):
        """Test volume confirmation check."""
        strategy = MomentumStrategy(name="test")
        
        conf = strategy._check_volume_confirmation(basic_context)
        
        assert 0 <= conf <= 1
    
    def test_parameter_update(self):
        """Test parameter update."""
        strategy = MomentumStrategy(name="test")
        
        strategy.update_parameters(
            momentum_period=20,
            momentum_threshold=0.03
        )
        
        assert strategy.momentum_period == 20
        assert strategy.momentum_threshold == 0.03


# Test MeanReversionStrategy
class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy."""
    
    def test_strategy_creation(self):
        """Test mean reversion strategy creation."""
        strategy = MeanReversionStrategy(
            name="mean_reversion",
            config={
                "params": {
                    "lookback_period": 20,
                    "entry_z_score": 2.0,
                }
            }
        )
        
        assert strategy.lookback_period == 20
        assert strategy.entry_z_score == 2.0
    
    def test_z_score_calculation(self, basic_context):
        """Test z-score calculation."""
        strategy = MeanReversionStrategy(name="test")
        
        z_score = strategy._calculate_z_score(basic_context.prices)
        
        assert z_score is not None
        assert isinstance(z_score, float)
    
    def test_signal_generation_oversold(self):
        """Test signal generation when oversold."""
        # Create prices that drop significantly
        np.random.seed(42)
        prices = np.concatenate([
            np.linspace(100, 100, 50),
            np.linspace(100, 85, 50)  # 15% drop
        ])
        volumes = np.random.uniform(1000, 5000, 100)
        
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=prices,
            volumes=volumes,
            timestamps=[datetime.now()] * len(prices),
        )
        
        strategy = MeanReversionStrategy(
            name="test",
            config={"params": {"entry_z_score": 1.5}}
        )
        
        signal = strategy.generate_signal(context)
        
        # Should generate BUY when oversold
        if signal:
            assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
    
    def test_signal_generation_overbought(self):
        """Test signal generation when overbought."""
        # Create prices that rise significantly
        np.random.seed(42)
        prices = np.concatenate([
            np.linspace(100, 100, 50),
            np.linspace(100, 115, 50)  # 15% rise
        ])
        volumes = np.random.uniform(1000, 5000, 100)
        
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=prices,
            volumes=volumes,
            timestamps=[datetime.now()] * len(prices),
        )
        
        strategy = MeanReversionStrategy(
            name="test",
            config={"params": {"entry_z_score": 1.5}}
        )
        
        signal = strategy.generate_signal(context)
        
        # Should generate SELL when overbought
        if signal:
            assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]
    
    def test_rsi_confirmation(self, basic_context):
        """Test RSI confirmation."""
        strategy = MeanReversionStrategy(name="test")
        
        # Add RSI indicator
        basic_context.indicators["rsi"] = np.random.uniform(20, 80, 100)
        
        conf = strategy._check_rsi(basic_context, -2.0)
        
        assert 0 <= conf <= 1
    
    def test_bollinger_check(self, basic_context):
        """Test Bollinger Band check."""
        strategy = MeanReversionStrategy(name="test")
        
        conf = strategy._check_bollinger(basic_context)
        
        assert 0 <= conf <= 1
    
    def test_parameter_update(self):
        """Test parameter update."""
        strategy = MeanReversionStrategy(name="test")
        
        strategy.update_parameters(
            lookback_period=30,
            entry_z_score=2.5,
            exit_z_score=0.3
        )
        
        assert strategy.lookback_period == 30
        assert strategy.entry_z_score == 2.5
        assert strategy.exit_z_score == 0.3


# Test Strategy Validation
class TestStrategyValidation:
    """Tests for strategy validation."""
    
    def test_confidence_threshold(self, basic_context):
        """Test minimum confidence threshold."""
        strategy = MomentumStrategy(
            name="test",
            config={"min_confidence": 0.8}
        )
        
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=0.5,  # Below threshold
            price=42000.0,
            timestamp=datetime.now(),
            strategy="test",
        )
        
        valid = strategy.validate_signal(signal, basic_context)
        
        assert not valid
    
    def test_critical_risk_rejection(self):
        """Test signal rejection on critical risk."""
        strategy = MomentumStrategy(name="test")
        
        context = StrategyContext(
            symbol="BTCUSDT",
            prices=np.random.uniform(100, 110, 100),
            volumes=np.random.uniform(1000, 5000, 100),
            timestamps=[datetime.now()] * 100,
            risk_level="critical",
        )
        
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.9,
            price=42000.0,
            timestamp=datetime.now(),
            strategy="test",
        )
        
        valid = strategy.validate_signal(signal, context)
        
        assert not valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
