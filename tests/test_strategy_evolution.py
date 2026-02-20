# tests/test_strategy_evolution.py
"""
Test Suite for Strategy and Evolution Modules
==============================================
Tests for trading strategies and AutoML evolution.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from src.strategy.base_strategy import (
    BaseStrategy,
    TradingSignal as Signal,  # Alias for backwards compatibility
    SignalType as SignalAction,  # Alias for backwards compatibility
    SignalStrength,
    StrategyContext,
)
from src.strategy.momentum import MomentumStrategy
from src.automl.evolution import (
    EvolutionEngine,
    EvolutionConfig,
    Individual,
)


class TestSignal:
    """Tests for Signal dataclass."""
    
    def test_signal_creation(self):
        """Test signal creation."""
        signal = Signal(
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            confidence=0.8,
            strength=SignalStrength.STRONG,
            price=42000.0,
            quantity=0.1,
            stop_loss=40000.0,
            take_profit=45000.0,
            reason="Test signal",
            strategy="test",
            timestamp=datetime.now(),
        )
        
        assert signal.symbol == "BTCUSDT"
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.8
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = Signal(
            symbol="BTCUSDT",
            action=SignalAction.SELL,
            confidence=0.6,
            strength=SignalStrength.MODERATE,
            price=42000.0,
            quantity=None,
            stop_loss=None,
            take_profit=None,
            reason="Test",
            strategy="test",
            timestamp=datetime.now(),
        )
        
        data = signal.to_dict()
        
        assert data["symbol"] == "BTCUSDT"
        assert data["action"] == "SELL"
        assert data["confidence"] == 0.6


class TestBaseStrategy:
    """Tests for BaseStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = {
            "max_position_size": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }
        
        # Create concrete implementation for testing
        class TestStrategy(BaseStrategy):
            def generate_signal(self, context):
                return Signal(
                    symbol=context.get("symbol", "TEST"),
                    action=SignalAction.HOLD,
                    confidence=0.5,
                    strength=SignalStrength.WEAK,
                    price=100.0,
                    quantity=None,
                    stop_loss=None,
                    take_profit=None,
                    reason="Test",
                    strategy=self.name,
                    timestamp=datetime.now(),
                )
            
            def get_required_data(self):
                return ["prices"]
        
        return TestStrategy(name="test_strategy", config=config)
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "test_strategy"
        assert strategy.max_position_size == 0.1
        assert strategy.stop_loss_pct == 0.02
    
    def test_start_stop(self, strategy):
        """Test strategy start/stop."""
        assert not strategy.is_active
        
        strategy.start()
        assert strategy.is_active
        
        strategy.stop()
        assert not strategy.is_active
    
    def test_position_size_calculation(self, strategy):
        """Test position size calculation."""
        size = strategy.calculate_position_size(
            price=100.0,
            portfolio_value=10000.0,
            risk_per_trade=0.02
        )
        
        assert size > 0
        assert size <= 10000.0 / 100.0 * 0.1  # Max position size
    
    def test_stop_loss_calculation(self, strategy):
        """Test stop loss calculation."""
        stop_loss = strategy.calculate_stop_loss(100.0, SignalAction.BUY)
        assert stop_loss == 98.0  # 2% below
        
        stop_loss = strategy.calculate_stop_loss(100.0, SignalAction.SELL)
        assert stop_loss == 102.0  # 2% above
    
    def test_take_profit_calculation(self, strategy):
        """Test take profit calculation."""
        take_profit = strategy.calculate_take_profit(100.0, SignalAction.BUY)
        assert take_profit == 104.0  # 4% above
        
        take_profit = strategy.calculate_take_profit(100.0, SignalAction.SELL)
        assert take_profit == 96.0  # 4% below
    
    def test_strength_determination(self, strategy):
        """Test signal strength determination."""
        assert strategy.determine_strength(0.9) == SignalStrength.STRONG
        assert strategy.determine_strength(0.6) == SignalStrength.MODERATE
        assert strategy.determine_strength(0.3) == SignalStrength.WEAK
    
    def test_metrics_update(self, strategy):
        """Test metrics update."""
        strategy.update_metrics(100.0, is_win=True)
        assert strategy.metrics.winning_signals == 1
        assert strategy.metrics.total_pnl == 100.0
        
        strategy.update_metrics(-50.0, is_win=False)
        assert strategy.metrics.losing_signals == 1
        assert strategy.metrics.total_pnl == 50.0


class TestMomentumStrategy:
    """Tests for MomentumStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create momentum strategy instance."""
        config = {
            "lookback_period": 14,
            "momentum_threshold": 0.02,
            "volume_threshold": 1.5,
            "use_ma_filter": True,
            "ma_fast": 10,
            "ma_slow": 30,
        }
        return MomentumStrategy(name="momentum", config=config)
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.lookback_period == 14
        assert strategy.momentum_threshold == 0.02
        assert strategy.use_ma_filter == True
    
    def test_required_data(self, strategy):
        """Test required data keys."""
        required = strategy.get_required_data()
        assert "prices" in required
        assert "volumes" in required
    
    def test_momentum_calculation(self, strategy):
        """Test momentum calculation."""
        # Upward momentum
        prices = np.array([100, 105, 110, 115, 120])
        momentum = strategy._calculate_momentum(prices)
        assert momentum > 0
        
        # Downward momentum
        prices = np.array([120, 115, 110, 105, 100])
        momentum = strategy._calculate_momentum(prices)
        assert momentum < 0
    
    def test_volume_ratio_calculation(self, strategy):
        """Test volume ratio calculation."""
        volumes = np.array([100] * 20 + [200])  # Last volume is 2x average
        ratio = strategy._calculate_volume_ratio(volumes)
        assert ratio > 1.5
    
    def test_ma_signal_calculation(self, strategy):
        """Test MA signal calculation."""
        # Bullish trend
        prices = np.arange(1, 51, dtype=float)
        signal = strategy._calculate_ma_signal(prices)
        assert signal > 0
        
        # Bearish trend
        prices = np.arange(50, 0, -1, dtype=float)
        signal = strategy._calculate_ma_signal(prices)
        assert signal < 0
    
    def test_signal_generation(self, strategy):
        """Test signal generation."""
        # Create context with bullish data
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.uniform(0.01, 0.03, 50))
        volumes = np.random.uniform(100, 200, 50)
        
        context = {
            "symbol": "BTCUSDT",
            "prices": prices.tolist(),
            "volumes": volumes.tolist(),
            "highs": prices.tolist(),
            "lows": (prices * 0.99).tolist(),
        }
        
        signal = strategy.generate_signal(context)
        
        assert signal is not None
        assert signal.symbol == "BTCUSDT"
        assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
    
    def test_signal_with_weak_momentum(self, strategy):
        """Test signal with weak momentum."""
        # Flat prices = weak momentum
        prices = np.ones(50) * 100
        volumes = np.ones(50) * 100
        
        context = {
            "symbol": "BTCUSDT",
            "prices": prices.tolist(),
            "volumes": volumes.tolist(),
            "highs": prices.tolist(),
            "lows": prices.tolist(),
        }
        
        signal = strategy.generate_signal(context)
        
        # Should be HOLD due to weak momentum
        assert signal.action == SignalAction.HOLD


class TestIndividual:
    """Tests for Individual dataclass."""
    
    def test_individual_creation(self):
        """Test individual creation."""
        individual = Individual(
            params={"threshold": 0.02, "period": 14},
            fitness=0.8,
            generation=1,
        )
        
        assert individual.params["threshold"] == 0.02
        assert individual.fitness == 0.8
        assert individual.generation == 1
    
    def test_individual_to_dict(self):
        """Test individual serialization."""
        individual = Individual(
            params={"test": 1},
            fitness=0.5,
            generation=0,
        )
        
        data = individual.to_dict()
        
        assert "id" in data
        assert data["params"]["test"] == 1
        assert data["fitness"] == 0.5


class TestEvolutionEngine:
    """Tests for EvolutionEngine."""
    
    @pytest.fixture
    def config(self):
        """Create evolution config."""
        return EvolutionConfig(
            population_size=10,
            elite_size=2,
            mutation_rate=0.1,
            crossover_rate=0.7,
            generations=5,
            param_ranges={
                "threshold": (0.01, 0.1, "float"),
                "period": (5, 30, "int"),
            }
        )
    
    @pytest.fixture
    def engine(self, config):
        """Create evolution engine."""
        return EvolutionEngine(config)
    
    def test_engine_initialization(self, engine, config):
        """Test engine initialization."""
        assert engine.config.population_size == 10
        assert engine.config.mutation_rate == 0.1
    
    def test_population_initialization(self, engine):
        """Test population initialization."""
        param_ranges = {
            "threshold": (0.01, 0.1, "float"),
            "period": (5, 30, "int"),
        }
        
        population = engine.initialize_population(param_ranges)
        
        assert len(population) == 10
        
        for individual in population:
            assert "threshold" in individual.params
            assert "period" in individual.params
            assert 0.01 <= individual.params["threshold"] <= 0.1
            assert 5 <= individual.params["period"] <= 30
    
    def test_evolution(self, engine):
        """Test evolution process."""
        # Simple evaluation function
        def evaluate(params):
            # Prefer higher threshold and middle period
            return params["threshold"] * 10 - abs(params["period"] - 15) * 0.1
        
        engine.initialize_population({
            "threshold": (0.01, 0.1, "float"),
            "period": (5, 30, "int"),
        })
        
        best = engine.evolve(evaluate, generations=3)
        
        assert best is not None
        assert best.fitness > 0
    
    def test_tournament_selection(self, engine):
        """Test tournament selection."""
        engine.population = [
            Individual(params={"x": 1}, fitness=0.1),
            Individual(params={"x": 2}, fitness=0.5),
            Individual(params={"x": 3}, fitness=0.9),
        ]
        
        # Run multiple selections
        selected = [engine._tournament_selection() for _ in range(10)]
        
        # Higher fitness individuals should be selected more often
        fitnesses = [s.fitness for s in selected]
        assert np.mean(fitnesses) > 0.1  # Should be biased toward higher fitness
    
    def test_crossover(self, engine):
        """Test crossover operation."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"a": 10, "b": 20, "c": 30}
        
        child = engine._crossover(params1, params2)
        
        # Child should have values from both parents
        assert child["a"] in [1, 10]
        assert child["b"] in [2, 20]
        assert child["c"] in [3, 30]
    
    def test_mutation(self, engine):
        """Test mutation operation."""
        engine._adaptive_mutation_rate = 1.0  # Force mutation
        engine.config.param_ranges = {
            "threshold": (0.01, 0.1, "float"),
        }
        
        params = {"threshold": 0.05}
        mutated = engine._mutate(params)
        
        # Should be mutated (but still in range)
        assert 0.01 <= mutated["threshold"] <= 0.1
    
    def test_adaptive_mutation(self, engine):
        """Test adaptive mutation rate."""
        initial_rate = engine._adaptive_mutation_rate
        
        # Simulate stagnation
        engine._stagnation_count = 5
        engine._update_adaptive_mutation()
        
        assert engine._adaptive_mutation_rate > initial_rate
    
    def test_statistics(self, engine):
        """Test statistics retrieval."""
        engine.initialize_population({
            "threshold": (0.01, 0.1, "float"),
        })
        
        stats = engine.get_statistics()
        
        assert "generation" in stats
        assert "population_size" in stats
    
    def test_best_params(self, engine):
        """Test best params retrieval."""
        engine.best_individual = Individual(
            params={"test": 1},
            fitness=0.9,
        )
        
        best_params = engine.get_best_params()
        
        assert best_params["test"] == 1


class TestParamRanges:
    """Tests for parameter range creation."""
    
    def test_momentum_params(self):
        """Test momentum parameter ranges."""
        ranges = create_param_ranges("momentum")
        
        assert "lookback_period" in ranges
        assert "momentum_threshold" in ranges
        assert "stop_loss_pct" in ranges
    
    def test_mean_reversion_params(self):
        """Test mean reversion parameter ranges."""
        ranges = create_param_ranges("mean_reversion")
        
        assert "lookback_period" in ranges
        assert "z_score_threshold" in ranges
    
    def test_breakout_params(self):
        """Test breakout parameter ranges."""
        ranges = create_param_ranges("breakout")
        
        assert "lookback_period" in ranges
        assert "breakout_threshold" in ranges
    
    def test_unknown_params(self):
        """Test unknown strategy type."""
        ranges = create_param_ranges("unknown")
        
        assert ranges == {}


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
