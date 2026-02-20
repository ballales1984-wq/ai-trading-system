# tests/test_evolution.py
"""
Test Suite for Evolution Engine
================================
Tests for genetic algorithm-based parameter optimization.
"""

import pytest
import tempfile
import os
import numpy as np

from src.automl.evolution import (
    EvolutionEngine,
    EvolutionConfig,
    Individual,
    StrategyOptimizer,
)


# Fixtures
@pytest.fixture
def evolution_config():
    """Create evolution config."""
    return EvolutionConfig(
        population_size=10,
        elite_size=2,
        mutation_rate=0.1,
        crossover_rate=0.7,
        generations=5,
        seed=42,
    )


@pytest.fixture
def evolution_engine(evolution_config):
    """Create evolution engine."""
    engine = EvolutionEngine(evolution_config)
    engine.set_param_space("threshold", 0.001, 0.05, "float")
    engine.set_param_space("period", 5, 30, "int")
    return engine


# Test Individual
class TestIndividual:
    """Tests for Individual dataclass."""
    
    def test_individual_creation(self):
        """Test individual creation."""
        ind = Individual(
            params={"threshold": 0.02, "period": 10},
            fitness=0.5,
            generation=0,
        )
        
        assert ind.params["threshold"] == 0.02
        assert ind.params["period"] == 10
        assert ind.fitness == 0.5
    
    def test_individual_serialization(self):
        """Test individual serialization."""
        ind = Individual(
            params={"threshold": 0.02},
            fitness=0.5,
            generation=1,
            history=[0.3, 0.4, 0.5],
        )
        
        data = ind.to_dict()
        
        assert data["params"]["threshold"] == 0.02
        assert data["fitness"] == 0.5
        assert data["generation"] == 1
        assert len(data["history"]) == 3
    
    def test_individual_deserialization(self):
        """Test individual deserialization."""
        data = {
            "params": {"threshold": 0.03},
            "fitness": 0.6,
            "generation": 2,
            "history": [0.4, 0.5, 0.6],
        }
        
        ind = Individual.from_dict(data)
        
        assert ind.params["threshold"] == 0.03
        assert ind.fitness == 0.6
        assert ind.generation == 2


# Test EvolutionConfig
class TestEvolutionConfig:
    """Tests for EvolutionConfig."""
    
    def test_config_defaults(self):
        """Test default config values."""
        config = EvolutionConfig()
        
        assert config.population_size == 20
        assert config.elite_size == 4
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7
        assert config.generations == 10
    
    def test_config_custom(self):
        """Test custom config values."""
        config = EvolutionConfig(
            population_size=50,
            mutation_rate=0.2,
            seed=123,
        )
        
        assert config.population_size == 50
        assert config.mutation_rate == 0.2
        assert config.seed == 123


# Test EvolutionEngine
class TestEvolutionEngine:
    """Tests for EvolutionEngine."""
    
    def test_engine_creation(self, evolution_config):
        """Test engine creation."""
        engine = EvolutionEngine(evolution_config)
        
        assert engine.config.population_size == 10
        assert engine.generation == 0
    
    def test_param_space_definition(self, evolution_engine):
        """Test parameter space definition."""
        assert "threshold" in evolution_engine._param_space
        assert "period" in evolution_engine._param_space
        
        min_val, max_val, param_type = evolution_engine._param_space["threshold"]
        assert min_val == 0.001
        assert max_val == 0.05
        assert param_type == "float"
    
    def test_population_initialization(self, evolution_engine):
        """Test population initialization."""
        population = evolution_engine.initialize_population()
        
        assert len(population) == 10
        
        for ind in population:
            assert "threshold" in ind.params
            assert "period" in ind.params
            assert 0.001 <= ind.params["threshold"] <= 0.05
            assert 5 <= ind.params["period"] <= 30
    
    def test_evaluation(self, evolution_engine):
        """Test population evaluation."""
        evolution_engine.initialize_population()
        
        def evaluate(params):
            # Simple fitness function
            return params["threshold"] * 10 + params["period"] * 0.1
        
        evolution_engine._evaluate_population(evaluate)
        
        for ind in evolution_engine.population:
            assert ind.fitness is not None
    
    def test_tournament_selection(self, evolution_engine):
        """Test tournament selection."""
        evolution_engine.initialize_population()
        
        # Set fitness values
        for i, ind in enumerate(evolution_engine.population):
            ind.fitness = i * 0.1
        
        selected = evolution_engine._tournament_selection()
        
        assert selected is not None
        assert selected.fitness is not None
    
    def test_crossover(self, evolution_engine):
        """Test crossover operation."""
        parent1 = Individual(params={"threshold": 0.01, "period": 10})
        parent2 = Individual(params={"threshold": 0.03, "period": 20})
        
        child_params = evolution_engine._crossover(
            parent1.params, parent2.params
        )
        
        assert "threshold" in child_params
        assert "period" in child_params
        # Child should inherit from either parent
        assert child_params["threshold"] in [0.01, 0.03]
        assert child_params["period"] in [10, 20]
    
    def test_mutation(self, evolution_engine):
        """Test mutation operation."""
        original_params = {"threshold": 0.02, "period": 15}
        
        # Run mutation multiple times to check it works
        for _ in range(10):
            mutated = evolution_engine._mutate(original_params.copy())
            
            # Check bounds
            assert 0.001 <= mutated["threshold"] <= 0.05
            assert 5 <= mutated["period"] <= 30
    
    def test_evolution_run(self, evolution_engine):
        """Test full evolution run."""
        def evaluate(params):
            # Optimal: threshold=0.025, period=17
            target_threshold = 0.025
            target_period = 17
            
            threshold_score = 1 - abs(params["threshold"] - target_threshold) / 0.05
            period_score = 1 - abs(params["period"] - target_period) / 25
            
            return threshold_score * 0.5 + period_score * 0.5
        
        best = evolution_engine.evolve(evaluate, generations=5)
        
        assert best is not None
        assert best.fitness is not None
        assert best.fitness > 0
    
    def test_early_stopping(self):
        """Test early stopping."""
        config = EvolutionConfig(
            population_size=10,
            generations=20,
            early_stopping=2,
            seed=42,
        )
        
        engine = EvolutionEngine(config)
        engine.set_param_space("x", -10, 10, "float")
        
        # Fitness function with global optimum
        def evaluate(params):
            return -params["x"] ** 2  # Maximum at x=0
        
        engine.evolve(evaluate)
        
        # Should stop before 20 generations
        assert engine.generation < 20
    
    def test_diversity_calculation(self, evolution_engine):
        """Test diversity calculation."""
        evolution_engine.initialize_population()
        
        for ind in evolution_engine.population:
            ind.fitness = 0.5
        
        diversity = evolution_engine._calculate_diversity()
        
        assert 0 <= diversity <= 1
    
    def test_get_best_params(self, evolution_engine):
        """Test getting best parameters."""
        evolution_engine.initialize_population()
        
        for i, ind in enumerate(evolution_engine.population):
            ind.fitness = i * 0.1
        
        evolution_engine.population.sort(
            key=lambda x: x.fitness, reverse=True
        )
        evolution_engine.best_individual = evolution_engine.population[0]
        
        best_params = evolution_engine.get_best_params()
        
        assert best_params is not None
        assert "threshold" in best_params
    
    def test_get_top_individuals(self, evolution_engine):
        """Test getting top individuals."""
        evolution_engine.initialize_population()
        
        for i, ind in enumerate(evolution_engine.population):
            ind.fitness = i * 0.1
        
        top = evolution_engine.get_top_individuals(3)
        
        assert len(top) == 3
        # Should be sorted by fitness
        assert top[0].fitness >= top[1].fitness >= top[2].fitness
    
    def test_checkpoint_save_load(self, evolution_engine):
        """Test checkpoint save and load."""
        evolution_engine.initialize_population()
        
        for ind in evolution_engine.population:
            ind.fitness = 0.5
        
        # Save checkpoint
        evolution_engine.save_checkpoint("test_checkpoint.json")
        
        # Create new engine and load
        new_engine = EvolutionEngine(evolution_engine.config)
        new_engine.set_param_space("threshold", 0.001, 0.05, "float")
        new_engine.set_param_space("period", 5, 30, "int")
        new_engine.load_checkpoint("test_checkpoint.json")
        
        assert new_engine.generation == evolution_engine.generation
        assert len(new_engine.population) == len(evolution_engine.population)
        
        # Cleanup
        os.unlink(evolution_engine._checkpoint_dir / "test_checkpoint.json")


# Test StrategyOptimizer
class TestStrategyOptimizer:
    """Tests for StrategyOptimizer."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        class MockStrategy:
            def __init__(self, name, config):
                pass
        
        optimizer = StrategyOptimizer(
            strategy_class=MockStrategy,
            data=None,
            config={"population_size": 10},
        )
        
        assert optimizer.strategy_class == MockStrategy
        assert optimizer.engine.config.population_size == 10
    
    def test_param_definition(self):
        """Test parameter definition."""
        class MockStrategy:
            def __init__(self, name, config):
                pass
        
        optimizer = StrategyOptimizer(MockStrategy, None)
        optimizer.define_param("threshold", 0.01, 0.1, "float")
        
        assert "threshold" in optimizer.engine._param_space
    
    def test_optimization_run(self):
        """Test optimization run."""
        class MockStrategy:
            def __init__(self, name, config):
                self.params = config.get("params", {})
        
        optimizer = StrategyOptimizer(
            MockStrategy,
            None,
            config={
                "population_size": 5,
                "generations": 2,
            }
        )
        optimizer.define_param("x", -5, 5, "float")
        
        # Mock backtest
        optimizer._run_backtest = lambda s: {
            "sharpe": -abs(s.params.get("x", 0)),
            "return": 0,
            "win_rate": 0,
        }
        
        best_params = optimizer.optimize(metric="sharpe")
        
        assert best_params is not None
        assert "x" in best_params


# Integration Tests
class TestEvolutionIntegration:
    """Integration tests for evolution engine."""
    
    def test_full_optimization_workflow(self):
        """Test full optimization workflow."""
        # Create engine
        config = EvolutionConfig(
            population_size=20,
            elite_size=4,
            generations=10,
            mutation_rate=0.15,
            crossover_rate=0.8,
            seed=42,
        )
        
        engine = EvolutionEngine(config)
        
        # Define parameter space
        engine.set_param_space("a", 0, 10, "float")
        engine.set_param_space("b", -5, 5, "float")
        engine.set_param_space("c", 1, 100, "int")
        
        # Define fitness function (find maximum)
        def evaluate(params):
            # Target: a=5, b=-2, c=50
            a_error = abs(params["a"] - 5) / 10
            b_error = abs(params["b"] - (-2)) / 10
            c_error = abs(params["c"] - 50) / 99
            
            return 1 - (a_error + b_error + c_error) / 3
        
        # Run evolution
        best = engine.evolve(evaluate)
        
        assert best.fitness > 0.5  # Should find reasonable solution
        
        # Check history
        assert len(engine.history) > 0
        
        # Check convergence
        first_gen = engine.history[0]
        last_gen = engine.history[-1]
        
        # Fitness should improve
        assert last_gen["best_fitness"] >= first_gen["best_fitness"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
