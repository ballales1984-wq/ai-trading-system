# src/automl/evolution.py
"""
Evolution Engine for AutoML
===========================
Genetic algorithm-based parameter optimization for trading strategies.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import json
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    Represents an individual in the population.
    
    Each individual has a set of parameters (genes) and a fitness score.
    """
    params: Dict[str, Any]
    fitness: Optional[float] = None
    generation: int = 0
    history: List[float] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"ind_{random.randint(10000, 99999)}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "params": self.params,
            "fitness": self.fitness,
            "generation": self.generation,
            "history": self.history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Individual":
        """Create from dictionary."""
        return cls(
            params=data["params"],
            fitness=data.get("fitness"),
            generation=data.get("generation", 0),
            history=data.get("history", []),
        )


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine."""
    population_size: int = 20
    elite_size: int = 4
    mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    crossover_rate: float = 0.7
    generations: int = 10
    early_stopping: int = 3
    n_jobs: int = 1
    seed: Optional[int] = None


class EvolutionEngine:
    """
    Genetic algorithm engine for strategy parameter optimization.
    
    Features:
    - Tournament selection
    - Crossover and mutation operators
    - Elitism preservation
    - Adaptive mutation rates
    - Multi-objective optimization support
    - Checkpoint save/restore
    """
    
    def __init__(self, config: EvolutionConfig = None):
        """
        Initialize Evolution Engine.
        
        Args:
            config: Evolution configuration
        """
        self.config = config or EvolutionConfig()
        
        # Set random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Population
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []
        
        # Parameter space
        self._param_space: Dict[str, Tuple] = {}
        
        # Checkpoint directory
        self._checkpoint_dir = Path("data/evolution_checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"EvolutionEngine initialized: "
            f"pop_size={self.config.population_size}, "
            f"generations={self.config.generations}"
        )
    
    def set_param_space(
        self,
        param_name: str,
        param_min: float,
        param_max: float,
        param_type: str = "float"
    ):
        """
        Define parameter search space.
        
        Args:
            param_name: Parameter name
            param_min: Minimum value
            param_max: Maximum value
            param_type: Parameter type ('float', 'int', 'choice')
        """
        self._param_space[param_name] = (param_min, param_max, param_type)
    
    def set_param_choices(self, param_name: str, choices: List[Any]):
        """
        Define parameter choices for categorical parameters.
        
        Args:
            param_name: Parameter name
            choices: List of possible values
        """
        self._param_space[param_name] = (choices, None, "choice")
    
    def initialize_population(self) -> List[Individual]:
        """
        Initialize random population.
        
        Returns:
            List of individuals
        """
        self.population = []
        
        for _ in range(self.config.population_size):
            params = {}
            
            for param_name, (min_val, max_val, param_type) in self._param_space.items():
                if param_type == "float":
                    params[param_name] = random.uniform(min_val, max_val)
                elif param_type == "int":
                    params[param_name] = random.randint(int(min_val), int(max_val))
                elif param_type == "choice":
                    params[param_name] = random.choice(min_val)
            
            self.population.append(Individual(params=params, generation=0))
        
        logger.info(f"Initialized population of {len(self.population)} individuals")
        return self.population
    
    def evolve(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        generations: int = None,
        callback: Callable[[int, Individual], None] = None
    ) -> Individual:
        """
        Run evolution process.
        
        Args:
            evaluate_fn: Function that takes params dict and returns fitness
            generations: Number of generations (overrides config)
            callback: Optional callback after each generation
            
        Returns:
            Best individual found
        """
        generations = generations or self.config.generations
        
        # Initialize if needed
        if not self.population:
            self.initialize_population()
        
        best_fitness = float('-inf')
        no_improvement_count = 0
        
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate population
            self._evaluate_population(evaluate_fn)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best
            current_best = self.population[0]
            
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                self.best_individual = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Record history
            gen_stats = {
                "generation": gen,
                "best_fitness": current_best.fitness,
                "avg_fitness": np.mean([i.fitness for i in self.population]),
                "worst_fitness": self.population[-1].fitness,
                "diversity": self._calculate_diversity(),
            }
            self.history.append(gen_stats)
            
            logger.info(
                f"Generation {gen}: best={current_best.fitness:.4f}, "
                f"avg={gen_stats['avg_fitness']:.4f}, "
                f"diversity={gen_stats['diversity']:.4f}"
            )
            
            # Callback
            if callback:
                callback(gen, current_best)
            
            # Early stopping
            if no_improvement_count >= self.config.early_stopping:
                logger.info(f"Early stopping at generation {gen}")
                break
            
            # Create next generation
            if gen < generations - 1:
                self.population = self._create_next_generation(gen + 1)
        
        return self.best_individual
    
    def _evaluate_population(self, evaluate_fn: Callable):
        """Evaluate fitness for all individuals."""
        for individual in self.population:
            if individual.fitness is None:
                try:
                    fitness = evaluate_fn(individual.params)
                    individual.fitness = fitness
                    individual.history.append(fitness)
                except Exception as e:
                    logger.error(f"Evaluation error: {e}")
                    individual.fitness = float('-inf')
    
    def _create_next_generation(self, generation: int) -> List[Individual]:
        """Create next generation through selection, crossover, mutation."""
        next_gen = []
        
        # Elitism - preserve best individuals
        elites = self.population[:self.config.elite_size]
        for elite in elites:
            next_gen.append(Individual(
                params=elite.params.copy(),
                fitness=elite.fitness,
                generation=generation,
                history=elite.history.copy(),
            ))
        
        # Fill rest with offspring
        while len(next_gen) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child_params = self._crossover(parent1.params, parent2.params)
            else:
                child_params = parent1.params.copy()
            
            # Mutation
            child_params = self._mutate(child_params)
            
            next_gen.append(Individual(
                params=child_params,
                generation=generation,
            ))
        
        return next_gen
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(
            self.population,
            min(tournament_size, len(self.population))
        )
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform crossover between two parameter sets."""
        child_params = {}
        
        for key in params1.keys():
            if random.random() < 0.5:
                child_params[key] = params1[key]
            else:
                child_params[key] = params2[key]
        
        return child_params
    
    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate parameters."""
        mutated = params.copy()
        
        for param_name, value in mutated.items():
            if random.random() < self.config.mutation_rate:
                min_val, max_val, param_type = self._param_space[param_name]
                
                if param_type == "float":
                    # Gaussian mutation
                    delta = (max_val - min_val) * self.config.mutation_strength
                    new_value = value + random.gauss(0, delta)
                    new_value = max(min_val, min(max_val, new_value))
                    mutated[param_name] = new_value
                    
                elif param_type == "int":
                    delta = int((max_val - min_val) * self.config.mutation_strength)
                    new_value = value + random.randint(-delta, delta)
                    new_value = max(int(min_val), min(int(max_val), new_value))
                    mutated[param_name] = new_value
                    
                elif param_type == "choice":
                    mutated[param_name] = random.choice(min_val)
        
        return mutated
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        
        for i, ind1 in enumerate(self.population):
            for ind2 in self.population[i+1:]:
                dist = self._param_distance(ind1.params, ind2.params)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _param_distance(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """Calculate normalized distance between parameter sets."""
        total_dist = 0.0
        
        for key in params1.keys():
            val1, val2 = params1[key], params2[key]
            
            min_val, max_val, param_type = self._param_space[key]
            
            if param_type in ("float", "int"):
                # Normalized distance
                range_val = max_val - min_val
                if range_val > 0:
                    dist = abs(val1 - val2) / range_val
                    total_dist += dist
            elif param_type == "choice":
                # Hamming distance
                total_dist += 0.0 if val1 == val2 else 1.0
        
        return total_dist / len(params1)
    
    def save_checkpoint(self, filename: str = None):
        """Save evolution state to checkpoint."""
        filename = filename or f"checkpoint_gen_{self.generation}.json"
        filepath = self._checkpoint_dir / filename
        
        checkpoint = {
            "generation": self.generation,
            "config": {
                "population_size": self.config.population_size,
                "elite_size": self.config.elite_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
            },
            "population": [ind.to_dict() for ind in self.population],
            "best_individual": (
                self.best_individual.to_dict() if self.best_individual else None
            ),
            "history": self.history,
            "param_space": {
                k: list(v) for k, v in self._param_space.items()
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load evolution state from checkpoint."""
        filepath = self._checkpoint_dir / filename
        
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.generation = checkpoint["generation"]
        self.population = [
            Individual.from_dict(d) for d in checkpoint["population"]
        ]
        self.best_individual = (
            Individual.from_dict(checkpoint["best_individual"])
            if checkpoint["best_individual"] else None
        )
        self.history = checkpoint["history"]
        
        logger.info(f"Checkpoint loaded: {filepath}")
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found."""
        if self.best_individual:
            return self.best_individual.params.copy()
        return None
    
    def get_top_individuals(self, n: int = 5) -> List[Individual]:
        """Get top N individuals."""
        sorted_pop = sorted(
            self.population,
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_pop[:n]


class StrategyOptimizer:
    """
    High-level strategy optimizer using evolution.
    
    Provides a simple interface for optimizing trading strategy parameters.
    """
    
    def __init__(self, strategy_class, data: Any, config: Dict = None):
        """
        Initialize strategy optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Data for backtesting
            config: Optimizer configuration
        """
        self.strategy_class = strategy_class
        self.data = data
        self.config = config or {}
        
        self.engine = EvolutionEngine(
            EvolutionConfig(
                population_size=self.config.get("population_size", 20),
                generations=self.config.get("generations", 10),
                mutation_rate=self.config.get("mutation_rate", 0.1),
            )
        )
        
        self._best_params = None
    
    def define_param(self, name: str, min_val: float, max_val: float, param_type: str = "float"):
        """Define a parameter to optimize."""
        self.engine.set_param_space(name, min_val, max_val, param_type)
    
    def optimize(self, metric: str = "sharpe") -> Dict[str, Any]:
        """
        Run optimization.
        
        Args:
            metric: Metric to optimize ('sharpe', 'return', 'win_rate')
            
        Returns:
            Best parameters found
        """
        def evaluate(params: Dict) -> float:
            return self._evaluate_strategy(params, metric)
        
        best = self.engine.evolve(evaluate)
        self._best_params = best.params if best else None
        
        return self._best_params
    
    def _evaluate_strategy(self, params: Dict, metric: str) -> float:
        """Evaluate strategy with given parameters."""
        try:
            strategy = self.strategy_class(
                name="optimize",
                config={"params": params}
            )
            
            # Run backtest
            result = self._run_backtest(strategy)
            
            # Return metric
            return result.get(metric, 0)
            
        except Exception as e:
            logger.error(f"Strategy evaluation error: {e}")
            return float('-inf')
    
    def _run_backtest(self, strategy) -> Dict:
        """Run backtest with strategy."""
        # Placeholder - implement actual backtest
        return {"sharpe": 0, "return": 0, "win_rate": 0}
