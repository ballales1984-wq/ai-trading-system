# src/automl/automl_engine.py
"""
AutoML Engine for Strategy Evolution
=====================================
Automated machine learning system that evolves and optimizes
trading strategies using genetic algorithms, hyperparameter
optimization, and feature selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random
import json
from pathlib import Path


@dataclass
class StrategyGenome:
    """
    A genome representing a trading strategy configuration.
    """
    # Feature selection (binary: use/don't use)
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = False
    use_stochastic: bool = False
    use_obv: bool = False
    use_vwap: bool = False
    
    # Feature parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    bb_period: int = 20
 bb_std: float = 2.0
    
    atr_period: int = 14
    stochastic_period: int = 14
    
    # Strategy parameters
    entry_threshold: float = 0.5
    exit_threshold: float = 0.5
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    
    # Position sizing
    position_size: float = 0.1
    use_kelly: bool = False
    kelly_fraction: float = 0.25
    
    # Timeframes
    primary_tf: str = "1h"
    confirmation_tf: str = "4h"
    
    # ML model params
    model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    
    # Fitness score
    fitness: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert genome to dictionary."""
        return {
            'use_rsi': self.use_rsi,
            'use_macd': self.use_macd,
            'use_bollinger': self.use_bollinger,
            'use_atr': self.use_atr,
            'use_stochastic': self.use_stochastic,
            'use_obv': self.use_obv,
            'use_vwap': self.use_vwap,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'atr_period': self.atr_period,
            'stochastic_period': self.stochastic_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'position_size': self.position_size,
            'use_kelly': self.use_kelly,
            'kelly_fraction': self.kelly_fraction,
            'primary_tf': self.primary_tf,
            'confirmation_tf': self.confirmation_tf,
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'fitness': self.fitness
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyGenome':
        """Create genome from dictionary."""
        genome = cls()
        for key, value in data.items():
            if hasattr(genome, key):
                setattr(genome, key, value)
        return genome
    
    def mutate(self, mutation_rate: float = 0.1) -> 'StrategyGenome':
        """
        Create a mutated copy of this genome.
        
        Args:
            mutation_rate: Probability of each gene mutating
            
        Returns:
            New mutated genome
        """
        new_genome = StrategyGenome()
        
        # Copy all genes
        for key, value in self.to_dict().items():
            if key != 'fitness' and hasattr(new_genome, key):
                setattr(new_genome, key, value)
        
        # Mutate boolean flags
        if random.random() < mutation_rate:
            new_genome.use_rsi = not new_genome.use_rsi
        if random.random() < mutation_rate:
            new_genome.use_macd = not new_genome.use_macd
        if random.random() < mutation_rate:
            new_genome.use_bollinger = not new_genome.use_bollinger
        if random.random() < mutation_rate:
            new_genome.use_atr = not new_genome.use_atr
        if random.random() < mutation_rate:
            new_genome.use_stochastic = not new_genome.use_stochastic
        if random.random() < mutation_rate:
            new_genome.use_obv = not new_genome.use_obv
        if random.random() < mutation_rate:
            new_genome.use_vwap = not new_genome.use_vwap
        
        # Mutate numeric parameters
        if random.random() < mutation_rate:
            new_genome.rsi_period = random.choice([7, 14, 21, 28])
        if random.random() < mutation_rate:
            new_genome.rsi_overbought = random.uniform(60, 80)
        if random.random() < mutation_rate:
            new_genome.rsi_oversold = random.uniform(20, 40)
        if random.random() < mutation_rate:
            new_genome.macd_fast = random.choice([8, 12, 16, 20])
        if random.random() < mutation_rate:
            new_genome.macd_slow = random.choice([20, 26, 32])
        if random.random() < mutation_rate:
            new_genome.macd_signal = random.choice([7, 9, 11])
        if random.random() < mutation_rate:
            new_genome.bb_period = random.choice([15, 20, 25, 30])
        if random.random() < mutation_rate:
            new_genome.bb_std = random.uniform(1.5, 2.5)
        if random.random() < mutation_rate:
            new_genome.entry_threshold = random.uniform(0.3, 0.7)
        if random.random() < mutation_rate:
            new_genome.exit_threshold = random.uniform(0.3, 0.7)
        if random.random() < mutation_rate:
            new_genome.stop_loss_pct = random.uniform(0.01, 0.05)
        if random.random() < mutation_rate:
            new_genome.take_profit_pct = random.uniform(0.02, 0.08)
        if random.random() < mutation_rate:
            new_genome.position_size = random.uniform(0.05, 0.2)
        if random.random() < mutation_rate:
            new_genome.n_estimators = random.choice([50, 100, 200])
        if random.random() < mutation_rate:
            new_genome.max_depth = random.choice([3, 5, 7, 10])
        
        return new_genome
    
    @staticmethod
    def crossover(parent1: 'StrategyGenome', parent2: 'StrategyGenome') -> 'StrategyGenome':
        """
        Create offspring from two parents.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Child genome
        """
        dict1 = parent1.to_dict()
        dict2 = parent2.to_dict()
        
        # Remove fitness from both
        dict1.pop('fitness')
        dict2.pop('fitness')
        
        # Random crossover
        child_dict = {}
        for key in dict1.keys():
            child_dict[key] = random.choice([dict1[key], dict2[key]])
        
        return StrategyGenome.from_dict(child_dict)


class AutoMLEvolver:
    """
    Genetic algorithm-based AutoML engine for trading strategy evolution.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.1,
        tournament_size: int = 3
    ):
        """
        Initialize AutoML Evolver.
        
        Args:
            population_size: Number of strategies in population
            generations: Number of evolution generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Ratio of top performers to keep
            tournament_size: Size of tournament selection
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        
        self.population: List[StrategyGenome] = []
        self.best_genome: Optional[StrategyGenome] = None
        self.fitness_history: List[float] = []
        
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        
        # Create random genomes
        for _ in range(self.population_size):
            genome = StrategyGenome()
            
            # Randomize parameters
            genome.use_rsi = random.choice([True, False])
            genome.use_macd = random.choice([True, False])
            genome.use_bollinger = random.choice([True, False])
            genome.use_atr = random.choice([True, False])
            genome.use_stochastic = random.choice([True, False])
            
            genome.rsi_period = random.choice([7, 14, 21])
            genome.macd_fast = random.choice([8, 12, 16])
            genome.macd_slow = random.choice([20, 26])
            genome.bb_period = random.choice([15, 20, 25])
            genome.entry_threshold = random.uniform(0.3, 0.7)
            genome.stop_loss_pct = random.uniform(0.01, 0.05)
            genome.take_profit_pct = random.uniform(0.02, 0.08)
            genome.position_size = random.uniform(0.05, 0.2)
            
            self.population.append(genome)
    
    def evaluate_population(
        self,
        fitness_fn: Callable[[StrategyGenome], float]
    ):
        """
        Evaluate fitness of all genomes in population.
        
        Args:
            fitness_fn: Function that takes a genome and returns fitness score
        """
        for genome in self.population:
            genome.fitness = fitness_fn(genome)
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Track best
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = self.population[0]
        
        self.fitness_history.append(self.population[0].fitness)
    
    def select_parent(self) -> StrategyGenome:
        """
        Tournament selection for parent selection.
        
        Returns:
            Selected parent genome
        """
        tournament = random.sample(self.population, self.tournament_size)
        tournament.sort(key=lambda g: g.fitness, reverse=True)
        return tournament[0]
    
    def evolve(self, fitness_fn: Callable[[StrategyGenome], float]) -> StrategyGenome:
        """
        Run the evolutionary process.
        
        Args:
            fitness_fn: Function to evaluate genome fitness
            
        Returns:
            Best genome found
        """
        # Initialize
        self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate
            self.evaluate_population(fitness_fn)
            
            print(f"Generation {generation + 1}/{self.generations}")
            print(f"  Best fitness: {self.population[0].fitness:.4f}")
            print(f"  Avg fitness: {sum(g.fitness for g in self.population) / len(self.population):.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep top performers
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            new_population.extend(self.population[:elite_count])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self.select_parent()
                    parent2 = self.select_parent()
                    child = StrategyGenome.crossover(parent1, parent2)
                    child = child.mutate(self.mutation_rate)
                else:
                    child = self.select_parent().mutate(self.mutation_rate)
                
                new_population.append(child)
            
            self.population = new_population
        
        # Final evaluation
        self.evaluate_population(fitness_fn)
        
        return self.best_genome
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on final population.
        
        Returns:
            Dictionary of feature usage frequency
        """
        if not self.population:
            return {}
        
        features = [
            'use_rsi', 'use_macd', 'use_bollinger', 'use_atr',
            'use_stochastic', 'use_obv', 'use_vwap'
        ]
        
        importance = {}
        for feature in features:
            count = sum(1 for g in self.population[:10] if getattr(g, feature))
            importance[feature] = count / 10
        
        return importance
    
    def save_results(self, filepath: str):
        """Save evolution results to file."""
        results = {
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'fitness_history': self.fitness_history,
            'feature_importance': self.get_feature_importance(),
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Bayesian optimization.
    """
    
    def __init__(
        self,
        param_space: Dict,
        n_iterations: int = 50,
        n_initial_random: int = 10
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            param_space: Dictionary of parameter names to search ranges
            n_iterations: Number of optimization iterations
            n_initial_random: Number of random samples before optimization
        """
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.n_initial_random = n_initial_random
        
        self.results: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_score: float = float('-inf')
    
    def _sample_params(self) -> Dict:
        """Sample random parameters from search space."""
        params = {}
        for name, config in self.param_space.items():
            if config['type'] == 'int':
                params[name] = random.randint(config['min'], config['max'])
            elif config['type'] == 'float':
                params[name] = random.uniform(config['min'], config['max'])
            elif config['type'] == 'choice':
                params[name] = random.choice(config['options'])
        return params
    
    def _get_score(self, params: Dict) -> float:
        """Get score for parameters (placeholder - implement actual evaluation)."""
        # This should be replaced with actual strategy evaluation
        return random.random()
    
    def optimize(self, eval_fn: Callable[[Dict], float]) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization.
        
        Args:
            eval_fn: Function that evaluates params and returns score
            
        Returns:
            (best_params, best_score)
        """
        # Random exploration
        for _ in range(self.n_initial_random):
            params = self._sample_params()
            score = eval_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        # Bayesian-inspired optimization
        for _ in range(self.n_iterations - self.n_initial_random):
            # Simple exploitation: sample near best params
            if random.random() < 0.7 and self.best_params:
                params = self.best_params.copy()
                # Perturb one parameter
                param_name = random.choice(list(self.param_space.keys()))
                config = self.param_space[param_name]
                
                if config['type'] == 'int':
                    params[param_name] = max(config['min'], min(config['max'], 
                        params[param_name] + random.choice([-1, 1])))
                elif config['type'] == 'float':
                    params[param_name] = max(config['min'], min(config['max'],
                        params[param_name] * random.uniform(0.9, 1.1)))
            else:
                params = self._sample_params()
            
            score = eval_fn(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        return self.best_params, self.best_score


def create_fitness_function(
    data: pd.DataFrame,
    initial_capital: float = 10000,
    risk_free_rate: float = 0.02
) -> Callable[[StrategyGenome], float]:
    """
    Create a fitness function for strategy evaluation.
    
    Args:
        data: Price data with OHLCV
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        
    Returns:
        Fitness function
    """
    from src.technical_analysis import TechnicalAnalysis
    
    def fitness(genome: StrategyGenome) -> float:
        """
        Evaluate strategy fitness using multiple metrics.
        """
        ta = TechnicalAnalysis(data)
        
        # Generate features based on genome
        signals = []
        
        if genome.use_rsi:
            rsi = ta.rsi(period=genome.rsi_period)
            rsi_signal = (rsi < genome.rsi_oversold).astype(int) - (rsi > genome.rsi_overbought).astype(int)
            signals.append(rsi_signal)
        
        if genome.use_macd:
            macd, signal, hist = ta.macd(
                fast=genome.macd_fast,
                slow=genome.macd_slow,
                signal=genome.macd_signal
            )
            macd_signal = (macd > signal).astype(int) - (macd < signal).astype(int)
            signals.append(macd_signal)
        
        if genome.use_bollinger:
            bb_upper, bb_middle, bb_lower = ta.bollinger_bands(
                period=genome.bb_period,
                std=genome.bb_std
            )
            bb_signal = (data['close'] < bb_lower).astype(int) - (data['close'] > bb_upper).astype(int)
            signals.append(bb_signal)
        
        if not signals:
            return 0
        
        # Combine signals
        combined = sum(signals) / len(signals)
        
        # Generate trades
        position = 0
        capital = initial_capital
        trades = []
        
        for i in range(len(data)):
            if combined.iloc[i] > genome.entry_threshold and position == 0:
                position = 1
                entry_price = data['close'].iloc[i]
                entry_idx = i
                
            elif combined.iloc[i] < -genome.exit_threshold and position == 1:
                exit_price = data['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price
                capital *= (1 + pnl * genome.position_size)
                trades.append(pnl)
                position = 0
        
        # Calculate metrics
        if not trades:
            return 0
        
        returns = np.array(trades)
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Profit factor
        gross_profit = returns[returns > 0].sum() if len(returns[returns > 0]) > 0 else 0
        gross_loss = abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Combined fitness score
        fitness = (
            0.3 * sharpe +
            0.2 * sortino +
            0.2 * win_rate +
            0.15 * profit_factor +
            0.15 * (1 - max_drawdown)
        )
        
        return fitness
    
    return fitness
