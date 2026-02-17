# src/meta/meta_evolution_engine.py
"""
Meta-Evolution Engine
====================
The highest level of the quant ecosystem: simultaneous evolution of
RL agents, ML models, GP strategies, and market parameters.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class HybridAgent:
    """
    A hybrid agent combining RL, ML, and GP components.
    """
    # Component references
    rl_component: any = None
    ml_component: any = None
    gp_component: any = None
    
    # Weights for combining decisions
    weight_rl: float = 0.33
    weight_ml: float = 0.33
    weight_gp: float = 0.34
    
    # Agent metadata
    agent_id: str = ""
    generation: int = 0
    fitness: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'weight_rl': self.weight_rl,
            'weight_ml': self.weight_ml,
            'weight_gp': self.weight_gp,
            'fitness': self.fitness
        }
    
    def get_decision(self, state: np.ndarray) -> float:
        """
        Get weighted decision from all components.
        
        Returns:
            Combined action value in [-1, 1]
        """
        decisions = []
        weights = []
        
        # RL decision
        if self.rl_component is not None:
            try:
                rl_action, _ = self.rl_component.forward(state)
                decisions.append(float(rl_action.numpy().flatten()[0]))
                weights.append(self.weight_rl)
            except:
                pass
        
        # ML decision
        if self.ml_component is not None:
            try:
                ml_pred = self.ml_component.predict([state])[0]
                ml_action = float(ml_pred) if isinstance(ml_pred, (int, float)) else 0.0
                decisions.append(ml_action)
                weights.append(self.weight_ml)
            except:
                pass
        
        # GP decision
        if self.gp_component is not None:
            try:
                gp_action = self.gp_component.evaluate(state)
                decisions.append(float(gp_action))
                weights.append(self.weight_gp)
            except:
                pass
        
        if not decisions:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        return sum(d * w for d, w in zip(decisions, weights))


class SimpleRLComponent:
    """Simple RL component for testing."""
    
    def __init__(self, state_dim: int = 8, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # Simple random weights
        self.weights = np.random.randn(state_dim, hidden_dim) * 0.1
        self.output_weights = np.random.randn(hidden_dim) * 0.1
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward pass."""
        state = np.array(state).flatten()
        
        # Simple forward
        hidden = np.tanh(np.dot(state, self.weights))
        output = np.tanh(np.dot(hidden, self.output_weights))
        
        return output.reshape(1, 1), 0.0


class SimpleMLComponent:
    """Simple ML component for testing."""
    
    def __init__(self):
        self.threshold = 0.0
    
    def predict(self, states: List[np.ndarray]) -> List[float]:
        """Predict action."""
        predictions = []
        for state in states:
            state = np.array(state).flatten()
            # Simple momentum strategy
            pred = np.mean(state[:4]) - np.mean(state[4:8])
            predictions.append(pred)
        return predictions


class GPComponent:
    """
    Genetic Programming component - evolves decision trees.
    """
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.tree = self._create_random_tree(max_depth)
        
    def _create_random_tree(self, depth: int) -> Dict:
        """Create random decision tree."""
        if depth == 0:
            return {'type': 'leaf', 'value': random.choice([-1, 0, 1])}
        
        return {
            'type': 'node',
            'feature': random.randint(0, 7),
            'threshold': random.uniform(-1, 1),
            'true_branch': self._create_random_tree(depth - 1),
            'false_branch': self._create_random_tree(depth - 1)
        }
    
    def evaluate(self, state: np.ndarray) -> float:
        """Evaluate decision tree."""
        state = np.array(state).flatten()
        
        def _eval(node):
            if node['type'] == 'leaf':
                return node['value']
            
            feature_value = state[node['feature']]
            
            if feature_value > node['threshold']:
                return _eval(node['true_branch'])
            else:
                return _eval(node['false_branch'])
        
        return _eval(self.tree)
    
    def mutate(self) -> 'GPComponent':
        """Mutate the tree."""
        new_gp = GPComponent(self.max_depth)
        new_gp.tree = self._mutate_tree(self.tree, self.max_depth)
        return new_gp
    
    def _mutate_tree(self, node: Dict, depth: int) -> Dict:
        """Mutate a node."""
        if random.random() < 0.1:
            return self._create_random_tree(depth)
        
        if node['type'] == 'leaf':
            return node
        
        return {
            'type': 'node',
            'feature': node['feature'],
            'threshold': node['threshold'] + random.uniform(-0.1, 0.1),
            'true_branch': self._mutate_tree(node['true_branch'], depth - 1),
            'false_branch': self._mutate_tree(node['false_branch'], depth - 1)
        }
    
    def crossover(self, other: 'GPComponent') -> 'GPComponent':
        """Crossover with another GP."""
        new_gp = GPComponent(self.max_depth)
        new_gp.tree = self._crossover_trees(self.tree, other.tree)
        return new_gp
    
    def _crossover_trees(self, t1: Dict, t2: Dict) -> Dict:
        """Crossover two trees."""
        if random.random() < 0.5:
            return t1
        return t2


class MetaEvolutionEngine:
    """
    Meta-Evolution Engine - evolves hybrid agents across generations.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        tournament_size: int = 3
    ):
        """
        Initialize Meta-Evolution Engine.
        
        Args:
            population_size: Number of hybrid agents
            elite_ratio: Ratio of top performers to keep
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Tournament selection size
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        self.population: List[HybridAgent] = []
        self.generation = 0
        self.best_agent: Optional[HybridAgent] = None
        self.fitness_history: List[float] = []
        
    def create_hybrid_agent(
        self,
        rl_agents: List = None,
        ml_models: List = None,
        gp_strategies: List = None
    ) -> HybridAgent:
        """
        Create a new hybrid agent.
        
        Args:
            rl_agents: List of available RL agents
            ml_models: List of available ML models
            gp_strategies: List of available GP strategies
        """
        agent = HybridAgent()
        agent.agent_id = f"agent_{len(self.population)}_{datetime.now().timestamp()}"
        agent.generation = self.generation
        
        # Select components
        if rl_agents:
            agent.rl_component = random.choice(rl_agents)
        else:
            agent.rl_component = SimpleRLComponent()
            
        if ml_models:
            agent.ml_component = random.choice(ml_models)
        else:
            agent.ml_component = SimpleMLComponent()
            
        if gp_strategies:
            agent.gp_component = random.choice(gp_strategies)
        else:
            agent.gp_component = GPComponent()
        
        # Random weights
        agent.weight_rl = random.random()
        agent.weight_ml = random.random()
        agent.weight_gp = random.random()
        
        return agent
    
    def initialize_population(
        self,
        rl_agents: List = None,
        ml_models: List = None,
        gp_strategies: List = None
    ):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            agent = self.create_hybrid_agent(rl_agents, ml_models, gp_strategies)
            self.population.append(agent)
    
    def mutate_agent(self, agent: HybridAgent) -> HybridAgent:
        """
        Mutate an agent's weights.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            Mutated agent
        """
        new_agent = HybridAgent()
        new_agent.agent_id = f"{agent.agent_id}_m"
        new_agent.generation = self.generation
        
        # Copy components
        new_agent.rl_component = agent.rl_component
        new_agent.ml_component = agent.ml_component
        new_agent.gp_component = agent.gp_component
        
        # Mutate weights
        if random.random() < self.mutation_rate:
            new_agent.weight_rl = max(0, min(1, agent.weight_rl + random.uniform(-0.2, 0.2)))
        else:
            new_agent.weight_rl = agent.weight_rl
            
        if random.random() < self.mutation_rate:
            new_agent.weight_ml = max(0, min(1, agent.weight_ml + random.uniform(-0.2, 0.2)))
        else:
            new_agent.weight_ml = agent.weight_ml
            
        if random.random() < self.mutation_rate:
            new_agent.weight_gp = max(0, min(1, agent.weight_gp + random.uniform(-0.2, 0.2)))
        else:
            new_agent.weight_gp = agent.weight_gp
        
        # Normalize weights
        total = new_agent.weight_rl + new_agent.weight_ml + new_agent.weight_gp
        if total > 0:
            new_agent.weight_rl /= total
            new_agent.weight_ml /= total
            new_agent.weight_gp /= total
        
        # Mutate GP component
        if random.random() < self.mutation_rate and agent.gp_component:
            new_agent.gp_component = agent.gp_component.mutate()
        
        return new_agent
    
    def crossover_agents(
        self,
        agent1: HybridAgent,
        agent2: HybridAgent
    ) -> HybridAgent:
        """
        Create child from two parents.
        
        Args:
            agent1: First parent
            agent2: Second parent
            
        Returns:
            Child agent
        """
        child = HybridAgent()
        child.agent_id = f"child_{datetime.now().timestamp()}"
        child.generation = self.generation
        
        # Crossover weights
        child.weight_rl = random.choice([agent1.weight_rl, agent2.weight_rl])
        child.weight_ml = random.choice([agent1.weight_ml, agent2.weight_ml])
        child.weight_gp = random.choice([agent1.weight_gp, agent2.weight_gp])
        
        # Use components from parents
        child.rl_component = random.choice([agent1.rl_component, agent2.rl_component])
        child.ml_component = random.choice([agent1.ml_component, agent2.ml_component])
        
        # Crossover GP
        if agent1.gp_component and agent2.gp_component:
            if random.random() < 0.5:
                child.gp_component = agent1.gp_component.crossover(agent2.gp_component)
            else:
                child.gp_component = random.choice([agent1.gp_component, agent2.gp_component])
        else:
            child.gp_component = agent1.gp_component or agent2.gp_component or GPComponent()
        
        return child
    
    def tournament_select(self) -> HybridAgent:
        """Tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda a: a.fitness)
    
    def evaluate_population(self, fitness_fn: Callable[[HybridAgent], float]):
        """
        Evaluate fitness of all agents.
        
        Args:
            fitness_fn: Function to evaluate agent fitness
        """
        for agent in self.population:
            agent.fitness = fitness_fn(agent)
        
        # Sort by fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        
        # Track best
        if not self.best_agent or self.population[0].fitness > self.best_agent.fitness:
            self.best_agent = self.population[0]
        
        self.fitness_history.append(self.population[0].fitness)
    
    def evolve(
        self,
        fitness_fn: Callable[[HybridAgent], float],
        n_generations: int = 20,
        rl_agents: List = None,
        ml_models: List = None,
        gp_strategies: List = None,
        verbose: bool = True
    ) -> HybridAgent:
        """
        Run the meta-evolution process.
        
        Args:
            fitness_fn: Function to evaluate agent fitness
            n_generations: Number of generations
            rl_agents: Available RL agents
            ml_models: Available ML models
            gp_strategies: Available GP strategies
            verbose: Print progress
            
        Returns:
            Best hybrid agent found
        """
        # Initialize
        self.initialize_population(rl_agents, ml_models, gp_strategies)
        
        for gen in range(n_generations):
            self.generation = gen
            
            # Evaluate
            self.evaluate_population(fitness_fn)
            
            if verbose:
                print(f"Generation {gen + 1}/{n_generations}")
                print(f"  Best fitness: {self.population[0].fitness:.4f}")
                print(f"  Avg fitness: {sum(a.fitness for a in self.population) / len(self.population):.4f}")
                print(f"  Weights: RL={self.population[0].weight_rl:.2f}, "
                      f"ML={self.population[0].weight_ml:.2f}, "
                      f"GP={self.population[0].weight_gp:.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            new_population.extend(self.population[:elite_count])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self.tournament_select()
                    parent2 = self.tournament_select()
                    child = self.crossover_agents(parent1, parent2)
                else:
                    child = self.tournament_select()
                
                # Mutate
                if random.random() < self.mutation_rate:
                    child = self.mutate_agent(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        # Final evaluation
        self.evaluate_population(fitness_fn)
        
        return self.best_agent
    
    def get_population_stats(self) -> Dict:
        """Get population statistics."""
        if not self.population:
            return {}
        
        weights = np.array([
            [a.weight_rl, a.weight_ml, a.weight_gp]
            for a in self.population
        ])
        
        return {
            'size': len(self.population),
            'avg_fitness': np.mean([a.fitness for a in self.population]),
            'best_fitness': max(a.fitness for a in self.population),
            'worst_fitness': min(a.fitness for a in self.population),
            'avg_weight_rl': np.mean(weights[:, 0]),
            'avg_weight_ml': np.mean(weights[:, 1]),
            'avg_weight_gp': np.mean(weights[:, 2])
        }
    
    def save_results(self, filepath: str):
        """Save evolution results to file."""
        results = {
            'best_agent': self.best_agent.to_dict() if self.best_agent else None,
            'fitness_history': self.fitness_history,
            'population_stats': self.get_population_stats(),
            'config': {
                'population_size': self.population_size,
                'elite_ratio': self.elite_ratio,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


class CoEvolutionEngine(MetaEvolutionEngine):
    """
    Co-Evolution engine with predator-prey dynamics.
    Different agent types compete and evolve together.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.species = {
            'market_maker': [],
            'taker': [],
            'arbitrageur': [],
            'rl_agent': []
        }
    
    def add_species(self, species_type: str, agent: HybridAgent):
        """Add agent to species population."""
        if species_type in self.species:
            self.species[species_type].append(agent)
    
    def evolve_with_competition(self, fitness_fn: Callable, n_generations: int = 20):
        """Evolve with inter-species competition."""
        # This would involve more complex fitness evaluation
        # where agents compete against each other
        return self.evolve(fitness_fn, n_generations)


def create_simple_fitness_fn(simulator) -> Callable:
    """
    Create a simple fitness function using a simulator.
    
    Args:
        simulator: Market simulator
        
    Returns:
        Fitness function
    """
    def fitness(agent: HybridAgent) -> float:
        """Evaluate agent fitness."""
        simulator.reset()
        
        total_reward = 0
        n_steps = 500
        
        for _ in range(n_steps):
            state = simulator.step(0)  # Get state
            state_array = np.array([
                state.get('mid_price', 0),
                state.get('spread', 0),
                state.get('bid_size', 0),
                state.get('ask_size', 0),
                state.get('imbalance', 0),
                0, 0, 0
            ])
            
            # Get action from hybrid agent
            action = agent.get_decision(state_array)
            action_int = int(np.sign(action))
            
            # Execute step
            _, reward, done = simulator.step(action_int)
            total_reward += reward
            
            if done:
                break
        
        return total_reward
    
    return fitness
