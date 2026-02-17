# src/meta/multi_market_evolution.py
"""
Multi-Market Evolution Engine
===========================
An ecosystem where intelligent agents migrate between markets:
- Crypto
- Forex
- Equities
- Commodities

Based on reward gradients, volatility, liquidity, and competition.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class Market:
    """Represents a market biome."""
    name: str
    base_volatility: float
    base_liquidity: float
    base_spread: float
    is_open: bool = True
    
    def get_reward_gradient(self) -> float:
        """Get expected reward gradient."""
        return (1.0 / self.base_volatility) * self.base_liquidity
    
    def get_migration_attractiveness(self, agent_type: str) -> float:
        """Get how attractive this market is for an agent type."""
        if agent_type == "MM":
            # Market makers like low volatility, high liquidity, wide spreads
            return (self.base_spread * self.base_liquidity) / (self.base_volatility + 0.01)
        elif agent_type == "ARB":
            # Arbitrageurs like volatility and mispricing
            return self.base_volatility * (1.0 / self.base_spread)
        elif agent_type == "TAK":
            # Takers like volatility and trends
            return self.base_volatility * self.base_liquidity
        else:  # RL
            return self.base_liquidity / (self.base_volatility + 0.01)


class MarketSimulator:
    """Simple market simulator for a single market."""
    
    def __init__(
        self,
        name: str,
        volatility: float = 0.001,
        liquidity: float = 1.0,
        spread: float = 0.0005,
        initial_price: float = 100.0
    ):
        self.name = name
        self.volatility = volatility
        self.liquidity = liquidity
        self.spread = spread
        self.price = initial_price
        self.position = 0
        self.cash = 10000
        self.step_count = 0
        
    def reset(self):
        """Reset simulator."""
        self.position = 0
        self.cash = 10000
        self.step_count = 0
        
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return np.array([
            self.price,
            self.position,
            self.cash,
            self.volatility,
            self.spread,
            self.liquidity,
            np.sin(self.step_count / 50),  # Cyclical component
            np.cos(self.step_count / 50)
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step.
        
        Args:
            action: -1 (sell), 0 (hold), 1 (buy)
            
        Returns:
            (state, reward, done)
        """
        self.step_count += 1
        
        # Price movement
        drift = 0.0001
        shock = np.random.normal(0, self.volatility)
        price_change = drift + shock
        
        # If we have position, add PnL
        pnl = self.position * self.price * price_change
        
        self.price *= (1 + price_change)
        
        # Execute action
        if action != 0:
            qty = 0.1
            if action > 0:
                self.position += qty
                self.cash -= self.price * qty * (1 + self.spread)
            else:
                self.position -= qty
                self.cash += self.price * qty * (1 - self.spread)
        
        # Update volatility (regime changes)
        self.volatility *= (1 + np.random.normal(0, 0.1))
        self.volatility = max(0.0001, min(0.1, self.volatility))
        
        done = self.step_count >= 500
        
        reward = pnl + self.cash - 10000
        
        return self.get_state(), reward, done


class MigratingAgent:
    """An agent that can migrate between markets."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        params: np.ndarray = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type  # "MM", "ARB", "TAK", "RL"
        self.params = params if params is not None else np.random.randn(8)
        self.current_market = None
        self.position = 0
        self.cash = 10000
        self.fitness = 0.0
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'current_market': self.current_market,
            'fitness': self.fitness
        }


class MultiMarketEvolution:
    """
    Multi-Market Evolution Engine with migrating agents.
    """
    
    def __init__(
        self,
        markets: Dict[str, Market],
        population_size: int = 20,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.3,
        migration_rate: float = 0.2
    ):
        """
        Initialize Multi-Market Evolution.
        
        Args:
            markets: Dictionary of market name to Market
            population_size: Number of agents
            elite_ratio: Ratio of top performers to keep
            mutation_rate: Probability of mutation
            migration_rate: Probability of migration per generation
        """
        self.markets = markets
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.migration_rate = migration_rate
        
        self.agents: List[MigratingAgent] = []
        self.generation = 0
        self.fitness_history: List[float] = []
        
        # Market simulators
        self.simulators: Dict[str, MarketSimulator] = {}
        for name, market in markets.items():
            self.simulators[name] = MarketSimulator(
                name=name,
                volatility=market.base_volatility,
                liquidity=market.base_liquidity,
                spread=market.base_spread
            )
    
    def create_random_agent(self, agent_id: int) -> MigratingAgent:
        """Create a random agent."""
        agent_type = random.choice(["MM", "ARB", "TAK", "RL"])
        params = np.random.randn(8) * 0.1
        
        agent = MigratingAgent(
            agent_id=f"agent_{agent_id}",
            agent_type=agent_type,
            params=params
        )
        
        # Assign to random market
        agent.current_market = random.choice(list(self.markets.keys()))
        
        return agent
    
    def initialize_population(self):
        """Initialize population."""
        self.agents = [
            self.create_random_agent(i)
            for i in range(self.population_size)
        ]
    
    def mutate_agent(self, agent: MigratingAgent) -> MigratingAgent:
        """Mutate an agent."""
        new_agent = MigratingAgent(
            agent_id=f"{agent.agent_id}_m",
            agent_type=agent.agent_type,
            params=agent.params.copy()
        )
        new_agent.current_market = agent.current_market
        
        # Mutate params
        if random.random() < self.mutation_rate:
            new_agent.params += np.random.normal(0, 0.2, size=new_agent.params.shape)
        
        return new_agent
    
    def crossover_agents(
        self,
        agent1: MigratingAgent,
        agent2: MigratingAgent
    ) -> MigratingAgent:
        """Crossover two agents."""
        child = MigratingAgent(
            agent_id=f"child_{random.randint(0, 99999)}",
            agent_type=random.choice([agent1.agent_type, agent2.agent_type]),
            params=(agent1.params + agent2.params) / 2
        )
        child.current_market = random.choice([
            agent1.current_market,
            agent2.current_market
        ])
        
        return child
    
    def migrate_agent(self, agent: MigratingAgent) -> MigratingAgent:
        """Migrate agent to better market."""
        if random.random() > self.migration_rate:
            return agent
        
        # Calculate attractiveness of each market
        current_market_obj = self.markets.get(agent.current_market)
        
        if current_market_obj is None:
            agent.current_market = random.choice(list(self.markets.keys()))
            return agent
        
        best_market = agent.current_market
        best_attractiveness = current_market_obj.get_migration_attractiveness(agent.agent_type)
        
        for market_name, market in self.markets.items():
            attractiveness = market.get_migration_attractiveness(agent.agent_type)
            
            # Add some noise
            attractiveness += np.random.normal(0, 0.1)
            
            if attractiveness > best_attractiveness:
                best_attractiveness = attractiveness
                best_market = market_name
        
        agent.current_market = best_market
        return agent
    
    def evaluate_agent(self, agent: MigratingAgent) -> float:
        """Evaluate agent fitness in its current market."""
        sim = self.simulators.get(agent.current_market)
        
        if sim is None:
            return 0.0
        
        sim.reset()
        
        total_reward = 0
        
        for _ in range(300):
            state = sim.get_state()
            
            # Get action based on agent type
            if agent.agent_type == "MM":
                # Market maker: neutral
                action = 0
            elif agent.agent_type == "ARB":
                # Arbitrageur: mean reversion
                action = -1 if state[0] > 100 else 1
            elif agent.agent_type == "TAK":
                # Taker: momentum
                action = 1 if state[6] > 0 else -1
            else:  # RL
                # RL agent: learned policy
                action = int(np.sign(np.dot(agent.params, state)))
            
            _, reward, done = sim.step(action)
            total_reward += reward
            
            if done:
                break
        
        return total_reward
    
    def tournament_select(self) -> MigratingAgent:
        """Tournament selection."""
        tournament = random.sample(self.agents, 3)
        return max(tournament, key=lambda a: a.fitness)
    
    def evolve(self, n_generations: int = 20, verbose: bool = True) -> MigratingAgent:
        """
        Run the evolution.
        
        Args:
            n_generations: Number of generations
            verbose: Print progress
            
        Returns:
            Best agent
        """
        self.initialize_population()
        
        best_agent = None
        
        for gen in range(n_generations):
            self.generation = gen
            
            # Evaluate all agents
            for agent in self.agents:
                agent.fitness = self.evaluate_agent(agent)
            
            # Sort by fitness
            self.agents.sort(key=lambda a: a.fitness, reverse=True)
            
            if best_agent is None or self.agents[0].fitness > best_agent.fitness:
                best_agent = self.agents[0]
            
            self.fitness_history.append(self.agents[0].fitness)
            
            if verbose:
                market_dist = {}
                for agent in self.agents:
                    market_dist[agent.current_market] = market_dist.get(agent.current_market, 0) + 1
                
                print(f"Generation {gen + 1}/{n_generations}")
                print(f"  Best fitness: {self.agents[0].fitness:.2f}")
                print(f"  Best type: {self.agents[0].agent_type} @ {self.agents[0].current_market}")
                print(f"  Market distribution: {market_dist}")
            
            # Create new population
            new_agents = []
            
            # Elitism
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            new_agents.extend(self.agents[:elite_count])
            
            # Generate offspring
            while len(new_agents) < self.population_size:
                parent1 = self.tournament_select()
                parent2 = self.tournament_select()
                
                child = self.crossover_agents(parent1, parent2)
                child = self.mutate_agent(child)
                child = self.migrate_agent(child)
                
                new_agents.append(child)
            
            self.agents = new_agents
        
        return best_agent
    
    def get_ecosystem_stats(self) -> Dict:
        """Get ecosystem statistics."""
        if not self.agents:
            return {}
        
        type_dist = {}
        market_dist = {}
        
        for agent in self.agents:
            type_dist[agent.agent_type] = type_dist.get(agent.agent_type, 0) + 1
            market_dist[agent.current_market] = market_dist.get(agent.current_market, 0) + 1
        
        return {
            'population_size': len(self.agents),
            'best_fitness': max(a.fitness for a in self.agents),
            'avg_fitness': sum(a.fitness for a in self.agents) / len(self.agents),
            'agent_type_distribution': type_dist,
            'market_distribution': market_dist,
            'generations': self.generation
        }
    
    def save_results(self, filepath: str):
        """Save evolution results."""
        results = {
            'best_agent': self.agents[0].to_dict() if self.agents else None,
            'fitness_history': self.fitness_history,
            'ecosystem_stats': self.get_ecosystem_stats(),
            'markets': {
                name: {
                    'volatility': m.base_volatility,
                    'liquidity': m.base_liquidity,
                    'spread': m.base_spread
                }
                for name, m in self.markets.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def create_multi_market_ecosystem() -> MultiMarketEvolution:
    """Create a multi-market ecosystem with different market types."""
    markets = {
        'crypto': Market(
            name='crypto',
            base_volatility=0.02,
            base_liquidity=0.5,
            base_spread=0.001
        ),
        'forex': Market(
            name='forex',
            base_volatility=0.001,
            base_liquidity=1.0,
            base_spread=0.0001
        ),
        'equities': Market(
            name='equities',
            base_volatility=0.005,
            base_liquidity=0.8,
            base_spread=0.0005
        ),
        'commodities': Market(
            name='commodities',
            base_volatility=0.008,
            base_liquidity=0.6,
            base_spread=0.0008
        )
    }
    
    return MultiMarketEvolution(
        markets=markets,
        population_size=20,
        elite_ratio=0.1,
        mutation_rate=0.3,
        migration_rate=0.2
    )
