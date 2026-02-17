# src/meta/emergent_communication.py
"""
Emergent Communication Engine
==========================
Multi-agent system where agents develop their own communication
protocols, language, and coordination strategies.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CommunicationAgent:
    """Agent with communication capabilities."""
    agent_id: str
    agent_type: str  # "MM", "ARB", "TAK", "RL"
    policy: np.ndarray
    msg_encoder: np.ndarray
    msg_decoder: np.ndarray
    fitness: float = 0.0
    
    def send_message(self, state: np.ndarray) -> np.ndarray:
        """Generate message from state."""
        return np.tanh(state @ self.msg_encoder)
    
    def interpret_message(self, msg: np.ndarray) -> np.ndarray:
        """Interpret received message."""
        return np.tanh(msg @ self.msg_decoder)
    
    def act(self, state: np.ndarray, received_msg: np.ndarray = None) -> int:
        """Take action based on state and received message."""
        if received_msg is not None:
            meaning = self.interpret_message(received_msg)
            combined = state + meaning
        else:
            combined = state
        
        action_value = combined @ self.policy
        return int(np.sign(action_value))
    
    def mutate(self) -> 'CommunicationAgent':
        """Create mutated copy."""
        new_agent = CommunicationAgent(
            agent_id=f"{self.agent_id}_m",
            agent_type=self.agent_type,
            policy=self.policy.copy(),
            msg_encoder=self.msg_encoder.copy(),
            msg_decoder=self.msg_decoder.copy()
        )
        
        # Mutate with probability
        if random.random() < 0.3:
            new_agent.policy += np.random.normal(0, 0.1, size=new_agent.policy.shape)
        if random.random() < 0.3:
            new_agent.msg_encoder += np.random.normal(0, 0.1, size=new_agent.msg_encoder.shape)
        if random.random() < 0.3:
            new_agent.msg_decoder += np.random.normal(0, 0.1, size=new_agent.msg_decoder.shape)
        
        return new_agent
    
    def crossover(self, other: 'CommunicationAgent') -> 'CommunicationAgent':
        """Create child from two parents."""
        child = CommunicationAgent(
            agent_id=f"child_{random.randint(0, 99999)}",
            agent_type=random.choice([self.agent_type, other.agent_type]),
            policy=(self.policy + other.policy) / 2,
            msg_encoder=(self.msg_encoder + other.msg_encoder) / 2,
            msg_decoder=(self.msg_decoder + other.msg_decoder) / 2
        )
        return child


class CommunicationEnvironment:
    """Environment where agents communicate and interact."""
    
    def __init__(self, simulator, agents: List[CommunicationAgent]):
        self.simulator = simulator
        self.agents = agents
        self.message_history: List[Dict] = []
    
    def reset(self):
        """Reset environment."""
        self.simulator.reset()
        self.message_history = []
    
    def run_episode(self) -> List[float]:
        """Run one episode with communication."""
        self.reset()
        rewards = [0.0 for _ in self.agents]
        
        for step in range(500):
            # Get current state
            state = self.simulator.get_state()
            state_array = np.array(state)
            
            # Agents send messages
            messages = [agent.send_message(state_array) for agent in self.agents]
            
            # Aggregate messages (simple: average)
            avg_message = np.mean(messages, axis=0)
            
            # Store message history
            self.message_history.append({
                'step': step,
                'messages': [m.tolist() for m in messages],
                'avg': avg_message.tolist()
            })
            
            # Each agent acts based on state + communication
            actions = []
            for i, agent in enumerate(self.agents):
                # Agent receives all other agents' messages
                other_messages = [messages[j] for j in range(len(messages)) if j != i]
                
                if other_messages:
                    combined_msg = np.mean(other_messages, axis=0)
                else:
                    combined_msg = None
                
                action = agent.act(state_array, combined_msg)
                actions.append(action)
            
            # Environment step with all actions
            _, step_rewards, done = self.simulator.step_multi(actions)
            
            # Accumulate rewards
            for i, r in enumerate(step_rewards):
                rewards[i] += r
            
            if done:
                break
        
        return rewards
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics."""
        if not self.message_history:
            return {}
        
        # Analyze message patterns
        message_variance = []
        for entry in self.message_history[-50:]:
            msgs = np.array(entry['messages'])
            message_variance.append(np.var(msgs))
        
        return {
            'avg_message_variance': np.mean(message_variance),
            'total_communications': len(self.message_history)
        }


class CommunicationEvolution:
    """Evolution of communication protocols."""
    
    def __init__(
        self,
        env: CommunicationEnvironment,
        population_size: int = 20,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.3
    ):
        self.env = env
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        
        self.agents: List[CommunicationAgent] = []
        self.generation = 0
        self.best_fitness = float('-inf')
    
    def initialize_population(self, state_dim: int = 8, msg_dim: int = 4):
        """Initialize random population."""
        self.agents = []
        
        for i in range(self.population_size):
            agent_type = random.choice(["MM", "ARB", "TAK", "RL"])
            
            agent = CommunicationAgent(
                agent_id=f"agent_{i}",
                agent_type=agent_type,
                policy=np.random.randn(state_dim) * 0.1,
                msg_encoder=np.random.randn(state_dim, msg_dim) * 0.1,
                msg_decoder=np.random.randn(msg_dim, state_dim) * 0.1
            )
            
            self.agents.append(agent)
    
    def tournament_select(self) -> CommunicationAgent:
        """Tournament selection."""
        tournament = random.sample(self.agents, 3)
        return max(tournament, key=lambda a: a.fitness)
    
    def evolve(self, n_generations: int = 20, verbose: bool = True) -> CommunicationAgent:
        """Run evolution."""
        state_dim = 8
        msg_dim = 4
        
        self.initialize_population(state_dim, msg_dim)
        
        best_agent = None
        
        for gen in range(n_generations):
            self.generation = gen
            
            # Evaluate all agents
            for agent in self.agents:
                # Replace agent in environment
                original_agents = self.env.agents.copy()
                self.env.agents = [agent]
                
                rewards = self.env.run_episode()
                agent.fitness = np.mean(rewards)
                
                self.env.agents = original_agents
            
            # Sort by fitness
            self.agents.sort(key=lambda a: a.fitness, reverse=True)
            
            if best_agent is None or self.agents[0].fitness > best_agent.fitness:
                best_agent = self.agents[0]
            
            if verbose:
                comm_stats = self.env.get_communication_stats()
                print(f"Generation {gen + 1}/{n_generations}")
                print(f"  Best fitness: {self.agents[0].fitness:.2f}")
                print(f"  Avg message variance: {comm_stats.get('avg_message_variance', 0):.4f}")
            
            # Create new population
            new_agents = []
            
            # Elitism
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            new_agents.extend(self.agents[:elite_count])
            
            # Generate offspring
            while len(new_agents) < self.population_size:
                parent1 = self.tournament_select()
                parent2 = self.tournament_select()
                
                child = parent1.crossover(parent2)
                
                if random.random() < self.mutation_rate:
                    child = child.mutate()
                
                new_agents.append(child)
            
            self.agents = new_agents
        
        return best_agent
    
    def analyze_language(self) -> Dict:
        """Analyze emergent language patterns."""
        if not self.env.message_history:
            return {}
        
        # Look at last 100 messages
        recent = self.env.message_history[-100:]
        
        # Calculate variance across agents (lower = more agreement)
        inter_agent_variance = []
        for entry in recent:
            msgs = np.array(entry['messages'])
            inter_agent_variance.append(np.mean(np.var(msgs, axis=0)))
        
        return {
            'language_agreement': 1.0 / (1.0 + np.mean(inter_agent_variance)),
            'vocabulary_diversity': len(set(
                tuple(m) for entry in recent 
                for m in entry['messages']
            )),
            'communication_frequency': len(recent)
        }


class EmergentDeceptionEngine:
    """
    Advanced engine where agents learn to deceive, bluff, and manipulate.
    """
    
    def __init__(self, env: CommunicationEnvironment):
        self.env = env
        self.trust_scores: Dict[str, float] = {}
    
    def evaluate_deception(
        self,
        agent: CommunicationAgent,
        other_agents: List[CommunicationAgent]
    ) -> float:
        """Evaluate an agent's ability to deceive."""
        # Agent tries to send misleading messages
        rewards = []
        
        for _ in range(50):
            state = self.env.simulator.get_state()
            state_array = np.array(state)
            
            # Send potentially deceptive message
            true_msg = agent.send_message(state_array)
            
            # Add deception (flip some bits or add noise)
            deceptive_msg = true_msg + np.random.normal(0, 0.5, size=true_msg.shape)
            
            # Other agents interpret
            other_responses = []
            for other in other_agents:
                response = other.interpret_message(deceptive_msg)
                other_responses.append(response)
            
            # Agent's reward from deception
            # Higher if other agents act differently than they would with truth
            reward = np.sum(np.abs(np.array(other_responses)))
            rewards.append(reward)
        
        return np.mean(rewards)
    
    def evolve_deception(self, n_generations: int = 20) -> CommunicationAgent:
        """Evolve agents that are good at deception."""
        self.env.reset()
        
        best_agent = None
        
        for gen in range(n_generations):
            # Evaluate deception for each agent
            for agent in self.env.agents:
                others = [a for a in self.env.agents if a.agent_id != agent.agent_id]
                agent.fitness = self.evaluate_deception(agent, others)
            
            # Sort by deception ability
            self.env.agents.sort(key=lambda a: a.fitness, reverse=True)
            
            best_agent = self.env.agents[0]
            
            # Create next generation (simplified)
            # ... (similar to CommunicationEvolution)
        
        return best_agent
