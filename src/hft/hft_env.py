# src/hft/hft_env.py
"""
HFT RL Environment (Gym-style)
=============================
A Gym-compatible environment for training RL agents on HFT strategies.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from src.hft.hft_simulator import HFTSimulator


class HFTRLEnv:
    """
    Reinforcement Learning environment for high-frequency trading.
    Compatible with OpenAI Gym interface.
    """
    
    def __init__(
        self,
        simulator: HFTSimulator,
        max_position: float = 1.0,
        reward_scaling: float = 1.0,
        penalty_for_holding: float = 0.0001
    ):
        """
        Initialize HFT RL Environment.
        
        Args:
            simulator: HFTSimulator instance
            max_position: Maximum allowed position size
            reward_scaling: Scale rewards by this factor
            penalty_for_holding: Penalty per step for holding position
        """
        self.sim = simulator
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.penalty_for_holding = penalty_for_holding
        
        # State dimensions
        self.state_dim = 8
        
        # Action space: [-1, 0, 1] * quantity tiers
        self.action_space_size = 9  # -1, 0, 1 * 3 quantity tiers
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.sim.reset()
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        State includes:
        - normalized bid price
        - normalized ask price
        - bid size
        - ask size
        - position (normalized)
        - unrealized PnL
        - orderbook imbalance
        - spread (normalized)
        """
        tick = self.sim.get_tick()
        
        mid = self.sim.get_midprice()
        spread = self.sim.get_spread()
        
        # Normalize prices
        bid_norm = tick['bid'] / mid
        ask_norm = tick['ask'] / mid
        
        # Normalize sizes
        bid_size = tick.get('bid_size', 1.0)
        ask_size = tick.get('ask_size', 1.0)
        
        # Position normalized
        position_norm = self.sim.position / self.max_position
        
        # Unrealized PnL (as % of portfolio)
        pnl = self.sim.get_pnl()
        pnl_pct = pnl / 10000.0  # Assume 10k initial
        
        # Orderbook imbalance
        imbalance = self.sim.get_imbalance()
        
        # Spread normalized
        spread_norm = spread / mid
        
        return np.array([
            bid_norm,
            ask_norm,
            bid_size,
            ask_size,
            position_norm,
            pnl_pct,
            imbalance,
            spread_norm
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer in [0, 8]
                - 0: HOLD
                - 1: BUY small
                - 2: BUY medium
                - 3: BUY large
                - 4: SELL small
                - 5: SELL medium
                - 6: SELL large
                - 7: FLATTEN (close position)
                - 8: REVERSE (flip position)
                
        Returns:
            (next_state, reward, done, info)
        """
        # Decode action
        action_map = {
            0: (0, 0),      # HOLD
            1: (1, 0.001),  # BUY small
            2: (1, 0.005),  # BUY medium
            3: (1, 0.01),   # BUY large
            4: (-1, 0.001), # SELL small
            5: (-1, 0.005), # SELL medium
            6: (-1, 0.01),  # SELL large
            7: (0, 0),      # FLATTEN (handled specially)
            8: (0, 0)       # REVERSE (handled specially)
        }
        
        direction, qty = action_map.get(action, (0, 0))
        
        # Handle special actions
        if action == 7:  # FLATTEN
            if self.sim.position > 0:
                direction = -1
                qty = abs(self.sim.position)
            elif self.sim.position < 0:
                direction = 1
                qty = abs(self.sim.position)
            else:
                direction = 0
                qty = 0
                
        elif action == 8:  # REVERSE
            if self.sim.position > 0:
                direction = -1
                qty = abs(self.sim.position) * 2
            elif self.sim.position < 0:
                direction = 1
                qty = abs(self.sim.position) * 2
            else:
                direction = 0
                qty = 0
        
        # Get PnL before
        pnl_before = self.sim.get_pnl()
        
        # Execute if action is not HOLD
        if direction != 0 and qty > 0:
            side = "BUY" if direction > 0 else "SELL"
            # Limit quantity to max position
            new_position = self.sim.position + direction * qty
            if abs(new_position) > self.max_position:
                qty = self.max_position - abs(self.sim.position)
            if qty > 0:
                self.sim.execute(side, qty)
        
        # Advance simulation
        self.sim.index += 1
        done = self.sim.index >= len(self.sim.ticks) - 1
        
        # Get PnL after
        pnl_after = self.sim.get_pnl()
        reward = pnl_after - pnl_before
        
        # Add holding penalty
        if self.sim.position != 0:
            reward -= self.penalty_for_holding * abs(self.sim.position)
        
        # Scale reward
        reward *= self.reward_scaling
        
        # Get next state
        next_state = self.get_state()
        
        # Additional info
        info = {
            'position': self.sim.position,
            'cash': self.sim.cash,
            'pnl': pnl_after,
            'action': action,
            'done': done
        }
        
        return next_state, reward, done, info
    
    def get_action_meanings(self) -> list:
        """Return list of action meanings."""
        return [
            "HOLD",
            "BUY 0.001",
            "BUY 0.005",
            "BUY 0.01",
            "SELL 0.001",
            "SELL 0.005",
            "SELL 0.01",
            "FLATTEN",
            "REVERSE"
        ]


class MultiAgentMarketSimulator:
    """
    Multi-agent market simulator for realistic market dynamics.
    Includes market maker, taker, and arbitrageur agents.
    """
    
    def __init__(
        self,
        base_price: float = 50000.0,
        volatility: float = 0.001,
        n_market_makers: int = 2,
        n_takers: int = 3,
        n_arbitrageurs: int = 1
    ):
        self.base_price = base_price
        self.volatility = volatility
        
        # Agent parameters
        self.market_makers = [
            self._create_market_maker(i) for i in range(n_market_makers)
        ]
        self.takers = [
            self._create_taker(i) for i in range(n_takers)
        ]
        self.arbitrageurs = [
            self._create_arbitrageur(i) for i in range(n_arbitrageurs)
        ]
        
        # Current state
        self.current_price = base_price
        self.orderbook = self._generate_orderbook()
        
    def _create_market_maker(self, idx: int) -> Dict:
        """Create market maker agent parameters."""
        return {
            'spread': 0.0005 * (1 + idx * 0.2),
            'size': 0.5 + idx * 0.2,
            'aggression': 0.3
        }
    
    def _create_taker(self, idx: int) -> Dict:
        """Create taker agent parameters."""
        return {
            'size': 0.1 + idx * 0.05,
            'frequency': 0.1 / (idx + 1),
            'direction_bias': np.random.choice([-1, 1])
        }
    
    def _create_arbitrageur(self, idx: int) -> Dict:
        """Create arbitrageur agent parameters."""
        return {
            'threshold': 0.001,
            'size': 0.2,
            'reaction_time': 0.05
        }
    
    def _generate_orderbook(self) -> Dict:
        """Generate synthetic orderbook."""
        mid = self.current_price
        spread = mid * 0.0005
        
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = mid - spread/2 - i * spread * 0.4
            ask_price = mid + spread/2 + i * spread * 0.4
            
            bid_size = 1.0 * np.exp(-i * 0.1)
            ask_size = 1.0 * np.exp(-i * 0.1)
            
            bids.append([bid_price, bid_size])
            asks.append([ask_price, ask_size])
        
        return {'bids': bids, 'asks': asks}
    
    def step(self) -> Dict:
        """Step all agents and update market."""
        # Market maker updates
        for mm in self.market_makers:
            self._update_market_maker(mm)
        
        # Taker trades
        for taker in self.takers:
            self._update_taker(taker)
        
        # Update price with random walk
        self.current_price *= 1 + np.random.normal(0, self.volatility)
        
        # Update orderbook
        self.orderbook = self._generate_orderbook()
        
        return {
            'price': self.current_price,
            'orderbook': self.orderbook
        }
    
    def _update_market_maker(self, mm: Dict):
        """Update market maker quotes."""
        # Adjust spread based on volatility
        mm['spread'] *= 1 + np.random.normal(0, 0.1)
        
    def _update_taker(self, taker: Dict):
        """Execute taker order."""
        if np.random.random() < taker['frequency']:
            # Execute trade
            direction = taker['direction_bias'] if np.random.random() < 0.6 else np.random.choice([-1, 1])
            size = taker['size'] * np.random.uniform(0.5, 1.5)
            
            # Impact price
            impact = size * 0.001
            self.current_price *= (1 + direction * impact)


def create_hft_env(
    price_history: np.ndarray,
    volume_history: np.ndarray,
    **kwargs
) -> HFTRLEnv:
    """
    Factory function to create HFT RL environment.
    
    Args:
        price_history: Historical prices
        volume_history: Historical volumes
        **kwargs: Additional arguments for HFTSimulator
        
    Returns:
        HFTRLEnv instance
    """
    from src.hft.hft_simulator import create_tick_data
    
    ticks = create_tick_data(price_history, volume_history)
    simulator = HFTSimulator(ticks, **kwargs)
    
    return HFTRLEnv(simulator)
