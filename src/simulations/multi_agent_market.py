# src/simulations/multi_agent_market.py
"""
Multi-Agent Market Simulator
============================
A realistic market environment with multiple trading agents including:
- Market Makers
- Takers (aggressive traders)
- Arbitrageurs
- Your RL Agent

This creates a realistic market microstructure for training and testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random


@dataclass
class Order:
    """Represents a single order in the market."""
    agent_id: str
    side: str  # 'bid' or 'ask'
    price: float
    quantity: float
    timestamp: int
    order_type: str = 'limit'  # 'limit', 'market', 'ioc'
    
    def __repr__(self):
        return f"Order({self.agent_id}, {self.side}, {self.price:.2f}, {self.quantity})"


@dataclass 
class Trade:
    """Represents a completed trade."""
    buyer_id: str
    seller_id: str
    price: float
    quantity: float
    timestamp: int
    
    def __repr__(self):
        return f"Trade({self.buyer_id} <- {self.seller_id}: {self.price:.2f} x {self.quantity})"


class MarketAgent:
    """Base class for market agents."""
    
    def __init__(self, agent_id: str, initial_capital: float = 10000):
        self.agent_id = agent_id
        self.capital = initial_capital
        self.position = 0.0
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        
    def reset(self):
        """Reset agent state."""
        self.position = 0.0
        self.trades = []
        self.orders = []
        
    def get_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        return self.capital + self.position * current_price - 10000
    
    def can_trade(self, side: str, price: float, quantity: float) -> bool:
        """Check if agent can execute trade."""
        if side == 'bid':
            required = price * quantity
            return self.capital >= required
        else:
            return self.position >= quantity


class MarketMaker(MarketAgent):
    """
    Market Maker Agent
    Provides liquidity by placing bid and ask orders.
    """
    
    def __init__(
        self,
        agent_id: str,
        spread: float = 0.0005,
        size: float = 1.0,
        refresh_rate: int = 100
    ):
        super().__init__(agent_id)
        self.spread = spread
        self.size = size
        self.refresh_rate = refresh_rate
        self.tick_count = 0
        
    def act(self, market_state: Dict) -> List[Order]:
        """Generate market maker orders."""
        self.tick_count += 1
        
        if self.tick_count % self.refresh_rate != 0:
            return []
        
        mid_price = market_state['mid_price']
        
        # Cancel existing orders
        self.orders = []
        
        # Place bid and ask
        bid_price = mid_price * (1 - self.spread / 2)
        ask_price = mid_price * (1 + self.spread / 2)
        
        orders = [
            Order(self.agent_id, 'bid', bid_price, self.size, self.tick_count),
            Order(self.agent_id, 'ask', ask_price, self.size, self.tick_count)
        ]
        
        self.orders = orders
        return orders
    
    def on_trade(self, trade: Trade):
        """Handle executed trade."""
        self.trades.append(trade)
        
        if trade.buyer_id == self.agent_id:
            self.capital -= trade.price * trade.quantity
            self.position += trade.quantity
        else:
            self.capital += trade.price * trade.quantity
            self.position -= trade.quantity


class Taker(MarketAgent):
    """
    Taker Agent (Aggressive Trader)
    Places market orders to capture opportunities quickly.
    """
    
    def __init__(
        self,
        agent_id: str,
        frequency: float = 0.1,
        size_range: Tuple[float, float] = (0.1, 1.0),
        direction_bias: float = 0.0
    ):
        super().__init__(agent_id)
        self.frequency = frequency
        self.size_range = size_range
        self.direction_bias = direction_bias
        self.tick_count = 0
        
    def act(self, market_state: Dict) -> List[Order]:
        """Generate taker orders."""
        self.tick_count += 1
        
        if random.random() > self.frequency:
            return []
        
        # Determine direction
        if random.random() < 0.5 + self.direction_bias * 0.5:
            side = 'bid'
        else:
            side = 'ask'
        
        # Random size
        size = random.uniform(*self.size_range)
        
        # Market order
        order = Order(
            self.agent_id,
            side,
            market_state['mid_price'],  # Will execute at current price
            size,
            self.tick_count,
            'market'
        )
        
        self.orders = [order]
        return self.orders
    
    def on_trade(self, trade: Trade):
        """Handle executed trade."""
        self.trades.append(trade)
        
        if trade.buyer_id == self.agent_id:
            self.capital -= trade.price * trade.quantity
            self.position += trade.quantity
        else:
            self.capital += trade.price * trade.quantity
            self.position -= trade.quantity


class Arbitrageur(MarketAgent):
    """
    Arbitrageur Agent
    Exploits price differences between venues or instruments.
    """
    
    def __init__(
        self,
        agent_id: str,
        threshold: float = 0.001,
        size: float = 0.5,
        secondary_price_offset: float = 0.0
    ):
        super().__init__(agent_id)
        self.threshold = threshold
        self.size = size
        self.secondary_price_offset = secondary_price_offset
        self.tick_count = 0
        
    def act(self, market_state: Dict) -> List[Order]:
        """Generate arbitrage orders."""
        self.tick_count += 1
        
        primary_price = market_state['mid_price']
        secondary_price = market_state.get('secondary_price', primary_price)
        
        # Apply offset
        secondary_price *= (1 + self.secondary_price_offset)
        
        # Check for arbitrage opportunity
        spread = (secondary_price - primary_price) / primary_price
        
        if spread > self.threshold:
            # Buy primary, sell secondary
            order = Order(
                self.agent_id,
                'bid',
                primary_price,
                self.size,
                self.tick_count,
                'market'
            )
        elif spread < -self.threshold:
            # Sell primary, buy secondary  
            order = Order(
                self.agent_id,
                'ask',
                primary_price,
                self.size,
                self.tick_count,
                'market'
            )
        else:
            return []
        
        self.orders = [order]
        return self.orders
    
    def on_trade(self, trade: Trade):
        """Handle executed trade."""
        self.trades.append(trade)


class RLAgentWrapper(MarketAgent):
    """
    Wrapper for your RL Agent to interact with the market.
    """
    
    def __init__(
        self,
        agent_id: str,
        rl_agent,
        action_space_size: int = 9
    ):
        super().__init__(agent_id)
        self.rl_agent = rl_agent
        self.action_space_size = action_space_size
        self.tick_count = 0
        self.max_position = 1.0
        
    def act(self, market_state: Dict) -> List[Order]:
        """Generate orders based on RL agent decision."""
        self.tick_count += 1
        
        # Convert market state to RL state
        state = self._convert_state(market_state)
        
        # Get action from RL agent
        action = self._get_action(state)
        
        # Convert action to order
        orders = self._action_to_orders(action, market_state)
        
        self.orders = orders
        return orders
    
    def _convert_state(self, market_state: Dict) -> np.ndarray:
        """Convert market state to RL state vector."""
        mid = market_state['mid_price']
        spread = market_state.get('spread', mid * 0.001)
        
        bid = market_state.get('bid', mid - spread/2)
        ask = market_state.get('ask', mid + spread/2)
        
        return np.array([
            bid / mid,
            ask / mid,
            market_state.get('bid_size', 1.0),
            market_state.get('ask_size', 1.0),
            self.position / self.max_position,
            self.get_pnl(mid) / 10000.0,
            market_state.get('imbalance', 0.0),
            spread / mid
        ], dtype=np.float32)
    
    def _get_action(self, state: np.ndarray) -> int:
        """Get action from RL agent."""
        # This would call your actual RL agent
        # For now, return random action
        return random.randint(0, self.action_space_size - 1)
    
    def _action_to_orders(self, action: int, market_state: Dict) -> List[Order]:
        """Convert RL action to orders."""
        action_map = {
            0: (0, 0),      # HOLD
            1: (1, 0.001),  # BUY small
            2: (1, 0.005),  # BUY medium
            3: (1, 0.01),   # BUY large
            4: (-1, 0.001), # SELL small
            5: (-1, 0.005), # SELL medium
            6: (-1, 0.01),  # SELL large
            7: (0, 0),      # FLATTEN
            8: (0, 0)       # REVERSE
        }
        
        direction, size = action_map.get(action, (0, 0))
        
        if direction == 0:
            return []
        
        side = 'bid' if direction > 0 else 'ask'
        
        return [Order(
            self.agent_id,
            side,
            market_state['mid_price'],
            size,
            self.tick_count,
            'market'
        )]
    
    def on_trade(self, trade: Trade):
        """Handle executed trade."""
        self.trades.append(trade)
        
        if trade.buyer_id == self.agent_id:
            self.capital -= trade.price * trade.quantity
            self.position += trade.quantity
        else:
            self.capital += trade.price * trade.quantity
            self.position -= trade.quantity


class Orderbook:
    """Simulated orderbook for order matching."""
    
    def __init__(self):
        self.bids: List[Tuple[float, float]] = []  # (price, quantity)
        self.asks: List[Tuple[float, float]] = []
        self.trades: List[Trade] = []
        self.tick_count = 0
        
    def add_order(self, order: Order):
        """Add order to book."""
        if order.side == 'bid':
            self.bids.append((order.price, order.quantity))
            self.bids.sort(key=lambda x: -x[0])  # Highest first
        else:
            self.asks.append((order.price, order.quantity))
            self.asks.sort(key=lambda x: x[0])  # Lowest first
        
        # Try to match
        self._match_orders()
        
    def _match_orders(self):
        """Match bid/ask orders."""
        self.bids.sort(key=lambda x: -x[0])
        self.asks.sort(key=lambda x: x[0])
        
        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            
            if best_bid[0] >= best_ask[0]:
                # Match!
                quantity = min(best_bid[1], best_ask[1])
                price = best_ask[0]  # Execute at ask price
                
                trade = Trade(
                    buyer_id='bid_taker',  # Simplified
                    seller_id='ask_taker',
                    price=price,
                    quantity=quantity,
                    timestamp=self.tick_count
                )
                
                self.trades.append(trade)
                
                # Update quantities
                self.bids[0] = (best_bid[0], best_bid[1] - quantity)
                self.asks[0] = (best_ask[0], best_ask[1] - quantity)
                
                # Remove empty levels
                if self.bids[0][1] <= 0:
                    self.bids.pop(0)
                if self.asks[0][1] <= 0:
                    self.asks.pop(0)
            else:
                break
    
    def get_state(self) -> Dict:
        """Get current orderbook state."""
        best_bid = self.bids[0] if self.bids else (0, 0)
        best_ask = self.asks[0] if self.asks else (0, 0)
        
        mid = (best_bid[0] + best_ask[0]) / 2 if best_bid[0] > 0 else 0
        spread = best_ask[0] - best_bid[0] if best_bid[0] > 0 else 0
        
        total_bid_size = sum(q for _, q in self.bids[:5])
        total_ask_size = sum(q for _, q in self.asks[:5])
        
        return {
            'bid': best_bid[0],
            'ask': best_ask[0],
            'mid_price': mid,
            'spread': spread,
            'bid_size': best_bid[1],
            'ask_size': best_ask[1],
            'imbalance': (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-8)
        }


class MultiAgentMarket:
    """
    Complete multi-agent market simulation.
    """
    
    def __init__(
        self,
        initial_price: float = 50000,
        n_market_makers: int = 2,
        n_takers: int = 3,
        n_arbitrageurs: int = 1,
        include_rl_agent: bool = True,
        rl_agent=None
    ):
        self.current_price = initial_price
        self.orderbook = Orderbook()
        self.tick_count = 0
        
        # Create agents
        self.agents: Dict[str, MarketAgent] = {}
        
        # Market makers
        for i in range(n_market_makers):
            mm = MarketMaker(
                f'mm_{i}',
                spread=0.0005 * (1 + i * 0.3),
                size=0.5 + i * 0.3,
                refresh_rate=50 + i * 20
            )
            self.agents[mm.agent_id] = mm
        
        # Takers
        for i in range(n_takers):
            taker = Taker(
                f'taker_{i}',
                frequency=0.05 + i * 0.02,
                size_range=(0.1, 0.5),
                direction_bias=random.uniform(-0.3, 0.3)
            )
            self.agents[taker.agent_id] = taker
        
        # Arbitrageurs
        for i in range(n_arbitrageurs):
            arb = Arbitrageur(
                f'arb_{i}',
                threshold=0.001,
                size=0.2,
                secondary_price_offset=random.uniform(-0.001, 0.001)
            )
            self.agents[arb.agent_id] = arb
        
        # RL Agent
        if include_rl_agent:
            self.rl_agent = RLAgentWrapper('rl_agent', rl_agent)
            self.agents['rl_agent'] = self.rl_agent
        
        # Price history
        self.price_history = [initial_price]
        
    def step(self) -> Dict:
        """Execute one market step."""
        self.tick_count += 1
        
        # Random walk for fundamental price
        drift = 0.0001
        shock = np.random.normal(0, 0.001)
        self.current_price *= (1 + drift + shock)
        
        # Get orderbook state
        market_state = self.orderbook.get_state()
        market_state['mid_price'] = self.current_price
        market_state['secondary_price'] = self.current_price * random.uniform(0.999, 1.001)
        
        # Process each agent
        for agent in self.agents.values():
            orders = agent.act(market_state)
            
            for order in orders:
                # Simulate execution
                if order.order_type == 'market':
                    # Execute immediately at current price
                    if order.side == 'bid':
                        exec_price = market_state['ask']
                    else:
                        exec_price = market_state['bid']
                    
                    trade = Trade(
                        buyer_id=order.agent_id if order.side == 'bid' else 'market',
                        seller_id=order.agent_id if order.side == 'ask' else 'market',
                        price=exec_price,
                        quantity=order.quantity,
                        timestamp=self.tick_count
                    )
                    
                    agent.on_trade(trade)
                    
                    # Update agent position/capital
                    if order.side == 'bid':
                        agent.capital -= exec_price * order.quantity
                        agent.position += order.quantity
                    else:
                        agent.capital += exec_price * order.quantity
                        agent.position -= order.quantity
                else:
                    self.orderbook.add_order(order)
        
        # Record price
        self.price_history.append(self.current_price)
        
        return market_state
    
    def get_rl_agent_stats(self) -> Dict:
        """Get RL agent performance statistics."""
        if 'rl_agent' not in self.agents:
            return {}
        
        rl = self.agents['rl_agent']
        
        return {
            'capital': rl.capital,
            'position': rl.position,
            'total_pnl': rl.get_pnl(self.current_price),
            'num_trades': len(rl.trades)
        }
    
    def reset(self, initial_price: float = 50000):
        """Reset the market."""
        self.current_price = initial_price
        self.tick_count = 0
        self.price_history = [initial_price]
        self.orderbook = Orderbook()
        
        for agent in self.agents.values():
            agent.reset()
    
    def run_simulation(
        self,
        n_steps: int,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Run a full simulation.
        
        Args:
            n_steps: Number of steps to simulate
            verbose: Print progress
            
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        for step in range(n_steps):
            state = self.step()
            
            if verbose and step % 100 == 0:
                rl_stats = self.get_rl_agent_stats()
                print(f"Step {step}: Price={state['mid_price']:.2f}, "
                      f"RL PnL={rl_stats.get('total_pnl', 0):.2f}")
            
            results.append({
                'step': step,
                'price': state['mid_price'],
                'spread': state['spread'],
                'imbalance': state['imbalance'],
                **self.get_rl_agent_stats()
            })
        
        return pd.DataFrame(results)
