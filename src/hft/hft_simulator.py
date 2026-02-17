# src/hft/hft_simulator.py
"""
HFT Tick-by-Tick Simulator
==========================
A realistic high-frequency trading simulator for backtesting,
RL training, and strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


class HFTSimulator:
    """
    High-Frequency Trading simulator with realistic orderbook,
    latency, slippage, and market impact modeling.
    """
    
    def __init__(
        self,
        ticks_df: pd.DataFrame,
        latency_ms: float = 20.0,
        impact_coeff: float = 0.1,
        maker_fee: float = 0.0001,
        taker_fee: float = 0.0005
    ):
        """
        Initialize HFT Simulator.
        
        Args:
            ticks_df: DataFrame with columns:
                - bid: best bid price
                - ask: best ask price
                - bid_size: bid order size
                - ask_size: ask order size
                - timestamp: (optional) tick timestamp
            latency_ms: Simulated exchange latency in milliseconds
            impact_coeff: Market impact coefficient
            maker_fee: Maker fee (0.01% = 0.0001)
            taker_fee: Taker fee (0.05% = 0.0005)
        """
        self.ticks = ticks_df.reset_index(drop=True)
        self.latency = latency_ms / 1000.0  # Convert to seconds
        self.impact_coeff = impact_coeff
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # State
        self.index = 0
        self.position = 0.0
        self.cash = 0.0
        self.trades = []
        self.pnl_history = []
        
    def get_tick(self) -> pd.Series:
        """Get current tick data."""
        if self.index >= len(self.ticks):
            return self.ticks.iloc[-1]
        return self.ticks.iloc[self.index]
    
    def get_orderbook(self, levels: int = 5) -> Dict:
        """
        Simulate L2 orderbook.
        
        Args:
            levels: Number of price levels to simulate
            
        Returns:
            Dict with 'bids' and 'asks' as lists of [price, size]
        """
        tick = self.get_tick()
        mid = (tick['bid'] + tick['ask']) / 2
        spread = tick['ask'] - tick['bid']
        
        bids = []
        asks = []
        
        for i in range(levels):
            bid_price = mid - spread/2 - i * spread * 0.5
            ask_price = mid + spread/2 + i * spread * 0.5
            
            # Size decreases with distance from best
            bid_size = tick.get('bid_size', 1.0) * (1 - i * 0.15)
            ask_size = tick.get('ask_size', 1.0) * (1 - i * 0.15)
            
            bids.append([bid_price, max(bid_size, 0.01)])
            asks.append([ask_price, max(ask_size, 0.01)])
        
        return {'bids': bids, 'asks': asks}
    
    def execute(
        self,
        side: str,
        qty: float,
        order_type: str = "market"
    ) -> Dict:
        """
        Execute a trade with simulated slippage and market impact.
        
        Args:
            side: 'BUY' or 'SELL'
            qty: Quantity to trade
            order_type: 'market', 'limit', or 'ioc'
            
        Returns:
            Dict with execution details
        """
        tick = self.get_tick()
        timestamp = tick.get('timestamp', self.index)
        
        # Calculate base price
        if side == "BUY":
            base_price = tick['ask']
            base_size = tick.get('ask_size', 1.0)
        else:
            base_price = tick['bid']
            base_size = tick.get('bid_size', 1.0)
        
        # Market impact model: price moves against you based on order size
        impact = self.impact_coeff * qty / base_size
        executed_price = base_price * (1 + impact) if side == "BUY" else base_price * (1 - impact)
        
        # Slippage model (random component)
        slippage = np.random.normal(0, spread * 0.1) if 'ask' in tick and 'bid' in tick else 0
        executed_price += slippage
        
        # Calculate fees
        fee = self.taker_fee if order_type == "market" else self.maker_fee
        total_fee = executed_price * qty * fee
        
        # Update position and cash
        if side == "BUY":
            self.position += qty
            self.cash -= executed_price * qty + total_fee
        else:
            self.position -= qty
            self.cash += executed_price * qty - total_fee
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'side': side,
            'qty': qty,
            'price': executed_price,
            'fee': total_fee,
            'slippage': abs(executed_price - base_price),
            'index': self.index
        }
        self.trades.append(trade)
        
        # Simulate latency - advance index
        latency_ticks = int(self.latency * 1000)
        self.index += latency_ticks
        self.index = min(self.index, len(self.ticks) - 1)
        
        return trade
    
    def get_midprice(self) -> float:
        """Get current mid price."""
        tick = self.get_tick()
        return (tick['bid'] + tick['ask']) / 2
    
    def get_spread(self) -> float:
        """Get current spread."""
        tick = self.get_tick()
        return tick['ask'] - tick['bid']
    
    def get_imbalance(self) -> float:
        """Get orderbook imbalance (positive = bid-heavy, negative = ask-heavy)."""
        tick = self.get_tick()
        bid_size = tick.get('bid_size', 1.0)
        ask_size = tick.get('ask_size', 1.0)
        total = bid_size + ask_size
        if total == 0:
            return 0
        return (bid_size - ask_size) / total
    
    def get_microprice(self) -> float:
        """
        Calculate microprice (volume-weighted mid price).
        Gives more weight to trades on the bid side.
        """
        tick = self.get_tick()
        mid = (tick['bid'] + tick['ask']) / 2
        imbalance = self.get_imbalance()
        return mid + imbalance * (tick['ask'] - tick['bid']) * 0.5
    
    def step(self, action: int, qty: float = 0.001) -> Tuple[float, bool]:
        """
        Step the simulator forward.
        
        Args:
            action: -1 (sell), 0 (hold), 1 (buy)
            qty: Quantity to trade if action != 0
            
        Returns:
            (pnl, done)
        """
        pnl_before = self.get_pnl()
        
        if action != 0:
            side = "BUY" if action > 0 else "SELL"
            self.execute(side, qty)
        
        # Advance one tick
        self.index += 1
        done = self.index >= len(self.ticks) - 1
        
        pnl_after = self.get_pnl()
        self.pnl_history.append(pnl_after)
        
        return pnl_after - pnl_before, done
    
    def get_pnl(self, mark_to_market: bool = True) -> float:
        """
        Calculate current PnL.
        
        Args:
            mark_to_market: Use current market price for unrealized PnL
        """
        if mark_to_market:
            mid = self.get_midprice()
            return self.cash + self.position * mid
        return self.cash
    
    def get_stats(self) -> Dict:
        """Get simulation statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'final_position': 0,
                'total_fees': 0
            }
        
        total_fees = sum(t['fee'] for t in self.trades)
        
        return {
            'total_trades': len(self.trades),
            'total_pnl': self.get_pnl(),
            'final_position': self.position,
            'total_fees': total_fees,
            'avg_slippage': np.mean([t['slippage'] for t in self.trades]),
            'max_position': max(abs(self.position), 0),
            'winning_trades': sum(1 for t in self.trades if t['slippage'] < 0),
            'losing_trades': sum(1 for t in self.trades if t['slippage'] > 0)
        }
    
    def reset(self, initial_cash: float = 10000.0):
        """Reset the simulator."""
        self.index = 0
        self.position = 0.0
        self.cash = initial_cash
        self.trades = []
        self.pnl_history = []


class OrderbookSimulator:
    """
    Simulates full orderbook dynamics for L2 simulation.
    """
    
    def __init__(self, base_spread: float = 0.01, base_depth: float = 10.0):
        self.base_spread = base_spread
        self.base_depth = base_depth
        
    def generate_snapshot(
        self,
        mid_price: float,
        volatility: float = 0.001
    ) -> Dict:
        """
        Generate a realistic orderbook snapshot.
        
        Args:
            mid_price: Current mid price
            volatility: Price volatility factor
            
        Returns:
            Dict with orderbook data
        """
        spread = self.base_spread * (1 + np.random.exponential(0.5))
        spread = max(spread, volatility * mid_price)
        
        bids = []
        asks = []
        
        for i in range(20):
            # Price levels
            bid_price = mid_price - spread/2 - i * spread * 0.3
            ask_price = mid_price + spread/2 + i * spread * 0.3
            
            # Size with random variation
            base_size = self.base_depth * np.exp(-i * 0.1)
            bid_size = max(0.1, base_size * np.random.lognormal(0, 0.5))
            ask_size = max(0.1, base_size * np.random.lognormal(0, 0.5))
            
            bids.append([round(bid_price, 2), round(bid_size, 4)])
            asks.append([round(ask_price, 2), round(ask_size, 4)])
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'bids': bids,
            'asks': asks,
            'timestamp': pd.Timestamp.now()
        }
    
    def simulate_market_impact(
        self,
        side: str,
        quantity: float,
        orderbook: Dict
    ) -> float:
        """
        Calculate market impact for a trade.
        
        Args:
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            orderbook: Current orderbook state
            
        Returns:
            Expected price impact
        """
        levels = orderbook['asks'] if side == 'BUY' else orderbook['bids']
        
        remaining_qty = quantity
        avg_price = 0
        cum_qty = 0
        
        for price, size in levels:
            if remaining_qty <= 0:
                break
                
            fill_qty = min(remaining_qty, size)
            avg_price += price * fill_qty
            cum_qty += fill_qty
            remaining_qty -= fill_qty
        
        if cum_qty > 0:
            avg_price /= cum_qty
            mid = orderbook['mid_price']
            impact = (avg_price - mid) / mid
            return impact
        
        return 0.01  # Default 1% slippage if orderbook insufficient


def create_tick_data(
    prices: np.ndarray,
    volumes: np.ndarray
) -> pd.DataFrame:
    """
    Create tick DataFrame from price and volume arrays.
    
    Args:
        prices: Array of prices
        volumes: Array of volumes
        
    Returns:
        DataFrame with bid, ask, bid_size, ask_size
    """
    mid = prices
    spread = np.abs(np.diff(prices)) * 0.5 + 0.01
    spread = np.insert(spread, 0, spread[0])
    
    bids = mid - spread / 2
    asks = mid + spread / 2
    
    bid_sizes = volumes * np.random.uniform(0.8, 1.2, len(volumes))
    ask_sizes = volumes * np.random.uniform(0.8, 1.2, len(volumes))
    
    return pd.DataFrame({
        'bid': bids,
        'ask': asks,
        'bid_size': bid_sizes,
        'ask_size': ask_sizes,
        'timestamp': pd.date_range(start='2024-01-01', periods=len(prices), freq='1ms')
    })
