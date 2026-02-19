# src/core/execution/orderbook_simulator.py
"""
Order Book Simulator
====================
Estimates market impact through order book modeling:
- Volume profile analysis
- Queue position estimation
- Adverse selection adjustment

Author: AI Trading System
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """Price level in the order book."""
    price: float
    quantity: float
    order_count: int = 0
    
    def __repr__(self):
        return f"PriceLevel(price={self.price:.4f}, qty={self.quantity:.4f}, orders={self.order_count})"


@dataclass
class OrderBookSnapshot:
    """Complete order book state."""
    timestamp: datetime
    symbol: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_levels: List[PriceLevel] = field(default_factory=list)
    ask_levels: List[PriceLevel] = field(default_factory=list)
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    imbalance: float = 0.0  # Bid/ask volume imbalance
    
    @property
    def depth(self) -> int:
        """Number of price levels."""
        return max(len(self.bid_levels), len(self.ask_levels))
    
    @property
    def mid_bid_ask(self) -> float:
        """Average of bid and ask."""
        return (self.best_bid + self.best_ask) / 2 if self.best_bid > 0 and self.best_ask > 0 else 0
    
    @property
    def book_depth_5pct(self) -> float:
        """Book depth within 5% of mid."""
        threshold = self.mid_price * 1.05
        bid_depth = sum(l.quantity for l in self.bid_levels if l.price >= self.mid_price * 0.95)
        ask_depth = sum(l.quantity for l in self.ask_levels if l.price <= threshold)
        return bid_depth + ask_depth


@dataclass
class MarketImpactEstimate:
    """Market impact estimation result."""
    immediate_impact: float = 0.0  # Price impact immediately after order
    permanent_impact: float = 0.0  # Permanent price impact
    temporary_impact: float = 0.0  # Temporary/volatile impact
    queue_position: int = 0  # Position in queue
    expected_fill_time: float = 0.0  # Expected time to fill
    fill_probability: float = 0.0  # Probability of fill
    adverse_selection_risk: float = 0.0  # Risk of adverse selection
    
    @property
    def total_impact(self) -> float:
        """Total expected impact."""
        return self.immediate_impact + self.temporary_impact


class OrderBookSimulator:
    """
    Order Book Simulator
    ====================
    Simulates order book dynamics to estimate market impact.
    """
    
    def __init__(
        self,
        num_levels: int = 10,
        price_precision: int = 4,
        volume_precision: int = 8,
    ):
        """
        Initialize order book simulator.
        
        Args:
            num_levels: Number of price levels to track
            price_precision: Decimal precision for prices
            volume_precision: Decimal precision for volumes
        """
        self.num_levels = num_levels
        self.price_precision = price_precision
        self.volume_precision = volume_precision
        
        # Historical snapshots for analysis
        self.snapshots: deque = deque(maxlen=1000)
        
        # Model parameters
        self.impact_coefficients = {
            'alpha': 0.0001,  # Linear coefficient
            'beta': 0.5,      # Volume elasticity
            'gamma': 0.1,     # Imbalance coefficient
        }
        
    def update_order_book(
        self,
        symbol: str,
        bid_prices: List[float],
        bid_quantities: List[float],
        ask_prices: List[float],
        ask_quantities: List[float],
    ) -> OrderBookSnapshot:
        """
        Update order book with new data.
        
        Args:
            symbol: Trading symbol
            bid_prices: List of bid prices
            bid_quantities: List of bid quantities
            ask_prices: List of ask prices
            ask_quantities: List of ask quantities
            
        Returns:
            OrderBookSnapshot
        """
        # Create price levels
        bid_levels = [
            PriceLevel(price=p, quantity=q, order_count=int(q / 0.001))
            for p, q in zip(bid_prices[:self.num_levels], bid_quantities[:self.num_levels])
        ]
        
        ask_levels = [
            PriceLevel(price=p, quantity=q, order_count=int(q / 0.001))
            for p, q in zip(ask_prices[:self.num_levels], ask_quantities[:self.num_levels])
        ]
        
        # Calculate metrics
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
        
        total_bid_volume = sum(bid_quantities)
        total_ask_volume = sum(ask_quantities)
        
        # Calculate imbalance
        total_volume = total_bid_volume + total_ask_volume
        imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
        
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            imbalance=imbalance,
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def estimate_market_impact(
        self,
        order_book: OrderBookSnapshot,
        order_side: str,
        order_quantity: float,
        order_value: float,
    ) -> MarketImpactEstimate:
        """
        Estimate market impact for an order.
        
        Args:
            order_book: Current order book snapshot
            order_side: 'buy' or 'sell'
            order_quantity: Order quantity
            order_value: Estimated order value
            
        Returns:
            MarketImpactEstimate
        """
        if order_side.lower() == 'buy':
            return self._estimate_buy_impact(order_book, order_quantity, order_value)
        else:
            return self._estimate_sell_impact(order_book, order_quantity, order_value)
    
    def _estimate_buy_impact(
        self,
        order_book: OrderBookSnapshot,
        order_quantity: float,
        order_value: float,
    ) -> MarketImpactEstimate:
        """Estimate market impact for a buy order."""
        # Calculate volume to
        remaining_q crossty = order_quantity
        total_cost = 0.0
        queue_position = 0
        
        for level in order_book.ask_levels:
            if remaining_qty <= 0:
                break
                
            # Fill at this level
            filled_qty = min(remaining_qty, level.quantity)
            
            # Add to total cost (price impact)
            total_cost += filled_qty * level.price
            
            # Track queue position
            queue_position += level.order_count * (filled_qty / level.quantity)
            
            remaining_qty -= filled_qty
        
        # Calculate immediate impact (vs mid price)
        if order_quantity > 0:
            avg_fill_price = total_cost / order_quantity
            immediate_impact = (avg_fill_price - order_book.mid_price) / order_book.mid_price
        else:
            immediate_impact = 0
        
        # Permanent impact (based on order book imbalance)
        permanent_impact = self._calculate_permanent_impact(
            order_quantity,
            order_book,
            is_buy=True
        )
        
        # Temporary impact (short-lived)
        temporary_impact = immediate_impact - permanent_impact
        
        # Expected fill time (based on volume)
        fill_time = self._estimate_fill_time(
            order_quantity,
            order_book.total_ask_volume
        )
        
        # Fill probability
        fill_prob = self._estimate_fill_probability(
            order_quantity,
            order_book
        )
        
        # Adverse selection risk
        adverse_selection = self._calculate_adverse_selection_risk(
            order_book,
            is_buy=True
        )
        
        return MarketImpactEstimate(
            immediate_impact=immediate_impact * 10000,  # Convert to bps
            permanent_impact=permanent_impact * 10000,
            temporary_impact=temporary_impact * 10000,
            queue_position=int(queue_position),
            expected_fill_time=fill_time,
            fill_probability=fill_prob,
            adverse_selection_risk=adverse_selection,
        )
    
    def _estimate_sell_impact(
        self,
        order_book: OrderBookSnapshot,
        order_quantity: float,
        order_value: float,
