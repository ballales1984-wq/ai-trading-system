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
        # Calculate volume to cross
        remaining_qty = order_quantity
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
    ) -> MarketImpactEstimate:
        """Estimate market impact for a sell order."""
        # Calculate volume to cross
        remaining_qty = order_quantity
        total_cost = 0.0
        queue_position = 0
        
        for level in order_book.bid_levels:
            if remaining_qty <= 0:
                break
                
            filled_qty = min(remaining_qty, level.quantity)
            total_cost += filled_qty * level.price
            queue_position += level.order_count * (filled_qty / level.quantity)
            remaining_qty -= filled_qty
        
        # Calculate immediate impact
        if order_quantity > 0:
            avg_fill_price = total_cost / order_quantity
            immediate_impact = (order_book.mid_price - avg_fill_price) / order_book.mid_price
        else:
            immediate_impact = 0
        
        # Permanent impact
        permanent_impact = self._calculate_permanent_impact(
            order_quantity,
            order_book,
            is_buy=False
        )
        
        temporary_impact = immediate_impact - permanent_impact
        
        fill_time = self._estimate_fill_time(
            order_quantity,
            order_book.total_bid_volume
        )
        
        fill_prob = self._estimate_fill_probability(
            order_quantity,
            order_book
        )
        
        adverse_selection = self._calculate_adverse_selection_risk(
            order_book,
            is_buy=False
        )
        
        return MarketImpactEstimate(
            immediate_impact=immediate_impact * 10000,
            permanent_impact=permanent_impact * 10000,
            temporary_impact=temporary_impact * 10000,
            queue_position=int(queue_position),
            expected_fill_time=fill_time,
            fill_probability=fill_prob,
            adverse_selection_risk=adverse_selection,
        )
    
    def _calculate_permanent_impact(
        self,
        order_quantity: float,
        order_book: OrderBookSnapshot,
        is_buy: bool,
    ) -> float:
        """
        Calculate permanent price impact.
        
        Uses a model based on order size relative to volume and order book imbalance.
        """
        # Participation rate
        if is_buy:
            market_volume = order_book.total_ask_volume
        else:
            market_volume = order_book.total_bid_volume
            
        if market_volume <= 0:
            return 0
            
        participation = order_quantity / market_volume
        
        # Base impact from order size
        alpha = self.impact_coefficients['alpha']
        beta = self.impact_coefficients['beta']
        
        size_impact = alpha * (participation ** beta)
        
        # Adjust for order book imbalance
        gamma = self.impact_coefficients['gamma']
        imbalance_impact = gamma * order_book.imbalance * participation
        
        # For buys, positive imbalance (more bids) increases impact
        # For sells, negative imbalance (more asks) increases impact
        if is_buy:
            total_impact = size_impact + imbalance_impact
        else:
            total_impact = size_impact - imbalance_impact
            
        return total_impact
    
    def _estimate_fill_time(
        self,
        order_quantity: float,
        available_volume: float,
    ) -> float:
        """
        Estimate time to fill order.
        
        Args:
            order_quantity: Order quantity
            available_volume: Available volume at best prices
            
        Returns:
            Estimated fill time in seconds
        """
        if available_volume <= 0:
            return float('inf')
            
        # Simple model: time = quantity / (volume per second)
        # Assume 1% of available volume per second trades
        trades_per_second = available_volume * 0.01
        
        if trades_per_second <= 0:
            return float('inf')
            
        return order_quantity / trades_per_second
    
    def _estimate_fill_probability(
        self,
        order_quantity: float,
        order_book: OrderBookSnapshot,
    ) -> float:
        """
        Estimate probability of order filling.
        
        Args:
            order_quantity: Order quantity
            order_book: Current order book
            
        Returns:
            Fill probability (0-1)
        """
        # Compare order size to available liquidity
        total_liquidity = order_book.total_bid_volume + order_book.total_ask_volume
        
        if total_liquidity <= 0:
            return 0
            
        # Simple fill probability model
        fill_ratio = min(order_quantity / total_liquidity, 1.0)
        
        # Higher ratio = lower probability
        fill_prob = 1.0 - fill_ratio * 0.5
        
        return max(0, min(1, fill_prob))
    
    def _calculate_adverse_selection_risk(
        self,
        order_book: OrderBookSnapshot,
        is_buy: bool,
    ) -> float:
        """
        Calculate adverse selection risk.
        
        Measures the risk that orders are picked off by informed traders.
        
        Args:
            order_book: Current order book
            is_buy: True for buy order, False for sell
            
        Returns:
            Adverse selection risk (0-1)
        """
        # Wide spreads often indicate adverse selection risk
        spread_risk = min(order_book.spread_bps / 50, 1.0)  # Cap at 50 bps
        
        # Low depth indicates adverse selection risk
        depth_risk = 1.0 - min(order_book.book_depth_5pct / 1000, 1.0)
        
        # Order book imbalance can indicate informed trading
        # Large imbalances may signal informed orders
        imbalance_risk = abs(order_book.imbalance)
        
        # Combine factors
        total_risk = (spread_risk * 0.4 + depth_risk * 0.3 + imbalance_risk * 0.3)
        
        return min(total_risk, 1.0)
    
    def analyze_volume_profile(
        self,
        symbol: str,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze volume profile over time.
        
        Args:
            symbol: Trading symbol
            window: Number of snapshots to analyze
            
        Returns:
            Volume profile analysis
        """
        if window is None:
            window = len(self.snapshots)
            
        recent_snapshots = list(self.snapshots)[-window:]
        
        if not recent_snapshots:
            return {}
            
        # Aggregate volume at each price level
        bid_volume_by_level = {}
        ask_volume_by_level = {}
        
        for snapshot in recent_snapshots:
            for i, level in enumerate(snapshot.bid_levels):
                bid_volume_by_level[i] = bid_volume_by_level.get(i, 0) + level.quantity
                
            for i, level in enumerate(snapshot.ask_levels):
                ask_volume_by_level[i] = ask_volume_by_level.get(i, 0) + level.quantity
        
        # Calculate average volumes
        n = len(recent_snapshots)
        avg_bid_volume = {k: v / n for k, v in bid_volume_by_level.items()}
        avg_ask_volume = {k: v / n for k, v in ask_volume_by_level.items()}
        
        # Find volume nodes (high volume levels)
        max_bid_level = max(avg_bid_volume.items(), key=lambda x: x[1])
        max_ask_level = max(avg_ask_volume.items(), key=lambda x: x[1])
        
        return {
            "symbol": symbol,
            "sample_count": n,
            "avg_bid_volume_by_level": avg_bid_volume,
            "avg_ask_volume_by_level": avg_ask_volume,
            "max_bid_level": {"level": max_bid_level[0], "volume": max_bid_level[1]},
            "max_ask_level": {"level": max_ask_level[0], "volume": max_ask_level[1]},
            "avg_spread_bps": np.mean([s.spread_bps for s in recent_snapshots]),
            "avg_imbalance": np.mean([s.imbalance for s in recent_snapshots]),
        }
    
    def get_queue_position(
        self,
        order_book: OrderBookSnapshot,
        order_side: str,
        order_price: float,
        order_quantity: float,
    ) -> Dict[str, Any]:
        """
        Estimate queue position for a limit order.
        
        Args:
            order_book: Current order book
            order_side: 'buy' or 'sell'
            order_price: Limit order price
            order_quantity: Order quantity
            
        Returns:
            Queue position information
        """
        if order_side.lower() == 'buy':
            # Find position in bid queue
            queue_ahead_qty = 0
            queue_ahead_orders = 0
            
            for level in order_book.bid_levels:
                if level.price >= order_price:
                    queue_ahead_qty += level.quantity
                    queue_ahead_orders += level.order_count
                    
            target_levels = order_book.ask_levels
            price_comparison = lambda p: p <= order_price
        else:
            # Find position in ask queue
            queue_ahead_qty = 0
            queue_ahead_orders = 0
            
            for level in order_book.ask_levels:
                if level.price <= order_price:
                    queue_ahead_qty += level.quantity
                    queue_ahead_orders += level.order_count
                    
            target_levels = order_book.bid_levels
            price_comparison = lambda p: p >= order_price
        
        # Calculate queue position
        queue_position = queue_ahead_orders
        
        # Estimate time to fill based on queue
        fill_time = self._estimate_fill_time(queue_ahead_qty, 
                                              sum(l.quantity for l in target_levels[:3]))
        
        return {
            "queue_position": queue_position,
            "queue_ahead_quantity": queue_ahead_qty,
            "queue_ahead_orders": queue_ahead_orders,
            "estimated_fill_time": fill_time,
            "fill_probability": self._estimate_fill_probability(order_quantity, order_book),
        }
    
    def simulate_market_impact_path(
        self,
        order_book: OrderBookSnapshot,
        order_side: str,
        order_quantity: float,
        num_slices: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Simulate market impact path for sliced orders.
        
        Args:
            order_book: Starting order book
            order_side: 'buy' or 'sell'
            order_quantity: Total order quantity
            num_slices: Number of order slices
            
        Returns:
            List of impact estimates for each slice
        """
        slice_quantity = order_quantity / num_slices
        impacts = []
        
        # Simulate each slice
        remaining_book = order_book
        
        for i in range(num_slices):
            # Estimate impact for this slice
            estimate = self.estimate_market_impact(
                remaining_book,
                order_side,
                slice_quantity,
                slice_quantity * remaining_book.mid_price
            )
            
            impacts.append({
                "slice": i + 1,
                "quantity": slice_quantity,
                "immediate_impact_bps": estimate.immediate_impact,
                "permanent_impact_bps": estimate.permanent_impact,
                "temporary_impact_bps": estimate.temporary_impact,
                "fill_probability": estimate.fill_probability,
            })
            
            # Update book for next slice (simplified - assume partial fill)
            # In reality, this would require more sophisticated modeling
            
        return impacts


def create_order_book_from_depth(
    symbol: str,
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
) -> OrderBookSnapshot:
    """
    Factory function to create order book from depth data.
    
    Args:
        symbol: Trading symbol
        bids: List of (price, quantity) tuples for bids
        asks: List of (price, quantity) tuples for asks
        
    Returns:
        OrderBookSnapshot
    """
    simulator = OrderBookSimulator()
    
    bid_prices = [b[0] for b in bids]
    bid_quantities = [b[1] for b in bids]
    ask_prices = [a[0] for a in asks]
    ask_quantities = [a[1] for a in asks]
    
    return simulator.update_order_book(
        symbol=symbol,
        bid_prices=bid_prices,
        bid_quantities=bid_quantities,
        ask_prices=ask_prices,
        ask_quantities=ask_quantities,
    )
