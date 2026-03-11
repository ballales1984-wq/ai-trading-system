"""
Market Making Strategy Module
============================
Implements market making trading strategies with inventory risk management.

Author: AI Trading System
Version: 1.0.0
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np


@dataclass
class Quote:
    """Market maker quote"""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketState:
    """Current market state"""
    mid_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    volatility: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.now)


class InventoryRiskManager:
    """
    Manages inventory risk for market making.
    """
    
    def __init__(
        self,
        max_position: float = 1.0,
        max_inventory_skew: float = 0.3
    ):
        """
        Initialize inventory risk manager.
        
        Args:
            max_position: Maximum allowed position size
            max_inventory_skew: Maximum inventory skew (0-1)
        """
        self.max_position = max_position
        self.max_inventory_skew = max_inventory_skew
        self.inventory = 0.0
        self.inventory_history = deque(maxlen=1000)
    
    def update_inventory(self, delta: float):
        """Update inventory position"""
        self.inventory += delta
        self.inventory_history.append(self.inventory)
    
    def get_inventory_skew(self) -> float:
        """
        Calculate inventory skew (0 = balanced, 1 = full long, -1 = full short)
        """
        if self.max_position == 0:
            return 0.0
        return self.inventory / self.max_position
    
    def adjust_spread_for_inventory(self, base_spread: float) -> Tuple[float, float]:
        """
        Adjust bid/ask spread based on inventory.
        
        Returns:
            (adjusted_bid_spread, adjusted_ask_spread)
        """
        skew = self.get_inventory_skew()
        
        # If long inventory, make bid more attractive, ask less attractive
        # If short inventory, make ask more attractive, bid less attractive
        skew_multiplier = self.max_inventory_skew
        
        # As inventory increases (long), bid spread decreases, ask spread increases
        bid_adjustment = base_spread * (1 - skew * skew_multiplier)
        ask_adjustment = base_spread * (1 + skew * skew_multiplier)
        
        return max(bid_adjustment, 0.0001), max(ask_adjustment, 0.0001)
    
    def should_quote(self) -> bool:
        """Check if should continue quoting based on inventory"""
        return abs(self.get_inventory_skew()) < self.max_inventory_skew
    
    def get_inventory_risk_penalty(self) -> float:
        """Get penalty factor for quotes based on inventory"""
        skew = abs(self.get_inventory_skew())
        return skew * 0.5  # Max 50% penalty


class SpreadCalculator:
    """
    Calculates optimal bid-ask spreads for market making.
    """
    
    def __init__(
        self,
        min_spread: float = 0.0001,
        max_spread: float = 0.05
    ):
        """
        Initialize spread calculator.
        
        Args:
            min_spread: Minimum spread (0.01%)
            max_spread: Maximum spread (5%)
        """
        self.min_spread = min_spread
        self.max_spread = max_spread
    
    def calculate_spread(
        self,
        volatility: float,
        volume_24h: float,
        inventory_risk: float = 0.0
    ) -> float:
        """
        Calculate optimal spread based on market conditions.
        
        Args:
            volatility: Current volatility (annualized)
            volume_24h: 24h trading volume
            inventory_risk: Current inventory risk (0-1)
            
        Returns:
            Optimal spread as fraction of price
        """
        # Base spread from volatility
        # Using a simplified version of the optimal spread formula
        # Spread = 2 * sigma * sqrt(t * (gamma + inventory_risk))
        # where sigma is volatility, t is time, gamma is risk aversion
        
        # Simplified: spread proportional to volatility
        vol_spread = volatility * 0.5  # 50% of volatility
        
        # Adjust for volume (higher volume = tighter spreads)
        volume_factor = min(1.0, 1000000 / max(volume_24h, 1))
        vol_spread *= (0.5 + 0.5 * volume_factor)
        
        # Add inventory risk premium
        risk_premium = inventory_risk * 0.002  # Up to 0.2% from inventory
        
        # Total spread
        total_spread = vol_spread + risk_premium
        
        # Clamp to min/max
        return max(self.min_spread, min(self.max_spread, total_spread))
    
    def calculate_half_spread(
        self,
        volatility: float,
        volume_24h: float,
        inventory_risk: float = 0.0
    ) -> float:
        """Calculate half-spread (distance from mid to quote)"""
        return self.calculate_spread(volatility, volume_24h, inventory_risk) / 2


class MarketMaker:
    """
    Market making strategy implementation.
    """
    
    def __init__(
        self,
        symbol: str,
        base_inventory: float = 0.0,
        min_order_size: float = 0.001,
        max_position: float = 1.0,
        target_inventory: float = 0.0
    ):
        """
        Initialize market maker.
        
        Args:
            symbol: Trading symbol
            base_inventory: Starting inventory
            min_order_size: Minimum order size
            max_position: Maximum position size
            target_inventory: Target inventory level (0 = neutral)
        """
        self.symbol = symbol
        self.min_order_size = min_order_size
        self.target_inventory = target_inventory
        
        # Risk management
        self.inventory_manager = InventoryRiskManager(
            max_position=max_position,
            max_inventory_skew=0.5
        )
        self.inventory_manager.inventory = base_inventory - target_inventory
        
        # Spread calculator
        self.spread_calculator = SpreadCalculator()
        
        # State
        self.current_mid_price = 0.0
        self.last_quote_time = None
        self.quote_count = 0
        self.trade_count = 0
        self.pnl = 0.0
        
        # History
        self.quote_history: List[Quote] = []
        self.price_history: List[float] = []
    
    def update_market_state(self, market_state: MarketState):
        """Update current market state"""
        self.current_mid_price = market_state.mid_price
        self.price_history.append(market_state.mid_price)
        
        # Keep only last 1000 prices
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
    
    def calculate_volatility(self) -> float:
        """Calculate current volatility from price history"""
        if len(self.price_history) < 2:
            return 0.5  # Default 50% volatility
        
        returns = np.diff(np.log(self.price_history))
        
        if len(returns) == 0:
            return 0.5
        
        # Annualized volatility
        vol = np.std(returns) * math.sqrt(365 * 24)  # Assuming hourly data
        
        return max(vol, 0.01)  # Minimum 1% volatility
    
    def generate_quote(
        self,
        market_state: MarketState,
        order_size: Optional[float] = None
    ) -> Optional[Quote]:
        """
        Generate market making quote.
        
        Args:
            market_state: Current market state
            order_size: Order size (optional, uses default if not provided)
            
        Returns:
            Quote if should quote, None otherwise
        """
        # Update market state
        self.update_market_state(market_state)
        
        # Check if should quote
        if not self.inventory_manager.should_quote():
            return None
        
        # Calculate volatility
        volatility = self.calculate_volatility()
        
        # Get inventory risk penalty
        inventory_risk = self.inventory_manager.get_inventory_risk_penalty()
        
        # Calculate spread
        half_spread = self.spread_calculator.calculate_half_spread(
            volatility=volatility,
            volume_24h=market_state.volume_24h,
            inventory_risk=inventory_risk
        )
        
        # Adjust for inventory skew
        bid_adj, ask_adj = self.inventory_manager.adjust_spread_for_inventory(half_spread)
        
        # Calculate quotes
        mid_price = market_state.mid_price
        
        # Bid: below mid
        bid_price = mid_price * (1 - bid_adj)
        # Ask: above mid
        ask_price = mid_price * (1 + ask_adj)
        
        # Ensure bid < ask
        if bid_price >= ask_price:
            mid = (bid_price + ask_price) / 2
            bid_price = mid * 0.999
            ask_price = mid * 1.001
        
        # Order size
        size = order_size or self.min_order_size
        
        # Cap size based on inventory
        inventory_skew = abs(self.inventory_manager.get_inventory_skew())
        if inventory_skew > 0.3:
            size *= (1 - inventory_skew)
        
        # Create quote
        quote = Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=size,
            ask_size=size
        )
        
        # Store quote
        self.quote_history.append(quote)
        self.quote_count += 1
        self.last_quote_time = datetime.now()
        
        return quote
    
    def process_trade(
        self,
        side: str,
        price: float,
        size: float
    ):
        """
        Process a trade that was filled.
        
        Args:
            side: "buy" or "sell"
            price: Fill price
            size: Fill size
        """
        if side.lower() == "buy":
            self.inventory_manager.update_inventory(size)
            # PnL: we bought at price, we can sell at mid
            self.pnl += (self.current_mid_price - price) * size
        else:  # sell
            self.inventory_manager.update_inventory(-size)
            # PnL: we sold at price, we can buy back at mid
            self.pnl += (price - self.current_mid_price) * size
        
        self.trade_count += 1
    
    def get_status(self) -> Dict:
        """Get current market maker status"""
        return {
            "symbol": self.symbol,
            "inventory": self.inventory_manager.inventory,
            "inventory_skew": self.inventory_manager.get_inventory_skew(),
            "mid_price": self.current_mid_price,
            "quote_count": self.quote_count,
            "trade_count": self.trade_count,
            "pnl": self.pnl,
            "should_quote": self.inventory_manager.should_quote(),
            "volatility": self.calculate_volatility()
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if self.quote_count == 0:
            return {
                "quotes_generated": 0,
                "trades_executed": 0,
                "fill_rate": 0.0,
                "total_pnl": self.pnl,
                "avg_inventory": 0.0
            }
        
        fill_rate = self.trade_count / self.quote_count if self.quote_count > 0 else 0
        avg_inventory = np.mean(list(self.inventory_manager.inventory_history)) if len(self.inventory_manager.inventory_history) > 0 else 0
        
        return {
            "quotes_generated": self.quote_count,
            "trades_executed": self.trade_count,
            "fill_rate": fill_rate,
            "total_pnl": self.pnl,
            "avg_inventory": avg_inventory,
            "inventory_risk": abs(self.inventory_manager.get_inventory_skew())
        }


class AdaptiveMarketMaker(MarketMaker):
    """
    Adaptive market maker that adjusts parameters based on market conditions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime = "normal"  # normal, volatile, illiquid, trending
        self.regime_confidence = 0.0
    
    def detect_market_regime(self, prices: List[float]) -> Tuple[str, float]:
        """
        Detect current market regime.
        
        Returns:
            (regime, confidence)
        """
        if len(prices) < 20:
            return "normal", 0.0
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Volatility regime
        vol = np.std(returns)
        
        # Trend detection
        trend = (prices[-1] - prices[0]) / prices[0]
        
        # Volume proxy (if we had volume)
        # Using volatility as proxy for now
        liquidity = 1.0 / (vol + 0.01)
        
        # Detect regime
        if abs(trend) > 0.1:  # Strong trend
            if trend > 0:
                return "trending_up", 0.8
            else:
                return "trending_down", 0.8
        elif vol > 0.02:  # High volatility
            return "volatile", 0.7
        elif liquidity < 0.5:  # Illiquid
            return "illiquid", 0.6
        else:
            return "normal", 0.5
    
    def adjust_for_regime(self, base_spread: float) -> float:
        """Adjust spread based on market regime"""
        regime, confidence = self.detect_market_regime(self.price_history)
        
        self.regime = regime
        self.regime_confidence = confidence
        
        # Adjust spread based on regime
        regime_multipliers = {
            "normal": 1.0,
            "volatile": 1.5,  # Wider spreads in volatile markets
            "illiquid": 2.0,  # Much wider in illiquid markets
            "trending_up": 1.2,
            "trending_down": 1.2
        }
        
        return base_spread * regime_multipliers.get(regime, 1.0)


# Demo
def run_demo():
    """Demo function showing market making."""
    print("=" * 50)
    print("MARKET MAKING DEMO")
    print("=" * 50)
    
    # Create market maker
    mm = AdaptiveMarketMaker(
        symbol="BTCUSDT",
        base_inventory=0.0,
        min_order_size=0.01,
        max_position=1.0
    )
    
    # Simulate market data
    base_price = 50000.0
    
    for i in range(20):
        # Simulate price movement
        price_change = random.uniform(-0.02, 0.02)
        mid_price = base_price * (1 + price_change)
        
        # Create market state
        market_state = MarketState(
            mid_price=mid_price,
            bid_price=mid_price * 0.999,
            ask_price=mid_price * 1.001,
            bid_size=1.0,
            ask_size=1.0,
            volatility=0.6,
            volume_24h=1000000000
        )
        
        # Generate quote
        quote = mm.generate_quote(market_state, order_size=0.1)
        
        if quote:
            print(f"Quote {i+1}: Bid={quote.bid_price:.2f}, Ask={quote.ask_price:.2f}")
        
        # Simulate occasional trades
        if random.random() < 0.3:
            side = random.choice(["buy", "sell"])
            fill_price = quote.ask_price if side == "buy" else quote.bid_price
            mm.process_trade(side, fill_price, 0.1)
            print(f"  -> Filled: {side.upper()} {fill_price:.2f}")
        
        base_price = mid_price
    
    # Print performance
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    
    metrics = mm.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    status = mm.get_status()
    print(f"\nCurrent regime: {status.get('regime', 'N/A')}")


if __name__ == "__main__":
    run_demo()
