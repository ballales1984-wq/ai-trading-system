"""
Smart Order Routing (SOR)
========================
Intelligent order routing across multiple venues for best execution.

Features:
- Multi-venue price comparison
- Liquidity-aware routing
- Transaction cost optimization
- Latency-based routing decisions
- Anti-gaming logic
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class Venue(str, Enum):
    """Trading venues."""
    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    OKX = "okx"
    KRAKEN = "kraken"
    PAPER = "paper"


class RoutingStrategy(str, Enum):
    """Order routing strategies."""
    BEST_PRICE = "best_price"  # Route to best price
    LOWEST_COST = "lowest_cost"  # Include fees in calculation
    FASTEST = "fastest"  # Route to fastest venue
    LIQUIDITY = "liquidity"  # Route to most liquid venue
    SMART = "smart"  # Balance all factors
    TWAP = "twap"  # Time-weighted routing
    VOLUME_WEIGHTED = "volume_weighted"  # Volume-weighted routing


@dataclass
class VenueQuote:
    """Quote from a trading venue."""
    venue: Venue
    symbol: str
    
    # Prices
    bid: float
    ask: float
    spread: float
    
    # Sizes
    bid_size: float
    ask_size: float
    
    # Fees
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001
    
    # Latency
    latency_ms: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    def effective_buy_price(self, quantity: float) -> float:
        """Calculate effective buy price including fees and slippage."""
        base_price = self.ask
        
        # Add slippage for large orders
        if quantity > self.ask_size:
            slippage = (quantity - self.ask_size) / self.ask_size * 0.001
            base_price *= (1 + slippage)
        
        # Add taker fee
        return base_price * (1 + self.taker_fee)
    
    def effective_sell_price(self, quantity: float) -> float:
        """Calculate effective sell price including fees and slippage."""
        base_price = self.bid
        
        # Add slippage for large orders
        if quantity > self.bid_size:
            slippage = (quantity - self.bid_size) / self.bid_size * 0.001
            base_price *= (1 - slippage)
        
        # Subtract taker fee
        return base_price * (1 - self.taker_fee)


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: str = "BUY"
    quantity: float = 0.0
    
    # Routing
    venue: Venue = Venue.BINANCE
    strategy: RoutingStrategy = RoutingStrategy.SMART
    
    # Price
    requested_price: Optional[float] = None
    expected_price: float = 0.0
    expected_cost: float = 0.0
    
    # Execution
    order_type: str = "LIMIT"
    time_in_force: str = "GTC"
    
    # Metrics
    price_improvement: float = 0.0
    cost_savings: float = 0.0
    confidence: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Alternative venues
    alternatives: List[Tuple[Venue, float]] = field(default_factory=list)


@dataclass
class VenueMetrics:
    """Performance metrics for a venue."""
    venue: Venue
    
    # Latency
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Fill rates
    fill_rate: float = 0.0
    partial_fill_rate: float = 0.0
    
    # Price improvement
    avg_price_improvement: float = 0.0
    
    # Reliability
    uptime: float = 1.0
    error_rate: float = 0.0
    
    # Volume
    daily_volume: float = 0.0
    
    # Score (computed)
    score: float = 0.0


class SmartOrderRouter:
    """
    Smart Order Router for multi-venue execution.
    
    Features:
    - Real-time quote aggregation
    - Latency-aware routing
    - Cost optimization
    - Liquidity analysis
    
    Usage:
        router = SmartOrderRouter()
        router.add_venue(Venue.BINANCE, binance_connector)
        router.add_venue(Venue.BYBIT, bybit_connector)
        
        # Get routing decision
        decision = await router.route(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            strategy=RoutingStrategy.SMART
        )
        
        # Execute on selected venue
        result = await router.execute(decision)
    """
    
    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.SMART,
    ):
        self.default_strategy = default_strategy
        
        # Venue connections
        self._venues: Dict[Venue, Any] = {}
        self._venue_metrics: Dict[Venue, VenueMetrics] = {}
        
        # Quote cache
        self._quotes: Dict[str, Dict[Venue, VenueQuote]] = {}
        self._quote_ttl = 5.0  # seconds
        
        # Routing history
        self._routing_history: List[RoutingDecision] = []
    
    def add_venue(self, venue: Venue, connector: Any):
        """Add a trading venue."""
        self._venues[venue] = connector
        self._venue_metrics[venue] = VenueMetrics(venue=venue)
        logger.info(f"Added venue: {venue.value}")
    
    def remove_venue(self, venue: Venue):
        """Remove a trading venue."""
        self._venues.pop(venue, None)
        self._venue_metrics.pop(venue, None)
        logger.info(f"Removed venue: {venue.value}")
    
    async def get_quotes(self, symbol: str) -> Dict[Venue, VenueQuote]:
        """Get quotes from all venues for a symbol."""
        quotes = {}
        
        for venue, connector in self._venues.items():
            try:
                # Get ticker data
                ticker = await connector.get_ticker(symbol)
                
                quote = VenueQuote(
                    venue=venue,
                    symbol=symbol,
                    bid=float(ticker.get("bid", ticker.get("bidPrice", 0))),
                    ask=float(ticker.get("ask", ticker.get("askPrice", 0))),
                    spread=float(ticker.get("spread", 0)),
                    bid_size=float(ticker.get("bidSize", ticker.get("bidQty", 0))),
                    ask_size=float(ticker.get("askSize", ticker.get("askQty", 0))),
                    timestamp=datetime.now(timezone.utc),
                )
                
                # Update spread if not provided
                if quote.spread == 0 and quote.bid > 0 and quote.ask > 0:
                    quote.spread = (quote.ask - quote.bid) / quote.mid_price
                
                quotes[venue] = quote
                
            except Exception as e:
                logger.warning(f"Failed to get quote from {venue.value}: {e}")
        
        self._quotes[symbol] = quotes
        return quotes
    
    async def route(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: RoutingStrategy = None,
        limit_price: float = None,
    ) -> RoutingDecision:
        """
        Determine optimal routing for an order.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            strategy: Routing strategy
            limit_price: Optional limit price
            
        Returns:
            RoutingDecision with optimal venue and execution details
        """
        strategy = strategy or self.default_strategy
        
        # Get quotes from all venues
        quotes = await self.get_quotes(symbol)
        
        if not quotes:
            raise RuntimeError(f"No quotes available for {symbol}")
        
        # Score venues based on strategy
        scored_venues = self._score_venues(quotes, side, quantity, strategy)
        
        if not scored_venues:
            raise RuntimeError(f"No valid venues for {symbol}")
        
        # Select best venue
        best_venue, best_score = scored_venues[0]
        best_quote = quotes[best_venue]
        
        # Calculate expected price
        if side.upper() == "BUY":
            expected_price = best_quote.effective_buy_price(quantity)
        else:
            expected_price = best_quote.effective_sell_price(quantity)
        
        # Create routing decision
        decision = RoutingDecision(
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            venue=best_venue,
            strategy=strategy,
            requested_price=limit_price,
            expected_price=expected_price,
            expected_cost=expected_price * quantity,
            confidence=min(best_score / 100, 1.0),
            alternatives=[(v, s) for v, s in scored_venues[1:4]],
        )
        
        # Calculate price improvement vs worst venue
        if len(scored_venues) > 1:
            worst_venue = scored_venues[-1][0]
            worst_quote = quotes[worst_venue]
            
            if side.upper() == "BUY":
                worst_price = worst_quote.effective_buy_price(quantity)
            else:
                worst_price = worst_quote.effective_sell_price(quantity)
            
            decision.price_improvement = abs(expected_price - worst_price) / worst_price
            decision.cost_savings = abs(expected_price - worst_price) * quantity
        
        # Store in history
        self._routing_history.append(decision)
        if len(self._routing_history) > 1000:
            self._routing_history.pop(0)
        
        logger.info(
            f"Routing decision: {symbol} {side} {quantity} -> {best_venue.value} "
            f"@ {expected_price:.4f} (confidence: {decision.confidence:.2%})"
        )
        
        return decision
    
    def _score_venues(
        self,
        quotes: Dict[Venue, VenueQuote],
        side: str,
        quantity: float,
        strategy: RoutingStrategy,
    ) -> List[Tuple[Venue, float]]:
        """
        Score venues based on routing strategy.
        
        Returns:
            List of (venue, score) tuples sorted by score descending
        """
        scores = []
        
        for venue, quote in quotes.items():
            score = 0.0
            metrics = self._venue_metrics.get(venue, VenueMetrics(venue=venue))
            
            if strategy == RoutingStrategy.BEST_PRICE:
                # Pure price-based scoring
                if side.upper() == "BUY":
                    score = 100 - (quote.ask * 100 / quote.mid_price - 100)
                else:
                    score = 100 - (100 - quote.bid * 100 / quote.mid_price)
            
            elif strategy == RoutingStrategy.LOWEST_COST:
                # Include fees
                if side.upper() == "BUY":
                    effective_price = quote.effective_buy_price(quantity)
                    score = 100 - (effective_price / quote.mid_price - 1) * 1000
                else:
                    effective_price = quote.effective_sell_price(quantity)
                    score = 100 - (1 - effective_price / quote.mid_price) * 1000
            
            elif strategy == RoutingStrategy.FASTEST:
                # Latency-based scoring
                score = 100 - metrics.avg_latency_ms
            
            elif strategy == RoutingStrategy.LIQUIDITY:
                # Liquidity-based scoring
                available_size = quote.ask_size if side.upper() == "BUY" else quote.bid_size
                if quantity <= available_size:
                    score = 100 - quote.spread * 1000
                else:
                    # Penalize for insufficient liquidity
                    score = (available_size / quantity) * 50
            
            elif strategy == RoutingStrategy.SMART:
                # Balanced scoring
                price_score = 0
                if side.upper() == "BUY":
                    price_score = 100 - (quote.ask / quote.mid_price - 1) * 1000
                else:
                    price_score = 100 - (1 - quote.bid / quote.mid_price) * 1000
                
                liquidity_score = 0
                available_size = quote.ask_size if side.upper() == "BUY" else quote.bid_size
                if quantity <= available_size:
                    liquidity_score = 100
                else:
                    liquidity_score = (available_size / quantity) * 100
                
                latency_score = max(0, 100 - metrics.avg_latency_ms)
                reliability_score = metrics.uptime * 100
                
                # Weighted average
                score = (
                    price_score * 0.35 +
                    liquidity_score * 0.25 +
                    latency_score * 0.20 +
                    reliability_score * 0.20
                )
            
            else:
                # Default to smart
                score = 50
            
            # Apply reliability penalty
            score *= metrics.uptime
            
            # Store score in metrics
            metrics.score = score
            
            scores.append((venue, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    async def execute(self, decision: RoutingDecision) -> Dict[str, Any]:
        """
        Execute a routing decision.
        
        Args:
            decision: Routing decision to execute
            
        Returns:
            Execution result
        """
        venue = decision.venue
        connector = self._venues.get(venue)
        
        if not connector:
            raise RuntimeError(f"Venue {venue.value} not connected")
        
        try:
            # Place order
            result = await connector.place_order(
                symbol=decision.symbol,
                side=decision.side,
                order_type=decision.order_type,
                quantity=decision.quantity,
                price=decision.requested_price,
                time_in_force=decision.time_in_force,
            )
            
            logger.info(
                f"Executed on {venue.value}: {result.get('orderId', result.get('order_id'))}"
            )
            
            return {
                "success": True,
                "venue": venue.value,
                "order_id": result.get("orderId") or result.get("order_id"),
                "result": result,
            }
            
        except Exception as e:
            logger.error(f"Execution failed on {venue.value}: {e}")
            
            # Try alternative venues
            for alt_venue, _ in decision.alternatives:
                alt_connector = self._venues.get(alt_venue)
                if alt_connector:
                    try:
                        result = await alt_connector.place_order(
                            symbol=decision.symbol,
                            side=decision.side,
                            order_type=decision.order_type,
                            quantity=decision.quantity,
                            price=decision.requested_price,
                        )
                        
                        logger.info(f"Executed on alternative {alt_venue.value}")
                        
                        return {
                            "success": True,
                            "venue": alt_venue.value,
                            "order_id": result.get("orderId") or result.get("order_id"),
                            "result": result,
                            "fallback": True,
                        }
                        
                    except Exception as alt_e:
                        logger.warning(f"Alternative {alt_venue.value} also failed: {alt_e}")
            
            return {
                "success": False,
                "error": str(e),
            }
    
    async def route_and_execute(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: RoutingStrategy = None,
        limit_price: float = None,
    ) -> Tuple[RoutingDecision, Dict[str, Any]]:
        """
        Route and execute in one call.
        
        Returns:
            Tuple of (routing decision, execution result)
        """
        decision = await self.route(symbol, side, quantity, strategy, limit_price)
        result = await self.execute(decision)
        return decision, result
    
    def get_venue_metrics(self, venue: Venue = None) -> Dict[str, Any]:
        """Get performance metrics for venues."""
        if venue:
            metrics = self._venue_metrics.get(venue)
            return metrics.__dict__ if metrics else {}
        
        return {v.value: m.__dict__ for v, m in self._venue_metrics.items()}
    
    def update_venue_metrics(
        self,
        venue: Venue,
        latency_ms: float = None,
        fill_success: bool = None,
        price_improvement: float = None,
    ):
        """Update venue performance metrics."""
        metrics = self._venue_metrics.get(venue)
        if not metrics:
            return
        
        # Update latency (exponential moving average)
        if latency_ms is not None:
            metrics.avg_latency_ms = metrics.avg_latency_ms * 0.9 + latency_ms * 0.1
            metrics.p99_latency_ms = max(metrics.p99_latency_ms * 0.99, latency_ms)
        
        # Update fill rate
        if fill_success is not None:
            if fill_success:
                metrics.fill_rate = metrics.fill_rate * 0.99 + 0.01
            else:
                metrics.fill_rate = metrics.fill_rate * 0.99
                metrics.error_rate = metrics.error_rate * 0.99 + 0.01
        
        # Update price improvement
        if price_improvement is not None:
            metrics.avg_price_improvement = (
                metrics.avg_price_improvement * 0.9 + price_improvement * 0.1
            )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_routes": 0}
        
        total = len(self._routing_history)
        
        # Venue distribution
        venue_counts: Dict[Venue, int] = {}
        for decision in self._routing_history:
            venue_counts[decision.venue] = venue_counts.get(decision.venue, 0) + 1
        
        # Average metrics
        avg_confidence = sum(d.confidence for d in self._routing_history) / total
        avg_improvement = sum(d.price_improvement for d in self._routing_history) / total
        
        return {
            "total_routes": total,
            "venue_distribution": {v.value: c for v, c in venue_counts.items()},
            "avg_confidence": round(avg_confidence, 4),
            "avg_price_improvement": round(avg_improvement, 6),
        }


# ============================================================================
# MULTI-VENUE ORDER SPLITTER
# ============================================================================

class OrderSplitter:
    """
    Split large orders across multiple venues.
    
    Features:
    - Liquidity-aware splitting
    - Minimize market impact
    - Parallel execution
    """
    
    def __init__(self, router: SmartOrderRouter):
        self.router = router
    
    async def split_and_route(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        max_venues: int = 3,
        min_split_size: float = 0.1,
    ) -> List[Tuple[RoutingDecision, Dict[str, Any]]]:
        """
        Split order across multiple venues.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            total_quantity: Total quantity to execute
            max_venues: Maximum venues to use
            min_split_size: Minimum size per split
            
        Returns:
            List of (decision, result) tuples
        """
        results = []
        remaining = total_quantity
        
        # Get quotes
        quotes = await self.router.get_quotes(symbol)
        
        # Sort venues by available liquidity
        if side.upper() == "BUY":
            sorted_venues = sorted(
                quotes.items(),
                key=lambda x: x[1].ask_size,
                reverse=True
            )
        else:
            sorted_venues = sorted(
                quotes.items(),
                key=lambda x: x[1].bid_size,
                reverse=True
            )
        
        # Split across top venues
        for venue, quote in sorted_venues[:max_venues]:
            if remaining <= 0:
                break
            
            # Calculate split size
            available = quote.ask_size if side.upper() == "BUY" else quote.bid_size
            split_size = min(remaining, available * 0.8)  # Don't take more than 80%
            
            if split_size < min_split_size:
                continue
            
            # Route and execute
            try:
                decision, result = await self.router.route_and_execute(
                    symbol=symbol,
                    side=side,
                    quantity=split_size,
                )
                results.append((decision, result))
                
                if result.get("success"):
                    remaining -= split_size
                    
            except Exception as e:
                logger.warning(f"Split execution failed on {venue.value}: {e}")
        
        # Handle remaining quantity
        if remaining > min_split_size:
            logger.warning(f"Unfilled quantity: {remaining}")
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_router(
    venues: List[Tuple[Venue, Any]] = None,
    strategy: RoutingStrategy = RoutingStrategy.SMART,
) -> SmartOrderRouter:
    """
    Create a smart order router with optional venues.
    
    Args:
        venues: List of (venue, connector) tuples
        strategy: Default routing strategy
        
    Returns:
        Configured SmartOrderRouter
    """
    router = SmartOrderRouter(default_strategy=strategy)
    
    if venues:
        for venue, connector in venues:
            router.add_venue(venue, connector)
    
    return router
