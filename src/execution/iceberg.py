"""
Iceberg Order Execution
======================
Iceberg orders hide the true order size by breaking it into smaller,
visible "tip" orders. As each tip is filled, a new one is revealed.

Features:
- Hidden order size
- Random tip sizes to avoid detection
- Price improvement strategies
- Anti-gaming measures
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class IcebergState(str, Enum):
    """State of an iceberg order."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class IcebergConfig:
    """Configuration for iceberg order execution."""
    # Tip settings
    min_tip_size: float = 0.1  # Minimum visible tip size
    max_tip_size: float = 1.0  # Maximum visible tip size
    tip_size_percent: float = 0.1  # Tip as % of total (10%)
    randomize_tip: bool = True  # Randomize tip sizes
    
    # Timing settings
    min_delay_ms: float = 100  # Minimum delay between orders (ms)
    max_delay_ms: float = 500  # Maximum delay between orders (ms)
    randomize_delay: bool = True  # Randomize delays
    
    # Price settings
    price_variance: float = 0.0001  # 0.01% price variance
    aggressive_fill: bool = True  # Be aggressive on fills
    price_improvement_enabled: bool = True
    
    # Anti-gaming settings
    max_participation_rate: float = 0.3  # Max 30% of volume
    detection_avoidance: bool = True
    vary_order_pattern: bool = True
    
    # Risk settings
    max_slippage: float = 0.001  # 0.1% max slippage
    timeout_seconds: float = 3600  # 1 hour timeout


@dataclass
class IcebergOrder:
    """Iceberg order representation."""
    order_id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: str = "BUY"  # BUY or SELL
    total_quantity: float = 0.0
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    
    # Price settings
    limit_price: Optional[float] = None
    avg_fill_price: float = 0.0
    
    # Tip tracking
    current_tip_size: float = 0.0
    tip_filled: float = 0.0
    tips_placed: int = 0
    tips_filled: int = 0
    
    # State
    state: IcebergState = IcebergState.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metrics
    total_value: float = 0.0
    slippage: float = 0.0
    
    # Broker
    broker_order_ids: List[str] = field(default_factory=list)
    current_broker_order_id: Optional[str] = None
    
    # Config
    config: IcebergConfig = field(default_factory=IcebergConfig)


class IcebergExecutor:
    """
    Executes iceberg orders with stealth and efficiency.
    
    Usage:
        executor = IcebergExecutor(broker_interface)
        
        order = IcebergOrder(
            symbol="BTCUSDT",
            side="BUY",
            total_quantity=10.0,
            limit_price=50000.0,
        )
        
        await executor.submit(order)
        
        # Monitor
        status = executor.get_status(order.order_id)
        
        # Cancel
        await executor.cancel(order.order_id)
    """
    
    def __init__(
        self,
        broker_interface,
        config: IcebergConfig = None,
    ):
        self.broker = broker_interface
        self.config = config or IcebergConfig()
        
        self._orders: Dict[str, IcebergOrder] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start(self):
        """Start the iceberg executor."""
        self._running = True
        logger.info("Iceberg executor started")
    
    async def stop(self):
        """Stop the iceberg executor."""
        self._running = False
        
        # Cancel all active orders
        for order_id in list(self._orders.keys()):
            await self.cancel(order_id)
        
        # Wait for tasks to complete
        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Iceberg executor stopped")
    
    async def submit(self, order: IcebergOrder) -> str:
        """
        Submit an iceberg order for execution.
        
        Returns:
            Order ID
        """
        if order.order_id in self._orders:
            raise ValueError(f"Order {order.order_id} already exists")
        
        # Initialize order
        order.remaining_quantity = order.total_quantity
        order.config = order.config or self.config
        order.state = IcebergState.PENDING
        
        self._orders[order.order_id] = order
        
        # Start execution task
        task = asyncio.create_task(self._execute_order(order.order_id))
        self._tasks[order.order_id] = task
        
        logger.info(
            f"Iceberg order submitted: {order.order_id} "
            f"{order.side} {order.total_quantity} {order.symbol} @ {order.limit_price}"
        )
        
        return order.order_id
    
    async def cancel(self, order_id: str) -> bool:
        """Cancel an iceberg order."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        order.state = IcebergState.CANCELLED
        
        # Cancel current tip order
        if order.current_broker_order_id:
            try:
                await self.broker.cancel_order(
                    order.current_broker_order_id,
                    order.symbol
                )
            except Exception as e:
                logger.warning(f"Failed to cancel tip order: {e}")
        
        # Cancel execution task
        if order_id in self._tasks:
            self._tasks[order_id].cancel()
        
        logger.info(f"Iceberg order cancelled: {order_id}")
        return True
    
    async def pause(self, order_id: str) -> bool:
        """Pause an iceberg order."""
        order = self._orders.get(order_id)
        if not order or order.state != IcebergState.ACTIVE:
            return False
        
        order.state = IcebergState.PAUSED
        
        # Cancel current tip
        if order.current_broker_order_id:
            await self.broker.cancel_order(
                order.current_broker_order_id,
                order.symbol
            )
        
        logger.info(f"Iceberg order paused: {order_id}")
        return True
    
    async def resume(self, order_id: str) -> bool:
        """Resume a paused iceberg order."""
        order = self._orders.get(order_id)
        if not order or order.state != IcebergState.PAUSED:
            return False
        
        order.state = IcebergState.ACTIVE
        logger.info(f"Iceberg order resumed: {order_id}")
        return True
    
    def get_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an iceberg order."""
        order = self._orders.get(order_id)
        if not order:
            return None
        
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "state": order.state.value,
            "total_quantity": order.total_quantity,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "fill_percent": (order.filled_quantity / order.total_quantity * 100) if order.total_quantity > 0 else 0,
            "avg_fill_price": order.avg_fill_price,
            "tips_placed": order.tips_placed,
            "tips_filled": order.tips_filled,
            "slippage": order.slippage,
            "created_at": order.created_at.isoformat(),
            "started_at": order.started_at.isoformat() if order.started_at else None,
            "completed_at": order.completed_at.isoformat() if order.completed_at else None,
        }
    
    async def _execute_order(self, order_id: str):
        """Execute an iceberg order."""
        order = self._orders[order_id]
        order.state = IcebergState.ACTIVE
        order.started_at = datetime.now(timezone.utc)
        
        try:
            while self._running and order.state == IcebergState.ACTIVE:
                # Check if complete
                if order.remaining_quantity <= 0:
                    order.state = IcebergState.COMPLETED
                    order.completed_at = datetime.now(timezone.utc)
                    break
                
                # Check timeout
                elapsed = (datetime.now(timezone.utc) - order.started_at).total_seconds()
                if elapsed > order.config.timeout_seconds:
                    logger.warning(f"Iceberg order timeout: {order_id}")
                    order.state = IcebergState.COMPLETED
                    break
                
                # Calculate next tip
                tip_size = self._calculate_tip_size(order)
                order.current_tip_size = tip_size
                
                # Place tip order
                success = await self._place_tip(order)
                
                if not success:
                    await asyncio.sleep(1.0)
                    continue
                
                order.tips_placed += 1
                
                # Wait for fill
                await self._wait_for_fill(order)
                
                # Random delay before next tip
                delay = self._calculate_delay(order.config)
                await asyncio.sleep(delay)
            
        except asyncio.CancelledError:
            logger.debug(f"Iceberg execution cancelled: {order_id}")
        except Exception as e:
            logger.error(f"Iceberg execution error: {e}")
            order.state = IcebergState.REJECTED
        
        finally:
            # Calculate final metrics
            if order.filled_quantity > 0:
                order.slippage = abs(order.avg_fill_price - (order.limit_price or 0)) / (order.limit_price or 1)
            
            logger.info(
                f"Iceberg order completed: {order_id} "
                f"filled={order.filled_quantity}/{order.total_quantity} "
                f"avg_price={order.avg_fill_price} slippage={order.slippage:.4%}"
            )
    
    def _calculate_tip_size(self, order: IcebergOrder) -> float:
        """Calculate the size of the next tip order."""
        config = order.config
        
        # Base tip size
        base_size = order.remaining_quantity * config.tip_size_percent
        
        # Apply min/max constraints
        base_size = max(config.min_tip_size, min(config.max_tip_size, base_size))
        
        # Don't exceed remaining quantity
        base_size = min(base_size, order.remaining_quantity)
        
        # Randomize if enabled
        if config.randomize_tip:
            variance = random.uniform(0.7, 1.3)
            base_size *= variance
            base_size = max(config.min_tip_size, min(config.max_tip_size, base_size))
            base_size = min(base_size, order.remaining_quantity)
        
        return round(base_size, 8)
    
    def _calculate_delay(self, config: IcebergConfig) -> float:
        """Calculate delay between tip orders."""
        if config.randomize_delay:
            delay_ms = random.uniform(config.min_delay_ms, config.max_delay_ms)
        else:
            delay_ms = (config.min_delay_ms + config.max_delay_ms) / 2
        
        return delay_ms / 1000.0  # Convert to seconds
    
    def _calculate_tip_price(self, order: IcebergOrder) -> Optional[float]:
        """Calculate price for tip order."""
        if not order.limit_price:
            return None
        
        config = order.config
        
        if config.price_variance > 0:
            variance = random.uniform(-config.price_variance, config.price_variance)
            price = order.limit_price * (1 + variance)
        else:
            price = order.limit_price
        
        # Price improvement for aggressive fills
        if config.aggressive_fill:
            if order.side == "BUY":
                price *= 1.0001  # Slightly higher for buys
            else:
                price *= 0.9999  # Slightly lower for sells
        
        return round(price, 8)
    
    async def _place_tip(self, order: IcebergOrder) -> bool:
        """Place a tip order."""
        try:
            price = self._calculate_tip_price(order)
            
            # Create order via broker
            result = await self.broker.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type="LIMIT" if price else "MARKET",
                quantity=order.current_tip_size,
                price=price,
            )
            
            order.current_broker_order_id = result.get("orderId") or result.get("order_id")
            if order.current_broker_order_id:
                order.broker_order_ids.append(order.current_broker_order_id)
            
            logger.debug(
                f"Tip order placed: {order.current_broker_order_id} "
                f"qty={order.current_tip_size} price={price}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to place tip order: {e}")
            return False
    
    async def _wait_for_fill(self, order: IcebergOrder):
        """Wait for tip order to be filled."""
        max_wait = 60  # Maximum wait time in seconds
        start_time = asyncio.get_event_loop().time()
        
        while self._running and order.state == IcebergState.ACTIVE:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > max_wait:
                # Cancel and move to next tip
                if order.current_broker_order_id:
                    try:
                        await self.broker.cancel_order(
                            order.current_broker_order_id,
                            order.symbol
                        )
                    except Exception:
                        pass
                return
            
            try:
                # Check order status
                status = await self.broker.get_order_status(
                    order.current_broker_order_id,
                    order.symbol
                )
                
                filled_qty = float(status.get("filled_quantity", 0) or status.get("executedQty", 0))
                avg_price = float(status.get("average_price", 0) or status.get("avgPrice", 0) or 0)
                order_status = status.get("status", "").upper()
                
                # Update fill tracking
                if filled_qty > order.tip_filled:
                    new_fill = filled_qty - order.tip_filled
                    order.tip_filled = filled_qty
                    
                    # Update totals
                    if avg_price > 0:
                        old_value = order.filled_quantity * order.avg_fill_price
                        new_value = new_fill * avg_price
                        total_value = old_value + new_value
                        order.filled_quantity += new_fill
                        order.avg_fill_price = total_value / order.filled_quantity if order.filled_quantity > 0 else 0
                        order.total_value = total_value
                    else:
                        order.filled_quantity += new_fill
                    
                    order.remaining_quantity = order.total_quantity - order.filled_quantity
                
                # Check if tip is complete
                if order_status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                    if order_status == "FILLED":
                        order.tips_filled += 1
                    
                    order.tip_filled = 0
                    order.current_broker_order_id = None
                    return
                
                # Small delay before next check
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error checking tip status: {e}")
                await asyncio.sleep(0.5)


# ============================================================================
# ICEBERG ORDER BUILDER
# ============================================================================

class IcebergBuilder:
    """
    Builder for creating iceberg orders with fluent API.
    
    Usage:
        order = (IcebergBuilder()
            .symbol("BTCUSDT")
            .side("BUY")
            .quantity(10.0)
            .limit_price(50000.0)
            .tip_size(0.5, 1.0)
            .randomize()
            .build())
    """
    
    def __init__(self):
        self._symbol = ""
        self._side = "BUY"
        self._quantity = 0.0
        self._limit_price: Optional[float] = None
        self._config = IcebergConfig()
    
    def symbol(self, symbol: str) -> "IcebergBuilder":
        self._symbol = symbol
        return self
    
    def side(self, side: str) -> "IcebergBuilder":
        self._side = side.upper()
        return self
    
    def buy(self) -> "IcebergBuilder":
        self._side = "BUY"
        return self
    
    def sell(self) -> "IcebergBuilder":
        self._side = "SELL"
        return self
    
    def quantity(self, qty: float) -> "IcebergBuilder":
        self._quantity = qty
        return self
    
    def limit_price(self, price: float) -> "IcebergBuilder":
        self._limit_price = price
        return self
    
    def market(self) -> "IcebergBuilder":
        self._limit_price = None
        return self
    
    def tip_size(self, min_size: float, max_size: float = None) -> "IcebergBuilder":
        self._config.min_tip_size = min_size
        self._config.max_tip_size = max_size or min_size * 2
        return self
    
    def tip_percent(self, percent: float) -> "IcebergBuilder":
        self._config.tip_size_percent = percent / 100
        return self
    
    def randomize(self, enabled: bool = True) -> "IcebergBuilder":
        self._config.randomize_tip = enabled
        self._config.randomize_delay = enabled
        return self
    
    def delay(self, min_ms: float, max_ms: float = None) -> "IcebergBuilder":
        self._config.min_delay_ms = min_ms
        self._config.max_delay_ms = max_ms or min_ms
        return self
    
    def timeout(self, seconds: float) -> "IcebergBuilder":
        self._config.timeout_seconds = seconds
        return self
    
    def aggressive(self, enabled: bool = True) -> "IcebergBuilder":
        self._config.aggressive_fill = enabled
        return self
    
    def build(self) -> IcebergOrder:
        if not self._symbol:
            raise ValueError("Symbol is required")
        if self._quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        return IcebergOrder(
            symbol=self._symbol,
            side=self._side,
            total_quantity=self._quantity,
            limit_price=self._limit_price,
            config=self._config,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_iceberg_order(
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float = None,
    tip_size: float = None,
) -> IcebergOrder:
    """Create a simple iceberg order."""
    builder = IcebergBuilder().symbol(symbol).side(side).quantity(quantity)
    
    if limit_price:
        builder = builder.limit_price(limit_price)
    else:
        builder = builder.market()
    
    if tip_size:
        builder = builder.tip_size(tip_size)
    
    return builder.build()
