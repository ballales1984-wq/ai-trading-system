"""
Order Management System (OMS)
=============================
Professional order lifecycle management with retry logic, partial fills,
slippage tracking, and commission calculation.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from app.core.logging import TradingLogger


logger = TradingLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class OrderStatus(str, Enum):
    """Order status states."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TimeInForce(str, Enum):
    """Time in force."""
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


# ============================================================================
# DATA MODELS
# ============================================================================

class Order(BaseModel):
    """Order model with full lifecycle tracking."""
    order_id: str = Field(default_factory=lambda: str(uuid4()))
    client_order_id: Optional[str] = None
    
    # Order details
    symbol: str
    side: str = "BUY"
    order_type: str = "MARKET"
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    
    # Execution details
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    
    # Status
    status: str = "PENDING"
    
    # Financial tracking
    commission: float = 0.0
    slippage: float = 0.0
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Broker info
    broker_order_id: Optional[str] = None
    exchange: str = "binance"
    
    # Strategy
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    
    class Config:
        use_enum_values = True


class OrderFill(BaseModel):
    """Order fill event."""
    fill_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    fill_quantity: float
    fill_price: float
    commission: float = 0.0
    fees: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trade_id: Optional[str] = None


class OrderBookUpdate(BaseModel):
    """Order book update event."""
    symbol: str
    best_bid: float
    best_ask: float
    spread: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# ORDER MANAGER CLASS
# ============================================================================

class OrderManager:
    """
    Professional Order Management System.
    
    Handles:
    - Order lifecycle management
    - Retry logic with exponential backoff
    - Partial fill handling
    - Slippage and commission tracking
    - Latency monitoring
    - Order validation
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_slippage_pct: float = 0.001,
        commission_rate: float = 0.001,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_slippage_pct = max_slippage_pct
        self.commission_rate = commission_rate
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, List[OrderFill]] = {}
        
        # Callbacks
        self.on_order_update: Optional[Callable] = None
        self.on_fill: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Metrics
        self.metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0.0,
            "total_commission": 0.0,
            "avg_fill_latency_ms": 0.0,
        }
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_id: Optional[str] = None,
        exchange: str = "binance",
    ) -> Order:
        """Create and register a new order."""
        order = Order(
            symbol=symbol.upper(),
            side=side.upper(),
            order_type=order_type.upper(),
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            remaining_quantity=quantity,
            strategy_id=strategy_id,
            exchange=exchange,
        )
        
        # Validate order
        self._validate_order(order)
        
        # Store order
        self.orders[order.order_id] = order
        self.fills[order.order_id] = []
        self.metrics["total_orders"] += 1
        
        logger.log_order(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price or 0.0,
        )
        
        return order
    
    def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order.order_type in ["LIMIT", "STOP_LIMIT"] and order.price is None:
            raise ValueError("Limit orders require a price")
        
        if order.order_type in ["STOP", "STOP_LIMIT"] and order.stop_price is None:
            raise ValueError("Stop orders require a stop price")
    
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker."""
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order with status {order.status}")
        
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()
        order.updated_at = datetime.utcnow()
        
        logger.info(f"Order submitted: {order.order_id}", event="order_submit")
        
        return order
    
    async def process_fill(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        trade_id: Optional[str] = None,
    ) -> Order:
        """Process an order fill."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        
        # Calculate commission
        commission = fill_quantity * fill_price * self.commission_rate
        
        # Calculate slippage
        slippage = 0.0
        if order.price:
            slippage = abs(fill_price - order.price) / order.price
        
        # Create fill record
        fill = OrderFill(
            order_id=order_id,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            trade_id=trade_id,
        )
        
        self.fills[order_id].append(fill)
        
        # Update order
        order.filled_quantity += fill_quantity
        order.remaining_quantity = order.quantity - order.filled_quantity
        order.commission += commission
        order.slippage = max(order.slippage, slippage)
        
        # Calculate weighted average price
        total_value = sum(
            f.fill_quantity * f.fill_price 
            for f in self.fills[order_id]
        )
        total_qty = sum(f.fill_quantity for f in self.fills[order_id])
        order.average_price = total_value / total_qty if total_qty > 0 else None
        
        # Update status
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
            self.metrics["filled_orders"] += 1
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        order.updated_at = datetime.utcnow()
        
        # Update metrics
        self.metrics["total_volume"] += fill_quantity * fill_price
        self.metrics["total_commission"] += commission
        
        logger.log_trade(
            trade_id=fill.fill_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            pnl=0.0,  # PnL calculated at portfolio level
        )
        
        return order
    
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an order."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Cannot cancel order with status {order.status}")
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        self.metrics["cancelled_orders"] += 1
        
        logger.info(f"Order cancelled: {order_id}", event="order_cancel")
        
        return order
    
    async def retry_order(self, order: Order) -> Order:
        """Retry a failed order with exponential backoff."""
        if order.retry_count >= self.max_retries:
            order.status = OrderStatus.ERROR
            order.error_message = "Max retries exceeded"
            self.metrics["rejected_orders"] += 1
            return order
        
        order.retry_count += 1
        delay = self.retry_delay * (2 ** (order.retry_count - 1))
        
        logger.warning(
            f"Retrying order {order.order_id}, attempt {order.retry_count}, "
            f"delay: {delay}s"
        )
        
        await asyncio.sleep(delay)
        
        return await self.submit_order(order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders with optional filters."""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        if status:
            orders = [o for o in orders if o.status == status]
        
        return orders[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get OMS metrics."""
        return self.metrics.copy()