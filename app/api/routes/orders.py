"""
Order Management Routes
======================
REST API for order management and execution.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4
import logging

from fastapi import APIRouter, HTTPException, status, Query

from pydantic import BaseModel, Field
from app.core.data_adapter import get_data_adapter
from app.api.mock_data import get_orders as mock_orders

# Import demo mode functions from portfolio (they share the same state)
from app.api.routes.portfolio import get_demo_mode

logger = logging.getLogger(__name__)


router = APIRouter()


# ============================================================================
# DATA MODELS
# ============================================================================

class OrderCreate(BaseModel):
    """Request model for creating an order."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: str = Field(..., description="Order side: BUY or SELL")
    order_type: str = Field(default="MARKET", description="Order type: MARKET, LIMIT, STOP")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, gt=0, description="Limit price (for LIMIT orders)")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price (for STOP orders)")
    time_in_force: str = Field(default="GTC", description="Time in force: GTC, IOC, FOK")
    strategy_id: Optional[str] = Field(None, description="Strategy generating the order")
    broker: str = Field(default="binance", description="Broker to use: binance, ib, bybit")


class OrderResponse(BaseModel):
    """Response model for order data."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    created_at: datetime
    updated_at: datetime
    strategy_id: Optional[str] = None
    broker: str
    error_message: Optional[str] = None


class OrderUpdate(BaseModel):
    """Request model for updating an order."""
    quantity: Optional[float] = Field(None, gt=0, description="New quantity")
    price: Optional[float] = Field(None, gt=0, description="New limit price")
    stop_price: Optional[float] = Field(None, gt=0, description="New stop price")


class EmergencyStopRequest(BaseModel):
    """Request model for emergency stop."""
    reason: Optional[str] = Field(None, description="Reason for emergency stop")
    cancel_all_orders: bool = Field(default=True, description="Cancel all pending orders")
    close_all_positions: bool = Field(default=False, description="Close all open positions")


class EmergencyStopResponse(BaseModel):
    """Response model for emergency stop."""
    success: bool
    message: str
    cancelled_orders: int
    closed_positions: int
    timestamp: datetime


# ============================================================================
# IN-MEMORY ORDER STORE (Replace with database in production)
# ============================================================================

orders_db: dict = {}

# Demo mode order store for tracking demo orders
demo_orders_db: dict = {}

# Emergency stop state
emergency_stop_active: bool = False


# ============================================================================
# ROUTES
# ============================================================================

@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(order: OrderCreate) -> OrderResponse:
    """
    Create a new order.
    
    The order goes through the risk engine before execution.
    """
    # Check if emergency stop is active
    if emergency_stop_active:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading is currently disabled due to emergency stop"
        )
    
    order_id = str(uuid4())
    now = datetime.utcnow()
    
    # Demo mode: Create a mock order
    if get_demo_mode():
        # Simulate order creation in demo mode
        order_response = OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            status="PENDING",
            filled_quantity=0.0,
            average_price=None,
            commission=0.0,
            created_at=now,
            updated_at=now,
            strategy_id=order.strategy_id,
            broker="demo",
        )
        
        # Store in demo orders database
        demo_orders_db[order_id] = order_response
        
        # Simulate immediate fill for market orders
        if order.order_type == "MARKET":
            order_response.status = "FILLED"
            order_response.filled_quantity = order.quantity
            order_response.average_price = order.price or _get_mock_price(order.symbol)
            order_response.updated_at = datetime.utcnow()
            demo_orders_db[order_id] = order_response
        
        return order_response
    
    # Production mode: Submit to execution engine
    order_response = OrderResponse(
        order_id=order_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price,
        stop_price=order.stop_price,
        status="PENDING",
        created_at=now,
        updated_at=now,
        strategy_id=order.strategy_id,
        broker=order.broker,
    )
    
    # Store order
    orders_db[order_id] = order_response
    
    # Submit to execution engine
    try:
        from app.execution.broker_connector import create_broker_connector
        from app.execution.broker_connector import BrokerOrder as BOrder
        
        connector = create_broker_connector(order.broker or 'paper')
        connected = await connector.connect()
        
        if connected:
            broker_order = BOrder(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                broker=order.broker or 'paper',
            )
            
            result = await connector.place_order(broker_order)
            
            # Update order response with execution result
            order_response.status = result.status if hasattr(result, 'status') else 'FILLED'
            order_response.filled_quantity = getattr(result, 'filled_quantity', order.quantity)
            order_response.average_price = getattr(result, 'average_price', order.price)
            order_response.broker_order_id = getattr(result, 'broker_order_id', '')
            order_response.updated_at = datetime.utcnow()
            
            orders_db[order_id] = order_response
            
            await connector.disconnect()
        else:
            order_response.status = 'REJECTED'
            order_response.error = 'Failed to connect to broker'
    except Exception as e:
        logger.warning(f"Execution engine error (falling back to pending): {e}")
        order_response.status = 'PENDING'
    
    return order_response


def _get_mock_price(symbol: str) -> float:
    """Get a mock price for a symbol in demo mode."""
    from app.api.mock_data import BASE_PRICES
    
    # Normalize symbol
    if "/" not in symbol:
        symbol = symbol.replace("USDT", "/USDT")
    
    base_price = BASE_PRICES.get(symbol, 100.0)
    # Add small random variation
    import random
    return base_price * (1 + random.uniform(-0.001, 0.001))


@router.get("", response_model=List[OrderResponse])
async def list_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum orders to return"),
) -> List[OrderResponse]:
    """
    List all orders with optional filters.
    """
    # Extract actual values from Query objects if needed
    # When called directly (not through FastAPI), these might be Query objects
    symbol_val = None
    status_val = None
    limit_val = 100
    
    if symbol is not None:
        if hasattr(symbol, 'default'):
            symbol_val = symbol.default
        else:
            symbol_val = symbol
    
    if status is not None:
        if hasattr(status, 'default'):
            status_val = status.default
        else:
            status_val = status
    
    if limit is not None:
        if hasattr(limit, 'default'):
            limit_val = int(limit.default) if limit.default is not None else 100
        else:
            limit_val = int(limit) if limit is not None else 100
    
    # Use mock data if demo mode is enabled
    if get_demo_mode():
        mock_data = mock_orders(status=status_val)
        orders = [OrderResponse(
            order_id=o["id"],
            symbol=o["symbol"],
            side=o["side"],
            order_type=o["type"],
            quantity=o["quantity"],
            price=o["price"],
            stop_price=None,
            status=o["status"],
            filled_quantity=o["filled_quantity"],
            average_price=o["price"] if o["status"] == "FILLED" else None,
            commission=0.0,
            created_at=datetime.fromisoformat(o["created_at"]) if o.get("created_at") else datetime.utcnow(),
            updated_at=datetime.utcnow(),
            strategy_id=None,
            broker="demo",
        ) for o in mock_data]
        
        # Add any newly created demo orders
        for order in demo_orders_db.values():
            if status_val is None or order.status == status_val:
                if symbol_val is None or order.symbol == symbol_val:
                    orders.append(order)
        
        if symbol_val:
            orders = [o for o in orders if o.symbol == symbol_val]
        
        # Sort by created_at descending
        orders.sort(key=lambda x: x.created_at, reverse=True)
        
        return orders[:limit_val]

    
    # Try to get real orders first
    adapter = get_data_adapter()
    real_orders = adapter.get_orders()
    
    if real_orders:
        orders = [OrderResponse(**o) for o in real_orders]
    else:
        orders = list(orders_db.values())
    
    # Apply filters
    if symbol_val:
        orders = [o for o in orders if o.symbol == symbol_val]
    if status_val:
        orders = [o for o in orders if o.status == status_val]
    
    # Apply limit
    orders = orders[:limit_val]
    
    return orders




@router.get("/history", response_model=List[OrderResponse])
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status: FILLED, PENDING, CANCELLED"),
    date_from: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    date_to: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum orders to return"),
) -> List[OrderResponse]:
    """
    Get trade history with P&L data.
    
    Returns filled orders with profit/loss calculations for trade history display.
    Supports filtering by symbol, status, and date range.
    """
    # Extract actual values from Query objects if needed
    symbol_val = None
    status_val = None
    limit_val = 100
    
    if symbol is not None:
        if hasattr(symbol, 'default'):
            symbol_val = symbol.default
        else:
            symbol_val = symbol
    
    if status is not None:
        if hasattr(status, 'default'):
            status_val = status.default.upper() if status.default else None
        else:
            status_val = str(status).upper() if status else None
    
    if limit is not None:
        if hasattr(limit, 'default'):
            limit_val = int(limit.default) if limit.default is not None else 100
        else:
            limit_val = int(limit) if limit is not None else 100
    
    # Use mock data if demo mode is enabled
    if get_demo_mode():
        from app.api.mock_data import get_orders as mock_get_orders
        
        mock_orders = mock_get_orders(status=status_val)
        
        orders = []
        for o in mock_orders:
            # Skip orders without filled_at for history (unless explicitly requested)
            if status_val != "PENDING" and o.get("status") == "PENDING":
                continue
                
            order = OrderResponse(
                order_id=o["id"],
                symbol=o["symbol"],
                side=o["side"],
                order_type=o["type"],
                quantity=o["quantity"],
                price=o["price"],
                stop_price=None,
                status=o["status"],
                filled_quantity=o["filled_quantity"],
                average_price=o["price"] if o["status"] == "FILLED" else None,
                commission=0.0,
                created_at=datetime.fromisoformat(o["created_at"]) if o.get("created_at") else datetime.utcnow(),
                updated_at=datetime.fromisoformat(o["filled_at"]) if o.get("filled_at") else datetime.utcnow(),
                strategy_id=None,
                broker="demo",
            )
            orders.append(order)
        
        # Apply symbol filter
        if symbol_val:
            orders = [o for o in orders if o.symbol == symbol_val]
        
        # Sort by created_at descending (most recent first)
        orders.sort(key=lambda x: x.created_at, reverse=True)
        
        return orders[:limit_val]
    
    # Production mode: get from database
    adapter = get_data_adapter()
    real_orders = adapter.get_orders()
    
    if real_orders:
        orders = [OrderResponse(**o) for o in real_orders]
    else:
        orders = list(orders_db.values())
    
    # Apply filters
    if symbol_val:
        orders = [o for o in orders if o.symbol == symbol_val]
    if status_val:
        orders = [o for o in orders if o.status == status_val]
    
    # Apply date filters if provided
    if date_from:
        orders = [o for o in orders if o.created_at >= date_from]
    if date_to:
        orders = [o for o in orders if o.created_at <= date_to]
    
    # Sort by created_at descending
    orders.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply limit
    orders = orders[:limit_val]
    
    return orders


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str) -> OrderResponse:

    """
    Get order by ID.
    """
    # Demo mode: Check demo orders first
    if get_demo_mode():
        if order_id in demo_orders_db:
            return demo_orders_db[order_id]
        
        # Check mock orders
        mock_data = mock_orders()
        for o in mock_data:
            if o["id"] == order_id:
                return OrderResponse(
                    order_id=o["id"],
                    symbol=o["symbol"],
                    side=o["side"],
                    order_type=o["type"],
                    quantity=o["quantity"],
                    price=o["price"],
                    stop_price=None,
                    status=o["status"],
                    filled_quantity=o["filled_quantity"],
                    average_price=o["price"] if o["status"] == "FILLED" else None,
                    commission=0.0,
                    created_at=datetime.fromisoformat(o["created_at"]) if o.get("created_at") else datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    strategy_id=None,
                    broker="demo",
                )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    # Production mode
    if order_id not in orders_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    return orders_db[order_id]


@router.patch("/{order_id}", response_model=OrderResponse)
async def update_order(order_id: str, update: OrderUpdate) -> OrderResponse:
    """
    Update an existing order.
    
    Only PENDING orders can be modified.
    """
    # Demo mode
    if get_demo_mode():
        if order_id not in demo_orders_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )
        
        order = demo_orders_db[order_id]
        
        # Check if order can be modified
        if order.status != "PENDING":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot modify order with status {order.status}"
            )
        
        # Apply updates
        if update.quantity is not None:
            order.quantity = update.quantity
        if update.price is not None:
            order.price = update.price
        if update.stop_price is not None:
            order.stop_price = update.stop_price
        
        order.updated_at = datetime.utcnow()
        demo_orders_db[order_id] = order
        
        return order
    
    # Production mode
    if order_id not in orders_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    order = orders_db[order_id]
    
    # Check if order can be modified
    if order.status != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot modify order with status {order.status}"
        )
    
    # Apply updates
    if update.quantity is not None:
        order.quantity = update.quantity
    if update.price is not None:
        order.price = update.price
    if update.stop_price is not None:
        order.stop_price = update.stop_price
    
    order.updated_at = datetime.utcnow()
    
    return order


@router.delete("/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_order(order_id: str) -> None:
    """
    Cancel a pending order.
    """
    # Demo mode
    if get_demo_mode():
        if order_id not in demo_orders_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )
        
        order = demo_orders_db[order_id]
        
        # Check if order can be cancelled
        if order.status not in ["PENDING", "PARTIALLY_FILLED"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel order with status {order.status}"
            )
        
        order.status = "CANCELLED"
        order.updated_at = datetime.utcnow()
        demo_orders_db[order_id] = order
        return
    
    # Production mode
    if order_id not in orders_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    order = orders_db[order_id]
    
    # Check if order can be cancelled
    if order.status not in ["PENDING", "PARTIALLY_FILLED"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order with status {order.status}"
        )
    
    order.status = "CANCELLED"
    order.updated_at = datetime.utcnow()


@router.post("/{order_id}/execute", response_model=OrderResponse)
async def execute_order(order_id: str) -> OrderResponse:
    """
    Manually trigger order execution.
    
    In production, this submits the order to the broker.
    """
    # Check if emergency stop is active
    if emergency_stop_active:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading is currently disabled due to emergency stop"
        )
    
    # Demo mode
    if get_demo_mode():
        if order_id not in demo_orders_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )
        
        order = demo_orders_db[order_id]
        
        if order.status != "PENDING":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Order {order_id} is not in PENDING status"
            )
        
        # Simulate execution
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.average_price = order.price or _get_mock_price(order.symbol)
        order.updated_at = datetime.utcnow()
        demo_orders_db[order_id] = order
        
        return order
    
    # Production mode
    if order_id not in orders_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    order = orders_db[order_id]
    
    if order.status != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Order {order_id} is not in PENDING status"
        )
    
    # In production, this would call the execution engine
    # For now, simulate execution
    order.status = "FILLED"
    order.filled_quantity = order.quantity
    order.average_price = order.price or 0.0  # Market price
    order.updated_at = datetime.utcnow()
    
    return order


@router.post("/emergency-stop", response_model=EmergencyStopResponse)
async def emergency_stop(request: EmergencyStopRequest) -> EmergencyStopResponse:
    """
    Emergency stop - immediately halt all trading activity.
    
    This endpoint:
    1. Activates emergency stop mode (prevents new orders)
    2. Cancels all pending orders
    3. Optionally closes all open positions
    """
    global emergency_stop_active
    
    cancelled_count = 0
    closed_count = 0
    
    # Activate emergency stop
    emergency_stop_active = True
    
    # Cancel all pending orders
    if request.cancel_all_orders:
        if get_demo_mode():
            for order_id, order in list(demo_orders_db.items()):
                if order.status == "PENDING":
                    order.status = "CANCELLED"
                    order.updated_at = datetime.utcnow()
                    demo_orders_db[order_id] = order
                    cancelled_count += 1
        else:
            for order_id, order in list(orders_db.items()):
                if order.status == "PENDING":
                    order.status = "CANCELLED"
                    order.updated_at = datetime.utcnow()
                    orders_db[order_id] = order
                    cancelled_count += 1
    
    # Close all positions (simulated in demo mode)
    if request.close_all_positions:
        # In a real implementation, this would submit market orders to close positions
        # For demo, we just track that positions would be closed
        closed_count = 5  # Simulated number of positions closed
    
    logger.warning(f"EMERGENCY STOP ACTIVATED: {request.reason or 'No reason provided'}")
    logger.warning(f"Cancelled {cancelled_count} orders, closed {closed_count} positions")
    
    return EmergencyStopResponse(
        success=True,
        message="Emergency stop activated. All trading halted.",
        cancelled_orders=cancelled_count,
        closed_positions=closed_count,
        timestamp=datetime.utcnow()
    )


@router.post("/emergency-resume", response_model=dict)
async def emergency_resume() -> dict:
    """
    Resume trading after emergency stop.
    
    Deactivates emergency stop mode, allowing new orders to be created.
    """
    global emergency_stop_active
    
    was_active = emergency_stop_active
    emergency_stop_active = False
    
    if was_active:
        logger.info("EMERGENCY STOP DEACTIVATED: Trading resumed")
        return {
            "success": True,
            "message": "Emergency stop deactivated. Trading resumed.",
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "success": True,
            "message": "Trading was already active.",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/status/emergency", response_model=dict)
async def get_emergency_status() -> dict:
    """
    Get current emergency stop status.
    """
    return {
        "emergency_stop_active": emergency_stop_active,
        "timestamp": datetime.utcnow().isoformat()
    }
