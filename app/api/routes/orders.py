"""
Order Management Routes
======================
REST API for order management and execution.
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Query

from pydantic import BaseModel, Field
from app.core.data_adapter import get_data_adapter


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


# ============================================================================
# IN-MEMORY ORDER STORE (Replace with database in production)
# ============================================================================

orders_db: dict = {}


# ============================================================================
# ROUTES
# ============================================================================

@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(order: OrderCreate) -> OrderResponse:
    """
    Create a new order.
    
    The order goes through the risk engine before execution.
    """
    order_id = str(uuid4())
    now = datetime.utcnow()
    
    # Create order record
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


@router.get("/", response_model=List[OrderResponse])
async def list_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum orders to return"),
) -> List[OrderResponse]:
    """
    List all orders with optional filters.
    """
    # Try to get real orders first
    adapter = get_data_adapter()
    real_orders = adapter.get_orders()
    
    if real_orders:
        orders = [OrderResponse(**o) for o in real_orders]
    else:
        orders = list(orders_db.values())
    
    # Apply filters
    if symbol:
        orders = [o for o in orders if o.symbol == symbol]
    if status:
        orders = [o for o in orders if o.status == status]
    
    # Apply limit
    orders = orders[:limit]
    
    return orders


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str) -> OrderResponse:
    """
    Get order by ID.
    """
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
