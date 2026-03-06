"""
Tests for Order Manager module.
"""

import pytest
from datetime import datetime
from app.execution.order_manager import (
    OrderStatus,
    OrderSide,
    OrderType,
    TimeInForce,
    Order
)


class TestOrderStatus:
    """Test OrderStatus enum."""

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"
        assert OrderStatus.ERROR.value == "ERROR"


class TestOrderSide:
    """Test OrderSide enum."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestOrderType:
    """Test OrderType enum."""

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
        assert OrderType.TRAILING_STOP.value == "TRAILING_STOP"


class TestTimeInForce:
    """Test TimeInForce enum."""

    def test_time_in_force_values(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"


class TestOrder:
    """Test Order model."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=1.0
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.order_type == "MARKET"
        assert order.quantity == 1.0

    def test_order_with_price(self):
        """Test creating an order with price."""
        order = Order(
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=10.0,
            price=2000.0
        )
        
        assert order.price == 2000.0
        assert order.order_type == "LIMIT"

    def test_order_with_stop_price(self):
        """Test creating an order with stop price."""
        order = Order(
            symbol="BTCUSDT",
            side="SELL",
            order_type="STOP",
            quantity=5.0,
            stop_price=45000.0
        )
        
        assert order.stop_price == 45000.0

    def test_order_default_values(self):
        """Test order default values."""
        order = Order(
            symbol="BTCUSDT",
            quantity=1.0
        )
        
        assert order.side == "BUY"
        assert order.order_type == "MARKET"
        assert order.time_in_force == "GTC"
        assert order.filled_quantity == 0.0
        assert order.remaining_quantity == 0.0

    def test_order_has_id(self):
        """Test order has generated ID."""
        order = Order(
            symbol="BTCUSDT",
            quantity=1.0
        )
        
        assert order.order_id is not None
        assert len(order.order_id) > 0

    def test_order_custom_client_id(self):
        """Test order with custom client order ID."""
        order = Order(
            symbol="BTCUSDT",
            quantity=1.0,
            client_order_id="my-custom-id-123"
        )
        
        assert order.client_order_id == "my-custom-id-123"

    def test_order_time_in_force(self):
        """Test order with different time in force."""
        order = Order(
            symbol="BTCUSDT",
            quantity=1.0,
            time_in_force="IOC"
        )
        
        assert order.time_in_force == "IOC"
