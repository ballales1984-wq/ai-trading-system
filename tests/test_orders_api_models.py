"""
Tests for Orders API Models
===========================
Tests for the data models in app/api/routes/orders.py
"""

import pytest
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOrderCreate:
    """Test OrderCreate pydantic model."""
    
    def test_order_create_required_fields(self):
        """Test OrderCreate with required fields only."""
        from app.api.routes.orders import OrderCreate
        
        order = OrderCreate(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.quantity == 0.5
        assert order.order_type == "MARKET"  # default
        assert order.time_in_force == "GTC"  # default
        assert order.broker == "binance"  # default
    
    def test_order_create_all_fields(self):
        """Test OrderCreate with all fields."""
        from app.api.routes.orders import OrderCreate
        
        order = OrderCreate(
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=1.0,
            price=3000.0,
            stop_price=2900.0,
            time_in_force="IOC",
            strategy_id="momentum_001",
            broker="bybit"
        )
        
        assert order.symbol == "ETHUSDT"
        assert order.side == "SELL"
        assert order.order_type == "LIMIT"
        assert order.quantity == 1.0
        assert order.price == 3000.0
        assert order.stop_price == 2900.0
        assert order.time_in_force == "IOC"
        assert order.strategy_id == "momentum_001"
        assert order.broker == "bybit"
    
    def test_order_create_validation_quantity_positive(self):
        """Test OrderCreate validates quantity > 0."""
        from app.api.routes.orders import OrderCreate
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            OrderCreate(
                symbol="BTCUSDT",
                side="BUY",
                quantity=-1.0
            )
    
    def test_order_create_validation_price_positive(self):
        """Test OrderCreate validates price > 0."""
        from app.api.routes.orders import OrderCreate
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            OrderCreate(
                symbol="BTCUSDT",
                side="BUY",
                quantity=1.0,
                price=-100.0
            )


class TestOrderResponse:
    """Test OrderResponse pydantic model."""
    
    def test_order_response_creation(self):
        """Test OrderResponse creation."""
        from app.api.routes.orders import OrderResponse
        
        now = datetime.utcnow()
        order = OrderResponse(
            order_id="order-123",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            price=50000.0,
            stop_price=49000.0,
            status="FILLED",
            filled_quantity=0.5,
            average_price=50000.0,
            commission=0.5,
            created_at=now,
            updated_at=now,
            broker="binance"
        )
        
        assert order.order_id == "order-123"
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.status == "FILLED"
        assert order.filled_quantity == 0.5
        assert order.average_price == 50000.0
        assert order.commission == 0.5
    
    def test_order_response_defaults(self):
        """Test OrderResponse default values."""
        from app.api.routes.orders import OrderResponse
        
        now = datetime.utcnow()
        order = OrderResponse(
            order_id="order-456",
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=1.0,
            price=3000.0,
            stop_price=2900.0,
            status="PENDING",
            created_at=now,
            updated_at=now,
            broker="bybit"
        )
        
        assert order.filled_quantity == 0.0
        assert order.average_price is None
        assert order.commission == 0.0
        assert order.strategy_id is None
        assert order.error_message is None
    
    def test_order_response_with_error(self):
        """Test OrderResponse with error message."""
        from app.api.routes.orders import OrderResponse
        
        now = datetime.utcnow()
        order = OrderResponse(
            order_id="order-789",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.1,
            price=50000.0,
            stop_price=49000.0,
            status="REJECTED",
            created_at=now,
            updated_at=now,
            broker="binance",
            error_message="Insufficient balance"
        )
        
        assert order.status == "REJECTED"
        assert order.error_message == "Insufficient balance"


class TestOrderUpdate:
    """Test OrderUpdate pydantic model."""
    
    def test_order_update_creation(self):
        """Test OrderUpdate creation."""
        from app.api.routes.orders import OrderUpdate
        
        update = OrderUpdate(
            quantity=2.0,
            price=3100.0,
            stop_price=3000.0
        )
        
        assert update.quantity == 2.0
        assert update.price == 3100.0
        assert update.stop_price == 3000.0
    
    def test_order_update_partial(self):
        """Test OrderUpdate with partial fields."""
        from app.api.routes.orders import OrderUpdate
        
        update = OrderUpdate(quantity=1.5)
        
        assert update.quantity == 1.5
        assert update.price is None
        assert update.stop_price is None
    
    def test_order_update_validation(self):
        """Test OrderUpdate validates positive values."""
        from app.api.routes.orders import OrderUpdate
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            OrderUpdate(quantity=-1.0)


class TestEmergencyStopRequest:
    """Test EmergencyStopRequest pydantic model."""
    
    def test_emergency_stop_request_defaults(self):
        """Test EmergencyStopRequest default values."""
        from app.api.routes.orders import EmergencyStopRequest
        
        request = EmergencyStopRequest()
        
        assert request.reason is None
        assert request.cancel_all_orders is True
        assert request.close_all_positions is False
    
    def test_emergency_stop_request_all_fields(self):
        """Test EmergencyStopRequest with all fields."""
        from app.api.routes.orders import EmergencyStopRequest
        
        request = EmergencyStopRequest(
            reason="Market crash detected",
            cancel_all_orders=True,
            close_all_positions=True
        )
        
        assert request.reason == "Market crash detected"
        assert request.cancel_all_orders is True
        assert request.close_all_positions is True


class TestEmergencyStopResponse:
    """Test EmergencyStopResponse pydantic model."""
    
    def test_emergency_stop_response_creation(self):
        """Test EmergencyStopResponse creation."""
        from app.api.routes.orders import EmergencyStopResponse
        
        now = datetime.utcnow()
        response = EmergencyStopResponse(
            success=True,
            message="Emergency stop activated",
            cancelled_orders=5,
            closed_positions=2,
            timestamp=now
        )
        
        assert response.success is True
        assert response.message == "Emergency stop activated"
        assert response.cancelled_orders == 5
        assert response.closed_positions == 2
        assert response.timestamp == now


class TestOrdersDatabase:
    """Test orders database functions."""
    
    def test_orders_db_exists(self):
        """Test orders_db dictionary exists."""
        from app.api.routes.orders import orders_db
        
        assert isinstance(orders_db, dict)
    
    def test_demo_orders_db_exists(self):
        """Test demo_orders_db dictionary exists."""
        from app.api.routes.orders import demo_orders_db
        
        assert isinstance(demo_orders_db, dict)
    
    def test_emergency_stop_flag(self):
        """Test emergency_stop_active flag."""
        from app.api.routes.orders import emergency_stop_active
        
        # Just check it exists and is a boolean
        assert isinstance(emergency_stop_active, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
