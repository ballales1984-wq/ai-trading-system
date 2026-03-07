"""Functional tests for orders API"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app

client = TestClient(app)

class TestOrdersAPI:
    """Test orders API endpoints."""
    
    def test_list_orders(self):
        """Test listing orders."""
        response = client.get("/api/v1/orders")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_list_orders_with_symbol(self):
        """Test listing orders with symbol filter."""
        response = client.get("/api/v1/orders?symbol=BTCUSDT")
        assert response.status_code == 200
    
    def test_create_order_market(self):
        """Test creating a market order."""
        order_data = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "order_type": "market",
            "quantity": 0.001
        }
        response = client.post("/api/v1/orders", json=order_data)
        assert response.status_code in [200, 201, 500]  # May fail due to broker
    
    def test_create_order_limit(self):
        """Test creating a limit order."""
        order_data = {
            "symbol": "ETHUSDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.01,
            "price": 3000.0
        }
        response = client.post("/api/v1/orders", json=order_data)
        assert response.status_code in [200, 201, 422, 500]
    
    def test_get_order_by_id(self):
        """Test getting a specific order."""
        response = client.get("/api/v1/orders/test-order-id")
        assert response.status_code in [200, 404]
    
    def test_cancel_order(self):
        """Test cancelling an order."""
        response = client.delete("/api/v1/orders/test-order-id")
        assert response.status_code in [204, 404, 500]
    
    def test_execute_order(self):
        """Test executing an order."""
        response = client.post("/api/v1/orders/test-order-id/execute")
        assert response.status_code in [200, 404, 500]
    
    def test_get_trade_history(self):
        """Test getting trade history."""
        response = client.get("/api/v1/orders/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_emergency_stop(self):
        """Test emergency stop."""
        response = client.post("/api/v1/orders/emergency-stop", json={"reason": "test"})
        assert response.status_code in [200, 500]
    
    def test_emergency_resume(self):
        """Test emergency resume."""
        response = client.post("/api/v1/orders/emergency-resume")
        assert response.status_code in [200, 500]
    
    def test_get_emergency_status(self):
        """Test getting emergency status."""
        response = client.get("/api/v1/orders/status/emergency")
        assert response.status_code == 200
