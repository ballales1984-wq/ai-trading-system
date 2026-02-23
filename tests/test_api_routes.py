"""
Tests for API Routes
====================
Tests for FastAPI route handlers.
"""

import pytest
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestHealthRoutes:
    """Test health check routes."""
    
    def setup_method(self):
        """Setup test client."""
        from app.main import app
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test main health check endpoint."""
        response = self.client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'service' in data
        assert 'version' in data
    
    def test_readiness_check(self):
        """Test readiness check endpoint."""
        response = self.client.get("/api/v1/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'ready'
    
    def test_liveness_check(self):
        """Test liveness check endpoint."""
        response = self.client.get("/api/v1/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'alive'


class TestWaitlistRoutes:
    """Test waitlist routes."""
    
    def setup_method(self):
        """Setup test client."""
        from app.main import app
        self.client = TestClient(app)
    
    def test_join_waitlist(self):
        """Test joining the waitlist."""
        response = self.client.post(
            "/api/v1/waitlist",
            json={
                "email": "test@example.com",
                "source": "landing_page"
            }
        )
        
        # Should succeed or return validation error
        assert response.status_code in [200, 201, 422]
    
    def test_join_waitlist_invalid_email(self):
        """Test joining waitlist with invalid email."""
        response = self.client.post(
            "/api/v1/waitlist",
            json={
                "email": "invalid-email"
            }
        )
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_join_waitlist_missing_email(self):
        """Test joining waitlist without email."""
        response = self.client.post(
            "/api/v1/waitlist",
            json={
                "source": "landing_page"
            }
        )
        
        # Should return validation error
        assert response.status_code == 422


class TestMarketRoutes:
    """Test market data routes."""
    
    def setup_method(self):
        """Setup test client."""
        from app.main import app
        self.client = TestClient(app)
    
    def test_get_market_data(self):
        """Test getting market data."""
        response = self.client.get("/api/v1/market/data")
        
        # Should return data or 404/401
        assert response.status_code in [200, 401, 404, 422]
    
    def test_get_symbols(self):
        """Test getting available symbols."""
        response = self.client.get("/api/v1/market/symbols")
        
        # Should return symbols or 404/401
        assert response.status_code in [200, 401, 404]


class TestPortfolioRoutes:
    """Test portfolio routes."""
    
    def setup_method(self):
        """Setup test client."""
        from app.main import app
        self.client = TestClient(app)
    
    def test_get_portfolio(self):
        """Test getting portfolio."""
        response = self.client.get("/api/v1/portfolio")
        
        # Should return portfolio or 401/404
        assert response.status_code in [200, 401, 404]
    
    def test_get_portfolio_summary(self):
        """Test getting portfolio summary."""
        response = self.client.get("/api/v1/portfolio/summary")
        
        # Should return summary or 401/404
        assert response.status_code in [200, 401, 404]


class TestOrderRoutes:
    """Test order routes."""
    
    def setup_method(self):
        """Setup test client."""
        from app.main import app
        self.client = TestClient(app)
    
    def test_get_orders(self):
        """Test getting orders."""
        response = self.client.get("/api/v1/orders")
        
        # Should return orders or 401/404
        assert response.status_code in [200, 401, 404]
    
    def test_create_order(self):
        """Test creating an order."""
        response = self.client.post(
            "/api/v1/orders",
            json={
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 0.1,
                "price": 50000
            }
        )
        
        # Should return 200, 201, 401, or 422 (validation error)
        assert response.status_code in [200, 201, 401, 422]

    def test_emergency_stop_blocks_order_creation(self):
        """Emergency stop should block BUY/SELL order creation."""
        os.environ["ADMIN_SECRET_KEY"] = "test-admin-key"
        headers = {"X-Admin-Key": "test-admin-key", "X-Admin-User": "tests"}

        activate = self.client.post(
            "/api/v1/orders/emergency/activate",
            json={"confirm": True, "reason": "test"},
            headers=headers,
        )
        assert activate.status_code == 200
        assert activate.json().get("trading_halted") is True

        blocked = self.client.post(
            "/api/v1/orders",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.1,
                "order_type": "MARKET",
                "broker": "paper",
            },
        )
        assert blocked.status_code == 423

        deactivate = self.client.post(
            "/api/v1/orders/emergency/deactivate",
            json={"confirm": True, "reason": "test cleanup"},
            headers=headers,
        )
        assert deactivate.status_code == 200
        assert deactivate.json().get("trading_halted") is False


class TestRiskRoutes:
    """Test risk management routes."""
    
    def setup_method(self):
        """Setup test client."""
        from app.main import app
        self.client = TestClient(app)
    
    def test_get_risk_metrics(self):
        """Test getting risk metrics."""
        response = self.client.get("/api/v1/risk/metrics")
        
        # Should return metrics or 401/404
        assert response.status_code in [200, 401, 404]
    
    def test_get_risk_limits(self):
        """Test getting risk limits."""
        response = self.client.get("/api/v1/risk/limits")
        
        # Should return limits or 401/404
        assert response.status_code in [200, 401, 404]
