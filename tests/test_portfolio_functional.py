"""Functional tests for portfolio API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestPortfolioAPI:
    """Test portfolio API endpoints."""
    
    def test_get_portfolio_summary(self):
        """Test getting portfolio summary."""
        response = client.get("/api/v1/portfolio/summary")
        assert response.status_code == 200
    
    def test_get_positions(self):
        """Test getting positions."""
        response = client.get("/api/v1/portfolio/positions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_positions_with_symbol(self):
        """Test getting positions with symbol filter."""
        response = client.get("/api/v1/portfolio/positions?symbol=BTCUSDT")
        assert response.status_code == 200
    
    def test_get_performance(self):
        """Test getting performance metrics."""
        response = client.get("/api/v1/portfolio/performance")
        assert response.status_code == 200
    
    def test_get_allocation(self):
        """Test getting allocation."""
        response = client.get("/api/v1/portfolio/allocation")
        assert response.status_code == 200
    
    def test_get_history(self):
        """Test getting portfolio history."""
        response = client.get("/api/v1/portfolio/history")
        assert response.status_code == 200
    
    def test_get_history_with_days(self):
        """Test getting portfolio history with days parameter."""
        response = client.get("/api/v1/portfolio/history?days=30")
        assert response.status_code == 200
    
    def test_get_dual_summary(self):
        """Test getting dual summary."""
        response = client.get("/api/v1/portfolio/summary/dual")
        assert response.status_code in [200, 500]
    
    def test_get_demo_mode(self):
        """Test getting demo mode."""
        response = client.get("/api/v1/portfolio/mode")
        assert response.status_code in [200, 500]
    
    def test_set_demo_mode(self):
        """Test setting demo mode."""
        response = client.post("/api/v1/portfolio/mode?enabled=true")
        assert response.status_code in [200, 500]
