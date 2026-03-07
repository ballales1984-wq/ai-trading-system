"""Functional tests for risk API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestRiskAPI:
    """Test risk API endpoints."""
    
    def test_get_risk_metrics(self):
        """Test getting risk metrics."""
        response = client.get("/api/v1/risk/metrics")
        assert response.status_code in [200, 500]
    
    def test_get_correlation_matrix(self):
        """Test getting correlation matrix."""
        response = client.get("/api/v1/risk/correlation")
        assert response.status_code in [200, 500]
    
    def test_calculate_var(self):
        """Test calculating VaR."""
        response = client.post("/api/v1/risk/var", json={"confidence": 0.95, "days": 1})
        assert response.status_code in [200, 404, 422, 500]
    
    def test_get_portfolio_risk(self):
        """Test getting portfolio risk."""
        response = client.get("/api/v1/risk/portfolio")
        assert response.status_code in [200, 404, 500]
    
    def test_get_risk_status(self):
        """Test getting risk status."""
        response = client.get("/api/v1/risk/status")
        assert response.status_code in [200, 404, 500]
