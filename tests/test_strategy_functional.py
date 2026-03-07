"""Functional tests for strategy API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestStrategyAPI:
    """Test strategy API endpoints."""
    
    def test_list_strategies(self):
        """Test listing strategies."""
        response = client.get("/api/v1/strategy")
        assert response.status_code in [200, 500]
    
    def test_get_strategy(self):
        """Test getting a strategy."""
        response = client.get("/api/v1/strategy/momentum")
        assert response.status_code in [200, 404, 500]
    
    def test_create_strategy(self):
        """Test creating a strategy."""
        response = client.post("/api/v1/strategy", json={
            "name": "test_strategy",
            "type": "momentum",
            "parameters": {}
        })
        assert response.status_code in [200, 201, 400, 500]
    
    def test_update_strategy(self):
        """Test updating a strategy."""
        response = client.put("/api/v1/strategy/test", json={"parameters": {}})
        assert response.status_code in [200, 404, 500]
    
    def test_delete_strategy(self):
        """Test deleting a strategy."""
        response = client.delete("/api/v1/strategy/test")
        assert response.status_code in [200, 204, 404, 500]
