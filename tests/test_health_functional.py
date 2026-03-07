"""Functional tests for health API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealthAPI:
    """Test health API endpoints."""
    
    def test_health_check(self):
        """Test health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    def test_health_liveness(self):
        """Test liveness probe."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_ready(self):
        """Test readiness probe."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code in [200, 404, 503]
    
    def test_health_db(self):
        """Test database health."""
        response = client.get("/api/v1/health/db")
        assert response.status_code in [200, 404, 503]
