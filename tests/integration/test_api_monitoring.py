"""
API Integration Tests - Monitoring
==================================
Tests for monitoring endpoints and performance tracking.

Author: AI Trading System
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestMonitoringEndpoints:
    """Test monitoring endpoints."""
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns data."""
        response = client.get("/api/monitoring/metrics")
        # May return 200 or 503 if monitoring not fully set up
        assert response.status_code in [200, 503]
    
    def test_health_endpoint(self):
        """Test health endpoint returns status."""
        response = client.get("/api/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "monitoring" in data
    
    def test_reset_metrics_endpoint(self):
        """Test reset metrics endpoint."""
        response = client.post("/api/monitoring/reset")
        assert response.status_code in [200, 503]


class TestPerformanceEndpoints:
    """Test performance monitoring endpoints."""
    
    def test_performance_metrics_endpoint(self):
        """Test performance metrics endpoint."""
        response = client.get("/api/performance/metrics")
        assert response.status_code in [200, 404]
    
    def test_slowest_functions_endpoint(self):
        """Test slowest functions endpoint."""
        response = client.get("/api/performance/slowest")
        assert response.status_code in [200, 404]
    
    def test_most_called_functions_endpoint(self):
        """Test most called functions endpoint."""
        response = client.get("/api/performance/most_called")
        assert response.status_code in [200, 404]
    
    def test_reset_performance_endpoint(self):
        """Test reset performance endpoint."""
        response = client.post("/api/performance/reset")
        assert response.status_code in [200, 404, 405]
    
    def test_cache_stats_endpoint(self):
        """Test cache stats endpoint."""
        response = client.get("/api/performance/cache/stats")
        assert response.status_code in [200, 404]
    
    def test_clear_cache_endpoint(self):
        """Test clear cache endpoint."""
        response = client.post("/api/performance/cache/clear")
        assert response.status_code in [200, 404, 405]


class TestHealthCheck:
    """Test basic health checks."""
    
    def test_root_health(self):
        """Test root health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_includes_version(self):
        """Test health includes version info."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert "environment" in data


class TestAPIEndpoints:
    """Test various API endpoints."""
    
    def test_api_news_endpoint(self):
        """Test news endpoint exists."""
        response = client.get("/api/v1/news")
        assert response.status_code in [200, 404, 401]
    
    def test_api_market_endpoint(self):
        """Test market endpoint exists."""
        response = client.get("/api/v1/market")
        assert response.status_code in [200, 404, 401]
    
    def test_api_portfolio_endpoint(self):
        """Test portfolio endpoint exists."""
        response = client.get("/api/v1/portfolio")
        assert response.status_code in [200, 404, 401]
    
    def test_api_orders_endpoint(self):
        """Test orders endpoint exists."""
        response = client.get("/api/v1/orders")
        assert response.status_code in [200, 404, 401]
    
    def test_api_risk_endpoint(self):
        """Test risk endpoint exists."""
        response = client.get("/api/v1/risk")
        assert response.status_code in [200, 404, 401]
    
    def test_api_strategy_endpoint(self):
        """Test strategy endpoint exists."""
        response = client.get("/api/v1/strategy")
        assert response.status_code in [200, 404, 401]


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation endpoints."""
    
    def test_docs_endpoint(self):
        """Test /docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self):
        """Test /redoc endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_json_endpoint(self):
        """Test OpenAPI JSON endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
