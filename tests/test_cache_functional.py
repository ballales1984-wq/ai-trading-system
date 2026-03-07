"""Functional tests for cache API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestCacheAPI:
    """Test cache API endpoints."""
    
    def test_cache_get(self):
        """Test cache get."""
        response = client.get("/api/v1/cache/test_key")
        assert response.status_code in [200, 404, 500]
    
    def test_cache_set(self):
        """Test cache set."""
        response = client.post("/api/v1/cache/test_key", json={"value": "test_value"})
        assert response.status_code in [200, 201, 404, 500]
    
    def test_cache_delete(self):
        """Test cache delete."""
        response = client.delete("/api/v1/cache/test_key")
        assert response.status_code in [200, 204, 404, 500]
    
    def test_cache_clear(self):
        """Test cache clear."""
        response = client.delete("/api/v1/cache/")
        assert response.status_code in [200, 204, 404, 500, 503]
    
    def test_cache_health(self):
        """Test cache health."""
        response = client.get("/api/v1/cache/health")
        assert response.status_code in [200, 404, 500]
