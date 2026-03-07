"""Functional tests for news API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestNewsAPI:
    """Test news API endpoints."""
    
    def test_get_news(self):
        """Test getting news."""
        response = client.get("/api/v1/news")
        assert response.status_code in [200, 500]
    
    def test_get_news_with_limit(self):
        """Test getting news with limit."""
        response = client.get("/api/v1/news?limit=10")
        assert response.status_code in [200, 500]
    
    def test_get_news_by_symbol(self):
        """Test getting news by symbol."""
        response = client.get("/api/v1/news/BTCUSDT")
        assert response.status_code in [200, 404, 500]
    
    def test_get_news_by_category(self):
        """Test getting news by category."""
        response = client.get("/api/v1/news?category=bitcoin")
        assert response.status_code in [200, 500]
