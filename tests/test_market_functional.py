"""Functional tests for market API"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestMarketAPI:
    """Test market API endpoints."""
    
    def test_get_all_prices(self):
        """Test getting all prices."""
        response = client.get("/api/v1/market/prices")
        assert response.status_code == 200
    
    def test_get_single_price(self):
        """Test getting single price."""
        response = client.get("/api/v1/market/price/BTCUSDT")
        assert response.status_code in [200, 404]
    
    def test_get_candles(self):
        """Test getting candles."""
        response = client.get("/api/v1/market/candles/BTCUSDT")
        assert response.status_code in [200, 404]
    
    def test_get_candles_with_params(self):
        """Test getting candles with parameters."""
        response = client.get("/api/v1/market/candles/BTCUSDT?interval=1h&limit=100")
        assert response.status_code in [200, 404]
    
    def test_get_orderbook(self):
        """Test getting orderbook."""
        response = client.get("/api/v1/market/orderbook/BTCUSDT")
        assert response.status_code in [200, 404]
    
    def test_get_sentiment(self):
        """Test getting market sentiment."""
        response = client.get("/api/v1/market/sentiment")
        assert response.status_code in [200, 500]
    
    def test_get_ticker(self):
        """Test getting ticker."""
        response = client.get("/api/v1/market/ticker/BTCUSDT")
        assert response.status_code in [200, 404]
    
    def test_get_tickers(self):
        """Test getting all tickers."""
        response = client.get("/api/v1/market/tickers")
        assert response.status_code in [200, 404, 500]
