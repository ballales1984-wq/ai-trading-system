"""
Test Suite for Market Data Module
================================
Comprehensive tests for market data feed.
"""

import pytest
from datetime import datetime
from app.market_data.data_feed import (
    MarketData,
    MarketDataFeed,
)


class TestMarketData:
    """Tests for MarketData dataclass."""
    
    def test_market_data_creation(self):
        """Test market data creation."""
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=45000.0,
            high=46000.0,
            low=44000.0,
            close=45500.0,
            volume=1000.0,
        )
        assert data.symbol == "BTCUSDT"
        assert data.open == 45000.0
        assert data.high == 46000.0
        assert data.low == 44000.0
        assert data.close == 45500.0
        assert data.volume == 1000.0
    
    def test_market_data_to_dict(self):
        """Test market data to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        data = MarketData(
            symbol="ETHUSDT",
            timestamp=timestamp,
            open=2500.0,
            high=2600.0,
            low=2400.0,
            close=2550.0,
            volume=500.0,
        )
        result = data.to_dict()
        assert result["symbol"] == "ETHUSDT"
        assert result["open"] == 2500.0
        assert result["high"] == 2600.0
        assert result["low"] == 2400.0
        assert result["close"] == 2550.0
        assert result["volume"] == 500.0


class TestMarketDataFeed:
    """Tests for MarketDataFeed class."""
    
    def test_feed_creation(self):
        """Test market data feed creation."""
        feed = MarketDataFeed()
        assert feed._prices == {}
        assert feed._candles == {}
        assert feed._callbacks == []
    
    def test_subscribe(self):
        """Test subscribing to market data."""
        feed = MarketDataFeed()
        
        def callback(data):
            pass
        
        feed.subscribe(callback)
        assert callback in feed._callbacks
    
    def test_unsubscribe(self):
        """Test unsubscribing from market data."""
        feed = MarketDataFeed()
        
        def callback(data):
            pass
        
        feed.subscribe(callback)
        feed.unsubscribe(callback)
        assert callback not in feed._callbacks
    
    def test_unsubscribe_not_in_list(self):
        """Test unsubscribing a callback that's not in the list."""
        feed = MarketDataFeed()
        
        def callback(data):
            pass
        
        # Should not raise an error
        feed.unsubscribe(callback)
    
    def test_set_price(self):
        """Test setting price."""
        feed = MarketDataFeed()
        feed.set_price("BTCUSDT", 45000.0)
        assert feed._prices["BTCUSDT"] == 45000.0
    
    def test_get_price(self):
        """Test getting price."""
        feed = MarketDataFeed()
        feed._prices["ETHUSDT"] = 2500.0
        price = feed.get_price("ETHUSDT")
        assert price == 2500.0
    
    def test_get_price_not_found(self):
        """Test getting price that doesn't exist."""
        feed = MarketDataFeed()
        price = feed.get_price("NONEXISTENT")
        assert price is None
    
    def test_add_candle(self):
        """Test adding candle data."""
        feed = MarketDataFeed()
        candle = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=45000.0,
            high=46000.0,
            low=44000.0,
            close=45500.0,
            volume=1000.0,
        )
        feed.add_candle("BTCUSDT", candle)
        assert "BTCUSDT" in feed._candles
        assert len(feed._candles["BTCUSDT"]) == 1
    
    def test_get_candles(self):
        """Test getting candles."""
        feed = MarketDataFeed()
        candle = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=45000.0,
            high=46000.0,
            low=44000.0,
            close=45500.0,
            volume=1000.0,
        )
        feed._candles["BTCUSDT"] = [candle]
        candles = feed.get_candles("BTCUSDT")
        assert len(candles) == 1
    
    def test_get_candles_empty(self):
        """Test getting candles that don't exist."""
        feed = MarketDataFeed()
        candles = feed.get_candles("NONEXISTENT")
        assert candles == []
    
    def test_clear_prices(self):
        """Test clearing prices."""
        feed = MarketDataFeed()
        feed._prices = {"BTCUSDT": 45000.0, "ETHUSDT": 2500.0}
        feed.clear_prices()
        assert feed._prices == {}
    
    def test_clear_candles(self):
        """Test clearing candles."""
        feed = MarketDataFeed()
        feed._candles = {"BTCUSDT": []}
        feed.clear_candles("BTCUSDT")
        assert feed._candles == {}
