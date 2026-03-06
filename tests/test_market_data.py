"""
Tests for Market Data module.
"""

import pytest
from datetime import datetime
from app.market_data.data_feed import (
    MarketData,
    MarketDataFeed,
    market_data_feed,
    DataFeed
)


class TestMarketData:
    """Test MarketData dataclass."""

    def test_market_data_creation(self):
        """Test creating market data."""
        now = datetime.now()
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=now,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        assert data.symbol == "BTCUSDT"
        assert data.open == 50000.0
        assert data.high == 51000.0
        assert data.low == 49000.0
        assert data.close == 50500.0
        assert data.volume == 1000.0

    def test_market_data_to_dict(self):
        """Test converting market data to dict."""
        now = datetime(2024, 1, 1, 12, 0, 0)
        data = MarketData(
            symbol="ETHUSDT",
            timestamp=now,
            open=2000.0,
            high=2100.0,
            low=1950.0,
            close=2050.0,
            volume=500.0
        )
        
        data_dict = data.to_dict()
        
        assert data_dict["symbol"] == "ETHUSDT"
        assert "timestamp" in data_dict
        assert data_dict["open"] == 2000.0
        assert data_dict["high"] == 2100.0
        assert data_dict["low"] == 1950.0
        assert data_dict["close"] == 2050.0
        assert data_dict["volume"] == 500.0


class TestMarketDataFeed:
    """Test MarketDataFeed class."""

    def test_feed_creation(self):
        """Test creating market data feed."""
        feed = MarketDataFeed()
        
        assert feed is not None
        assert feed._prices == {}
        assert feed._candles == {}
        assert feed._callbacks == []

    def test_feed_update_price(self):
        """Test updating price."""
        feed = MarketDataFeed()
        feed.update_price("BTCUSDT", 50000.0)
        
        assert feed.get_price("BTCUSDT") == 50000.0

    def test_feed_get_price_not_exists(self):
        """Test getting price for non-existent symbol."""
        feed = MarketDataFeed()
        
        assert feed.get_price("NONEXISTENT") is None

    def test_feed_get_all_prices(self):
        """Test getting all prices."""
        feed = MarketDataFeed()
        feed.update_price("BTCUSDT", 50000.0)
        feed.update_price("ETHUSDT", 3000.0)
        
        prices = feed.get_all_prices()
        
        assert "BTCUSDT" in prices
        assert "ETHUSDT" in prices
        assert prices["BTCUSDT"] == 50000.0
        assert prices["ETHUSDT"] == 3000.0

    def test_feed_add_candle(self):
        """Test adding candle data."""
        feed = MarketDataFeed()
        now = datetime.now()
        
        candle = MarketData(
            symbol="BTCUSDT",
            timestamp=now,
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        feed.add_candle(candle)
        
        candles = feed.get_candles("BTCUSDT")
        assert len(candles) == 1
        assert candles[0].symbol == "BTCUSDT"

    def test_feed_get_candles_limit(self):
        """Test getting candles with limit."""
        feed = MarketDataFeed()
        now = datetime.now()
        
        for i in range(10):
            candle = MarketData(
                symbol="BTCUSDT",
                timestamp=now,
                open=50000.0 + i,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0
            )
            feed.add_candle(candle)
        
        candles = feed.get_candles("BTCUSDT", limit=5)
        assert len(candles) == 5

    def test_feed_get_candles_not_exists(self):
        """Test getting candles for non-existent symbol."""
        feed = MarketDataFeed()
        
        candles = feed.get_candles("NONEXISTENT")
        assert candles == []

    def test_feed_get_dataframe(self):
        """Test getting DataFrame from candles."""
        feed = MarketDataFeed()
        now = datetime.now()
        
        for i in range(5):
            candle = MarketData(
                symbol="BTCUSDT",
                timestamp=now,
                open=50000.0 + i,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0
            )
            feed.add_candle(candle)
        
        df = feed.get_dataframe("BTCUSDT")
        
        assert len(df) == 5

    def test_feed_get_dataframe_empty(self):
        """Test getting empty DataFrame."""
        feed = MarketDataFeed()
        
        df = feed.get_dataframe("NONEXISTENT")
        
        assert len(df) == 0

    def test_feed_subscribe(self):
        """Test subscribing to feed."""
        feed = MarketDataFeed()
        
        def callback(symbol, price):
            pass
        
        feed.subscribe(callback)
        
        assert callback in feed._callbacks

    def test_feed_unsubscribe(self):
        """Test unsubscribing from feed."""
        feed = MarketDataFeed()
        
        def callback(symbol, price):
            pass
        
        feed.subscribe(callback)
        feed.unsubscribe(callback)
        
        assert callback not in feed._callbacks


class TestMarketDataSingleton:
    """Test market data feed singleton."""

    def test_market_data_feed_singleton(self):
        """Test market_data_feed singleton."""
        assert market_data_feed is not None

    def test_datafeed_alias(self):
        """Test DataFeed alias."""
        assert DataFeed is MarketDataFeed
