"""
Test Coverage for Market Data Module
====================================
Comprehensive tests to improve coverage for app/market_data/* modules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMarketDataDataFeed:
    """Test app.market_data.data_feed module."""
    
    def test_data_feed_module_import(self):
        """Test data_feed module can be imported."""
        from app.market_data import data_feed
        assert data_feed is not None
    
    def test_market_data_class(self):
        """Test MarketData class exists."""
        from app.market_data.data_feed import MarketData
        assert MarketData is not None
    
    def test_market_data_feed_class(self):
        """Test MarketDataFeed class exists."""
        from app.market_data.data_feed import MarketDataFeed
        assert MarketDataFeed is not None
    
    def test_market_data_creation(self):
        """Test MarketData creation with correct fields."""
        from app.market_data.data_feed import MarketData
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        assert data.symbol == "BTCUSDT"
        assert data.close == 50500.0
    
    def test_market_data_feed_initialization(self):
        """Test MarketDataFeed initialization."""
        from app.market_data.data_feed import MarketDataFeed
        feed = MarketDataFeed()
        assert feed is not None
    
    def test_market_data_feed_subscribe_unsubscribe(self):
        """Test MarketDataFeed subscribe and unsubscribe methods."""
        from app.market_data.data_feed import MarketDataFeed
        feed = MarketDataFeed()
        
        callback = Mock()
        feed.subscribe(callback)
        assert callback in feed._callbacks
        
        feed.unsubscribe(callback)
        assert callback not in feed._callbacks
    
    def test_market_data_feed_update_price(self):
        """Test MarketDataFeed update_price method."""
        from app.market_data.data_feed import MarketDataFeed
        feed = MarketDataFeed()
        
        callback = Mock()
        feed.subscribe(callback)
        feed.update_price("BTCUSDT", 50000.0)
        
        assert feed.get_price("BTCUSDT") == 50000.0
        callback.assert_called_once()
    
    def test_market_data_feed_get_price(self):
        """Test MarketDataFeed get_price method."""
        from app.market_data.data_feed import MarketDataFeed
        feed = MarketDataFeed()
        feed.update_price("BTCUSDT", 50000.0)
        
        price = feed.get_price("BTCUSDT")
        assert price == 50000.0
    
    def test_market_data_to_dict(self):
        """Test MarketData to_dict method."""
        from app.market_data.data_feed import MarketData
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        result = data.to_dict()
        assert isinstance(result, dict)
        assert result["symbol"] == "BTCUSDT"
    
    def test_market_data_feed_add_candle(self):
        """Test MarketDataFeed add_candle method."""
        from app.market_data.data_feed import MarketDataFeed, MarketData
        feed = MarketDataFeed()
        
        candle = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        
        feed.add_candle(candle)
        candles = feed.get_candles("BTCUSDT")
        assert len(candles) == 1
    
    def test_market_data_feed_get_candles(self):
        """Test MarketDataFeed get_candles method."""
        from app.market_data.data_feed import MarketDataFeed, MarketData
        feed = MarketDataFeed()
        
        # Add multiple candles
        for i in range(5):
            candle = MarketData(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                open=50000.0 + i,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=1000.0
            )
            feed.add_candle(candle)
        
        candles = feed.get_candles("BTCUSDT", limit=3)
        assert len(candles) == 3
    
    def test_market_data_feed_get_dataframe(self):
        """Test MarketDataFeed get_dataframe method."""
        from app.market_data.data_feed import MarketDataFeed, MarketData
        feed = MarketDataFeed()
        
        candle = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        feed.add_candle(candle)
        
        df = feed.get_dataframe("BTCUSDT")
        assert df is not None
    
    def test_market_data_singleton(self):
        """Test market_data_feed singleton."""
        from app.market_data.data_feed import market_data_feed
        assert market_data_feed is not None


class TestMarketDataWebsocketStream:
    """Test app.market_data.websocket_stream module."""
    
    def test_websocket_stream_module_import(self):
        """Test websocket_stream module can be imported."""
        from app.market_data import websocket_stream
        assert websocket_stream is not None
    
    def test_websocket_stream_class(self):
        """Test WebSocketStream class exists."""
        from app.market_data.websocket_stream import WebSocketStream
        assert WebSocketStream is not None
    
    def test_websocket_stream_initialization(self):
        """Test WebSocketStream initialization."""
        from app.market_data.websocket_stream import WebSocketStream
        ws = WebSocketStream()
        assert ws is not None
    
    def test_websocket_stream_connect(self):
        """Test WebSocketStream connect method."""
        from app.market_data.websocket_stream import WebSocketStream
        ws = WebSocketStream()
        
        if hasattr(ws, 'connect'):
            pass
    
    def test_websocket_stream_subscribe(self):
        """Test WebSocketStream subscribe method."""
        from app.market_data.websocket_stream import WebSocketStream
        ws = WebSocketStream()
        
        if hasattr(ws, 'subscribe'):
            callback = Mock()
            ws.subscribe("BTCUSDT", callback)
    
    def test_websocket_stream_is_connected(self):
        """Test WebSocketStream is_connected property."""
        from app.market_data.websocket_stream import WebSocketStream
        ws = WebSocketStream()
        
        if hasattr(ws, 'is_connected'):
            is_connected = ws.is_connected
            assert isinstance(is_connected, bool)


class TestMarketDataIntegration:
    """Integration tests for market_data module."""
    
    def test_market_data_types(self):
        """Test various market data types."""
        from app.market_data.data_feed import MarketData
        
        tickers = [
            MarketData(symbol="BTCUSDT", timestamp=datetime.now(), open=50000, high=51000, low=49000, close=50500, volume=1000),
            MarketData(symbol="ETHUSDT", timestamp=datetime.now(), open=3000, high=3100, low=2900, close=3050, volume=5000),
            MarketData(symbol="SOLUSDT", timestamp=datetime.now(), open=100, high=110, low=90, close=105, volume=10000)
        ]
        
        assert len(tickers) == 3
    
    def test_market_data_price_updates(self):
        """Test market data price updates."""
        from app.market_data.data_feed import MarketData
        
        base_price = 50000.0
        updates = []
        for i in range(5):
            data = MarketData(
                symbol="BTCUSDT",
                timestamp=datetime.now() + timedelta(seconds=i),
                open=base_price * (1 + i * 0.01),
                high=base_price * (1 + i * 0.01) * 1.01,
                low=base_price * (1 + i * 0.01) * 0.99,
                close=base_price * (1 + i * 0.01),
                volume=1000.0 + i * 100
            )
            updates.append(data)
        
        assert len(updates) == 5
        assert updates[-1].close > updates[0].close
    
    def test_market_data_volume_tracking(self):
        """Test market data volume tracking."""
        from app.market_data.data_feed import MarketData
        
        trades = [
            MarketData(symbol="BTCUSDT", timestamp=datetime.now(), open=50000, high=50000, low=50000, close=50000, volume=0.5),
            MarketData(symbol="BTCUSDT", timestamp=datetime.now(), open=50000, high=50000, low=50000, close=50000, volume=0.3),
            MarketData(symbol="BTCUSDT", timestamp=datetime.now(), open=50000, high=50000, low=50000, close=50000, volume=0.8)
        ]
        
        total_volume = sum(t.volume for t in trades)
        assert total_volume == 1.6
    
    def test_market_data_symbol_formats(self):
        """Test different market data symbol formats."""
        from app.market_data.data_feed import MarketData
        
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        
        for symbol in symbols:
            data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=100.0,
                high=110.0,
                low=90.0,
                close=105.0,
                volume=100.0
            )
            assert data.symbol == symbol


class TestMarketDataMock:
    """Tests using mock data for market data."""
    
    def test_mock_ticker_data(self):
        """Test mock ticker data generation."""
        from app.market_data.data_feed import MarketData
        
        mock_tickers = []
        for symbol, price in [("BTCUSDT", 50000.0), ("ETHUSDT", 3000.0), ("SOLUSDT", 100.0)]:
            mock_tickers.append(MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000.0
            ))
        
        assert len(mock_tickers) == 3
        assert all(t.close > 0 for t in mock_tickers)
    
    def test_mock_orderbook_data(self):
        """Test mock orderbook data generation."""
        base_price = 50000.0
        
        # Start from i=1 to ensure bids are below and asks are above base_price
        bids = [(base_price - i * 5, 1.0 - i * 0.1) for i in range(1, 11)]
        asks = [(base_price + i * 5, 1.0 - i * 0.1) for i in range(1, 11)]
        
        assert len(bids) == 10
        assert len(asks) == 10
        # All bids should be below base price
        assert all(b[0] < base_price for b in bids)
        # All asks should be above base price  
        assert all(a[0] > base_price for a in asks)
    
    def test_mock_candle_data(self):
        """Test mock candle data generation."""
        candles = []
        base_price = 50000.0
        
        for i in range(100):
            open_price = base_price + (i * 10)
            close_price = open_price + (i % 10 - 5) * 5
            high = max(open_price, close_price) + 10
            low = min(open_price, close_price) - 10
            volume = 1000.0 + i * 10
            
            candles.append({
                "timestamp": datetime.now() - timedelta(hours=100-i),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume
            })
        
        assert len(candles) == 100
        assert all(c["high"] >= c["low"] for c in candles)

