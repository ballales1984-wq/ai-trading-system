"""Extended tests for market data module"""
import pytest

class TestMarketDataExtended:
    def test_data_feed_exists(self):
        from app.market_data import data_feed
        assert data_feed is not None
    
    def test_websocket_stream_exists(self):
        from app.market_data import websocket_stream
        assert websocket_stream is not None
    
    def test_data_feed_class(self):
        from app.market_data.data_feed import DataFeed
        assert DataFeed is not None
    
    def test_websocket_class(self):
        from app.market_data.websocket_stream import WebSocketStream
        assert WebSocketStream is not None
