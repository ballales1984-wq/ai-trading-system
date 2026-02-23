"""
Tests for Event Bus System
===========================
Tests for the core event bus system that handles communication between components.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.event_bus import (
    EventBus, EventType, Event, EventHandler,
    SignalEventHandler, OrderEventHandler, RiskEventHandler
)


class TestEventBus:
    """Test the EventBus class."""
    
    def setup_method(self):
        """Setup test data."""
        self.event_bus = EventBus()
        self.test_dir = Path("logs/events")
        self.test_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test data."""
        # Remove test event logs
        for log_file in self.test_dir.glob("*.jsonl"):
            log_file.unlink()
        self.test_dir.rmdir()
    
    def test_initialization(self):
        """Test EventBus initialization."""
        assert self.event_bus is not None
        assert len(self.event_bus._subscribers) == 0
        assert len(self.event_bus._event_history) == 0
        assert self.event_bus._max_history == 10000
        assert self.test_dir.exists()
    
    def test_subscribe(self):
        """Test subscribing to events."""
        # Create mock handler
        mock_handler = Mock(spec=EventHandler)
        
        # Subscribe to MARKET_DATA
        self.event_bus.subscribe(EventType.MARKET_DATA, mock_handler)
        
        # Verify subscription
        assert EventType.MARKET_DATA in self.event_bus._subscribers
        assert mock_handler in self.event_bus._subscribers[EventType.MARKET_DATA]
    
    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        # Create mock handler
        mock_handler = Mock(spec=EventHandler)
        
        # Subscribe and then unsubscribe
        self.event_bus.subscribe(EventType.MARKET_DATA, mock_handler)
        self.event_bus.unsubscribe(EventType.MARKET_DATA, mock_handler)
        
        # Verify unsubscription
        assert EventType.MARKET_DATA in self.event_bus._subscribers
        assert mock_handler not in self.event_bus._subscribers[EventType.MARKET_DATA]
    
    def test_publish(self):
        """Test publishing events."""
        # Create mock handlers
        market_handler = Mock(spec=EventHandler)
        signal_handler = Mock(spec=EventHandler)
        
        # Subscribe handlers
        self.event_bus.subscribe(EventType.MARKET_DATA, market_handler)
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, signal_handler)
        
        # Create test event
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Publish event
        asyncio.run(self.event_bus.publish(test_event))
        
        # Verify handler was called
        market_handler.handle.assert_called_once()
        signal_handler.handle.assert_not_called()
        
        # Verify event was added to history
        assert len(self.event_bus._event_history) == 1
        assert self.event_bus._event_history[0] == test_event
    
    def test_publish_multiple_handlers(self):
        """Test publishing to multiple handlers."""
        # Create multiple mock handlers
        handlers = [Mock(spec=EventHandler) for _ in range(3)]
        
        # Subscribe all handlers
        for handler in handlers:
            self.event_bus.subscribe(EventType.MARKET_DATA, handler)
        
        # Create test event
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Publish event
        asyncio.run(self.event_bus.publish(test_event))
        
        # Verify all handlers were called
        for handler in handlers:
            handler.handle.assert_called_once()
    
    def test_publish_error_handling(self):
        """Test error handling in event handlers."""
        # Create mock handlers (one that raises exception)
        good_handler = Mock(spec=EventHandler)
        bad_handler = Mock(spec=EventHandler)
        bad_handler.handle = AsyncMock(side_effect=Exception("Test error"))
        
        # Subscribe handlers
        self.event_bus.subscribe(EventType.MARKET_DATA, good_handler)
        self.event_bus.subscribe(EventType.MARKET_DATA, bad_handler)
        
        # Create test event
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Publish event
        asyncio.run(self.event_bus.publish(test_event))
        
        # Verify good handler was called
        good_handler.handle.assert_called_once()
        
        # Verify bad handler was called and error was logged
        bad_handler.handle.assert_called_once()
    
    def test_event_logging(self):
        """Test event logging to file."""
        # Create test event
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Publish event
        asyncio.run(self.event_bus.publish(test_event))
        
        # Verify log file was created
        log_file = self.test_dir / f"events_{test_event.timestamp.strftime('%Y%m%d')}.jsonl"
        assert log_file.exists()
        
        # Verify log content
        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            log_data = json.loads(lines[0])
            assert log_data['event_type'] == 'market_data'
            assert log_data['data'] == {'symbol': 'BTCUSDT', 'price': 50000}
    
    def test_get_event_history(self):
        """Test getting event history."""
        # Create multiple events
        events = [
            Event(event_type=EventType.MARKET_DATA, data={'symbol': f'TEST{i}'})
            for i in range(5)
        ]
        
        # Publish all events
        for event in events:
            asyncio.run(self.event_bus.publish(event))
        
        # Test getting all history
        history = self.event_bus.get_event_history()
        assert len(history) == 5
        assert history == events
        
        # Test getting filtered history
        filtered = self.event_bus.get_event_history(event_type=EventType.MARKET_DATA)
        assert len(filtered) == 5
        
        # Test getting limited history
        limited = self.event_bus.get_event_history(limit=3)
        assert len(limited) == 3
        assert limited == events[-3:]
    
    def test_get_event_stats(self):
        """Test getting event statistics."""
        # Create events of different types
        events = [
            Event(event_type=EventType.MARKET_DATA),
            Event(event_type=EventType.SIGNAL_GENERATED),
            Event(event_type=EventType.MARKET_DATA),
            Event(event_type=EventType.ORDER_PLACED),
            Event(event_type=EventType.MARKET_DATA)
        ]
        
        # Publish all events
        for event in events:
            asyncio.run(self.event_bus.publish(event))
        
        # Get stats
        stats = self.event_bus.get_event_stats()
        
        # Verify stats
        assert stats['market_data'] == 3
        assert stats['signal_generated'] == 1
        assert stats['order_placed'] == 1
        assert stats['ticker_update'] == 0
    
    def test_event_history_limit(self):
        """Test event history limit."""
        # Create more events than max history
        events = [
            Event(event_type=EventType.MARKET_DATA, data={'symbol': f'TEST{i}'})
            for i in range(15000)
        ]
        
        # Publish all events
        for event in events:
            asyncio.run(self.event_bus.publish(event))
        
        # Verify history is limited
        assert len(self.event_bus._event_history) == 10000
        assert self.event_bus._event_history[-1] == events[-1]
        assert self.event_bus._event_history[0] == events[-10000]


class TestEvent:
    """Test the Event class."""
    
    def test_initialization(self):
        """Test Event initialization."""
        event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000},
            source='test'
        )
        
        assert event.event_type == EventType.MARKET_DATA
        assert event.data == {'symbol': 'BTCUSDT', 'price': 50000}
        assert event.source == 'test'
        assert isinstance(event.timestamp, datetime)
    
    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000},
            source='test'
        )
        
        event_dict = event.to_dict()
        
        assert event_dict['event_type'] == 'market_data'
        assert event_dict['data'] == {'symbol': 'BTCUSDT', 'price': 50000}
        assert event_dict['source'] == 'test'
        assert 'timestamp' in event_dict
    
    def test_timestamp_default(self):
        """Test default timestamp."""
        event1 = Event(event_type=EventType.MARKET_DATA)
        event2 = Event(event_type=EventType.MARKET_DATA)
        
        # Timestamps should be different
        assert event1.timestamp != event2.timestamp


class TestEventHandlers:
    """Test event handler classes."""
    
    def test_signal_event_handler(self):
        """Test SignalEventHandler."""
        # Create mock callback
        mock_callback = AsyncMock()
        
        # Create handler
        handler = SignalEventHandler(mock_callback)
        
        # Create test event
        test_event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            data={'signal': 'buy', 'symbol': 'BTCUSDT'}
        )
        
        # Handle event
        asyncio.run(handler.handle(test_event))
        
        # Verify callback was called
        mock_callback.assert_called_once_with({'signal': 'buy', 'symbol': 'BTCUSDT'})
    
    def test_signal_event_handler_wrong_type(self):
        """Test SignalEventHandler with wrong event type."""
        # Create mock callback
        mock_callback = AsyncMock()
        
        # Create handler
        handler = SignalEventHandler(mock_callback)
        
        # Create wrong event type
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Handle event
        asyncio.run(handler.handle(test_event))
        
        # Verify callback was not called
        mock_callback.assert_not_called()
    
    def test_order_event_handler(self):
        """Test OrderEventHandler."""
        # Create mock callbacks
        on_filled = AsyncMock()
        on_rejected = AsyncMock()
        
        # Create handler
        handler = OrderEventHandler(on_filled=on_filled, on_rejected=on_rejected)
        
        # Test filled event
        filled_event = Event(
            event_type=EventType.ORDER_FILLED,
            data={'order_id': '123', 'symbol': 'BTCUSDT'}
        )
        asyncio.run(handler.handle(filled_event))
        on_filled.assert_called_once_with({'order_id': '123', 'symbol': 'BTCUSDT'})
        on_rejected.assert_not_called()
        
        # Test rejected event
        rejected_event = Event(
            event_type=EventType.ORDER_REJECTED,
            data={'order_id': '123', 'reason': 'insufficient_balance'}
        )
        asyncio.run(handler.handle(rejected_event))
        on_rejected.assert_called_once_with({'order_id': '123', 'reason': 'insufficient_balance'})
        
        # Test other event type
        other_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        asyncio.run(handler.handle(other_event))
        on_filled.assert_called_once()
        on_rejected.assert_called_once()


class TestEventBusAsync:
    """Test async features of EventBus."""

    def setup_method(self):
        """Setup event bus instance for async tests."""
        self.event_bus = EventBus()
    
    async def test_publish_async(self):
        """Test async publishing."""
        # Create mock handler
        mock_handler = AsyncMock(spec=EventHandler)
        
        # Subscribe handler
        self.event_bus.subscribe(EventType.MARKET_DATA, mock_handler)
        
        # Create test event
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Publish event asynchronously
        await self.event_bus.publish(test_event)
        
        # Verify handler was called
        mock_handler.handle.assert_called_once()
    
    def test_publish_sync(self):
        """Test sync publishing."""
        # Create mock handler
        mock_handler = AsyncMock(spec=EventHandler)
        
        # Subscribe handler
        self.event_bus.subscribe(EventType.MARKET_DATA, mock_handler)
        
        # Create test event
        test_event = Event(
            event_type=EventType.MARKET_DATA,
            data={'symbol': 'BTCUSDT', 'price': 50000}
        )
        
        # Publish event synchronously
        asyncio.run(self.event_bus.publish_sync(test_event))
        
        # Verify handler was called
        mock_handler.handle.assert_called_once()
