# src/core/event_bus.py
"""
Event Bus for Core Engine
========================
Central event dispatcher for event-driven architecture.
Handles all communication between components.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the trading system."""
    # Market events
    MARKET_DATA = "market_data"
    TICKER_UPDATE = "ticker_update"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXECUTED = "signal_executed"
    SIGNAL_REJECTED = "signal_rejected"
    
    # Order events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Risk events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_ALERT = "risk_alert"
    EMERGENCY_EXIT = "emergency_exit"
    
    # Portfolio events
    PORTFOLIO_UPDATED = "portfolio_updated"
    BALANCE_UPDATED = "balance_updated"
    
    # System events
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"
    ENGINE_ERROR = "engine_error"
    MODEL_LOADED = "model_loaded"
    STATE_SAVED = "state_saved"


@dataclass
class Event:
    """
    Base event class.
    """
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source
        }


class EventHandler(ABC):
    """Abstract event handler."""
    
    @abstractmethod
    async def handle(self, event: Event):
        """Handle the event."""
        pass


class EventBus:
    """
    Central event bus for pub/sub communication.
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._event_history: List[Event] = []
        self._max_history = 10000
        
        # Create event log directory
        self._log_dir = Path("logs/events")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Event bus initialized")
    
    def subscribe(self, event_type: EventType, handler: EventHandler):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Event handler
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__class__.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Event handler to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
    
    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Log event
        self._log_event(event)
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Get subscribers
        handlers = self._subscribers.get(event.event_type, [])
        
        if not handlers:
            return
        
        # Notify all handlers
        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__class__.__name__}: {e}")
    
    async def publish_sync(self, event: Event):
        """Publish event synchronously."""
        await self.publish(event)
    
    def _log_event(self, event: Event):
        """Log event to file."""
        log_file = self._log_dir / f"events_{event.timestamp.strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events
            
        Returns:
            List of events
        """
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        
        return history[-limit:]
    
    def get_event_stats(self) -> Dict:
        """Get event statistics."""
        stats = {}
        
        for event_type in EventType:
            count = sum(1 for e in self._event_history if e.event_type == event_type)
            stats[event_type.value] = count
        
        return stats


class SignalEventHandler(EventHandler):
    """Handler for signal events."""
    
    def __init__(self, callback: Callable):
        """Initialize handler."""
        self.callback = callback
    
    async def handle(self, event: Event):
        """Handle signal event."""
        if event.event_type == EventType.SIGNAL_GENERATED:
            await self.callback(event.data)


class OrderEventHandler(EventHandler):
    """Handler for order events."""
    
    def __init__(self, on_filled: Callable = None, on_rejected: Callable = None):
        """Initialize handler."""
        self.on_filled = on_filled
        self.on_rejected = on_rejected
    
    async def handle(self, event: Event):
        """Handle order event."""
        if event.event_type == EventType.ORDER_FILLED and self.on_filled:
            await self.on_filled(event.data)
        elif event.event_type == EventType.ORDER_REJECTED and self.on_rejected:
            await self.on_rejected(event.data)


class RiskEventHandler(EventHandler):
    """Handler for risk events."""
    
    def __init__(self, on_alert: Callable = None, on_emergency: Callable = None):
        """Initialize handler."""
        self.on_alert = on_alert
        self.on_emergency = on_emergency
    
    async def handle(self, event: Event):
        """Handle risk event."""
        if event.event_type == EventType.RISK_ALERT and self.on_alert:
            await self.on_alert(event.data)
        elif event.event_type == EventType.EMERGENCY_EXIT and self.on_emergency:
            await self.on_emergency(event.data)


def create_event(event_type: EventType, data: Dict[str, Any], source: str = "system") -> Event:
    """
    Factory function to create events.
    
    Args:
        event_type: Type of event
        data: Event data
        source: Event source
        
    Returns:
        Event instance
    """
    return Event(
        event_type=event_type,
        data=data,
        source=source
    )
