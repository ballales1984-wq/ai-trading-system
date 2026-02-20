"""
Redis Message Bus
=================
Redis pub/sub wrapper for distributed trading system.
Enables communication between microservices.

Usage:
    from src.core.performance.message_bus import MessageBus, get_message_bus
    
    bus = get_message_bus()
    await bus.publish("signals", {"symbol": "BTC", "action": "BUY"})
    
    # Subscribe
    async def on_signal(msg):
        print(f"Received: {msg}")
    
    await bus.subscribe("signals", on_signal)
"""

import asyncio
import json
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid


logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis-py not available, message bus will be disabled")


class Channel(Enum):
    """Predefined message channels."""
    SIGNALS = "trading:signals"
    ORDERS = "trading:orders"
    RISK = "trading:risk"
    EXECUTION = "trading:execution"
    MARKET_DATA = "trading:market_data"
    SYSTEM = "trading:system"


@dataclass
class Message:
    """Message structure."""
    channel: str
    data: Dict[str, Any]
    timestamp: str
    message_id: str
    
    @classmethod
    def create(cls, channel: str, data: Dict[str, Any]) -> 'Message':
        """Create a new message."""
        return cls(
            channel=channel,
            data=data,
            timestamp=datetime.now().isoformat(),
            message_id=str(uuid.uuid4())
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create from JSON string."""
        obj = json.loads(json_str)
        return cls(
            channel=obj["channel"],
            data=obj["data"],
            timestamp=obj["timestamp"],
            message_id=obj["message_id"]
        )


class MessageBus:
    """
    Redis-based message bus for distributed trading.
    
    Features:
    - Pub/Sub messaging
    - Channel management
    - Automatic reconnection
    - Message serialization
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True,
        encoding: str = "utf-8"
    ):
        """
        Initialize message bus.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            decode_responses: Decode responses to strings
            encoding: Message encoding
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.decode_responses = decode_responses
        self.encoding = encoding
        
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._running = False
        
        # Message statistics
        self._messages_sent = 0
        self._messages_received = 0
        self._errors = 0
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis-py is not available")
        
        self._redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=self.decode_responses,
            encoding=self.encoding
        )
        
        # Test connection
        await self._redis.ping()
        logger.info(f"Connected to Redis at {self.host}:{self.port}")
        
        # Start pubsub
        self._pubsub = self._redis.pubsub()
        self._running = True
        
        # Start listener
        self._listener_task = asyncio.create_task(self._listen())
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
        
        logger.info("Disconnected from Redis")
    
    async def publish(self, channel: str, data: Dict[str, Any]) -> int:
        """
        Publish message to channel.
        
        Args:
            channel: Channel name
            data: Message data
            
        Returns:
            Number of subscribers
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        
        message = Message.create(channel, data)
        
        try:
            subscribers = await self._redis.publish(channel, message.to_json())
            self._messages_sent += 1
            return subscribers
        except Exception as e:
            self._errors += 1
            logger.error(f"Error publishing to {channel}: {e}")
            raise
    
    async def subscribe(self, channel: str, callback: Callable[[Message], Any]) -> None:
        """
        Subscribe to channel.
        
        Args:
            channel: Channel name
            callback: Function to call on message
        """
        if not self._pubsub:
            raise RuntimeError("Not connected to Redis")
        
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
            await self._pubsub.subscribe(channel)
        
        self._subscriptions[channel].append(callback)
        logger.info(f"Subscribed to {channel}")
    
    async def unsubscribe(self, channel: str, callback: Optional[Callable] = None) -> None:
        """
        Unsubscribe from channel.
        
        Args:
            channel: Channel name
            callback: Specific callback to remove
        """
        if channel not in self._subscriptions:
            return
        
        if callback:
            self._subscriptions[channel].remove(callback)
        else:
            self._subscriptions[channel].clear()
        
        if not self._subscriptions[channel]:
            del self._subscriptions[channel]
            if self._pubsub:
                await self._pubsub.unsubscribe(channel)
            logger.info(f"Unsubscribed from {channel}")
    
    async def _listen(self) -> None:
        """Listen for messages (runs in background)."""
        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break
                
                if message["type"] != "message":
                    continue
                
                try:
                    msg = Message.from_json(message["data"])
                    self._messages_received += 1
                    
                    # Call all callbacks for this channel
                    if msg.channel in self._subscriptions:
                        for callback in self._subscriptions[msg.channel]:
                            try:
                                result = callback(msg)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.error(f"Error in callback for {msg.channel}: {e}")
                                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._errors += 1
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in listener: {e}")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        return await self._redis.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        return await self._redis.set(key, value, ex=ex)
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        return await self._redis.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        return await self._redis.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "errors": self._errors,
            "subscriptions": list(self._subscriptions.keys()),
            "connected": self._running
        }


# Global message bus instance
_message_bus: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_message_bus(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None
) -> MessageBus:
    """
    Get or create global message bus.
    
    Args:
        host: Redis host
        port: Redis port
        db: Redis database
        password: Redis password
        
    Returns:
        MessageBus instance
    """
    global _message_bus
    
    if _message_bus is None:
        with _bus_lock:
            if _message_bus is None:
                _message_bus = MessageBus(
                    host=host,
                    port=port,
                    db=db,
                    password=password
                )
    
    return _message_bus


# Simplified interface for common operations
class TradingMessageBus:
    """Simplified interface for trading operations."""
    
    def __init__(self, bus: Optional[MessageBus] = None):
        self.bus = bus or get_message_bus()
    
    async def connect(self) -> None:
        await self.bus.connect()
    
    async def disconnect(self) -> None:
        await self.bus.disconnect()
    
    # Signal operations
    async def publish_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        strategy: str,
        **kwargs
    ) -> int:
        """Publish trading signal."""
        data = {
            "symbol": symbol,
            "action": action,  # BUY, SELL, CLOSE
            "confidence": confidence,
            "strategy": strategy,
            **kwargs
        }
        return await self.bus.publish(Channel.SIGNALS.value, data)
    
    async def on_signal(self, callback: Callable[[Message], Any]) -> None:
        """Subscribe to signals."""
        await self.bus.subscribe(Channel.SIGNALS.value, callback)
    
    # Order operations
    async def publish_order(self, order: Dict[str, Any]) -> int:
        """Publish order."""
        return await self.bus.publish(Channel.ORDERS.value, order)
    
    async def on_order(self, callback: Callable[[Message], Any]) -> None:
        """Subscribe to orders."""
        await self.bus.subscribe(Channel.ORDERS.value, callback)
    
    # Risk operations
    async def publish_risk_event(self, event: Dict[str, Any]) -> int:
        """Publish risk event."""
        return await self.bus.publish(Channel.RISK.value, event)
    
    async def on_risk_event(self, callback: Callable[[Message], Any]) -> None:
        """Subscribe to risk events."""
        await self.bus.subscribe(Channel.RISK.value, callback)
    
    # Execution operations
    async def publish_execution(self, execution: Dict[str, Any]) -> int:
        """Publish execution."""
        return await self.bus.publish(Channel.EXECUTION.value, execution)
    
    async def on_execution(self, callback: Callable[[Message], Any]) -> None:
        """Subscribe to executions."""
        await self.bus.subscribe(Channel.EXECUTION.value, callback)


# Example usage
if __name__ == "__main__":
    print("Testing Message Bus...")
    
    async def test():
        bus = MessageBus(host="localhost", port=6379)
        
        # Define callbacks
        async def on_signal(msg: Message):
            print(f"Signal received: {msg.data}")
        
        async def on_order(msg: Message):
            print(f"Order received: {msg.data}")
        
        try:
            # Connect
            await bus.connect()
            
            # Subscribe
            await bus.subscribe(Channel.SIGNALS.value, on_signal)
            await bus.subscribe(Channel.ORDERS.value, on_order)
            
            # Publish messages
            await bus.publish(
                Channel.SIGNALS.value,
                {"symbol": "BTCUSDT", "action": "BUY", "confidence": 0.95}
            )
            
            await bus.publish(
                Channel.ORDERS.value,
                {"order_id": "12345", "symbol": "BTCUSDT", "side": "BUY"}
            )
            
            # Wait for messages
            await asyncio.sleep(1)
            
            # Print stats
            print(f"\nStats: {bus.get_stats()}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            await bus.disconnect()
    
    asyncio.run(test())

