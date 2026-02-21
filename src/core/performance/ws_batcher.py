"""
WebSocket Batch Processing
=========================
High-performance batch processing for WebSocket messages.
Reduces overhead by processing messages in batches.

Features:
- Configurable batch size and timeout
- Automatic flushing on size or time threshold
- Backpressure handling
- Metrics collection
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 100  # Max messages per batch
    max_wait_time: float = 0.01  # Max wait time in seconds (10ms)
    max_queue_size: int = 10000  # Max messages in queue
    enable_compression: bool = False  # Compress batches
    metrics_interval: float = 60.0  # Metrics logging interval


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    total_messages: int = 0
    total_batches: int = 0
    messages_dropped: int = 0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    queue_size: int = 0
    batches_per_second: float = 0.0
    
    # Internal tracking
    _latency_samples: List[float] = field(default_factory=list)
    _batch_sizes: List[int] = field(default_factory=list)
    _last_reset: float = field(default_factory=time.time)
    
    def update_latency(self, latency_ms: float):
        """Update latency metrics."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 1000:
            self._latency_samples.pop(0)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
    
    def update_batch_size(self, size: int):
        """Update batch size metrics."""
        self._batch_sizes.append(size)
        if len(self._batch_sizes) > 1000:
            self._batch_sizes.pop(0)
        self.avg_batch_size = sum(self._batch_sizes) / len(self._batch_sizes)
    
    def reset_periodic(self):
        """Reset periodic metrics."""
        now = time.time()
        elapsed = now - self._last_reset
        if elapsed > 0:
            self.batches_per_second = self.total_batches / elapsed
        self._last_reset = now


class MessageBatcher(Generic[T]):
    """
    Generic message batcher for WebSocket data.
    
    Collects messages and processes them in batches for efficiency.
    
    Usage:
        batcher = MessageBatcher(
            process_callback=process_market_data,
            config=BatchConfig(max_batch_size=50, max_wait_time=0.005)
        )
        
        await batcher.start()
        
        # Add messages
        await batcher.add(message)
        
        # Stop
        await batcher.stop()
    """
    
    def __init__(
        self,
        process_callback: Callable[[List[T]], Any],
        config: BatchConfig = None,
        name: str = "default",
    ):
        self.process_callback = process_callback
        self.config = config or BatchConfig()
        self.name = name
        
        self._queue: deque = deque(maxlen=self.config.max_queue_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._flush_event = asyncio.Event()
        self._metrics = BatchMetrics()
        self._last_flush = time.time()
    
    async def start(self):
        """Start the batch processor."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(f"MessageBatcher '{self.name}' started")
    
    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        
        # Flush remaining messages
        if self._queue:
            await self._flush()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"MessageBatcher '{self.name}' stopped")
    
    async def add(self, message: T) -> bool:
        """
        Add a message to the batch queue.
        
        Returns:
            True if message was queued, False if dropped
        """
        if not self._running:
            logger.warning(f"MessageBatcher '{self.name}' not running, message dropped")
            return False
        
        if len(self._queue) >= self.config.max_queue_size:
            self._metrics.messages_dropped += 1
            logger.warning(
                f"MessageBatcher '{self.name}' queue full, "
                f"dropped {self._metrics.messages_dropped} messages"
            )
            return False
        
        self._queue.append(message)
        self._metrics.queue_size = len(self._queue)
        
        # Signal flush if batch size reached
        if len(self._queue) >= self.config.max_batch_size:
            self._flush_event.set()
        
        return True
    
    async def add_batch(self, messages: List[T]) -> int:
        """
        Add multiple messages to the queue.
        
        Returns:
            Number of messages successfully queued
        """
        queued = 0
        for msg in messages:
            if await self.add(msg):
                queued += 1
        return queued
    
    async def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                # Wait for flush event or timeout
                try:
                    await asyncio.wait_for(
                        self._flush_event.wait(),
                        timeout=self.config.max_wait_time
                    )
                except asyncio.TimeoutError:
                    pass  # Timeout reached, flush anyway
                
                self._flush_event.clear()
                
                # Flush if we have messages
                if self._queue:
                    await self._flush()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MessageBatcher '{self.name}' error: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop
    
    async def _flush(self):
        """Flush the current batch."""
        if not self._queue:
            return
        
        start_time = time.perf_counter()
        
        # Extract batch
        batch = []
        while len(batch) < self.config.max_batch_size and self._queue:
            batch.append(self._queue.popleft())
        
        if not batch:
            return
        
        # Update metrics
        self._metrics.total_batches += 1
        self._metrics.total_messages += len(batch)
        self._metrics.update_batch_size(len(batch))
        self._metrics.queue_size = len(self._queue)
        
        try:
            # Process batch
            if asyncio.iscoroutinefunction(self.process_callback):
                await self.process_callback(batch)
            else:
                self.process_callback(batch)
            
            # Update latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._metrics.update_latency(latency_ms)
            
        except Exception as e:
            logger.error(f"MessageBatcher '{self.name}' process error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        self._metrics.reset_periodic()
        return {
            "name": self.name,
            "total_messages": self._metrics.total_messages,
            "total_batches": self._metrics.total_batches,
            "messages_dropped": self._metrics.messages_dropped,
            "avg_batch_size": round(self._metrics.avg_batch_size, 2),
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "max_latency_ms": round(self._metrics.max_latency_ms, 2),
            "queue_size": self._metrics.queue_size,
            "batches_per_second": round(self._metrics.batches_per_second, 2),
        }


# ============================================================================
# SPECIALIZED BATCHERS
# ============================================================================

class MarketDataBatcher(MessageBatcher[Dict]):
    """
    Specialized batcher for market data messages.
    
    Features:
    - Symbol-based aggregation
    - OHLCV updates batching
    - Trade aggregation
    """
    
    def __init__(
        self,
        process_callback: Callable[[List[Dict]], Any],
        config: BatchConfig = None,
    ):
        super().__init__(process_callback, config, name="market_data")
        self._symbol_data: Dict[str, Dict] = {}
    
    async def add_trade(self, symbol: str, price: float, quantity: float, timestamp: datetime = None):
        """Add a trade to be batched."""
        message = {
            "type": "trade",
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        }
        return await self.add(message)
    
    async def add_ticker(self, symbol: str, bid: float, ask: float, last: float):
        """Add a ticker update."""
        message = {
            "type": "ticker",
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return await self.add(message)
    
    async def add_kline(self, symbol: str, interval: str, ohlcv: Dict):
        """Add a kline/candlestick update."""
        message = {
            "type": "kline",
            "symbol": symbol,
            "interval": interval,
            "open": ohlcv.get("open"),
            "high": ohlcv.get("high"),
            "low": ohlcv.get("low"),
            "close": ohlcv.get("close"),
            "volume": ohlcv.get("volume"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return await self.add(message)


class OrderBookBatcher(MessageBatcher[Dict]):
    """
    Specialized batcher for order book updates.
    
    Features:
    - Price level aggregation
    - Delta updates
    - Depth snapshot handling
    """
    
    def __init__(
        self,
        process_callback: Callable[[List[Dict]], Any],
        config: BatchConfig = None,
    ):
        super().__init__(process_callback, config, name="orderbook")
    
    async def add_depth_update(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
        is_snapshot: bool = False,
    ):
        """Add an order book depth update."""
        message = {
            "type": "depth_snapshot" if is_snapshot else "depth_update",
            "symbol": symbol,
            "bids": bids,  # [[price, qty], ...]
            "asks": asks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return await self.add(message)


class SignalBatcher(MessageBatcher[Dict]):
    """
    Specialized batcher for trading signals.
    
    Features:
    - Signal deduplication
    - Priority handling
    - Signal aggregation
    """
    
    def __init__(
        self,
        process_callback: Callable[[List[Dict]], Any],
        config: BatchConfig = None,
    ):
        super().__init__(process_callback, config, name="signals")
        self._seen_signals: Dict[str, float] = {}
    
    async def add_signal(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        direction: str,
        strength: float,
        metadata: Dict = None,
    ):
        """Add a trading signal."""
        # Deduplicate
        now = time.time()
        if signal_id in self._seen_signals:
            if now - self._seen_signals[signal_id] < 1.0:  # 1 second dedup window
                return True
        
        self._seen_signals[signal_id] = now
        
        # Cleanup old entries
        if len(self._seen_signals) > 1000:
            cutoff = now - 60.0  # Keep 1 minute
            self._seen_signals = {
                k: v for k, v in self._seen_signals.items()
                if v > cutoff
            }
        
        message = {
            "signal_id": signal_id,
            "symbol": symbol,
            "type": signal_type,
            "direction": direction,
            "strength": strength,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return await self.add(message)


# ============================================================================
# BATCH PROCESSOR MANAGER
# ============================================================================

class BatchProcessorManager:
    """
    Manages multiple batch processors.
    
    Usage:
        manager = BatchProcessorManager()
        
        # Create batchers
        market_batcher = manager.create_market_data_batcher(process_market)
        orderbook_batcher = manager.create_orderbook_batcher(process_orderbook)
        
        # Start all
        await manager.start_all()
        
        # Stop all
        await manager.stop_all()
    """
    
    def __init__(self):
        self._batchers: Dict[str, MessageBatcher] = {}
    
    def add_batcher(self, batcher: MessageBatcher, name: str = None):
        """Add a batcher to manage."""
        name = name or batcher.name
        self._batchers[name] = batcher
    
    def create_market_data_batcher(
        self,
        process_callback: Callable,
        config: BatchConfig = None,
    ) -> MarketDataBatcher:
        """Create and register a market data batcher."""
        batcher = MarketDataBatcher(process_callback, config)
        self.add_batcher(batcher)
        return batcher
    
    def create_orderbook_batcher(
        self,
        process_callback: Callable,
        config: BatchConfig = None,
    ) -> OrderBookBatcher:
        """Create and register an order book batcher."""
        batcher = OrderBookBatcher(process_callback, config)
        self.add_batcher(batcher)
        return batcher
    
    def create_signal_batcher(
        self,
        process_callback: Callable,
        config: BatchConfig = None,
    ) -> SignalBatcher:
        """Create and register a signal batcher."""
        batcher = SignalBatcher(process_callback, config)
        self.add_batcher(batcher)
        return batcher
    
    async def start_all(self):
        """Start all batchers."""
        for batcher in self._batchers.values():
            await batcher.start()
        logger.info(f"Started {len(self._batchers)} batch processors")
    
    async def stop_all(self):
        """Stop all batchers."""
        for batcher in self._batchers.values():
            await batcher.stop()
        logger.info(f"Stopped {len(self._batchers)} batch processors")
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics from all batchers."""
        return {
            name: batcher.get_metrics()
            for name, batcher in self._batchers.items()
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_fast_batcher(
    process_callback: Callable[[List[T]], Any],
    name: str = "fast",
) -> MessageBatcher[T]:
    """Create a fast batcher optimized for low latency."""
    config = BatchConfig(
        max_batch_size=50,
        max_wait_time=0.005,  # 5ms
        max_queue_size=5000,
    )
    return MessageBatcher(process_callback, config, name)


def create_throughput_batcher(
    process_callback: Callable[[List[T]], Any],
    name: str = "throughput",
) -> MessageBatcher[T]:
    """Create a batcher optimized for high throughput."""
    config = BatchConfig(
        max_batch_size=500,
        max_wait_time=0.05,  # 50ms
        max_queue_size=50000,
    )
    return MessageBatcher(process_callback, config, name)
