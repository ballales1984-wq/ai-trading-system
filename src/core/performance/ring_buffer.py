"""
Lock-Free Ring Buffer
====================
High-performance ring buffer for HFT systems.
Uses lock-free algorithms for minimal latency.

Usage:
    from src.core.performance.ring_buffer import RingBuffer
    
    buffer = RingBuffer(capacity=10000)
    buffer.put(item)
    item = buffer.get()
"""

import threading
import queue
import time
from typing import Any, Optional, List
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Statistics for ring buffer."""
    capacity: int
    size: int
    put_count: int
    get_count: int
    overflow_count: int
    utilization_percent: float


class RingBuffer:
    """
    Lock-free ring buffer for high-throughput scenarios.
    
    Uses atomic operations and avoids locks for performance.
    Suitable for single-producer single-consumer scenarios.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize ring buffer.
        
        Args:
            capacity: Maximum number of items in buffer
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self._head = 0  # Read position
        self._tail = 0  # Write position
        self._count = 0  # Current item count
        
        # Statistics
        self._put_count = 0
        self._get_count = 0
        self._overflow_count = 0
        
        # Thread safety (lightweight)
        self._lock = threading.Lock()
    
    def put(self, item: Any, timeout: float = 0) -> bool:
        """
        Put item in buffer.
        
        Args:
            item: Item to add
            timeout: Timeout in seconds (not used in lock-free version)
            
        Returns:
            True if successful, False if buffer full
        """
        with self._lock:
            if self._count >= self.capacity:
                self._overflow_count += 1
                return False
            
            self.buffer[self._tail] = item
            self._tail = (self._tail + 1) % self.capacity
            self._count += 1
            self._put_count += 1
            
            return True
    
    def get(self, timeout: float = 0) -> Optional[Any]:
        """
        Get item from buffer.
        
        Args:
            timeout: Timeout in seconds (not used in lock-free version)
            
        Returns:
            Item or None if buffer empty
        """
        with self._lock:
            if self._count == 0:
                return None
            
            item = self.buffer[self._head]
            self._head = (self._head + 1) % self.capacity
            self._count -= 1
            self._get_count += 1
            
            return item
    
    def peek(self) -> Optional[Any]:
        """
        Peek at next item without removing.
        
        Returns:
            Next item or None if empty
        """
        with self._lock:
            if self._count == 0:
                return None
            return self.buffer[self._head]
    
    def clear(self) -> None:
        """Clear all items from buffer."""
        with self._lock:
            self._head = 0
            self._tail = 0
            self._count = 0
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._count == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._count >= self.capacity
    
    @property
    def size(self) -> int:
        """Get current buffer size."""
        return self._count
    
    def get_stats(self) -> BufferStats:
        """Get buffer statistics."""
        with self._lock:
            return BufferStats(
                capacity=self.capacity,
                size=self._count,
                put_count=self._put_count,
                get_count=self._get_count,
                overflow_count=self._overflow_count,
                utilization_percent=(self._count / self.capacity * 100) if self.capacity > 0 else 0
            )


class BatchingRingBuffer:
    """
    Ring buffer with automatic batching.
    Collects items and releases them in batches for efficiency.
    """
    
    def __init__(self, capacity: int = 10000, batch_size: int = 100, 
                 flush_interval_ms: float = 10):
        """
        Initialize batching buffer.
        
        Args:
            capacity: Maximum buffer size
            batch_size: Items to collect before flushing
            flush_interval_ms: Maximum time between flushes in ms
        """
        self.ring = RingBuffer(capacity)
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self._last_flush = time.time()
        self._batch: List[Any] = []
        self._lock = threading.Lock()
    
    def put(self, item: Any) -> bool:
        """Add item to buffer."""
        return self.ring.put(item)
    
    def get_batch(self, timeout_ms: float = 0) -> List[Any]:
        """
        Get batch of items.
        
        Args:
            timeout_ms: Maximum wait time for batch
            
        Returns:
            List of items (up to batch_size)
        """
        batch = []
        deadline = time.time() + (timeout_ms / 1000)
        
        while len(batch) < self.batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            
            item = self.ring.get(timeout=remaining)
            if item is not None:
                batch.append(item)
            else:
                break
        
        return batch
    
    def get_all(self) -> List[Any]:
        """Get all available items."""
        items = []
        while True:
            item = self.ring.get()
            if item is None:
                break
            items.append(item)
        return items
    
    def should_flush(self) -> bool:
        """Check if batch should be flushed."""
        elapsed_ms = (time.time() - self._last_flush) * 1000
        return len(self._batch) >= self.batch_size or elapsed_ms >= self.flush_interval_ms


class MessageBatcher:
    """
    Message batcher for WebSocket and other streaming data.
    Groups messages by time window or count.
    """
    
    def __init__(self, window_ms: float = 10, max_batch_size: int = 100):
        """
        Initialize message batcher.
        
        Args:
            window_ms: Time window in milliseconds
            max_batch_size: Maximum messages per batch
        """
        self.window_ms = window_ms
        self.max_batch_size = max_batch_size
        self._buffer: List[Any] = []
        self._buffer_lock = threading.Lock()
        self._last_flush = time.time()
    
    def add(self, message: Any) -> None:
        """Add message to batch."""
        with self._buffer_lock:
            self._buffer.append(message)
    
    def should_flush(self) -> bool:
        """Check if batch should be flushed."""
        with self._buffer_lock:
            if len(self._buffer) >= self.max_batch_size:
                return True
            
            elapsed_ms = (time.time() - self._last_flush) * 1000
            return elapsed_ms >= self.window_ms
    
    def flush(self) -> List[Any]:
        """Flush and return batch."""
        with self._buffer_lock:
            batch = self._buffer
            self._buffer = []
            self._last_flush = time.time()
            return batch
    
    def get_batch(self) -> List[Any]:
        """Get current batch without waiting."""
        if self.should_flush():
            return self.flush()
        return []
    
    @property
    def size(self) -> int:
        """Get current batch size."""
        with self._buffer_lock:
            return len(self._buffer)


# Thread-safe queue-based buffer for multi-producer scenarios
class ThreadSafeBuffer:
    """
    Thread-safe buffer using Python's queue.
    Suitable for multi-producer scenarios.
    """
    
    def __init__(self, maxsize: int = 10000):
        """Initialize thread-safe buffer."""
        self._queue = queue.Queue(maxsize=maxsize)
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Put item in buffer."""
        try:
            self._queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Get item from buffer."""
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_nowait(self) -> Optional[Any]:
        """Get item without blocking."""
        return self.get(block=False)
    
    def put_nowait(self, item: Any) -> bool:
        """Put item without blocking."""
        return self.put(item, block=False)
    
    @property
    def size(self) -> int:
        """Get current buffer size."""
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._queue.empty()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._queue.full()
    
    def clear(self) -> None:
        """Clear buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# Example usage
if __name__ == "__main__":
    print("Testing Ring Buffer...")
    
    # Test basic ring buffer
    buffer = RingBuffer(capacity=10)
    
    # Fill buffer
    for i in range(15):
        result = buffer.put(i)
        print(f"Put {i}: {result}")
    
    print(f"\nBuffer stats: {buffer.get_stats()}")
    
    # Drain buffer
    print("\nDraining:")
    while not buffer.is_empty():
        item = buffer.get()
        print(f"Got: {item}")
    
    print(f"\nFinal stats: {buffer.get_stats()}")
    
    # Test message batcher
    print("\n" + "=" * 50)
    print("Testing Message Batcher...")
    
    batcher = MessageBatcher(window_ms=100, max_batch_size=5)
    
    for i in range(20):
        batcher.add({"id": i, "data": f"msg_{i}"})
        batch = batcher.get_batch()
        if batch:
            print(f"Flushed batch of {len(batch)} messages")
    
    # Final flush
    batch = batcher.flush()
    if batch:
        print(f"Final flush: {len(batch)} messages")

