"""
Database Write Batcher
=====================
High-performance database write batching for TimescaleDB.
Groups writes to reduce I/O and improve throughput.

Usage:
    from src.core.performance.db_batcher import DatabaseBatcher, get_db_batcher
    
    batcher = get_db_batcher()
    await batcher.write_order(order)
    await batcher.flush()  # Force flush
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading
import json


logger = logging.getLogger(__name__)


@dataclass
class WriteBatch:
    """Batch of database writes."""
    table: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    @property
    def size(self) -> int:
        return len(self.records)
    
    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000


class DatabaseBatcher:
    """
    Database write batcher for high-throughput scenarios.
    
    Features:
    - Automatic batching by count or time window
    - Table-specific batches
    - Background async flushing
    - Manual flush support
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        flush_interval_ms: float = 1000,
        max_queue_size: int = 10000,
        flush_callback: Optional[Callable] = None
    ):
        """
        Initialize database batcher.
        
        Args:
            batch_size: Maximum records per batch
            flush_interval_ms: Maximum time between flushes in ms
            max_queue_size: Maximum total queued records
            flush_callback: Async function to call for flushing
        """
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self.max_queue_size = max_queue_size
        self.flush_callback = flush_callback
        
        # Table-specific batches
        self._batches: Dict[str, WriteBatch] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self._total_writes = 0
        self._total_flushes = 0
        self._total_records = 0
        self._errors = 0
        self._last_flush_time = time.time()
        
        # Background task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"DatabaseBatcher initialized (batch_size={batch_size}, flush_interval={flush_interval_ms}ms)")
    
    async def start(self) -> None:
        """Start background flush task."""
        if self._running:
            return
        
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("DatabaseBatcher background flush started")
    
    async def stop(self) -> None:
        """Stop background flush and flush remaining."""
        if not self._running:
            return
        
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        logger.info("DatabaseBatcher stopped")
    
    def write(self, table: str, record: Dict[str, Any]) -> None:
        """
        Queue a record for batched writing.
        
        Args:
            table: Table name
            record: Record to write
        """
        with self._lock:
            # Create batch if needed
            if table not in self._batches:
                self._batches[table] = WriteBatch(table=table)
            
            batch = self._batches[table]
            batch.records.append(record)
            self._total_writes += 1
            
            # Check if should flush
            if len(batch.records) >= self.batch_size:
                asyncio.create_task(self._flush_table(table))
    
    async def write_async(self, table: str, record: Dict[str, Any]) -> None:
        """Async version of write."""
        self.write(table, record)
    
    def write_many(self, table: str, records: List[Dict[str, Any]]) -> None:
        """
        Queue multiple records.
        
        Args:
            table: Table name
            records: List of records
        """
        for record in records:
            self.write(table, record)
    
    async def flush(self) -> None:
        """Flush all pending writes."""
        tables = []
        with self._lock:
            tables = list(self._batches.keys())
        
        for table in tables:
            await self._flush_table(table)
    
    async def _flush_table(self, table: str) -> None:
        """Flush a specific table."""
        batch = None
        with self._lock:
            if table in self._batches:
                batch = self._batches.pop(table)
        
        if batch is None or not batch.records:
            return
        
        try:
            await self._execute_flush(table, batch.records)
            self._total_flushes += 1
            self._total_records += len(batch.records)
            self._last_flush_time = time.time()
        except Exception as e:
            self._errors += 1
            logger.error(f"Error flushing {table}: {e}")
            # Re-queue records on error
            with self._lock:
                if table not in self._batches:
                    self._batches[table] = WriteBatch(table=table)
                self._batches[table].records.extend(batch.records)
    
    async def _execute_flush(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Execute the actual database write."""
        if self.flush_callback:
            await self.flush_callback(table, records)
        else:
            # Default: just log
            logger.debug(f"Flushing {len(records)} records to {table}")
    
    async def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_ms / 1000)
                
                # Check all batches for age
                with self._lock:
                    for table, batch in list(self._batches.items()):
                        age_ms = batch.age_ms
                        if age_ms >= self.flush_interval_ms or len(batch.records) >= self.batch_size:
                            asyncio.create_task(self._flush_table(table))
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        with self._lock:
            pending = sum(len(b.records) for b in self._batches.values())
        
        return {
            "total_writes": self._total_writes,
            "total_flushes": self._total_flushes,
            "total_records": self._total_records,
            "errors": self._errors,
            "pending_records": pending,
            "tables": list(self._batches.keys()),
            "last_flush_age_ms": (time.time() - self._last_flush_time) * 1000
        }


# Global batcher instance
_batcher: Optional[DatabaseBatcher] = None
_batcher_lock = threading.Lock()


def get_db_batcher(
    batch_size: int = 100,
    flush_interval_ms: float = 1000,
    flush_callback: Optional[Callable] = None
) -> DatabaseBatcher:
    """
    Get or create global database batcher.
    
    Args:
        batch_size: Records per batch
        flush_interval_ms: Flush interval in ms
        flush_callback: Async flush function
        
    Returns:
        DatabaseBatcher instance
    """
    global _batcher
    
    if _batcher is None:
        with _batcher_lock:
            if _batcher is None:
                _batcher = DatabaseBatcher(
                    batch_size=batch_size,
                    flush_interval_ms=flush_interval_ms,
                    flush_callback=flush_callback
                )
    
    return _batcher


# Simplified batcher for orders
class OrderBatcher:
    """Specialized batcher for order records."""
    
    def __init__(self, batcher: Optional[DatabaseBatcher] = None):
        self.batcher = batcher or get_db_batcher()
    
    async def write_order(self, order: Dict[str, Any]) -> None:
        """Write order record."""
        # Ensure required fields
        order["timestamp"] = order.get("timestamp", datetime.now().isoformat())
        self.batcher.write("orders", order)
    
    async def write_orders(self, orders: List[Dict[str, Any]]) -> None:
        """Write multiple orders."""
        for order in orders:
            await self.write_order(order)
    
    async def flush(self) -> None:
        """Flush pending orders."""
        await self.batcher.flush()


# Specialized batcher for price data
class PriceBatcher:
    """Specialized batcher for price/tick data."""
    
    def __init__(self, batcher: Optional[DatabaseBatcher] = None):
        self.batcher = batcher or get_db_batcher(batch_size=500, flush_interval_ms=100)
    
    async def write_price(self, symbol: str, price: float, volume: float = 0) -> None:
        """Write price tick."""
        record = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now().isoformat()
        }
        self.batcher.write("prices", record)
    
    async def write_prices(self, prices: List[Dict[str, Any]]) -> None:
        """Write multiple prices."""
        for price in prices:
            self.batcher.write("prices", price)
    
    async def flush(self) -> None:
        """Flush pending prices."""
        await self.batcher.flush()


# Example usage
if __name__ == "__main__":
    print("Testing Database Batcher...")
    
    async def test():
        # Create batcher with callback
        async def my_flush(table: str, records: List[Dict]):
            print(f"Flushing {len(records)} records to {table}")
            # In real usage, this would be an async DB insert
        
        batcher = DatabaseBatcher(
            batch_size=5,
            flush_interval_ms=1000,
            flush_callback=my_flush
        )
        
        # Start background flush
        await batcher.start()
        
        # Write some records
        for i in range(20):
            batcher.write("orders", {"id": i, "amount": i * 10})
        
        print(f"Stats after writes: {batcher.get_stats()}")
        
        # Wait for background flush
        await asyncio.sleep(1.5)
        
        print(f"Stats after flush: {batcher.get_stats()}")
        
        # Stop
        await batcher.stop()
        
        # Test specialized batcher
        print("\nTesting PriceBatcher...")
        price_batcher = PriceBatcher()
        await price_batcher.start()
        
        for i in range(10):
            await price_batcher.write_price("BTCUSDT", 50000 + i)
        
        await asyncio.sleep(0.2)
        await price_batcher.flush()
        await price_batcher.batcher.stop()
    
    asyncio.run(test())

