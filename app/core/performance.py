"""
Performance Optimization Module
==============================
Performance profiling, caching decorators, and optimization utilities.

Author: AI Trading System
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function."""
    function_name: str = ""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_called: datetime = field(default_factory=datetime.now)
    
    def record_call(self, duration: float):
        """Record a function call duration."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.call_count
        self.last_called = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time * 1000, 2),
            "min_time_ms": round(self.min_time * 1000, 2) if self.min_time != float('inf') else 0,
            "max_time_ms": round(self.max_time * 1000, 2),
            "avg_time_ms": round(self.avg_time * 1000, 2),
            "last_called": self.last_called.isoformat(),
        }


class PerformanceProfiler:
    """
    Performance Profiler
    ====================
    Tracks performance metrics for functions.
    """
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
    
    def record(self, function_name: str, duration: float):
        """Record a function call."""
        if function_name not in self._metrics:
            self._metrics[function_name] = PerformanceMetrics(function_name=function_name)
        self._metrics[function_name].record_call(duration)
    
    def get_metrics(self, function_name: str = None) -> Dict:
        """Get metrics for a specific function or all functions."""
        if function_name:
            if function_name in self._metrics:
                return self._metrics[function_name].to_dict()
            return {}
        
        return {name: metrics.to_dict() for name, metrics in self._metrics.items()}
    
    def reset(self, function_name: str = None):
        """Reset metrics for a specific function or all functions."""
        if function_name:
            if function_name in self._metrics:
                self._metrics[function_name] = PerformanceMetrics(function_name=function_name)
        else:
            self._metrics.clear()
    
    def get_slowest_functions(self, limit: int = 10) -> list:
        """Get the slowest functions by average time."""
        sorted_metrics = sorted(
            self._metrics.values(),
            key=lambda m: m.avg_time,
            reverse=True
        )
        return [m.to_dict() for m in sorted_metrics[:limit]]
    
    def get_most_called_functions(self, limit: int = 10) -> list:
        """Get the most frequently called functions."""
        sorted_metrics = sorted(
            self._metrics.values(),
            key=lambda m: m.call_count,
            reverse=True
        )
        return [m.to_dict() for m in sorted_metrics[:limit]]


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _profiler


def profile(func: Callable = None, *, name: str = None) -> Callable:
    """
    Decorator to profile function execution time.
    
    Usage:
        @profile
        def my_function():
            pass
        
        @profile(name="custom_name")
        def another_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        func_name = name or f.__name__
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                _profiler.record(func_name, duration)
                if duration > 1.0:  # Log if > 1 second
                    logger.warning(f"Slow function: {func_name} took {duration:.3f}s")
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                _profiler.record(func_name, duration)
                if duration > 1.0:
                    logger.warning(f"Slow async function: {func_name} took {duration:.3f}s")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        return sync_wrapper
    
    # Handle both @profile and @profile() syntax
    if func is None:
        return decorator
    return decorator(func)


# Cache implementation with TTL support

class CacheEntry:
    """Cache entry with TTL."""
    
    def __init__(self, value: Any, ttl: float):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl
    
    def get_value(self) -> Optional[Any]:
        """Get value if not expired."""
        return self.value if not self.is_expired() else None


class TimedCache:
    """
    Timed Cache
    ==========
    Simple in-memory cache with TTL support.
    """
    
    def __init__(self, default_ttl: float = 60.0):
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            value = entry.get_value()
            if value is not None:
                self._hits += 1
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: float = None):
        """Set value in cache."""
        ttl = ttl or self._default_ttl
        self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str):
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate * 100, 2),
        }


# Global cache instance
_cache = TimedCache()


def get_cache() -> TimedCache:
    """Get the global cache instance."""
    return _cache


def cached(ttl: float = 60.0, key_func: Callable = None):
    """
    Decorator to cache function results.
    
    Usage:
        @cached(ttl=60)
        def expensive_function(arg1, arg2):
            # Expensive computation
            return result
    """
    def decorator(func: Callable) -> Callable:
        func_name = func.__name__
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func_name}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func_name}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = await func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Batch processing utilities

async def batch_process(
    items: list,
    processor: Callable,
    batch_size: int = 10,
    delay: float = 0.0,
) -> list:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        delay: Delay between batches in seconds
    
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[processor(item) for item in batch],
            return_exceptions=True
        )
        
        results.extend(batch_results)
        
        # Delay between batches
        if delay > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay)
    
    return results


def chunk_list(items: list, chunk_size: int) -> list:
    """Split list into chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# Rate limiter with backoff

class ExponentialBackoff:
    """
    Exponential Backoff
    ==================
    Simple exponential backoff for retries.
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.attempt = 0
    
    def reset(self):
        """Reset the backoff."""
        self.attempt = 0
    
    def get_delay(self) -> float:
        """Get the next delay."""
        delay = min(self.base_delay * (self.multiplier ** self.attempt), self.max_delay)
        self.attempt += 1
        return delay
    
    async def wait(self):
        """Wait for the backoff delay."""
        delay = self.get_delay()
        await asyncio.sleep(delay)


# Connection pool utilities

class SimpleConnectionPool:
    """
    Simple Connection Pool
    =====================
    Basic connection pool for managing resources.
    """
    
    def __init__(
        self,
        factory: Callable,
        min_size: int = 1,
        max_size: int = 10,
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self._pool = []
        self._in_use = set()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self._lock:
            # Try to get from pool
            while self._pool:
                conn = self._pool.pop()
                if conn is not None:
                    self._in_use.add(conn)
                    return conn
            
            # Create new if under max
            if len(self._in_use) < self.max_size:
                conn = await self.factory()
                self._in_use.add(conn)
                return conn
        
        # Wait for available connection
        while True:
            await asyncio.sleep(0.1)
            async with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use.add(conn)
                    return conn
    
    async def release(self, conn):
        """Release a connection back to the pool."""
        async with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._pool.append(conn)
    
    async def close_all(self):
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool + list(self._in_use):
                if hasattr(conn, 'close'):
                    await conn.close()
            self._pool.clear()
            self._in_use.clear()
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            "available": len(self._pool),
            "in_use": len(self._in_use),
            "total": len(self._pool) + len(self._in_use),
            "max_size": self.max_size,
        }


# API Routes for performance monitoring

def create_performance_routes(app):
    """Create performance monitoring API routes."""
    
    @app.get("/api/performance/metrics")
    async def get_performance_metrics():
        """Get performance metrics for all tracked functions."""
        return _profiler.get_metrics()
    
    @app.get("/api/performance/slowest")
    async def get_slowest_functions(limit: int = 10):
        """Get the slowest functions."""
        return _profiler.get_slowest_functions(limit)
    
    @app.get("/api/performance/most_called")
    async def get_most_called_functions(limit: int = 10):
        """Get the most frequently called functions."""
        return _profiler.get_most_called_functions(limit)
    
    @app.post("/api/performance/reset")
    async def reset_performance_metrics():
        """Reset all performance metrics."""
        _profiler.reset()
        return {"message": "Performance metrics reset"}
    
    @app.get("/api/performance/cache/stats")
    async def get_cache_stats():
        """Get cache statistics."""
        return _cache.get_stats()
    
    @app.post("/api/performance/cache/clear")
    async def clear_cache():
        """Clear the cache."""
        _cache.clear()
        return {"message": "Cache cleared"}
    
    @app.post("/api/performance/cache/cleanup")
    async def cleanup_cache():
        """Remove expired cache entries."""
        _cache.cleanup_expired()
        return {"message": "Expired cache entries removed"}


__all__ = [
    "PerformanceMetrics",
    "PerformanceProfiler",
    "get_profiler",
    "profile",
    "TimedCache",
    "get_cache",
    "cached",
    "batch_process",
    "chunk_list",
    "ExponentialBackoff",
    "SimpleConnectionPool",
    "create_performance_routes",
]
