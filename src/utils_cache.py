"""
Caching utility for API responses
Provides in-memory caching with TTL support
"""

import time
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class Cache:
    """
    Simple in-memory cache with TTL support
    """
    
    def __init__(self, default_ttl: int = 60):
        """
        Initialize cache
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from args and kwargs"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry['expires_at']:
                self._hits += 1
                logger.debug(f"Cache HIT: {key}")
                return entry['value']
            else:
                # Expired - remove
                del self._cache[key]
        
        self._misses += 1
        logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
        logger.debug(f"Cache SET: {key} (ttl={ttl}s)")
    
    def delete(self, key: str):
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all cache"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self._cache)
        }
    
    def cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if now >= v['expires_at']
        ]
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


# Global cache instance
_global_cache = Cache(default_ttl=60)


def get_cache() -> Cache:
    """Get global cache instance"""
    return _global_cache


def cached(ttl: int = 60, key_prefix: str = ""):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{cache._make_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def invalidate_cache(key_prefix: str = ""):
    """
    Invalidate cache entries matching prefix
    
    Args:
        key_prefix: Prefix to match for invalidation
    """
    cache = get_cache()
    if not key_prefix:
        cache.clear()
    else:
        keys_to_delete = [k for k in cache._cache.keys() if k.startswith(key_prefix)]
        for key in keys_to_delete:
            cache.delete(key)
        logger.info(f"Invalidated {len(keys_to_delete)} cache entries")


# Convenience functions for common caching patterns
class DataCache:
    """Pre-configured cache for market data"""
    
    OHLCV_TTL = 60  # 1 minute
    TICKER_TTL = 10  # 10 seconds
    ORDERBOOK_TTL = 5  # 5 seconds
    
    @staticmethod
    @cached(ttl=OHLCV_TTL, key_prefix="ohlcv")
    def get_ohlcv_cached(symbol: str, timeframe: str) -> Any:
        """Placeholder - actual implementation would fetch data"""
        pass
    
    @staticmethod
    @cached(ttl=TICKER_TTL, key_prefix="ticker")
    def get_ticker_cached(symbol: str) -> Any:
        """Placeholder - actual implementation would fetch data"""
        pass


# Cache for ML predictions
class MLPredictionCache:
    """Cache for ML model predictions"""
    
    PREDICTION_TTL = 300  # 5 minutes
    
    @staticmethod
    @cached(ttl=PREDICTION_TTL, key_prefix="ml_pred")
    def get_prediction_cached(symbol: str, features: Dict) -> Any:
        """Get cached ML prediction"""
        pass


if __name__ == "__main__":
    # Test cache
    cache = Cache(default_ttl=2)
    
    # Test set/get
    cache.set("test_key", {"data": "test_value"})
    result = cache.get("test_key")
    print(f"Test get: {result}")
    
    # Test expiration
    import time
    time.sleep(3)
    result = cache.get("test_key")
    print(f"After expiry: {result}")
    
    # Test stats
    cache.set("key1", "value1")
    cache.get("key1")
    cache.get("key2")  # miss
    print(f"Stats: {cache.get_stats()}")
