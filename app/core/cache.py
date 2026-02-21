"""
Redis Cache Manager
==================
Centralized Redis caching service for the trading system.
Provides caching for market data, signals, and session management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from functools import wraps

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings


logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Redis-based cache manager for the trading system.
    
    Features:
    - Async operations for high performance
    - Automatic serialization/deserialization
    - TTL-based expiration
    - Pattern-based key management
    - Health monitoring
    """
    
    def __init__(
        self,
        redis_url: str = None,
        max_connections: int = 20,
        decode_responses: bool = True,
    ):
        self.redis_url = redis_url or settings.redis_url
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None
        self._connected = False
        
    async def connect(self) -> bool:
        """Establish connection to Redis."""
        try:
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses,
            )
            self._redis = Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            
            logger.info(f"Connected to Redis: {self.redis_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        
        self._redis = None
        self._pool = None
        self._connected = False
        logger.info("Disconnected from Redis")
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._redis is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        result = {
            "connected": False,
            "latency_ms": None,
            "memory_used": None,
            "keys_count": None,
            "error": None,
        }
        
        if not self._redis:
            result["error"] = "Redis client not initialized"
            return result
        
        try:
            start = time.time()
            await self._redis.ping()
            latency = (time.time() - start) * 1000
            
            info = await self._redis.info("memory")
            db_size = await self._redis.dbsize()
            
            result.update({
                "connected": True,
                "latency_ms": round(latency, 2),
                "memory_used": info.get("used_memory_human", "unknown"),
                "keys_count": db_size,
            })
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis:
            return None
            
        try:
            value = await self._redis.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in cache with optional TTL."""
        if not self._redis:
            return False
            
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)
            
            result = await self._redis.set(key, value, ex=ttl, nx=nx, xx=xx)
            return result is not None or result is True
            
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not self._redis or not keys:
            return 0
            
        try:
            return await self._redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return 0
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not self._redis or not keys:
            return 0
            
        try:
            return await self._redis.exists(*keys)
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on a key."""
        if not self._redis:
            return False
            
        try:
            return await self._redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key '{key}': {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL of a key."""
        if not self._redis:
            return -2
            
        try:
            return await self._redis.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL error for key '{key}': {e}")
            return -2
    
    # =========================================================================
    # HASH OPERATIONS
    # =========================================================================
    
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get value from hash."""
        if not self._redis:
            return None
            
        try:
            value = await self._redis.hget(name, key)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis HGET error for '{name}.{key}': {e}")
            return None
    
    async def hset(
        self,
        name: str,
        key: str = None,
        value: Any = None,
        mapping: Dict[str, Any] = None,
    ) -> int:
        """Set value in hash."""
        if not self._redis:
            return 0
            
        try:
            if mapping:
                # Serialize values in mapping
                serialized = {}
                for k, v in mapping.items():
                    if isinstance(v, (dict, list)):
                        serialized[k] = json.dumps(v)
                    else:
                        serialized[k] = str(v) if v is not None else ""
                return await self._redis.hset(name, mapping=serialized)
            elif key and value is not None:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                return await self._redis.hset(name, key, str(value))
            return 0
                
        except Exception as e:
            logger.error(f"Redis HSET error for '{name}': {e}")
            return 0
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all values from hash."""
        if not self._redis:
            return {}
            
        try:
            data = await self._redis.hgetall(name)
            result = {}
            
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
                    
            return result
                
        except Exception as e:
            logger.error(f"Redis HGETALL error for '{name}': {e}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete keys from hash."""
        if not self._redis or not keys:
            return 0
            
        try:
            return await self._redis.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Redis HDEL error for '{name}': {e}")
            return 0
    
    # =========================================================================
    # LIST OPERATIONS
    # =========================================================================
    
    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to left of list."""
        if not self._redis or not values:
            return 0
            
        try:
            serialized = []
            for v in values:
                if isinstance(v, (dict, list)):
                    serialized.append(json.dumps(v))
                else:
                    serialized.append(str(v))
            return await self._redis.lpush(key, *serialized)
        except Exception as e:
            logger.error(f"Redis LPUSH error for '{key}': {e}")
            return 0
    
    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to right of list."""
        if not self._redis or not values:
            return 0
            
        try:
            serialized = []
            for v in values:
                if isinstance(v, (dict, list)):
                    serialized.append(json.dumps(v))
                else:
                    serialized.append(str(v))
            return await self._redis.rpush(key, *serialized)
        except Exception as e:
            logger.error(f"Redis RPUSH error for '{key}': {e}")
            return 0
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of elements from list."""
        if not self._redis:
            return []
            
        try:
            data = await self._redis.lrange(key, start, end)
            result = []
            
            for v in data:
                try:
                    result.append(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.append(v)
                    
            return result
                
        except Exception as e:
            logger.error(f"Redis LRANGE error for '{key}': {e}")
            return []
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        if not self._redis:
            return False
            
        try:
            await self._redis.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.error(f"Redis LTRIM error for '{key}': {e}")
            return False
    
    # =========================================================================
    # PATTERN OPERATIONS
    # =========================================================================
    
    async def keys(self, pattern: str) -> List[str]:
        """Find keys matching pattern."""
        if not self._redis:
            return []
            
        try:
            return await self._redis.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS error for pattern '{pattern}': {e}")
            return []
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self._redis:
            return 0
            
        try:
            keys = await self.keys(pattern)
            if keys:
                return await self.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis DELETE_PATTERN error for '{pattern}': {e}")
            return 0
    
    # =========================================================================
    # TRADING-SPECIFIC CACHING
    # =========================================================================
    
    async def cache_price(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
        ttl: int = 60,
    ) -> bool:
        """Cache current price for a symbol."""
        key = f"price:{symbol}"
        data = {
            "price": price,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        }
        return await self.set(key, data, ttl=ttl)
    
    async def get_cached_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached price for a symbol."""
        key = f"price:{symbol}"
        return await self.get(key)
    
    async def cache_orderbook(
        self,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]],
        ttl: int = 5,
    ) -> bool:
        """Cache order book snapshot."""
        key = f"orderbook:{symbol}"
        data = {
            "bids": bids[:20],  # Top 20 levels
            "asks": asks[:20],
            "timestamp": datetime.utcnow().isoformat(),
        }
        return await self.set(key, data, ttl=ttl)
    
    async def get_cached_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached order book."""
        key = f"orderbook:{symbol}"
        return await self.get(key)
    
    async def cache_signal(
        self,
        signal_id: str,
        signal_data: Dict[str, Any],
        ttl: int = 3600,
    ) -> bool:
        """Cache trading signal."""
        key = f"signal:{signal_id}"
        return await self.set(key, signal_data, ttl=ttl)
    
    async def get_cached_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get cached signal."""
        key = f"signal:{signal_id}"
        return await self.get(key)
    
    async def cache_portfolio_state(
        self,
        portfolio_id: str,
        state: Dict[str, Any],
        ttl: int = 300,
    ) -> bool:
        """Cache portfolio state."""
        key = f"portfolio:{portfolio_id}"
        return await self.set(key, state, ttl=ttl)
    
    async def get_cached_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get cached portfolio state."""
        key = f"portfolio:{portfolio_id}"
        return await self.get(key)
    
    async def cache_market_data(
        self,
        symbol: str,
        interval: str,
        data: List[Dict[str, Any]],
        ttl: int = 300,
    ) -> bool:
        """Cache OHLCV market data."""
        key = f"market:{symbol}:{interval}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_cached_market_data(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached market data."""
        key = f"market:{symbol}:{interval}"
        return await self.get(key)
    
    async def set_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
    ) -> tuple[bool, int]:
        """
        Check and update rate limit using sliding window.
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        if not self._redis:
            return True, limit
            
        try:
            now = time.time()
            window_start = now - window
            
            pipe = self._redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window)
            
            results = await pipe.execute()
            current_count = results[1]
            
            remaining = max(0, limit - current_count - 1)
            is_allowed = current_count < limit
            
            return is_allowed, remaining
                
        except Exception as e:
            logger.error(f"Rate limit error for '{key}': {e}")
            return True, limit


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_cache_manager: Optional[RedisCacheManager] = None


async def get_cache_manager() -> RedisCacheManager:
    """Get or create the cache manager singleton."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = RedisCacheManager()
        await _cache_manager.connect()
    
    return _cache_manager


async def close_cache_manager() -> None:
    """Close the cache manager singleton."""
    global _cache_manager
    
    if _cache_manager is not None:
        await _cache_manager.disconnect()
        _cache_manager = None


# =============================================================================
# DECORATOR FOR CACHING
# =============================================================================

def cached(
    key_prefix: str,
    ttl: int = 300,
    key_builder: Optional[callable] = None,
):
    """
    Decorator for caching function results.
    
    Usage:
        @cached("prices", ttl=60)
        async def get_price(symbol: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_cache_manager()
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key from args
                key_parts = [key_prefix]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator
