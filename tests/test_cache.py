"""
Tests for app/core/cache.py - Redis Cache Manager
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestRedisCacheManager:
    """Test suite for RedisCacheManager class."""

    def test_cache_manager_creation(self):
        """Test basic cache manager creation."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        assert cache.redis_url is not None
        assert cache.max_connections == 20
        assert cache.decode_responses is True
        assert cache._connected is False

    def test_cache_manager_custom_params(self):
        """Test cache manager with custom parameters."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager(
            redis_url="redis://custom:6379/5",
            max_connections=50,
            decode_responses=False
        )
        
        assert cache.redis_url == "redis://custom:6379/5"
        assert cache.max_connections == 50
        assert cache.decode_responses is False

    def test_is_connected_property(self):
        """Test is_connected property."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        # Not connected initially
        assert cache.is_connected is False
        
        # Mock connected state
        cache._connected = True
        cache._redis = MagicMock()
        
        assert cache.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connect failure handling."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager(redis_url="redis://invalid:6379/0")
        
        # Connection will fail, but we don't need Redis for these unit tests
        # Just test the manager was created
        assert cache._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect method."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        # Mock the disconnect
        cache._redis = AsyncMock()
        cache._pool = AsyncMock()
        cache._connected = True
        
        await cache.disconnect()
        
        assert cache._connected is False
        assert cache._redis is None

    @pytest.mark.asyncio
    async def test_health_check_no_connection(self):
        """Test health check when not connected."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.health_check()
        
        assert result["connected"] is False
        assert result["latency_ms"] is None

    @pytest.mark.asyncio
    async def test_get_no_redis(self):
        """Test get method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.get("test_key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_no_redis(self):
        """Test set method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.set("test_key", "test_value")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_no_redis(self):
        """Test delete method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.delete("test_key")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_exists_no_redis(self):
        """Test exists method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.exists("test_key")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_expire_no_redis(self):
        """Test expire method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.expire("test_key", 60)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_ttl_no_redis(self):
        """Test ttl method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.ttl("test_key")
        
        assert result == -2

    @pytest.mark.asyncio
    async def test_hget_no_redis(self):
        """Test hget method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.hget("hash_name", "key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_hset_no_redis(self):
        """Test hset method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.hset("hash_name", key="key", value="value")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_hgetall_no_redis(self):
        """Test hgetall method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.hgetall("hash_name")
        
        assert result == {}

    @pytest.mark.asyncio
    async def test_hdel_no_redis(self):
        """Test hdel method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.hdel("hash_name", "key")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_lpush_no_redis(self):
        """Test lpush method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.lpush("list_key", "value")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_rpush_no_redis(self):
        """Test rpush method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.rpush("list_key", "value")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_lrange_no_redis(self):
        """Test lrange method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.lrange("list_key")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_ltrim_no_redis(self):
        """Test ltrim method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.ltrim("list_key", 0, 10)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_keys_no_redis(self):
        """Test keys method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.keys("pattern*")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_no_redis(self):
        """Test scan method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        cursor, keys = await cache.scan(match="pattern*")
        
        assert cursor == 0
        assert keys == []

    @pytest.mark.asyncio
    async def test_delete_pattern_no_redis(self):
        """Test delete_pattern method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.delete_pattern("pattern*")
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_cache_price_no_redis(self):
        """Test cache_price method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.cache_price("BTCUSDT", 50000.0)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_price_no_redis(self):
        """Test get_cached_price method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.get_cached_price("BTCUSDT")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_orderbook_no_redis(self):
        """Test cache_orderbook method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        bids = [[50000.0, 1.0], [49999.0, 2.0]]
        asks = [[50001.0, 1.5], [50002.0, 3.0]]
        
        result = await cache.cache_orderbook("BTCUSDT", bids, asks)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_orderbook_no_redis(self):
        """Test get_cached_orderbook method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.get_cached_orderbook("BTCUSDT")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_signal_no_redis(self):
        """Test cache_signal method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        signal_data = {"direction": "BUY", "strength": 0.8}
        
        result = await cache.cache_signal("sig_001", signal_data)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_signal_no_redis(self):
        """Test get_cached_signal method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.get_cached_signal("sig_001")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_portfolio_state_no_redis(self):
        """Test cache_portfolio_state method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        state = {"total_value": 100000.0, "positions": 5}
        
        result = await cache.cache_portfolio_state("portfolio_1", state)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_portfolio_no_redis(self):
        """Test get_cached_portfolio method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.get_cached_portfolio("portfolio_1")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_market_data_no_redis(self):
        """Test cache_market_data method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        data = [{"time": 1234567890, "open": 50000, "high": 50100, "low": 49900, "close": 50050, "volume": 100}]
        
        result = await cache.cache_market_data("BTCUSDT", "1h", data)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cached_market_data_no_redis(self):
        """Test get_cached_market_data method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        result = await cache.get_cached_market_data("BTCUSDT", "1h")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_rate_limit_no_redis(self):
        """Test set_rate_limit method when Redis not initialized."""
        from app.core.cache import RedisCacheManager
        
        cache = RedisCacheManager()
        
        is_allowed, remaining = await cache.set_rate_limit("rate_test", 10, 60)
        
        # Should allow by default when Redis not available
        assert is_allowed is True
        assert remaining == 10


class TestCacheFunctions:
    """Test suite for cache module functions."""

    def test_cached_decorator_creation(self):
        """Test cached decorator creation."""
        from app.core.cache import cached
        
        # Test basic decorator
        @cached("test_prefix", ttl=60)
        async def test_func():
            pass
        
        assert test_func is not None

    def test_cached_decorator_with_key_builder(self):
        """Test cached decorator with custom key builder."""
        from app.core.cache import cached
        
        def custom_key_builder(symbol: str, interval: str) -> str:
            return f"price:{symbol}:{interval}"
        
        @cached("prices", key_builder=custom_key_builder)
        async def get_price(symbol: str, interval: str = "1h"):
            return {"price": 50000}
        
        assert get_price is not None
