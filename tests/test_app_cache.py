"""
Tests for Redis Cache Manager
==============================
Tests for the Redis cache manager implementation with mocking.
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.cache import RedisCacheManager


class TestRedisCacheManager:
    """Test Redis cache manager."""
    
    @pytest.fixture
    def mock_redis_client(self, event_loop):
        """Create a mock Redis client."""
        mock_client = Mock()
        
        # Helper to create asyncio.Future with return value
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        # Mock basic operations
        mock_client.get.return_value = async_result(None)
        mock_client.set.return_value = async_result(True)
        mock_client.delete.return_value = async_result(0)
        mock_client.exists.return_value = async_result(0)
        mock_client.expire.return_value = async_result(False)
        mock_client.keys.return_value = async_result([])
        mock_client.flushdb.return_value = async_result(True)
        mock_client.hget.return_value = async_result(None)
        mock_client.hset.return_value = async_result(0)
        mock_client.hgetall.return_value = async_result({})
        mock_client.hdel.return_value = async_result(0)
        mock_client.lpush.return_value = async_result(0)
        mock_client.rpush.return_value = async_result(0)
        mock_client.lrange.return_value = async_result([])
        mock_client.ltrim.return_value = async_result(True)
        
        return mock_client
    
    @pytest.fixture
    def cache_manager(self, mock_redis_client):
        """Create cache manager with mock dependencies."""
        manager = RedisCacheManager(
            redis_url='redis://localhost:6379/0'
        )
        # Mock connection
        manager._connected = True
        manager._redis = mock_redis_client
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test cache manager initialization."""
        manager = RedisCacheManager(redis_url='redis://localhost:6379/0')
        assert manager is not None
        assert not manager.is_connected
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager, mock_redis_client, event_loop):
        """Test setting and getting cache entries."""
        # Helper to create async result
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        # Test set
        mock_value = {"key": "value"}
        await cache_manager.set('test_key', mock_value, ttl=3600)
        
        # Test get
        mock_redis_client.get.return_value = async_result(json.dumps(mock_value))
        result = await cache_manager.get('test_key')
        assert result == mock_value
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache_manager, mock_redis_client, event_loop):
        """Test getting a nonexistent key returns None."""
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        mock_redis_client.get.return_value = async_result(None)
        result = await cache_manager.get('nonexistent_key')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_key(self, cache_manager, mock_redis_client, event_loop):
        """Test deleting a cache entry."""
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        mock_redis_client.delete.return_value = async_result(1)
        result = await cache_manager.delete('test_key')
        assert result == 1
        mock_redis_client.delete.assert_called_once_with('test_key')
    
    @pytest.mark.asyncio
    async def test_exists(self, cache_manager, mock_redis_client, event_loop):
        """Test checking if a key exists."""
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        # Test key exists
        mock_redis_client.exists.return_value = async_result(1)
        result = await cache_manager.exists('existing_key')
        assert result == 1
        
        # Test key doesn't exist
        mock_redis_client.exists.return_value = async_result(0)
        result = await cache_manager.exists('nonexistent_key')
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_expire(self, cache_manager, mock_redis_client, event_loop):
        """Test setting key expiration."""
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        mock_redis_client.expire.return_value = async_result(True)
        result = await cache_manager.expire('test_key', 300)
        assert result is True
        mock_redis_client.expire.assert_called_once_with('test_key', 300)
    
    @pytest.mark.asyncio
    async def test_keys(self, cache_manager, mock_redis_client, event_loop):
        """Test getting keys by pattern."""
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        mock_redis_client.keys.return_value = async_result(['key1', 'key2', 'key3'])
        keys = await cache_manager.keys('key*')
        assert len(keys) == 3
        assert 'key1' in keys
        assert 'key2' in keys
        assert 'key3' in keys
    
    @pytest.mark.asyncio
    async def test_cache_price(self, cache_manager, mock_redis_client):
        """Test caching price data."""
        await cache_manager.cache_price('BTC/USDT', 50000.0)
        mock_redis_client.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cached_price(self, cache_manager, mock_redis_client, event_loop):
        """Test getting cached price data."""
        def async_result(value):
            future = event_loop.create_future()
            future.set_result(value)
            return future
        
        mock_price_data = {
            "price": 50000.0,
            "timestamp": "2024-01-01T00:00:00"
        }
        mock_redis_client.get.return_value = async_result(json.dumps(mock_price_data))
        
        result = await cache_manager.get_cached_price('BTC/USDT')
        assert result is not None
        assert result['price'] == 50000.0
    
    @pytest.mark.asyncio
    async def test_cache_orderbook(self, cache_manager, mock_redis_client):
        """Test caching orderbook data."""
        bids = [[50000, 0.1], [49999, 0.2]]
        asks = [[50001, 0.1], [50002, 0.2]]
        
        await cache_manager.cache_orderbook('BTC/USDT', bids, asks)
        mock_redis_client.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_signal(self, cache_manager, mock_redis_client):
        """Test caching trading signals."""
        signal_data = {
            "strategy": "momentum",
            "symbol": "BTC/USDT",
            "action": "buy",
            "confidence": 0.85
        }
        
        await cache_manager.cache_signal('signal_123', signal_data)
        mock_redis_client.set.assert_called_once()
