"""
Tests for Redis Cache Manager
==============================
Tests for the Redis caching functionality using mocks.
"""

import pytest
import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRedisCacheManagerMocked:
    """Test RedisCacheManager with mocked Redis."""
    
    def setup_method(self):
        """Setup test data with mocked Redis."""
        # Mock the settings to avoid import issues
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379/0"
            from app.core.cache import RedisCacheManager
            self.cache = RedisCacheManager(redis_url="redis://localhost:6379/0")
    
    def test_initialization(self):
        """Test RedisCacheManager initialization."""
        assert self.cache is not None
        assert self.cache.redis_url == "redis://localhost:6379/0"
        assert self.cache.max_connections == 20
        assert self.cache.decode_responses is True
        assert self.cache._connected is False
    
    def test_is_connected_false(self):
        """Test is_connected when not connected."""
        assert self.cache.is_connected is False
    
    def test_connect_success(self):
        """Test successful connection to Redis."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        
        mock_pool = MagicMock()
        
        with patch('redis.asyncio.ConnectionPool.from_url', return_value=mock_pool), \
             patch('app.core.cache.Redis', return_value=mock_redis):
            
            result = asyncio.run(self.cache.connect())
            
            assert result is True
            assert self.cache._connected is True
    
    def test_connect_failure(self):
        """Test failed connection to Redis."""
        with patch('redis.asyncio.ConnectionPool.from_url') as mock_pool:
            mock_pool.side_effect = Exception("Connection refused")
            
            result = asyncio.run(self.cache.connect())
            
            assert result is False
            assert self.cache._connected is False
    
    def test_get_not_connected(self):
        """Test get when not connected."""
        result = asyncio.run(self.cache.get("test_key"))
        
        assert result is None
    
    def test_set_not_connected(self):
        """Test set when not connected."""
        result = asyncio.run(self.cache.set("test_key", "test_value"))
        
        assert result is False
    
    def test_get_with_mock_redis(self):
        """Test get with mocked Redis."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps({"key": "value"}))
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.get("test_key"))
        
        assert result == {"key": "value"}
    
    def test_get_none_value(self):
        """Test get when key doesn't exist."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.get("nonexistent_key"))
        
        assert result is None
    
    def test_get_string_value(self):
        """Test get with string value."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="plain_string")
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.get("test_key"))
        
        assert result == "plain_string"
    
    def test_set_with_mock_redis(self):
        """Test set with mocked Redis."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.set("test_key", {"key": "value"}))
        
        assert result is True
    
    def test_set_with_ttl(self):
        """Test set with TTL."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.set("test_key", "value", ttl=60))
        
        assert result is True
        # Verify set was called with ex parameter
        mock_redis.set.assert_called_once()
    
    def test_delete_with_mock_redis(self):
        """Test delete with mocked Redis."""
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.delete("test_key"))
        
        # delete returns int (number of keys deleted)
        assert result == 1
    
    def test_delete_not_connected(self):
        """Test delete when not connected."""
        result = asyncio.run(self.cache.delete("test_key"))
        
        # delete returns 0 when not connected
        assert result == 0
    
    def test_health_check_not_connected(self):
        """Test health check when not connected."""
        result = asyncio.run(self.cache.health_check())
        
        assert result['connected'] is False
        assert result['error'] == "Redis client not initialized"
    
    def test_health_check_connected(self):
        """Test health check when connected."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.info = AsyncMock(return_value={"used_memory_human": "1.5M"})
        mock_redis.dbsize = AsyncMock(return_value=100)
        self.cache._redis = mock_redis
        self.cache._connected = True
        
        result = asyncio.run(self.cache.health_check())
        
        assert result['connected'] is True
        assert result['latency_ms'] is not None
        assert result['memory_used'] == "1.5M"
        assert result['keys_count'] == 100
    
    def test_disconnect(self):
        """Test disconnecting from Redis."""
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.disconnect = AsyncMock()
        
        self.cache._redis = mock_redis
        self.cache._pool = mock_pool
        self.cache._connected = True
        
        asyncio.run(self.cache.disconnect())
        
        assert self.cache._redis is None
        assert self.cache._pool is None
        assert self.cache._connected is False
