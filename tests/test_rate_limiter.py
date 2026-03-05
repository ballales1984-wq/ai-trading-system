"""
Tests for Rate Limiter Module
=============================
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTokenBucket:
    """Test TokenBucket class."""
    
    def test_token_bucket_creation(self):
        """Test TokenBucket initialization."""
        from app.core.rate_limiter import TokenBucket
        
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0
    
    def test_token_bucket_consume_success(self):
        """Test consuming tokens successfully."""
        from app.core.rate_limiter import TokenBucket
        
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should be able to consume
        result = bucket.consume(5)
        assert result is True
    
    def test_token_bucket_consume_failure(self):
        """Test failing to consume when not enough tokens."""
        from app.core.rate_limiter import TokenBucket
        
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Consume more than available
        result = bucket.consume(15)
        assert result is False
    
    def test_token_bucket_get_wait_time_available(self):
        """Test get_wait_time when tokens are available."""
        from app.core.rate_limiter import TokenBucket
        
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        wait_time = bucket.get_wait_time(5)
        assert wait_time == 0
    
    def test_token_bucket_get_wait_time_unavailable(self):
        """Test get_wait_time when tokens are not available."""
        from app.core.rate_limiter import TokenBucket
        
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.consume(10)  # Use all tokens
        
        wait_time = bucket.get_wait_time(5)
        assert wait_time > 0


class TestRateLimitConfig:
    """Test RateLimitConfig class."""
    
    def test_rate_limit_config_defaults(self):
        """Test default configuration."""
        from app.core.rate_limiter import RateLimitConfig
        
        config = RateLimitConfig()
        
        assert config.requests_per_minute == 60
        assert config.burst_size == 10
    
    def test_rate_limit_config_custom(self):
        """Test custom configuration."""
        from app.core.rate_limiter import RateLimitConfig
        
        config = RateLimitConfig(requests_per_minute=100, burst_size=20)
        
        assert config.requests_per_minute == 100
        assert config.burst_size == 20


class TestRateLimitEntry:
    """Test RateLimitEntry class."""
    
    def test_rate_limit_entry_defaults(self):
        """Test default entry values."""
        from app.core.rate_limiter import RateLimitEntry
        
        entry = RateLimitEntry()
        
        assert entry.count == 0
        assert entry.window_start is not None


class TestRateLimiter:
    """Test RateLimiter class."""
    
    def test_rate_limiter_creation_default(self):
        """Test rate limiter creation with defaults."""
        from app.core.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        assert limiter is not None
    
    def test_rate_limiter_creation_custom_config(self):
        """Test rate limiter with custom config."""
        from app.core.rate_limiter import RateLimiter, RateLimitConfig
        
        config = RateLimitConfig(requests_per_minute=100, burst_size=20)
        limiter = RateLimiter(config=config)
        
        assert limiter is not None
    
    def test_rate_limiter_allow_under_limit(self):
        """Test allowing request under limit."""
        from app.core.rate_limiter import RateLimiter
        
        limiter = RateLimiter(max_requests=10)
        
        # Should allow requests
        result = limiter.allow("test_client")
        assert result is True


class TestAsyncRateLimiter:
    """Test AsyncRateLimiter class."""
    
    def test_async_rate_limiter_creation(self):
        """Test async rate limiter creation."""
        from app.core.rate_limiter import AsyncRateLimiter
        
        limiter = AsyncRateLimiter()
        
        assert limiter is not None
    
    def test_async_rate_limiter_with_config(self):
        """Test async rate limiter with config."""
        from app.core.rate_limiter import AsyncRateLimiter, RateLimitConfig
        
        config = RateLimitConfig(requests_per_minute=50, burst_size=10)
        limiter = AsyncRateLimiter(config=config)
        
        assert limiter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
