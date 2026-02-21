"""
Rate Limiter Module
===================
API rate limiting for security.

Author: AI Trading System
"""

import time
import asyncio
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


@dataclass
class RateLimitEntry:
    """Rate limit entry for tracking."""
    count: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    last_request: datetime = field(default_factory=datetime.now)
    blocked_until: Optional[datetime] = None


class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time in seconds until tokens are available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Rate Limiter
    ============
    Multi-strategy rate limiter for API endpoints.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter."""
        self.config = config or RateLimitConfig()
        
        # Track requests per client
        self._client_limits: Dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        
        # Token buckets per client
        self._token_buckets: Dict[str, TokenBucket] = {}
        
        # In-memory store for distributed systems (use Redis in production)
        self._store: Dict[str, Dict] = defaultdict(dict)
    
    def _get_client_id(self, identifier: str) -> str:
        """Get client identifier (can be IP, API key, user ID, etc.)."""
        # Hash the identifier for privacy
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def _is_blocked(self, client_id: str) -> bool:
        """Check if client is blocked."""
        entry = self._client_limits.get(client_id)
        
        if entry and entry.blocked_until:
            if datetime.now() < entry.blocked_until:
                return True
            else:
                # Unblock client
                entry.blocked_until = None
                entry.count = 0
                entry.window_start = datetime.now()
        
        return False
    
    def _get_retry_after(self, client_id: str) -> int:
        """Get seconds until client can retry."""
        entry = self._client_limits.get(client_id)
        
        if entry and entry.blocked_until:
            delta = entry.blocked_until - datetime.now()
            return max(1, int(delta.total_seconds()))
        
        return 60
    
    def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = "default",
    ) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Client identifier (IP, API key, user ID)
            endpoint: API endpoint name
            
        Returns:
            True if allowed, raises exception if rate limited
        """
        client_id = self._get_client_id(identifier)
        
        # Check if blocked
        if self._is_blocked(client_id):
            retry_after = self._get_retry_after(client_id)
            raise RateLimitExceeded(
                f"Rate limit exceeded. Retry after {retry_after} seconds.",
                retry_after
            )
        
        # Get or create entry
        entry = self._client_limits[client_id]
        now = datetime.now()
        
        # Reset window if expired (1 minute window)
        if now - entry.window_start > timedelta(minutes=1):
            entry.count = 0
            entry.window_start = now
        
        # Check minute limit
        if entry.count >= self.config.requests_per_minute:
            # Block client for 1 minute
            entry.blocked_until = now + timedelta(minutes=1)
            raise RateLimitExceeded(
                "Minute rate limit exceeded.",
                60
            )
        
        # Increment count
        entry.count += 1
        entry.last_request = now
        
        return True
    
    def check_rate_limit_with_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check rate limit with custom parameters.
        
        Args:
            identifier: Client identifier
            limit: Maximum requests in window
            window_seconds: Window size in seconds
            
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        client_id = self._get_client_id(identifier)
        now = datetime.now()
        
        # Get or create token bucket
        if client_id not in self._token_buckets:
            bucket_key = f"{client_id}_{limit}_{window_seconds}"
            refill_rate = limit / window_seconds
            self._token_buckets[bucket_key] = TokenBucket(limit, refill_rate)
        
        bucket = self._token_buckets[f"{client_id}_{limit}_{window_seconds}"]
        
        allowed = bucket.consume()
        remaining = int(bucket.tokens)
        
        if not allowed:
            wait_time = int(bucket.get_wait_time())
            raise RateLimitExceeded(
                f"Rate limit exceeded. Wait {wait_time} seconds.",
                wait_time
            )
        
        return True, remaining
    
    def reset(self, identifier: str):
        """Reset rate limit for a client."""
        client_id = self._get_client_id(identifier)
        
        if client_id in self._client_limits:
            del self._client_limits[client_id]
        
        if client_id in self._token_buckets:
            del self._token_buckets[client_id]
        
        logger.info(f"Reset rate limit for client: {client_id}")
    
    def get_stats(self, identifier: str) -> Dict:
        """Get rate limit statistics for a client."""
        client_id = self._get_client_id(identifier)
        entry = self._client_limits.get(client_id)
        
        if not entry:
            return {"status": "no_requests"}
        
        return {
            "count": entry.count,
            "window_start": entry.window_start.isoformat(),
            "last_request": entry.last_request.isoformat(),
            "blocked": entry.blocked_until is not None,
            "blocked_until": entry.blocked_until.isoformat() if entry.blocked_until else None,
        }


class AsyncRateLimiter:
    """
    Async Rate Limiter
    ==================
    Async-compatible rate limiter for high-performance applications.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize async rate limiter."""
        self.config = config or RateLimitConfig()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._limits: Dict[str, TokenBucket] = {}
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    async def acquire(
        self,
        key: str,
        tokens: int = 1,
    ) -> bool:
        """
        Acquire rate limit permission.
        
        Args:
            key: Rate limit key (e.g., user ID, IP)
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False otherwise
        """
        lock = self._get_lock(key)
        
        async with lock:
            if key not in self._limits:
                refill_rate = self.config.requests_per_minute / 60.0
                self._limits[key] = TokenBucket(
                    self.config.burst_size,
                    refill_rate
                )
            
            bucket = self._limits[key]
            return bucket.consume(tokens)
    
    async def wait_for(
        self,
        key: str,
        tokens: int = 1,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for rate limit permission.
        
        Args:
            key: Rate limit key
            tokens: Number of tokens
            timeout: Maximum wait time in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if await self.acquire(key, tokens):
                return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)


# Default rate limiter instance
default_rate_limiter = RateLimiter()
default_async_rate_limiter = AsyncRateLimiter()


# FastAPI integration example:
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Get client identifier (IP or API key)
    client_id = request.client.host
    
    # Check rate limit
    try:
        default_rate_limiter.check_rate_limit(client_id)
    except RateLimitExceeded as e:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": e.retry_after
            },
            headers={"Retry-After": str(e.retry_after)}
        )
    
    response = await call_next(request)
    return response
"""

