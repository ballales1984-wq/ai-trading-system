"""
Async utilities for parallel data fetching
Provides async/await support for API calls
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class AsyncFetcher:
    """
    Async data fetcher for parallel API calls
    """
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize async fetcher
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = None
    
    async def fetch_multiple(self, fetch_funcs: List[Callable]) -> List[Any]:
        """
        Fetch multiple data sources in parallel
        
        Args:
            fetch_funcs: List of fetch functions to execute
            
        Returns:
            List of results
        """
        loop = asyncio.get_event_loop()
        
        # Execute all fetch functions in parallel
        tasks = [
            loop.run_in_executor(self.executor, func)
            for func in fetch_funcs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Fetch {i} failed: {result}")
                processed.append(None)
            else:
                processed.append(result)
        
        return processed
    
    async def fetch_with_timeout(self, fetch_func: Callable, timeout: float = 30) -> Any:
        """
        Fetch with timeout
        
        Args:
            fetch_func: Function to execute
            timeout: Timeout in seconds
            
        Returns:
            Fetch result or None on timeout
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, fetch_func),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Fetch timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return None
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


# Global async fetcher
_async_fetcher = None


def get_async_fetcher(max_workers: int = 5) -> AsyncFetcher:
    """Get global async fetcher"""
    global _async_fetcher
    if _async_fetcher is None:
        _async_fetcher = AsyncFetcher(max_workers)
    return _async_fetcher


async def fetch_symbols_async(symbols: List[str], fetch_func: Callable) -> Dict[str, Any]:
    """
    Fetch data for multiple symbols in parallel
    
    Args:
        symbols: List of trading symbols
        fetch_func: Function to fetch data for a single symbol
        
    Returns:
        Dict mapping symbol to result
    """
    fetcher = get_async_fetcher()
    
    # Create fetch functions
    fetch_funcs = [lambda s=s: fetch_func(s) for s in symbols]
    
    # Execute in parallel
    results = await fetcher.fetch_multiple(fetch_funcs)
    
    # Map results to symbols
    return dict(zip(symbols, results))


class RateLimiter:
    """
    Rate limiter for API calls
    """
    
    def __init__(self, calls_per_second: float = 10):
        """
        Initialize rate limiter
        
        Args:
            calls_per_second: Maximum calls per second
        """
        self.calls_per_second = calls_per_second
        self.last_call = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a call"""
        async with self._lock:
            import time
            
            now = time.time()
            elapsed = now - self.last_call
            
            # Wait if needed
            min_interval = 1.0 / self.calls_per_second
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            
            self.last_call = time.time()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Decorator for rate-limited async functions
def rate_limited(calls_per_second: float = 10):
    """
    Decorator to rate limit async functions
    
    Args:
        calls_per_second: Maximum calls per second
    """
    limiter = RateLimiter(calls_per_second)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with limiter:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


from functools import wraps


if __name__ == "__main__":
    # Test async fetching
    import time
    
    def slow_fetch(n):
        time.sleep(0.5)
        return f"Result {n}"
    
    async def main():
        fetcher = AsyncFetcher(max_workers=3)
        
        # Test parallel fetch
        start = time.time()
        results = await fetcher.fetch_multiple([
            lambda: slow_fetch(1),
            lambda: slow_fetch(2),
            lambda: slow_fetch(3)
        ])
        elapsed = time.time() - start
        
        print(f"Results: {results}")
        print(f"Time: {elapsed:.2f}s (should be ~0.5s with parallel)")
        
        fetcher.shutdown()
    
    asyncio.run(main())
