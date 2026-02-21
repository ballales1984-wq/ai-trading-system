"""
Network Retry Utilities
Provides retry logic for API calls and network operations
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Callable, Any, Optional, List, Type
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0  # exponential backoff
DEFAULT_TIMEOUT = 30  # seconds


# ==================== EXCEPTIONS ====================

class RetryableError(Exception):
    """Base exception for errors that should trigger a retry"""
    pass


class NetworkError(RetryableError):
    """Network-related error (DNS, connection, timeout)"""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded error"""
    pass


# ==================== RETRY DECORATOR ====================

def retry_on_network_error(
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    exceptions: tuple = (requests.RequestException, ConnectionError, TimeoutError),
    retriable_status_codes: tuple = (429, 500, 502, 503, 504)
):
    """
    Decorator to retry a function on network errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        retriable_status_codes: HTTP status codes that should trigger a retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Check if it's a rate limit error
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if status_code == 429:
                            # Rate limited - longer wait
                            current_delay = delay * 10
                            logger.warning(f"Rate limited, waiting {current_delay}s before retry...")
                        elif status_code not in retriable_status_codes:
                            # Non-retriable status code
                            logger.error(f"Non-retriable HTTP error {status_code}: {e}")
                            raise
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            return None
            
        return wrapper
    return decorator


def retry_async(
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    exceptions: tuple = (asyncio.TimeoutError, ConnectionError)
):
    """
    Async version of retry decorator.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Async error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
            
            if last_exception:
                raise last_exception
            return None
            
        return wrapper
    return decorator


# ==================== HTTP CLIENT WITH RETRY ====================

class RetryableHTTPClient:
    """
    HTTP client with built-in retry logic.
    """
    
    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        timeout: float = DEFAULT_TIMEOUT
    ):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = requests.Session()
        
        # Configure retry strategy for requests library
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"RetryableHTTPClient initialized (max_retries={max_retries}, timeout={timeout})")
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with retry"""
        return self.session.get(url, timeout=self.timeout, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request with retry"""
        return self.session.post(url, timeout=self.timeout, **kwargs)
    
    def close(self):
        """Close the session"""
        self.session.close()


# ==================== API CALL WRAPPER ====================

class APICallWrapper:
    """
    Wrapper for API calls with retry logic and error handling.
    """
    
    def __init__(
        self,
        name: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        delay: float = DEFAULT_RETRY_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    ):
        self.name = name
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.client = RetryableHTTPClient(max_retries, delay, backoff_factor)
        self.last_success: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Make an API call with retry logic.
        
        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
        """
        self.total_calls += 1
        current_delay = self.delay
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                self.successful_calls += 1
                self.last_success = datetime.now()
                self.last_error = None
                return result
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"[{self.name}] API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor
                else:
                    self.failed_calls += 1
                    self.last_error = str(e)
                    logger.error(f"[{self.name}] API call failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    def get_stats(self) -> dict:
        """Get statistics for this API wrapper"""
        return {
            'name': self.name,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            'last_success': self.last_success,
            'last_error': self.last_error
        }
    
    def close(self):
        """Close the HTTP client"""
        self.client.close()


# ==================== CIRCUIT BREAKER ====================

class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        
        logger.info(f"CircuitBreaker initialized (failure_threshold={failure_threshold})")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logger.info("CircuitBreaker: Moving to half_open state")
            else:
                raise Exception(f"Circuit breaker is open. Call rejected.")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("CircuitBreaker: Circuit closed again")
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"CircuitBreaker: Opening circuit after {self.failure_count} failures")
            
            raise
    
    def reset(self):
        """Manually reset the circuit breaker"""
        self.failure_count = 0
        self.state = "closed"
        logger.info("CircuitBreaker: Manually reset")


# ==================== BATCH RETRY ====================

def retry_batch(
    items: List[Any],
    fetch_func: Callable,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay: float = DEFAULT_RETRY_DELAY,
    batch_size: int = 10
) -> List[Any]:
    """
    Retry fetching data in batches with individual item retry.
    
    Args:
        items: List of items to fetch
        fetch_func: Function to fetch a single item
        max_retries: Max retries per item
        delay: Delay between retries
        batch_size: Number of items per batch
        
    Returns:
        List of successfully fetched items
    """
    results = []
    failed_items = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        for item in batch:
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = fetch_func(item)
                    if result is not None:
                        results.append(result)
                        break
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Failed to fetch {item} (attempt {attempt + 1}): {e}")
                        time.sleep(current_delay)
                        current_delay *= 2
                    else:
                        logger.error(f"Failed to fetch {item} after {max_retries + 1} attempts")
                        failed_items.append(item)
    
    if failed_items:
        logger.warning(f"Failed to fetch {len(failed_items)} items: {failed_items}")
    
    return results


# ==================== UTILITY FUNCTIONS ====================

def is_retriable_error(error: Exception) -> bool:
    """
    Check if an error is retriable.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is retriable
    """
    # Network errors
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    
    # Requests errors
    if isinstance(error, requests.RequestException):
        if hasattr(error, 'response') and error.response is not None:
            status_code = error.response.status_code
            # Retry on rate limit or server errors
            if status_code in (429, 500, 502, 503, 504):
                return True
        return True
    
    return False


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)

