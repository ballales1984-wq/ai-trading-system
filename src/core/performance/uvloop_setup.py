"""
uvloop Setup for High-Performance Async
========================================
uvloop is a ultra-fast drop-in replacement for asyncio event loop.
It can make asyncio 2-4x faster than the default implementation.

Usage:
    # At application startup
    from src.core.performance.uvloop_setup import setup_uvloop, get_event_loop
    
    # Setup uvloop (call once at startup)
    setup_uvloop()
    
    # Get optimized event loop
    loop = get_event_loop()
"""

import asyncio
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Track if uvloop is available and enabled
_UVLOOP_AVAILABLE = False
_UVLOOP_ENABLED = False


def is_uvloop_available() -> bool:
    """Check if uvloop is available."""
    return _UVLOOP_AVAILABLE


def is_uvloop_enabled() -> bool:
    """Check if uvloop is currently enabled."""
    return _UVLOOP_ENABLED


def setup_uvloop() -> bool:
    """
    Setup uvloop for high-performance async operations.
    
    Returns:
        True if uvloop was successfully enabled, False otherwise.
    
    Note:
        - On Windows, uvloop is not available (uses ProactorEventLoop)
        - On Linux/macOS, uvloop provides 2-4x performance improvement
    """
    global _UVLOOP_AVAILABLE, _UVLOOP_ENABLED
    
    # Check if already enabled
    if _UVLOOP_ENABLED:
        logger.debug("uvloop already enabled")
        return True
    
    # Windows doesn't support uvloop
    if sys.platform == "win32":
        logger.info("Windows detected - using ProactorEventLoop (uvloop not supported)")
        _UVLOOP_AVAILABLE = False
        _UVLOOP_ENABLED = False
        
        # Optimize Windows event loop
        if sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.info("WindowsProactorEventLoopPolicy configured")
        
        return False
    
    # Try to import and setup uvloop
    try:
        import uvloop
        
        _UVLOOP_AVAILABLE = True
        
        # Install uvloop as the default event loop
        uvloop.install()
        _UVLOOP_ENABLED = True
        
        logger.info("uvloop enabled - async performance optimized (2-4x faster)")
        return True
        
    except ImportError:
        logger.warning(
            "uvloop not installed - using default asyncio. "
            "Install with: pip install uvloop"
        )
        _UVLOOP_AVAILABLE = False
        _UVLOOP_ENABLED = False
        return False
    
    except Exception as e:
        logger.error(f"Failed to setup uvloop: {e}")
        _UVLOOP_AVAILABLE = False
        _UVLOOP_ENABLED = False
        return False


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get the current event loop, creating one if necessary.
    
    Returns:
        The current event loop instance.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in current thread, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def create_optimized_loop() -> asyncio.AbstractEventLoop:
    """
    Create a new optimized event loop.
    
    Returns:
        A new event loop instance (uvloop if available).
    """
    if _UVLOOP_AVAILABLE and not _UVLOOP_ENABLED:
        # Setup uvloop if available but not yet enabled
        setup_uvloop()
    
    return asyncio.new_event_loop()


def run_async(coro):
    """
    Run an async coroutine in an optimized event loop.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
    """
    if _UVLOOP_ENABLED:
        import uvloop
        return uvloop.run(coro)
    else:
        return asyncio.run(coro)


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class LoopMonitor:
    """
    Monitor event loop performance.
    
    Tracks:
    - Loop lag (time spent blocked)
    - Task count
    - Callback processing time
    """
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._running = False
        self._task = None
        
        # Metrics
        self.lag_samples = []
        self.max_lag = 0.0
        self.avg_lag = 0.0
        self.callback_count = 0
    
    async def _monitor_loop(self):
        """Monitor loop performance."""
        import time
        
        while self._running:
            start = time.perf_counter()
            await asyncio.sleep(self.interval)
            elapsed = time.perf_counter() - start
            
            # Calculate lag (time spent not in sleep)
            lag = elapsed - self.interval
            if lag < 0:
                lag = 0
            
            self.lag_samples.append(lag)
            if len(self.lag_samples) > 100:
                self.lag_samples.pop(0)
            
            self.max_lag = max(self.max_lag, lag)
            self.avg_lag = sum(self.lag_samples) / len(self.lag_samples)
            
            # Log if lag is significant
            if lag > 0.1:  # 100ms lag
                logger.warning(f"Event loop lag detected: {lag*1000:.1f}ms")
    
    def start(self):
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.debug("Loop monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.debug("Loop monitor stopped")
    
    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            "avg_lag_ms": round(self.avg_lag * 1000, 2),
            "max_lag_ms": round(self.max_lag * 1000, 2),
            "samples": len(self.lag_samples),
            "uvloop_enabled": _UVLOOP_ENABLED,
        }


# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class OptimizedAsync:
    """
    Context manager for optimized async execution.
    
    Usage:
        async with OptimizedAsync():
            # Your async code here
            await some_async_function()
    """
    
    def __init__(self, enable_monitoring: bool = False):
        self.enable_monitoring = enable_monitoring
        self.monitor: Optional[LoopMonitor] = None
        self._uvloop_was_enabled = False
    
    async def __aenter__(self):
        self._uvloop_was_enabled = _UVLOOP_ENABLED
        
        if not self._uvloop_was_enabled:
            setup_uvloop()
        
        if self.enable_monitoring:
            self.monitor = LoopMonitor()
            self.monitor.start()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.monitor:
            self.monitor.stop()
        
        return False


# ============================================================================
# DECORATOR FOR OPTIMIZED ASYNC FUNCTIONS
# ============================================================================

def optimized_async(func):
    """
    Decorator to run async function with uvloop optimization.
    
    Usage:
        @optimized_async
        async def my_function():
            ...
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure uvloop is setup
        if not _UVLOOP_ENABLED:
            setup_uvloop()
        
        # Run the coroutine
        return run_async(func(*args, **kwargs))
    
    return wrapper


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_performance():
    """
    Initialize performance optimizations.
    Call this at application startup.
    """
    # Setup uvloop
    uvloop_ok = setup_uvloop()
    
    # Log status
    if uvloop_ok:
        logger.info("Performance optimizations enabled: uvloop")
    else:
        logger.info("Using standard asyncio event loop")
    
    return uvloop_ok


# Auto-initialize on import (optional, can be disabled)
_AUTO_INIT = True

if _AUTO_INIT:
    try:
        setup_uvloop()
    except Exception:
        pass  # Silently fail on import
