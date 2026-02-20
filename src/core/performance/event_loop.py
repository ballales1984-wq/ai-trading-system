"""
Optimized Event Loop with uvloop
================================
High-performance event loop for HFT and real-time trading.
uvloop provides 2-4x faster event loop than asyncio default.

Usage:
    from src.core.performance.event_loop import get_optimized_event_loop
    
    # Replace default event loop
    loop = get_optimized_event_loop()
    asyncio.set_event_loop(loop)
"""

import asyncio
import sys
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

# Try to import uvloop - fall back to default if not available
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False
    logger.warning("uvloop not available, using default event loop")


def get_optimized_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get optimized event loop with uvloop if available.
    
    Returns:
        Optimized event loop instance
    """
    if UVLOOP_AVAILABLE:
        # Create uvloop event loop with optimized settings
        loop = uvloop.new_event_loop()
        
        # Optimize for low latency
        loop.set_debug(False)  # Disable debug for performance
        loop.slow_callback_duration = 0.1  # Log slow callbacks
        
        logger.info("Using uvloop event loop (2-4x faster than default)")
        return loop
    else:
        # Fall back to default event loop with optimizations
        loop = asyncio.new_event_loop()
        loop.set_debug(False)
        logger.info("Using default asyncio event loop")
        return loop


def configure_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Configure event loop with performance optimizations.
    
    Args:
        loop: Event loop to configure
    """
    # Enable high performance settings
    if hasattr(loop, 'set_coroutine_debug_mode'):
        loop.set_coroutine_debug_mode(False)
    
    # Set default executor for CPU-bound tasks
    from concurrent.futures import ThreadPoolExecutor
    import atexit
    
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async_worker")
    loop.set_default_executor(executor)
    
    # Cleanup executor on exit
    def shutdown_executor():
        executor.shutdown(wait=True)
    
    atexit.register(shutdown_executor)
    
    logger.info("Event loop configured with ThreadPoolExecutor")


class HighPerformanceRunner:
    """
    Context manager for running async code with optimized event loop.
    
    Usage:
        async with HighPerformanceRunner() as runner:
            await runner.run(main())
    """
    
    def __init__(self, use_uvloop: bool = True):
        """
        Initialize runner.
        
        Args:
            use_uvloop: Whether to use uvloop (default: True)
        """
        self.use_uvloop = use_uvloop and UVLOOP_AVAILABLE
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = None
    
    async def __aenter__(self):
        """Enter context - setup optimized event loop."""
        if self.use_uvloop:
            self.loop = uvloop.new_event_loop()
        else:
            self.loop = asyncio.new_event_loop()
        
        asyncio.set_event_loop(self.loop)
        
        # Configure with thread pool
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.loop.set_default_executor(self._executor)
        
        logger.info(f"High performance runner started (uvloop: {self.use_uvloop})")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup."""
        if self.loop:
            # Cancel all running tasks
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            
            # Wait for cancellation
            if pending:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            
            self.loop.close()
        
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info("High performance runner stopped")
    
    @staticmethod
    def run(coro):
        """Run coroutine with optimized event loop."""
        if UVLOOP_AVAILABLE:
            return uvloop.run(coro)
        else:
            return asyncio.run(coro)


def run_async_main(main_func: Callable) -> Any:
    """
    Run async main function with optimized event loop.
    
    Args:
        main_func: Async function to run
        
    Returns:
        Result of main function
    """
    if UVLOOP_AVAILABLE:
        logger.info(f"Running {main_func.__name__} with uvloop")
        return uvloop.run(main_func())
    else:
        logger.info(f"Running {main_func.__name__} with default asyncio")
        return asyncio.run(main_func())


# Check uvloop availability and platform
def check_performance_readiness() -> dict:
    """
    Check system readiness for high-performance trading.
    
    Returns:
        Dict with readiness status and recommendations
    """
    status = {
        "uvloop_available": UVLOOP_AVAILABLE,
        "platform": sys.platform,
        "python_version": sys.version,
        "recommendations": []
    }
    
    if not UVLOOP_AVAILABLE:
        status["recommendations"].append(
            "Install uvloop for 2-4x performance: pip install uvloop"
        )
    
    if sys.platform == "win32":
        status["recommendations"].append(
            "Windows detected: Consider using WSL2 for better async performance"
        )
    
    # Check for other performance libraries
    try:
        import numpy
        status["numpy_version"] = numpy.__version__
    except ImportError:
        status["recommendations"].append("Install numpy for vectorized operations")
    
    return status


if __name__ == "__main__":
    # Test and display performance readiness
    import json
    
    status = check_performance_readiness()
    print("=" * 50)
    print("Performance Readiness Check")
    print("=" * 50)
    print(json.dumps(status, indent=2))
    print("=" * 50)
    
    # Quick benchmark
    async def benchmark():
        """Quick benchmark to compare event loops."""
        import time
        
        iterations = 100000
        
        # Test with current event loop
        loop = asyncio.get_event_loop()
        
        async def tiny_task():
            pass
        
        start = time.perf_counter()
        for _ in range(iterations):
            await tiny_task()
        elapsed = time.perf_counter() - start
        
        print(f"\nEvent loop benchmark ({iterations:,} simple tasks):")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {iterations/elapsed:,.0f} tasks/sec")
        
        if UVLOOP_AVAILABLE:
            # Benchmark with uvloop
            uv_loop = uvloop.new_event_loop()
            
            start = time.perf_counter()
            uv_loop.run_until_complete(asyncio.gather(*[tiny_task() for _ in range(iterations)]))
            uv_elapsed = time.perf_counter() - start
            
            print(f"\nuvloop benchmark ({iterations:,} tasks):")
            print(f"  Time: {uv_elapsed:.3f}s")
            print(f"  Rate: {iterations/uv_elapsed:,.0f} tasks/sec")
            print(f"\n  Speedup: {elapsed/uv_elapsed:.2f}x")
            
            uv_loop.close()
    
    asyncio.run(benchmark())

