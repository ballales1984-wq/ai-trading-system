"""
Async Logging with Queue Handler
===============================
High-performance async logging for HFT systems.
Uses queue-based logging to avoid blocking the main event loop.

Usage:
    from src.core.performance.async_logging import setup_async_logging
    
    # Setup at application startup
    setup_async_logging()
"""

import asyncio
import logging
import queue
import threading
import time
import json
from typing import Optional, Any, Dict
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import ThreadPoolExecutor
import atexit


class AsyncLogQueue:
    """
    Async-safe log queue with batching support.
    """
    
    def __init__(self, maxsize: int = 10000, batch_size: int = 100, flush_interval: float = 1.0):
        """
        Initialize async log queue.
        
        Args:
            maxsize: Maximum queue size
            batch_size: Number of logs to batch before flushing
            flush_interval: Seconds between forced flushes
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batch: list = []
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log_writer")
        self._running = True
        
        # Start flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def put(self, record: logging.LogRecord) -> None:
        """Add log record to queue."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Drop log if queue is full (prefer performance over logging)
            pass
    
    def _flush_loop(self) -> None:
        """Background thread to flush logs."""
        while self._running:
            time.sleep(0.1)  # Check every 100ms
            
            # Flush if batch is full or interval exceeded
            now = time.time()
            should_flush = (
                len(self._batch) >= self.batch_size or
                now - self._last_flush >= self.flush_interval
            )
            
            if should_flush and self._batch:
                self._do_flush()
    
    def _do_flush(self) -> None:
        """Flush batch to files (called from background thread)."""
        if not self._batch:
            return
        
        # Sort by timestamp and write
        try:
            self._batch.sort(key=lambda r: r.created)
            
            # Group by logger name for file organization
            by_logger: Dict[str, list] = {}
            for record in self._batch:
                logger_name = record.name
                if logger_name not in by_logger:
                    by_logger[logger_name] = []
                by_logger[logger_name].append(record)
            
            # Write to files (simplified - in production use proper file handling)
            for logger_name, records in by_logger.items():
                # This is where you'd write to actual log files
                pass  # Placeholder
            
        except Exception:
            pass  # Never let logging fail
        
        finally:
            self._batch.clear()
            self._last_flush = time.time()
    
    def flush(self) -> None:
        """Force flush."""
        self._do_flush()
    
    def stop(self) -> None:
        """Stop the queue."""
        self._running = False
        self.flush()
        self._executor.shutdown(wait=True)


# Global log queue
_log_queue: Optional[AsyncLogQueue] = None


def setup_async_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    batch_size: int = 100,
    flush_interval: float = 1.0
) -> QueueListener:
    """
    Setup async logging with queue handler.
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        batch_size: Number of logs to batch
        flush_interval: Seconds between flushes
        
    Returns:
        QueueListener instance
    """
    global _log_queue
    
    # Create queue
    log_queue = queue.Queue(-1)
    _log_queue = AsyncLogQueue(batch_size=batch_size, flush_interval=flush_interval)
    
    # Create queue handler
    queue_handler = QueueHandler(log_queue)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)
    
    # Configure specific loggers
    for logger_name in ["src", "app", "trading"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.handlers.clear()
        logger.addHandler(queue_handler)
    
    # Create console handler for immediate output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings to console
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create listener
    handlers = []
    
    # File handlers for different levels
    from logging.handlers import RotatingFileHandler
    
    # All logs
    all_handler = RotatingFileHandler(
        f"{log_dir}/all.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(all_handler)
    
    # Error logs
    error_handler = RotatingFileHandler(
        f"{log_dir}/error.log",
        maxBytes=5_000_000,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d'
    ))
    handlers.append(error_handler)
    
    # Trading logs
    trading_handler = RotatingFileHandler(
        f"{log_dir}/trading.log",
        maxBytes=20_000_000,
        backupCount=10
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    handlers.append(trading_handler)
    
    # Create listener
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    
    # Register cleanup
    atexit.register(listener.stop)
    
    logging.info(f"Async logging initialized (batch_size={batch_size}, flush_interval={flush_interval}s)")
    
    return listener


def get_log_queue() -> Optional[AsyncLogQueue]:
    """Get global log queue."""
    return _log_queue


class PerformanceLogger:
    """
    Logger specifically for performance metrics.
    Writes structured JSON logs for easy parsing.
    """
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
        self._enabled = True
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
    
    def log_latency(self, operation: str, latency_ms: float, **kwargs) -> None:
        """Log operation latency."""
        if not self._enabled:
            return
        
        self.logger.info(json.dumps({
            "type": "latency",
            "operation": operation,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }))
    
    def log_throughput(self, operation: str, count: int, duration_ms: float, **kwargs) -> None:
        """Log throughput metrics."""
        if not self._enabled:
            return
        
        rate = count / (duration_ms / 1000) if duration_ms > 0 else 0
        
        self.logger.info(json.dumps({
            "type": "throughput",
            "operation": operation,
            "count": count,
            "duration_ms": duration_ms,
            "rate_per_sec": rate,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }))
    
    def log_error(self, operation: str, error: str, **kwargs) -> None:
        """Log error with context."""
        self.logger.error(json.dumps({
            "type": "error",
            "operation": operation,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }))
    
    def log_trade(self, symbol: str, side: str, quantity: float, price: float, 
                  latency_ms: float, **kwargs) -> None:
        """Log trade execution."""
        if not self._enabled:
            return
        
        self.logger.info(json.dumps({
            "type": "trade",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }))


# Global performance logger
perf_logger = PerformanceLogger()


# Context manager for timing operations
class timing_context:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, logger: Optional[PerformanceLogger] = None):
        self.operation = operation
        self.logger = logger or perf_logger
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.logger.log_latency(self.operation, elapsed_ms)


# Decorator for timing functions
def timed(operation: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            perf_logger.log_latency(
                operation or f"{func.__module__}.{func.__name__}", 
                elapsed_ms
            )
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test async logging
    print("Testing async logging...")
    
    # Setup
    listener = setup_async_logging(
        level=logging.DEBUG,
        log_dir="logs",
        batch_size=10,
        flush_interval=0.5
    )
    
    # Test logging
    logger = logging.getLogger("test")
    
    for i in range(100):
        logger.info(f"Test message {i}")
        logger.debug(f"Debug message {i}")
    
    # Test performance logger
    perf_logger.log_latency("test_operation", 12.5)
    perf_logger.log_throughput("test_batch", 100, 50.0)
    perf_logger.log_trade("BTCUSDT", "BUY", 0.1, 50000, 15.2)
    
    # Test timing context
    with timing_context("test_context"):
        time.sleep(0.01)
    
    print("Async logging test complete")
    print("Check logs/ directory for output files")

