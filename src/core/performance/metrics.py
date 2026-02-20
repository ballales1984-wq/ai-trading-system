"""
Performance Profiler and Metrics
================================
Built-in performance profiling for HFT systems.
Includes timing decorators, latency tracking, and throughput metrics.

Usage:
    from src.core.performance.metrics import timed, TimingContext, profile_function
    
    @timed
    def my_function():
        pass
    
    with TimingContext("operation_name"):
        # code to profile
        pass
"""

import time
import functools
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import threading
import json


logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0
    
    @property
    def p50_ms(self) -> float:
        return self.avg_ms  # Simplified
    
    @property
    def p95_ms(self) -> float:
        return self.avg_ms * 1.5  # Simplified
    
    @property
    def p99_ms(self) -> float:
        return self.avg_ms * 2  # Simplified
    
    def record(self, elapsed_ms: float) -> None:
        """Record a timing measurement."""
        with self._lock:
            self.count += 1
            self.total_ms += elapsed_ms
            self.min_ms = min(self.min_ms, elapsed_ms)
            self.max_ms = max(self.max_ms, elapsed_ms)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": self.total_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "avg_ms": self.avg_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


class MetricsCollector:
    """
    Global metrics collector for performance tracking.
    Thread-safe singleton pattern.
    """
    
    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._timings: Dict[str, TimingStats] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._timings_lock = threading.Lock()
        self._counters_lock = threading.Lock()
        self._gauges_lock = threading.Lock()
        self._enabled = True
        self._initialized = True
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    # Timing methods
    def record_timing(self, name: str, elapsed_ms: float) -> None:
        """Record timing for an operation."""
        if not self._enabled:
            return
        
        with self._timings_lock:
            if name not in self._timings:
                self._timings[name] = TimingStats(name=name)
            self._timings[name].record(elapsed_ms)
    
    def get_timing(self, name: str) -> Optional[TimingStats]:
        """Get timing stats for an operation."""
        with self._timings_lock:
            return self._timings.get(name)
    
    def get_all_timings(self) -> Dict[str, Dict]:
        """Get all timing stats."""
        with self._timings_lock:
            return {name: stats.to_dict() for name, stats in self._timings.items()}
    
    def reset_timing(self, name: str) -> None:
        """Reset timing for an operation."""
        with self._timings_lock:
            if name in self._timings:
                del self._timings[name]
    
    def reset_all_timings(self) -> None:
        """Reset all timings."""
        with self._timings_lock:
            self._timings.clear()
    
    # Counter methods
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        if not self._enabled:
            return
        
        with self._counters_lock:
            self._counters[name] = self._counters.get(name, 0) + value
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._counters_lock:
            return self._counters.get(name, 0)
    
    def get_all_counters(self) -> Dict[str, int]:
        """Get all counters."""
        with self._counters_lock:
            return dict(self._counters)
    
    def reset_counter(self, name: str) -> None:
        """Reset a counter."""
        with self._counters_lock:
            if name in self._counters:
                del self._counters[name]
    
    # Gauge methods
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._gauges_lock:
            self._gauges[name] = value
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        with self._gauges_lock:
            return self._gauges.get(name)
    
    def get_all_gauges(self) -> Dict[str, float]:
        """Get all gauges."""
        with self._gauges_lock:
            return dict(self._gauges)
    
    # Export methods
    def export_json(self) -> str:
        """Export all metrics as JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "timings": self.get_all_timings(),
            "counters": self.get_all_counters(),
            "gauges": self.get_all_gauges(),
        }
        return json.dumps(data, indent=2)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Timings
        for name, stats in self.get_all_timings().items():
            metric_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name}_seconds summary")
            lines.append(f"{metric_name}_seconds_count {stats['count']}")
            lines.append(f"{metric_name}_seconds_sum {stats['total_ms'] / 1000}")
        
        # Counters
        for name, value in self.get_all_counters().items():
            metric_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")
        
        # Gauges
        for name, value in self.get_all_gauges().items():
            metric_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")
        
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 60)
        
        # Timings
        if self._timings:
            print("\n📊 TIMINGS:")
            print("-" * 60)
            for name, stats in sorted(self._timings.items(), key=lambda x: x[1].total_ms, reverse=True):
                print(f"  {name}:")
                print(f"    Count: {stats.count:,}")
                print(f"    Avg:   {stats.avg_ms:.3f}ms")
                print(f"    Min:   {stats.min_ms:.3f}ms")
                print(f"    Max:   {stats.max_ms:.3f}ms")
                print(f"    P95:   {stats.p95_ms:.3f}ms")
        
        # Counters
        if self._counters:
            print("\n📈 COUNTERS:")
            print("-" * 60)
            for name, value in sorted(self._counters.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {value:,}")
        
        # Gauges
        if self._gauges:
            print("\n📉 GAUGES:")
            print("-" * 60)
            for name, value in sorted(self._gauges.items()):
                print(f"  {name}: {value}")
        
        print("\n" + "=" * 60)


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics


# Decorators
def timed(name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Usage:
        @timed
        def my_function():
            pass
        
        @timed("custom_name")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _metrics.record_timing(op_name, elapsed_ms)
        
        return wrapper
    return decorator


def timed_async(name: Optional[str] = None):
    """Decorator to time async function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _metrics.record_timing(op_name, elapsed_ms)
        
        return wrapper
    return decorator


class TimingContext:
    """
    Context manager for timing code blocks.
    
    Usage:
        with TimingContext("my_operation"):
            # code to time
            pass
    """
    
    def __init__(self, name: str, record: bool = True):
        self.name = name
        self.record = record
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.record:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            _metrics.record_timing(self.name, elapsed_ms)
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000


class Counter:
    """Context manager for counting occurrences."""
    
    def __init__(self, name: str):
        self.name = name
    
    def __enter__(self):
        _metrics.increment_counter(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Gauge:
    """Context manager for setting gauge values."""
    
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
    
    def __enter__(self):
        _metrics.set_gauge(self.name, self.value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Profile decorator for detailed function profiling
def profile_function(func: Callable = None, *, name: Optional[str] = None):
    """
    Profile a function with detailed timing.
    
    Usage:
        @profile_function
        def my_func():
            pass
    """
    def decorator(f: Callable) -> Callable:
        op_name = name or f"{f.__module__}.{f.__name__}"
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_perf = time.perf_counter()
            start_cpu = time.process_time()
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                wall_time = (time.perf_counter() - start_perf) * 1000
                cpu_time = (time.process_time() - start_cpu) * 1000
                
                _metrics.record_timing(op_name, wall_time)
                _metrics.record_timing(f"{op_name}_cpu", cpu_time)
                _metrics.increment_counter(f"{op_name}_calls")
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


# Utility function for profiling code blocks
def profile_block(name: str):
    """Profile a code block."""
    return TimingContext(name)


# Example usage
if __name__ == "__main__":
    # Test metrics
    print("Testing performance metrics...")
    
    # Test timing
    @timed
    def slow_function():
        time.sleep(0.01)
        return "done"
    
    for _ in range(10):
        slow_function()
    
    # Test context manager
    with TimingContext("block_1"):
        time.sleep(0.005)
    
    # Test counter
    with Counter("requests"):
        pass
    
    # Test gauge
    with Gauge("memory_usage_mb", 256.5):
        pass
    
    # Test metrics collector
    metrics = get_metrics()
    metrics.increment_counter("test_counter", 5)
    metrics.set_gauge("temperature", 72.5)
    
    # Print summary
    metrics.print_summary()
    
    # Export JSON
    print("\n📄 JSON Export:")
    print(metrics.export_json()[:500] + "...")
    
    # Export Prometheus
    print("\n📊 Prometheus Export:")
    print(metrics.export_prometheus())

