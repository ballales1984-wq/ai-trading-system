"""
Performance Module
================
High-performance components for HFT trading system.

Modules:
- event_loop: uvloop integration for 2-4x faster async
- async_logging: Queue-based async logging
- metrics: Performance profiling and custom metrics
- prometheus_metrics: Prometheus exporters
- ring_buffer: Lock-free ring buffer for streaming
- db_batcher: Database write batching
- message_bus: Redis pub/sub for microservices
"""

from src.core.performance.event_loop import (
    get_optimized_event_loop,
    configure_event_loop,
    HighPerformanceRunner,
    run_async_main,
    check_performance_readiness,
    UVLOOP_AVAILABLE
)

from src.core.performance.async_logging import (
    setup_async_logging,
    get_log_queue,
    PerformanceLogger,
    timing_context,
    timed,
    perf_logger
)

from src.core.performance.metrics import (
    MetricsCollector,
    get_metrics,
    timed,
    timed_async,
    TimingContext,
    Counter,
    Gauge,
    profile_function,
    profile_block
)

from src.core.performance.ring_buffer import (
    RingBuffer,
    BatchingRingBuffer,
    MessageBatcher,
    ThreadSafeBuffer,
    BufferStats
)

from src.core.performance.db_batcher import (
    DatabaseBatcher,
    get_db_batcher,
    OrderBatcher,
    PriceBatcher,
    WriteBatch
)

from src.core.performance.message_bus import (
    MessageBus,
    get_message_bus,
    TradingMessageBus,
    Channel,
    Message
)

# Try to import prometheus metrics
try:
    from src.core.performance.prometheus_metrics import (
        init_metrics,
        get_metrics_text,
        get_content_type,
        record_order,
        record_signal,
        record_risk_check,
        record_ws_message,
        record_db_write,
        record_db_read,
        Timer,
        timed
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


__all__ = [
    # Event loop
    "get_optimized_event_loop",
    "configure_event_loop", 
    "HighPerformanceRunner",
    "run_async_main",
    "check_performance_readiness",
    "UVLOOP_AVAILABLE",
    
    # Async logging
    "setup_async_logging",
    "get_log_queue",
    "PerformanceLogger",
    "timing_context",
    "timed",
    "perf_logger",
    
    # Metrics
    "MetricsCollector",
    "get_metrics",
    "timed_async",
    "TimingContext",
    "Counter",
    "Gauge",
    "profile_function",
    "profile_block",
    
    # Ring buffer
    "RingBuffer",
    "BatchingRingBuffer",
    "MessageBatcher",
    "ThreadSafeBuffer",
    "BufferStats",
    
    # DB batcher
    "DatabaseBatcher",
    "get_db_batcher",
    "OrderBatcher",
    "PriceBatcher",
    "WriteBatch",
    
    # Message bus
    "MessageBus",
    "get_message_bus",
    "TradingMessageBus",
    "Channel",
    "Message",
    
    # Prometheus
    "init_metrics",
    "get_metrics_text",
    "get_content_type",
    "record_order",
    "record_signal",
    "record_risk_check",
    "record_ws_message",
    "record_db_write",
    "record_db_read",
    "Timer",
    "PROMETHEUS_AVAILABLE",
]

