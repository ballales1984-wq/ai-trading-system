"""
Prometheus Metrics Export
=========================
Prometheus metrics exporter for HFT trading system.
Provides custom metrics for trading operations.

Usage:
    from src.core.performance.prometheus_metrics import (
        get_trading_metrics,
        order_latency,
        signal_latency,
        risk_latency,
        ws_message_rate
    )
    
    # Record metrics
    order_latency.observe(15.2)  # 15.2ms
    signal_latency.observe(5.1)
    ws_message_rate.inc()
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info, CollectorRegistry
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics will be disabled")


# Default registry
_registry = None
_metrics: Dict = {}


def get_registry() -> Optional['CollectorRegistry']:
    """Get or create Prometheus registry."""
    global _registry
    if _registry is None and PROMETHEUS_AVAILABLE:
        _registry = CollectorRegistry()
    return _registry


def _create_metric(name: str, metric_type: str, description: str, **kwargs):
    """Create a metric if Prometheus is available."""
    if not PROMETHEUS_AVAILABLE:
        return None
    
    registry = get_registry()
    if registry is None:
        return None
    
    # Check if already exists
    if name in _metrics:
        return _metrics[name]
    
    # Create based on type
    if metric_type == "counter":
        metric = Counter(name, description, **kwargs, registry=registry)
    elif metric_type == "gauge":
        metric = Gauge(name, description, **kwargs, registry=registry)
    elif metric_type == "histogram":
        # Define buckets for latency metrics
        buckets = kwargs.get('buckets', (.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
        metric = Histogram(name, description, buckets=buckets, **kwargs, registry=registry)
    elif metric_type == "summary":
        metric = Summary(name, description, **kwargs, registry=registry)
    else:
        return None
    
    _metrics[name] = metric
    return metric


# ============================================================================
# Trading Metrics
# ============================================================================

# Order Execution Metrics
order_latency = None
order_count = None
order_errors = None
order_value_total = None

# Signal Generation Metrics
signal_latency = None
signal_count = None
signal_signals_generated = None

# Risk Check Metrics
risk_latency = None
risk_checks_total = None
risk_breaches = None

# WebSocket Metrics
ws_messages_total = None
ws_message_rate = None
ws_connections_active = None
ws_latency = None

# Database Metrics
db_write_latency = None
db_read_latency = None
db_connections_active = None

# System Metrics
cpu_usage = None
memory_usage = None
event_loop_delay = None


def init_metrics() -> None:
    """Initialize all trading metrics."""
    global order_latency, order_count, order_errors, order_value_total
    global signal_latency, signal_count, signal_signals_generated
    global risk_latency, risk_checks_total, risk_breaches
    global ws_messages_total, ws_message_rate, ws_connections_active, ws_latency
    global db_write_latency, db_read_latency, db_connections_active
    global cpu_usage, memory_usage, event_loop_delay
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus not available, skipping metrics initialization")
        return
    
    # Order metrics
    order_latency = _create_metric(
        "order_execution_latency_seconds",
        "histogram",
        "Order execution latency in seconds",
        ["symbol", "side", "broker"]
    )
    order_count = _create_metric(
        "orders_total",
        "counter",
        "Total number of orders",
        ["symbol", "side", "broker", "status"]
    )
    order_errors = _create_metric(
        "order_errors_total",
        "counter",
        "Total number of order errors",
        ["symbol", "error_type"]
    )
    order_value_total = _create_metric(
        "order_value_total",
        "counter",
        "Total order value",
        ["symbol", "side"]
    )
    
    # Signal metrics
    signal_latency = _create_metric(
        "signal_generation_time_seconds",
        "histogram",
        "Signal generation time in seconds",
        ["strategy", "symbol"]
    )
    signal_count = _create_metric(
        "signals_total",
        "counter",
        "Total signals generated",
        ["strategy", "signal_type"]
    )
    signal_signals_generated = _create_metric(
        "signals_generated_total",
        "counter",
        "Total signals by type",
        ["strategy", "signal_type", "direction"]
    )
    
    # Risk metrics
    risk_latency = _create_metric(
        "risk_check_time_seconds",
        "histogram",
        "Risk check latency in seconds"
    )
    risk_checks_total = _create_metric(
        "risk_checks_total",
        "counter",
        "Total risk checks performed",
        ["result"]
    )
    risk_breaches = _create_metric(
        "risk_breaches_total",
        "counter",
        "Total risk breaches",
        ["breach_type"]
    )
    
    # WebSocket metrics
    ws_messages_total = _create_metric(
        "websocket_messages_total",
        "counter",
        "Total WebSocket messages",
        ["source", "message_type"]
    )
    ws_message_rate = _create_metric(
        "websocket_message_rate",
        "gauge",
        "Current WebSocket message rate per second"
    )
    ws_connections_active = _create_metric(
        "websocket_connections_active",
        "gauge",
        "Number of active WebSocket connections"
    )
    ws_latency = _create_metric(
        "websocket_message_latency_seconds",
        "histogram",
        "WebSocket message processing latency",
        ["source"]
    )
    
    # Database metrics
    db_write_latency = _create_metric(
        "db_write_latency_seconds",
        "histogram",
        "Database write latency in seconds",
        ["table"]
    )
    db_read_latency = _create_metric(
        "db_read_latency_seconds",
        "histogram",
        "Database read latency in seconds",
        ["table", "query_type"]
    )
    db_connections_active = _create_metric(
        "db_connections_active",
        "gauge",
        "Number of active database connections"
    )
    
    # System metrics
    cpu_usage = _create_metric(
        "system_cpu_usage_percent",
        "gauge",
        "CPU usage percentage"
    )
    memory_usage = _create_metric(
        "system_memory_usage_bytes",
        "gauge",
        "Memory usage in bytes"
    )
    event_loop_delay = _create_metric(
        "event_loop_delay_seconds",
        "histogram",
        "Event loop callback delay in seconds"
    )
    
    logger.info("Prometheus metrics initialized")


def get_metrics_text() -> str:
    """Get metrics in Prometheus text format."""
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus not available"
    
    registry = get_registry()
    if registry is None:
        return "# No registry"
    
    return generate_latest(registry).decode('utf-8')


def get_content_type() -> str:
    """Get Prometheus content type."""
    return CONTENT_TYPE_LATEST


# ============================================================================
# Context Managers for Easy Timing
# ============================================================================

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, histogram, labels: Optional[Dict] = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        if self.histogram is not None:
            self.histogram.labels(**self.labels).observe(elapsed)


def timed(histogram_metric, labels: Optional[Dict] = None):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if histogram_metric is not None:
                    histogram_metric.labels(**(labels or {})).observe(elapsed)
        return wrapper
    return decorator


# ============================================================================
# Trading-specific helpers
# ============================================================================

def record_order(symbol: str, side: str, broker: str, status: str, 
                 latency_ms: float, value: float = 0) -> None:
    """Record order execution metrics."""
    if order_latency:
        order_latency.labels(symbol=symbol, side=side, broker=broker).observe(latency_ms / 1000)
    if order_count:
        order_count.labels(symbol=symbol, side=side, broker=broker, status=status).inc()
    if value > 0 and order_value_total:
        order_value_total.labels(symbol=symbol, side=side).inc(value)


def record_signal(strategy: str, symbol: str, signal_type: str, 
                  latency_ms: float) -> None:
    """Record signal generation metrics."""
    if signal_latency:
        signal_latency.labels(strategy=strategy, symbol=symbol).observe(latency_ms / 1000)
    if signal_count:
        signal_count.labels(strategy=strategy, signal_type=signal_type).inc()


def record_risk_check(result: str, latency_ms: float) -> None:
    """Record risk check metrics."""
    if risk_latency:
        risk_latency.observe(latency_ms / 1000)
    if risk_checks_total:
        risk_checks_total.labels(result=result).inc()


def record_ws_message(source: str, message_type: str, latency_ms: float) -> None:
    """Record WebSocket message metrics."""
    if ws_messages_total:
        ws_messages_total.labels(source=source, message_type=message_type).inc()
    if ws_latency:
        ws_latency.labels(source=source).observe(latency_ms / 1000)


def record_db_write(table: str, latency_ms: float) -> None:
    """Record database write metrics."""
    if db_write_latency:
        db_write_latency.labels(table=table).observe(latency_ms / 1000)


def record_db_read(table: str, query_type: str, latency_ms: float) -> None:
    """Record database read metrics."""
    if db_read_latency:
        db_read_latency.labels(table=table, query_type=query_type).observe(latency_ms / 1000)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Prometheus metrics...")
    
    # Initialize metrics
    init_metrics()
    
    # Record some test metrics
    record_order("BTCUSDT", "BUY", "binance", "filled", 15.2, 50000)
    record_order("ETHUSDT", "SELL", "bybit", "filled", 12.8, 25000)
    
    record_signal("momentum", "BTCUSDT", "entry", 5.1)
    record_signal("mean_reversion", "ETHUSDT", "exit", 3.2)
    
    record_risk_check("passed", 1.5)
    record_risk_check("passed", 1.2)
    
    record_ws_message("binance", "trade", 0.5)
    record_ws_message("bybit", "orderbook", 0.3)
    
    record_db_write("orders", 2.1)
    record_db_read("prices", "latest", 0.8)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("PROMETHEUS METRICS")
    print("=" * 60)
    print(get_metrics_text())
    print("=" * 60)

