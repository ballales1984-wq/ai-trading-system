"""
Prometheus Metrics for AI Trading System
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from fastapi import Request
from contextlib import asynccontextmanager

# Metrics registry
REQUEST_COUNT = Counter('trading_requests_total', 'Total trading requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('trading_request_duration_seconds', 'Request latency seconds')
ACTIVE_TRADES = Gauge('active_trades', 'Number of active trades')
PORTFOLIO_VALUE = Gauge('portfolio_total_value_usd', 'Portfolio total value USD')
PNL_TOTAL = Gauge('portfolio_pnl_usd', 'Total PnL USD')
RISK_VA_R95 = Gauge('portfolio_var_95_usd', 'VaR 95% USD')
ERROR_COUNT = Counter('trading_errors_total', 'Trading errors', ['type'])

@asynccontextmanager
async def lifespan(app):
    # Startup
    global REGISTRY
    REGISTRY = make_asgi_app()
    yield
    # Shutdown

def instrument_requests():
    """Instrument FastAPI with Prometheus metrics."""
    return REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT

def get_metrics_app():
    """Get Prometheus metrics ASGI app."""
    return make_asgi_app()

