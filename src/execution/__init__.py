"""
Execution Module
================
Moduli per l'esecuzione automatica degli ordini sugli exchange.
"""

from .auto_executor import (
    AutoExecutor,
    SimulatedExchangeClient,
    SafetyConfig,
    RateLimiter,
    ExecutedOrder,
    OrderStatus,
    OrderType,
    create_executor_from_config
)

__all__ = [
    "AutoExecutor",
    "SimulatedExchangeClient",
    "SafetyConfig",
    "RateLimiter",
    "ExecutedOrder",
    "OrderStatus",
    "OrderType",
    "create_executor_from_config"
]
