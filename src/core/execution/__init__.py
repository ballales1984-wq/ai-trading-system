# src/core/execution/__init__.py
"""
Execution Module
================
Order execution components:
- Broker Interface: Paper and Live trading
- Order Manager: Order execution with retry logic
"""

from src.core.execution.broker_interface import (
    Broker,
    PaperBroker,
    LiveBroker,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position,
    AccountBalance,
    create_broker
)

from src.core.execution.order_manager import (
    OrderManager,
    OrderRequest,
    ExecutionResult,
    RetryConfig,
    EmergencyExit
)


__all__ = [
    'Broker',
    'PaperBroker',
    'LiveBroker',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'Position',
    'AccountBalance',
    'create_broker',
    'OrderManager',
    'OrderRequest',
    'ExecutionResult',
    'RetryConfig',
    'EmergencyExit'
]
