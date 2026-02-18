# src/production/__init__.py
"""
Production Trading Module
========================
Production-ready trading infrastructure for real money trading.
"""

from src.production.broker_interface import (
    BrokerInterface,
    create_broker,
    PaperTradingBroker,
    BinanceBroker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    AccountBalance,
    MarketTicker
)

from src.production.order_manager import (
    OrderManager,
    RiskManager,
    RetryConfig,
    OrderRequest,
    RetryStrategy
)

from src.production.trading_engine import (
    ProductionTradingEngine,
    create_production_engine,
    TradingConfig,
    TradingMode,
    EngineState
)

__all__ = [
    # Broker
    'BrokerInterface',
    'create_broker',
    'PaperTradingBroker',
    'BinanceBroker',
    'Order',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    'Position',
    'AccountBalance',
    'MarketTicker',
    
    # Order Manager
    'OrderManager',
    'RiskManager',
    'RetryConfig',
    'OrderRequest',
    'RetryStrategy',
    
    # Trading Engine
    'ProductionTradingEngine',
    'create_production_engine',
    'TradingConfig',
    'TradingMode',
    'EngineState'
]
