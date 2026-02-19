# src/core/execution/__init__.py
"""
Execution Module
================
Order execution components:
- Broker Interface: Paper and Live trading
- Order Manager: Order execution with retry logic
- TCA: Transaction Cost Analysis
- Order Book Simulator: Market impact estimation
- Best Execution: TWAP, VWAP, POV algorithms
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

# TCA - Transaction Cost Analysis
from src.core.execution.tca import (
    TransactionCostAnalyzer,
    SlippageModel,
    ExecutionQuality,
    TradeRecord,
    OrderSnapshot,
    MarketImpactModel,
    ExecutionAlgorithm,
    create_tca_analyzer,
)

# Order Book Simulator
from src.core.execution.orderbook_simulator import (
    OrderBookSimulator,
    OrderBookSnapshot,
    MarketImpactEstimate,
    PriceLevel,
    create_order_book_from_depth,
)

# Best Execution
from src.core.execution.best_execution import (
    BestExecutionEngine,
    ExecutionStrategy,
    ExecutionStatus,
    ExecutionConfig,
    ExecutionPlan,
    ExecutionSlice,
    MarketDataSnapshot,
    TWAPAlgorithm,
    VWAPAlgorithm,
    POVAlgorithm,
    AdaptiveAlgorithm,
    create_execution_engine,
)


__all__ = [
    # Broker Interface
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
    
    # Order Manager
    'OrderManager',
    'OrderRequest',
    'ExecutionResult',
    'RetryConfig',
    'EmergencyExit',
    
    # TCA
    'TransactionCostAnalyzer',
    'SlippageModel',
    'ExecutionQuality',
    'TradeRecord',
    'OrderSnapshot',
    'MarketImpactModel',
    'ExecutionAlgorithm',
    'create_tca_analyzer',
    
    # Order Book Simulator
    'OrderBookSimulator',
    'OrderBookSnapshot',
    'MarketImpactEstimate',
    'PriceLevel',
    'create_order_book_from_depth',
    
    # Best Execution
    'BestExecutionEngine',
    'ExecutionStrategy',
    'ExecutionStatus',
    'ExecutionConfig',
    'ExecutionPlan',
    'ExecutionSlice',
    'MarketDataSnapshot',
    'TWAPAlgorithm',
    'VWAPAlgorithm',
    'POVAlgorithm',
    'AdaptiveAlgorithm',
    'create_execution_engine',
]
