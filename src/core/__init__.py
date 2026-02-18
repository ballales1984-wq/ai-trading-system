# src/core/__init__.py
"""
Core Module
==========
Trading engine core components:
- Event Bus: Event-driven communication
- State Manager: Persistent state management
- Engine: Main orchestrator
- Portfolio: Portfolio management
- Risk: Risk management
- Execution: Order execution and broker interfaces
"""

from src.core.event_bus import (
    EventBus,
    EventType,
    Event,
    EventHandler,
    SignalEventHandler,
    RiskEventHandler,
    create_event
)

from src.core.state_manager import (
    StateManager,
    PortfolioState,
    PositionState,
    OrderState,
    ModelState
)

from src.core.engine import (
    TradingEngine,
    EngineConfig,
    EngineState,
    TradingMode,
    create_engine
)

from src.core.portfolio.portfolio_manager import (
    PortfolioManager,
    Position,
    PositionSide,
    PortfolioMetrics
)

from src.core.risk.risk_engine import (
    RiskEngine,
    RiskLimits,
    RiskState,
    RiskLevel,
    RiskCheckResult
)

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
    # Event Bus
    'EventBus',
    'EventType',
    'Event',
    'EventHandler',
    'SignalEventHandler',
    'RiskEventHandler',
    'create_event',
    
    # State Manager
    'StateManager',
    'PortfolioState',
    'PositionState',
    'OrderState',
    'ModelState',
    
    # Engine
    'TradingEngine',
    'EngineConfig',
    'EngineState',
    'TradingMode',
    'create_engine',
    
    # Portfolio
    'PortfolioManager',
    'Position',
    'PositionSide',
    'PortfolioMetrics',
    
    # Risk
    'RiskEngine',
    'RiskLimits',
    'RiskState',
    'RiskLevel',
    'RiskCheckResult',
    
    # Execution
    'Broker',
    'PaperBroker',
    'LiveBroker',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'AccountBalance',
    'create_broker',
    
    # Order Manager
    'OrderManager',
    'OrderRequest',
    'ExecutionResult',
    'RetryConfig',
    'EmergencyExit'
]


__version__ = '2.0.0'
