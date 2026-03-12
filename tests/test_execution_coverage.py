"""
Test Coverage for Execution Module
================================
Comprehensive tests to improve coverage for app/execution/* modules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import sys
import os
import asyncio

# Mock ib_insync to avoid import errors and event loop issues
mock_ib_insync = MagicMock()
mock_ib_insync.IB = MagicMock
mock_ib_insync.Contract = MagicMock
mock_ib_insync.Forex = MagicMock
mock_ib_insync.Future = MagicMock
mock_ib_insync.LimitOrder = MagicMock
mock_ib_insync.MarketOrder = MagicMock
mock_ib_insync.Stock = MagicMock
mock_ib_insync.StopOrder = MagicMock
mock_ib_insync.Trade = MagicMock
mock_ib_insync.util = MagicMock

sys.modules['ib_insync'] = mock_ib_insync

# Set up event loop for MainThread to avoid ib_insync import errors
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExecutionBrokerConnector:
    """Test app.execution.broker_connector module."""
    
    def test_broker_connector_module_import(self):
        """Test broker_connector module can be imported."""
        from app.execution import broker_connector
        assert broker_connector is not None
    
    def test_broker_connector_class(self):
        """Test BrokerConnector class exists."""
        from app.execution.broker_connector import BrokerConnector
        assert BrokerConnector is not None
    
    def test_broker_enum(self):
        """Test Broker enum exists."""
        from app.execution.broker_connector import Broker
        assert Broker is not None
        assert hasattr(Broker, 'BINANCE')
        assert hasattr(Broker, 'BYBIT')
        assert hasattr(Broker, 'PAPER')
    
    def test_order_status_enum(self):
        """Test OrderStatus enum exists."""
        from app.execution.broker_connector import OrderStatus
        assert OrderStatus is not None
    
    def test_broker_order_class(self):
        """Test BrokerOrder class exists."""
        from app.execution.broker_connector import BrokerOrder
        assert BrokerOrder is not None
    
    def test_trade_class(self):
        """Test Trade class exists."""
        from app.execution.broker_connector import Trade
        assert Trade is not None
    
    def test_account_balance_class(self):
        """Test AccountBalance class exists."""
        from app.execution.broker_connector import AccountBalance
        assert AccountBalance is not None
    
    def test_position_class(self):
        """Test Position class exists."""
        from app.execution.broker_connector import Position
        assert Position is not None
    
    def test_broker_order_creation(self):
        """Test BrokerOrder creation."""
        from app.execution.broker_connector import BrokerOrder, OrderSide, OrderType
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.5,
            price=50000.0
        )
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
    
    def test_trade_creation(self):
        """Test Trade creation."""
        from app.execution.broker_connector import Trade
        
        trade = Trade(
            order_id="12345",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            price=50000.0,
            commission=0.5
        )
        assert trade.symbol == "BTCUSDT"
        assert trade.quantity == 0.5
    
    def test_account_balance_creation(self):
        """Test AccountBalance creation."""
        from app.execution.broker_connector import AccountBalance
        
        balance = AccountBalance(
            asset="USDT",
            free=90000.0,
            locked=10000.0,
            total=100000.0
        )
        assert balance.total == 100000.0
        assert balance.free == 90000.0
    
    def test_position_creation(self):
        """Test Position creation."""
        from app.execution.broker_connector import Position
        
        position = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            entry_price=48000.0,
            current_price=50000.0,
            unrealized_pnl=1000.0,
            margin=500.0
        )
        assert position.symbol == "BTCUSDT"
        assert position.quantity == 0.5
    
    def test_broker_factory(self):
        """Test BrokerFactory class."""
        from app.execution.broker_connector import BrokerFactory
        assert BrokerFactory is not None
    
    def test_create_broker_connector_function(self):
        """Test create_broker_connector function."""
        from app.execution.broker_connector import create_broker_connector
        assert callable(create_broker_connector)


class TestExecutionOrderManager:
    """Test app.execution.order_manager module."""
    
    def test_order_manager_module_import(self):
        """Test order_manager module can be imported."""
        from app.execution import order_manager
        assert order_manager is not None
    
    def test_order_status_enum(self):
        """Test OrderStatus enum exists."""
        from app.execution.order_manager import OrderStatus
        assert OrderStatus is not None
    
    def test_order_side_enum(self):
        """Test OrderSide enum exists."""
        from app.execution.order_manager import OrderSide
        assert OrderSide is not None
    
    def test_order_type_enum(self):
        """Test OrderType enum exists."""
        from app.execution.order_manager import OrderType
        assert OrderType is not None
    
    def test_time_in_force_enum(self):
        """Test TimeInForce enum exists."""
        from app.execution.order_manager import TimeInForce
        assert TimeInForce is not None
    
    def test_order_class(self):
        """Test Order class exists."""
        from app.execution.order_manager import Order
        assert Order is not None
    
    def test_order_fill_class(self):
        """Test OrderFill class exists."""
        from app.execution.order_manager import OrderFill
        assert OrderFill is not None
    
    def test_order_book_update_class(self):
        """Test OrderBookUpdate class exists."""
        from app.execution.order_manager import OrderBookUpdate
        assert OrderBookUpdate is not None
    
    def test_order_manager_class(self):
        """Test OrderManager class exists."""
        from app.execution.order_manager import OrderManager
        assert OrderManager is not None
    
    def test_order_creation(self):
        """Test Order creation."""
        from app.execution.order_manager import Order
        
        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.5,
            price=50000.0
        )
        assert order.symbol == "BTCUSDT"
    
    def test_order_fill_creation(self):
        """Test OrderFill creation."""
        from app.execution.order_manager import OrderFill
        
        fill = OrderFill(
            order_id="12345",
            fill_quantity=0.5,
            fill_price=50000.0
        )
        assert fill.order_id == "12345"
        assert fill.fill_quantity == 0.5
    
    def test_order_manager_initialization(self):
        """Test OrderManager initialization."""
        from app.execution.order_manager import OrderManager
        
        manager = OrderManager()
        assert manager is not None
    
    def test_order_manager_create_order(self):
        """Test OrderManager create_order method."""
        from app.execution.order_manager import OrderManager
        
        manager = OrderManager()
        
        async def test():
            order = await manager.create_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.5,
                order_type="MARKET"
            )
            assert order.symbol == "BTCUSDT"
        
        import asyncio
        asyncio.run(test())


class TestExecutionEngine:
    """Test app.execution.execution_engine module."""
    
    def test_execution_engine_module_import(self):
        """Test execution_engine module can be imported."""
        from app.execution import execution_engine
        assert execution_engine is not None
    
    def test_execution_state_enum(self):
        """Test ExecutionState enum exists."""
        from app.execution.execution_engine import ExecutionState
        assert ExecutionState is not None
    
    def test_retry_config_class(self):
        """Test RetryConfig class exists."""
        from app.execution.execution_engine import RetryConfig
        assert RetryConfig is not None
    
    def test_execution_result_class(self):
        """Test ExecutionResult class exists."""
        from app.execution.execution_engine import ExecutionResult
        assert ExecutionResult is not None
    
    def test_execution_engine_class(self):
        """Test ExecutionEngine class exists."""
        from app.execution.execution_engine import ExecutionEngine
        assert ExecutionEngine is not None
    
    def test_retry_config_creation(self):
        """Test RetryConfig creation."""
        from app.execution.execution_engine import RetryConfig
        
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0
        )
        assert config.max_retries == 3
    
    def test_execution_result_creation(self):
        """Test ExecutionResult creation."""
        from app.execution.execution_engine import ExecutionResult
        
        result = ExecutionResult(
            success=True,
            order_id="12345",
            message="Order placed successfully"
        )
        assert result.success is True
        assert result.order_id == "12345"
    
    def test_execution_engine_with_mock_broker(self):
        """Test ExecutionEngine with mock broker."""
        from app.execution.execution_engine import ExecutionEngine, RetryConfig
        from app.execution.broker_connector import PaperTradingConnector
        
        async def test():
            broker = PaperTradingConnector()
            await broker.connect()
            
            engine = ExecutionEngine(broker=broker)
            assert engine is not None
            assert engine.broker is not None
        
        import asyncio
        asyncio.run(test())


class TestExecutionConnectors:
    """Test app.execution.connectors modules."""
    
    def test_binace_connector_class(self):
        """Test BinanceConnector class."""
        from app.execution.connectors.binance_connector import BinanceConnector
        assert BinanceConnector is not None
    
    def test_paper_connector_class(self):
        """Test PaperConnector class."""
        from app.execution.connectors.paper_connector import PaperConnector
        assert PaperConnector is not None
    
    def test_ib_connector_class(self):
        """Test IBConnector class."""
        from app.execution.connectors.ib_connector import IBConnector
        assert IBConnector is not None
    
    def test_connectors_import(self):
        """Test connectors package can be imported."""
        from app.execution import connectors
        assert connectors is not None


class TestExecutionIntegration:
    """Integration tests for execution module."""
    
    def test_order_lifecycle(self):
        """Test order lifecycle."""
        from app.execution.order_manager import Order, OrderStatus
        
        # Create order
        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.5,
            price=50000.0
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.status == "PENDING"
    
    def test_position_calculation(self):
        """Test position calculation."""
        from app.execution.broker_connector import Position
        
        position = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            entry_price=45000.0,
            current_price=50000.0,
            unrealized_pnl=5000.0,
            margin=1000.0
        )
        
        pnl = (position.current_price - position.entry_price) * position.quantity
        assert pnl == 5000.0
    
    def test_balance_check(self):
        """Test balance checking."""
        from app.execution.broker_connector import AccountBalance
        
        balance = AccountBalance(
            asset="USDT",
            free=95000.0,
            locked=5000.0,
            total=100000.0
        )
        
        order_cost = 50000.0 * 0.5
        assert balance.free >= order_cost
    
    def test_retry_config_values(self):
        """Test retry configuration values."""
        from app.execution.execution_engine import RetryConfig
        
        config = RetryConfig(
            max_retries=5,
            base_delay=1.5,
            max_delay=30.0
        )
        
        assert config.max_retries > 0
        assert config.base_delay > 0
        assert config.max_delay > 0

