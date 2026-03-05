"""
Tests for Broker Connector Module
================================
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBroker:
    """Test Broker enum."""
    
    def test_broker_values(self):
        """Test Broker enum values."""
        from app.execution.broker_connector import Broker
        
        assert Broker.BINANCE.value == "binance"
        assert Broker.INTERACTIVE_BROKERS.value == "ib"
        assert Broker.BYBIT.value == "bybit"
        assert Broker.COINBASE.value == "coinbase"
        assert Broker.PAPER.value == "paper"


class TestOrderStatus:
    """Test OrderStatus enum."""
    
    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        from app.execution.broker_connector import OrderStatus
        
        assert OrderStatus.NEW.value == "NEW"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"


class TestOrderSide:
    """Test OrderSide enum."""
    
    def test_order_side_values(self):
        """Test OrderSide enum values."""
        from app.execution.broker_connector import OrderSide
        
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestOrderType:
    """Test OrderType enum."""
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        from app.execution.broker_connector import OrderType
        
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"


class TestBrokerOrder:
    """Test BrokerOrder model."""
    
    def test_broker_order_creation(self):
        """Test creating a BrokerOrder."""
        from app.execution.broker_connector import BrokerOrder
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.01
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.quantity == 0.01
    
    def test_broker_order_defaults(self):
        """Test BrokerOrder default values."""
        from app.execution.broker_connector import BrokerOrder
        
        order = BrokerOrder(
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=0.1
        )
        
        assert order.status == "NEW"  # Default status
        assert order.broker == "binance"  # Default broker


class TestTrade:
    """Test Trade model."""
    
    def test_trade_creation(self):
        """Test creating a Trade."""
        from app.execution.broker_connector import Trade
        
        trade = Trade(
            order_id="order_123",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            price=50000.0,
            commission=0.01
        )
        
        assert trade.order_id == "order_123"
        assert trade.symbol == "BTCUSDT"
        assert trade.price == 50000.0


class TestAccountBalance:
    """Test AccountBalance model."""
    
    def test_account_balance_creation(self):
        """Test creating AccountBalance."""
        from app.execution.broker_connector import AccountBalance
        
        balance = AccountBalance(
            asset="USDT",
            free=9000.0,
            locked=1000.0,
            total=10000.0
        )
        
        assert balance.total == 10000.0
        assert balance.free == 9000.0
        assert balance.asset == "USDT"


class TestPosition:
    """Test Position model."""
    
    def test_position_creation(self):
        """Test creating a Position."""
        from app.execution.broker_connector import Position
        
        position = Position(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            entry_price=45000.0,
            current_price=50000.0,
            unrealized_pnl=2500.0,
            margin=500.0
        )
        
        assert position.symbol == "BTCUSDT"
        assert position.quantity == 0.5
        assert position.entry_price == 45000.0


class TestBrokerFactory:
    """Test BrokerFactory class."""
    
    def test_broker_factory_creation(self):
        """Test creating BrokerFactory."""
        from app.execution.broker_connector import BrokerFactory
        
        factory = BrokerFactory()
        
        assert factory is not None


class TestCreateBrokerConnector:
    """Test create_broker_connector function."""
    
    def test_create_broker_connector_paper(self):
        """Test creating paper trading connector."""
        from app.execution.broker_connector import create_broker_connector, Broker
        
        connector = create_broker_connector("paper")
        
        assert connector is not None
        assert isinstance(connector.broker, Broker)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
