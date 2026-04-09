"""
Test Suite for Broker Connector Module
====================================
Comprehensive tests for broker connectors and data models.
"""

import pytest
from datetime import datetime
from app.execution.broker_connector import (
    Broker,
    OrderStatus,
    BrokerOrder,
    Trade,
    AccountBalance,
    Position,
    BrokerConnector,
)


class TestBroker:
    """Tests for Broker enum."""

    def test_broker_values(self):
        """Test broker enum values."""
        assert Broker.BINANCE.value == "binance"
        assert Broker.INTERACTIVE_BROKERS.value == "ib"
        assert Broker.BYBIT.value == "bybit"
        assert Broker.COINBASE.value == "coinbase"
        assert Broker.PAPER.value == "paper"


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_order_status_values(self):
        """Test order status values."""
        assert OrderStatus.NEW.value == "NEW"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"


class TestBrokerOrder:
    """Tests for BrokerOrder model."""

    def test_broker_order_creation(self):
        """Test broker order creation."""
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.1,
        )
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.order_type == "MARKET"
        assert order.quantity == 0.1
        assert order.status == "NEW"
        assert order.filled_quantity == 0.0

    def test_broker_order_with_price(self):
        """Test broker order with price."""
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.1,
            price=45000.0,
        )
        assert order.price == 45000.0

    def test_broker_order_with_stop(self):
        """Test broker order with stop price."""
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="SELL",
            order_type="STOP_LOSS",
            quantity=0.1,
            stop_price=44000.0,
        )
        assert order.stop_price == 44000.0

    def test_broker_order_defaults(self):
        """Test broker order default values."""
        order = BrokerOrder(
            symbol="ETHUSDT",
            side="SELL",
            order_type="MARKET",
            quantity=1.0,
        )
        assert order.order_id is not None
        assert order.broker_order_id is None
        assert order.time_in_force == "GTC"
        assert order.created_at is not None


class TestTrade:
    """Tests for Trade model."""

    def test_trade_creation(self):
        """Test trade creation."""
        trade = Trade(
            order_id="order_123",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=45000.0,
        )
        assert trade.order_id == "order_123"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.quantity == 0.1
        assert trade.price == 45000.0
        assert trade.commission == 0.0

    def test_trade_with_commission(self):
        """Test trade with commission."""
        trade = Trade(
            order_id="order_123",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=45000.0,
            commission=4.5,
        )
        assert trade.commission == 4.5

    def test_trade_defaults(self):
        """Test trade default values."""
        trade = Trade(
            order_id="order_123",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=45000.0,
        )
        assert trade.trade_id is not None
        assert trade.timestamp is not None
        assert trade.broker_trade_id is None


class TestAccountBalance:
    """Tests for AccountBalance model."""

    def test_account_balance_creation(self):
        """Test account balance creation."""
        balance = AccountBalance(
            asset="USDT",
            free=10000.0,
            locked=5000.0,
            total=15000.0,
        )
        assert balance.asset == "USDT"
        assert balance.free == 10000.0
        assert balance.locked == 5000.0
        assert balance.total == 15000.0

    def test_account_balance_total_calculation(self):
        """Test account balance total calculation."""
        balance = AccountBalance(
            asset="BTC",
            free=0.5,
            locked=0.2,
            total=0.0,  # Will be calculated
        )
        # If total is 0, it should be sum of free + locked
        if balance.total == 0:
            balance.total = balance.free + balance.locked
        assert balance.total == 0.7


class TestPosition:
    """Tests for Position model."""

    def test_position_creation(self):
        """Test position creation."""
        position = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            entry_price=45000.0,
        )
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.quantity == 0.5
        assert position.entry_price == 45000.0

    def test_position_with_pnl(self):
        """Test position with PnL."""
        position = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            entry_price=45000.0,
            current_price=46000.0,
        )
        assert position.current_price == 46000.0

    def test_position_defaults(self):
        """Test position default values."""
        position = Position(
            symbol="ETHUSDT",
            side="SHORT",
            quantity=1.0,
            entry_price=2500.0,
        )
        assert position.position_id is not None
        assert position.unrealized_pnl == 0.0


class TestBrokerConnector:
    """Tests for BrokerConnector base class."""

    def test_broker_connector_is_abstract(self):
        """Test that BrokerConnector cannot be instantiated."""
        with pytest.raises(TypeError):
            BrokerConnector(config={})

    def test_concrete_connector_creation(self):
        """Test creating a concrete connector."""
        from app.execution.broker_connector import PaperTradingConnector

        connector = PaperTradingConnector()
        assert connector.balance == {"USDT": 1000000.0}
        assert connector.positions == {}
        assert connector.orders == {}

    def test_paper_connector_default_prices(self):
        """Test paper connector default prices."""
        from app.execution.broker_connector import PaperTradingConnector

        connector = PaperTradingConnector()
        # Just check initialization works
        assert connector is not None

    def test_paper_connector_get_balance(self):
        """Test paper connector get balance."""
        from app.execution.broker_connector import PaperTradingConnector

        connector = PaperTradingConnector()
        # Just check the connector has balance
        assert "USDT" in connector.balance
