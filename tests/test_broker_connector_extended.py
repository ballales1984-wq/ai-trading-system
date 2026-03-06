"""
Tests for Broker Connector Extended Coverage
==========================================
Additional tests to improve broker_connector coverage.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBrokerEnumExtended:
    """Extended tests for Broker enum."""
    
    def test_broker_values_extended(self):
        """Test all Broker enum values."""
        from app.execution.broker_connector import Broker
        
        assert Broker.BINANCE.value == "binance"
        assert Broker.INTERACTIVE_BROKERS.value == "ib"
        assert Broker.BYBIT.value == "bybit"
        assert Broker.COINBASE.value == "coinbase"
        assert Broker.PAPER.value == "paper"
    
    def test_broker_from_string(self):
        """Test creating Broker from string."""
        from app.execution.broker_connector import Broker
        
        assert Broker("binance") == Broker.BINANCE
        assert Broker("bybit") == Broker.BYBIT


class TestOrderStatusExtended:
    """Extended tests for OrderStatus enum."""
    
    def test_order_status_values_extended(self):
        """Test all OrderStatus values."""
        from app.execution.broker_connector import OrderStatus
        
        assert OrderStatus.NEW.value == "NEW"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"


class TestBrokerOrderExtended:
    """Extended tests for BrokerOrder model."""
    
    def test_broker_order_with_all_fields(self):
        """Test BrokerOrder with all fields."""
        from app.execution.broker_connector import BrokerOrder
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.5,
            price=50000.0,
            stop_price=49000.0,
            time_in_force="GTC",
            status="NEW",
            broker="binance"
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.order_type == "LIMIT"
        assert order.quantity == 0.5
        assert order.price == 50000.0
        assert order.stop_price == 49000.0
        assert order.time_in_force == "GTC"
        assert order.status == "NEW"
        assert order.broker == "binance"
    
    def test_broker_order_defaults(self):
        """Test BrokerOrder default values."""
        from app.execution.broker_connector import BrokerOrder
        
        order = BrokerOrder(
            symbol="ETHUSDT",
            side="SELL",
            order_type="MARKET",
            quantity=1.0
        )
        
        assert order.order_id is not None
        assert order.broker_order_id is None
        assert order.price is None
        assert order.stop_price is None
        assert order.time_in_force == "GTC"
        assert order.status == "NEW"
        assert order.filled_quantity == 0.0
        assert order.average_price is None
        assert order.error_message is None
    
    def test_broker_order_update_status(self):
        """Test updating BrokerOrder status."""
        from app.execution.broker_connector import BrokerOrder
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.1
        )
        
        # Update status to filled
        order.status = "FILLED"
        order.filled_quantity = 0.1
        order.average_price = 50000.0
        
        assert order.status == "FILLED"
        assert order.filled_quantity == 0.1
        assert order.average_price == 50000.0


class TestTradeExtended:
    """Extended tests for Trade model."""
    
    def test_trade_with_all_fields(self):
        """Test Trade with all fields."""
        from app.execution.broker_connector import Trade
        
        trade = Trade(
            order_id="order-123",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            price=50000.0,
            commission=0.5,
            broker_trade_id="trade-456"
        )
        
        assert trade.order_id == "order-123"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.quantity == 0.5
        assert trade.price == 50000.0
        assert trade.commission == 0.5
        assert trade.broker_trade_id == "trade-456"
        assert trade.trade_id is not None
    
    def test_trade_defaults(self):
        """Test Trade default values."""
        from app.execution.broker_connector import Trade
        
        trade = Trade(
            order_id="order-123",
            symbol="ETHUSDT",
            side="SELL",
            quantity=1.0,
            price=3000.0
        )
        
        assert trade.trade_id is not None
        assert trade.commission == 0.0
        assert trade.timestamp is not None


class TestAccountBalanceExtended:
    """Extended tests for AccountBalance model."""
    
    def test_account_balance_calculations(self):
        """Test AccountBalance total calculation."""
        from app.execution.broker_connector import AccountBalance
        
        balance = AccountBalance(
            asset="USDT",
            free=5000.0,
            locked=1000.0,
            total=6000.0
        )
        
        assert balance.asset == "USDT"
        assert balance.free == 5000.0
        assert balance.locked == 1000.0
        assert balance.total == 6000.0


class TestPositionExtended:
    """Extended tests for Position model."""
    
    def test_position_pnl_calculation(self):
        """Test Position PnL calculation."""
        from app.execution.broker_connector import Position
        
        position = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.5,
            entry_price=48000.0,
            current_price=50000.0,
            unrealized_pnl=1000.0,
            leverage=1.0,
            margin=24000.0
        )
        
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.quantity == 0.5
        assert position.entry_price == 48000.0
        assert position.current_price == 50000.0
        assert position.unrealized_pnl == 1000.0
    
    def test_position_short_side(self):
        """Test Position with SHORT side."""
        from app.execution.broker_connector import Position
        
        position = Position(
            symbol="BTCUSDT",
            side="SHORT",
            quantity=0.5,
            entry_price=50000.0,
            current_price=48000.0,
            unrealized_pnl=1000.0,
            leverage=1.0,
            margin=25000.0
        )
        
        assert position.side == "SHORT"
        assert position.unrealized_pnl == 1000.0


class TestBinanceConnector:
    """Tests for BinanceConnector class."""
    
    def test_binance_connector_creation(self):
        """Test BinanceConnector creation."""
        from app.execution.broker_connector import BinanceConnector, Broker
        
        connector = BinanceConnector(
            api_key="test_api_key",
            secret_key="test_secret",
            testnet=True
        )
        
        assert connector.api_key == "test_api_key"
        assert connector.secret_key == "test_secret"
        assert connector.testnet is True
        assert connector.broker == Broker.BINANCE
        assert connector.connected is False
    
    def test_binance_connector_testnet_url(self):
        """Test BinanceConnector testnet URL."""
        from app.execution.broker_connector import BinanceConnector
        
        connector = BinanceConnector(testnet=True)
        assert "testnet.binance.vision" in connector.base_url
        
        connector_prod = BinanceConnector(testnet=False)
        assert "api.binance.com" in connector_prod.base_url
    
    def test_binance_connector_default(self):
        """Test BinanceConnector defaults."""
        from app.execution.broker_connector import BinanceConnector
        
        connector = BinanceConnector()
        
        assert connector.api_key == ""
        assert connector.secret_key == ""
        assert connector.testnet is True
        assert connector._session is None


class TestBrokerConnectorFactory:
    """Tests for broker connector factory."""
    
    def test_factory_creation(self):
        """Test creating broker factory."""
        from app.execution.broker_connector import create_broker_connector
        
        # Test creating Binance connector
        binance_conn = create_broker_connector("binance", testnet=True)
        assert binance_conn is not None
        
        # Test creating Paper connector
        paper_conn = create_broker_connector("paper")
        assert paper_conn is not None
    
    def test_factory_unknown_broker(self):
        """Test factory with unknown broker."""
        from app.execution.broker_connector import create_broker_connector
        
        # Should raise ValueError for unknown broker
        with pytest.raises(ValueError):
            create_broker_connector("unknown_broker")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
