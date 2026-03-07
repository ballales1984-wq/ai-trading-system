"""
Broker Connectors Coverage Tests
================================
Test coverage for broker connector modules.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBybitConnector:
    """Tests for BybitConnector class."""
    
    @pytest.mark.asyncio
    async def test_bybit_connector_creation(self):
        """Test BybitConnector creation."""
        from app.execution.broker_connector import BybitConnector, Broker
        
        connector = BybitConnector(
            api_key="test_api_key",
            secret_key="test_secret",
            testnet=True
        )
        
        assert connector.api_key == "test_api_key"
        assert connector.secret_key == "test_secret"
        assert connector.testnet is True
        assert connector.broker == Broker.BYBIT
        assert connector.connected is False
    
    @pytest.mark.asyncio
    async def test_bybit_connector_testnet_url(self):
        """Test BybitConnector testnet URL."""
        from app.execution.broker_connector import BybitConnector
        
        connector = BybitConnector(testnet=True)
        assert "api-testnet.bybit.com" in connector.base_url
        
        connector_prod = BybitConnector(testnet=False)
        assert "api.bybit.com" in connector_prod.base_url
    
    @pytest.mark.asyncio
    async def test_bybit_connector_default(self):
        """Test BybitConnector defaults."""
        from app.execution.broker_connector import BybitConnector
        
        connector = BybitConnector()
        
        assert connector.api_key == ""
        assert connector.secret_key == ""
        assert connector.testnet is True
        assert connector._session is None
    
    @pytest.mark.asyncio
    async def test_bybit_connect_success(self):
        """Test BybitConnector connect method."""
        from app.execution.broker_connector import BybitConnector
        
        connector = BybitConnector(testnet=True)
        
        with patch.object(connector, '_ensure_session', new_callable=AsyncMock) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value = mock_response
            
            with patch('aiohttp.ClientSession') as mock_client:
                mock_resp = AsyncMock()
                mock_resp.status = 200
                
                mock_session_instance = AsyncMock()
                mock_session_instance.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_session_instance.get.return_value.__aexit__ = AsyncMock()
                mock_client.return_value = mock_session_instance
                
                result = await connector.connect()


class TestPaperTradingConnector:
    """Tests for PaperTradingConnector class."""
    
    @pytest.mark.asyncio
    async def test_paper_connector_creation(self):
        """Test PaperTradingConnector creation."""
        from app.execution.broker_connector import PaperTradingConnector, Broker
        
        connector = PaperTradingConnector(initial_balance=50000.0)
        
        assert connector.broker == Broker.PAPER
        assert connector.balance == {"USDT": 50000.0}
        assert connector.connected is False
        assert connector.positions == {}
        assert connector.orders == {}
    
    @pytest.mark.asyncio
    async def test_paper_connector_default_balance(self):
        """Test PaperTradingConnector default balance."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector()
        
        assert connector.balance == {"USDT": 1000000.0}
    
    @pytest.mark.asyncio
    async def test_paper_connector_connect(self):
        """Test PaperTradingConnector connect method."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector()
        result = await connector.connect()
        
        assert result is True
        assert connector.connected is True
    
    @pytest.mark.asyncio
    async def test_paper_connector_disconnect(self):
        """Test PaperTradingConnector disconnect method."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector()
        await connector.connect()
        await connector.disconnect()
        
        assert connector.connected is False
    
    @pytest.mark.asyncio
    async def test_paper_connector_place_order_buy(self):
        """Test PaperTradingConnector place_order for BUY."""
        from app.execution.broker_connector import PaperTradingConnector, BrokerOrder
        
        connector = PaperTradingConnector(initial_balance=100000.0)
        await connector.connect()
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=1.0,
            price=43000.0
        )
        
        result = await connector.place_order(order)
        
        assert result.status == "FILLED"
        assert result.filled_quantity == 1.0
        assert result.average_price == 43000.0
    
    @pytest.mark.asyncio
    async def test_paper_connector_place_order_sell(self):
        """Test PaperTradingConnector place_order for SELL."""
        from app.execution.broker_connector import PaperTradingConnector, BrokerOrder
        
        connector = PaperTradingConnector(initial_balance=100000.0)
        await connector.connect()
        
        # First buy to have position
        buy_order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=1.0,
            price=43000.0
        )
        await connector.place_order(buy_order)
        
        # Now sell
        sell_order = BrokerOrder(
            symbol="BTCUSDT",
            side="SELL",
            order_type="MARKET",
            quantity=1.0,
            price=44000.0
        )
        result = await connector.place_order(sell_order)
        
        assert result.status == "FILLED"
        assert result.filled_quantity == 1.0
    
    @pytest.mark.asyncio
    async def test_paper_connector_cancel_order(self):
        """Test PaperTradingConnector cancel_order."""
        from app.execution.broker_connector import PaperTradingConnector, BrokerOrder
        
        connector = PaperTradingConnector()
        await connector.connect()
        
        order = BrokerOrder(
            symbol="ETHUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=10.0,
            price=2000.0
        )
        
        # Place order first
        await connector.place_order(order)
        
        # Cancel it
        result = await connector.cancel_order(order.order_id, order.symbol)
        
        assert result is True
        assert connector.orders[order.order_id].status == "CANCELLED"
    
    @pytest.mark.asyncio
    async def test_paper_connector_cancel_nonexistent_order(self):
        """Test PaperTradingConnector cancel_order for nonexistent order."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector()
        await connector.connect()
        
        result = await connector.cancel_order("nonexistent_id", "ETHUSDT")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_paper_connector_get_order_status(self):
        """Test PaperTradingConnector get_order_status."""
        from app.execution.broker_connector import PaperTradingConnector, BrokerOrder
        
        connector = PaperTradingConnector()
        await connector.connect()
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=1.0
        )
        
        await connector.place_order(order)
        
        result = await connector.get_order_status(order.order_id, order.symbol)
        
        assert result.order_id == order.order_id
        assert result.status == "FILLED"
    
    @pytest.mark.asyncio
    async def test_paper_connector_get_order_status_not_found(self):
        """Test PaperTradingConnector get_order_status for nonexistent order."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector()
        await connector.connect()
        
        result = await connector.get_order_status("nonexistent", "BTCUSDT")
        
        assert result.order_id == "nonexistent"
        assert result.status == "NEW"
    
    @pytest.mark.asyncio
    async def test_paper_connector_get_balance(self):
        """Test PaperTradingConnector get_balance."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector(initial_balance=100000.0)
        await connector.connect()
        
        result = await connector.get_balance()
        
        assert len(result) == 1
        assert result[0].asset == "USDT"
        assert result[0].free == 100000.0
        assert result[0].locked == 0.0
        assert result[0].total == 100000.0
    
    @pytest.mark.asyncio
    async def test_paper_connector_get_positions(self):
        """Test PaperTradingConnector get_positions."""
        from app.execution.broker_connector import PaperTradingConnector, BrokerOrder
        
        connector = PaperTradingConnector()
        await connector.connect()
        
        order = BrokerOrder(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=2.0,
            price=43000.0
        )
        await connector.place_order(order)
        
        result = await connector.get_positions()
        
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"
        assert result[0].quantity == 2.0
    
    @pytest.mark.asyncio
    async def test_paper_connector_get_symbol_price(self):
        """Test PaperTradingConnector get_symbol_price."""
        from app.execution.broker_connector import PaperTradingConnector
        
        connector = PaperTradingConnector()
        
        price_btc = await connector.get_symbol_price("BTCUSDT")
        assert price_btc == 43500.0
        
        price_eth = await connector.get_symbol_price("ETHUSDT")
        assert price_eth == 2350.0
        
        price_unknown = await connector.get_symbol_price("UNKNOWNUSDT")
        assert price_unknown == 100.0


class TestBrokerFactoryExtended:
    """Extended tests for BrokerFactory class."""
    
    def test_factory_create_bybit(self):
        """Test creating Bybit connector via factory."""
        from app.execution.broker_connector import BrokerFactory, Broker
        
        connector = BrokerFactory.create_broker("bybit", testnet=True)
        
        assert connector is not None
        assert connector.broker == Broker.BYBIT
    
    def test_factory_create_bybit_lowercase(self):
        """Test creating Bybit connector with lowercase."""
        from app.execution.broker_connector import BrokerFactory, Broker
        
        connector = BrokerFactory.create_broker("bybit")
        
        assert connector is not None
        assert connector.broker == Broker.BYBIT
    
    def test_factory_create_paper_with_custom_balance(self):
        """Test creating Paper connector with custom balance."""
        from app.execution.broker_connector import BrokerFactory, Broker
        
        connector = BrokerFactory.create_broker("paper")
        
        assert connector is not None
        assert connector.broker == Broker.PAPER
        assert connector.balance == {"USDT": 1000000.0}
    
    def test_factory_unsupported_broker(self):
        """Test factory with unsupported broker."""
        from app.execution.broker_connector import BrokerFactory
        
        with pytest.raises(ValueError) as exc_info:
            BrokerFactory.create_broker("unsupported_broker")
        
        assert "Unsupported broker" in str(exc_info.value)
    
    def test_create_broker_connector_alias(self):
        """Test create_broker_connector alias function."""
        from app.execution.broker_connector import create_broker_connector
        
        connector = create_broker_connector("binance", testnet=True)
        
        assert connector is not None
    
    def test_factory_case_insensitive(self):
        """Test factory case insensitivity."""
        from app.execution.broker_connector import BrokerFactory
        
        connector1 = BrokerFactory.create_broker("BINANCE")
        connector2 = BrokerFactory.create_broker("binance")
        connector3 = BrokerFactory.create_broker("Binance")
        
        assert connector1 is not None
        assert connector2 is not None
        assert connector3 is not None


class TestTradeModel:
    """Tests for Trade model."""
    
    def test_trade_creation(self):
        """Test Trade model creation."""
        from app.execution.broker_connector import Trade
        
        trade = Trade(
            order_id="order123",
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=43000.0,
            commission=10.0
        )
        
        assert trade.order_id == "order123"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.quantity == 1.0
        assert trade.price == 43000.0
        assert trade.commission == 10.0
    
    def test_trade_defaults(self):
        """Test Trade model defaults."""
        from app.execution.broker_connector import Trade
        
        trade = Trade(
            order_id="order123",
            symbol="ETHUSDT",
            side="SELL",
            quantity=5.0,
            price=2000.0
        )
        
        assert trade.trade_id is not None
        assert trade.commission == 0.0
        assert trade.broker_trade_id is None


class TestAccountBalanceModel:
    """Tests for AccountBalance model."""
    
    def test_account_balance_creation(self):
        """Test AccountBalance model creation."""
        from app.execution.broker_connector import AccountBalance
        
        balance = AccountBalance(
            asset="USDT",
            free=50000.0,
            locked=10000.0,
            total=60000.0
        )
        
        assert balance.asset == "USDT"
        assert balance.free == 50000.0
        assert balance.locked == 10000.0
        assert balance.total == 60000.0


class TestOrderSideOrderType:
    """Tests for OrderSide and OrderType enums."""
    
    def test_order_side_values(self):
        """Test OrderSide enum values."""
        from app.execution.broker_connector import OrderSide
        
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        from app.execution.broker_connector import OrderType
        
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
