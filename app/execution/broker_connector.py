"""
Broker Connectors
================
Multi-broker execution adapter pattern.
Supports Binance, Interactive Brokers, Bybit, and other brokers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import uuid4

from pydantic import BaseModel, Field
from app.core.logging import TradingLogger


logger = TradingLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class Broker(str, Enum):
    """Supported brokers."""
    BINANCE = "binance"
    INTERACTIVE_BROKERS = "ib"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    PAPER = "paper"


class OrderStatus(str, Enum):
    """Broker order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ============================================================================
# DATA MODELS
# ============================================================================

class BrokerOrder(BaseModel):
    """Broker-agnostic order model."""
    order_id: str = Field(default_factory=lambda: str(uuid4()))
    broker_order_id: Optional[str] = None
    
    symbol: str
    side: str  # BUY, SELL
    order_type: str  # MARKET, LIMIT, STOP, STOP_LIMIT
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    
    # Status
    status: str = "NEW"
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Broker
    broker: str = "binance"
    
    # Error
    error_message: Optional[str] = None


class Trade(BaseModel):
    """Trade execution model."""
    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    broker_trade_id: Optional[str] = None


class AccountBalance(BaseModel):
    """Account balance model."""
    asset: str
    free: float
    locked: float
    total: float


class Position(BaseModel):
    """Position model."""
    symbol: str
    side: str  # LONG, SHORT
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float = 1.0
    margin: float


# ============================================================================
# ABSTRACT BROKER CONNECTOR
# ============================================================================

class BrokerConnector(ABC):
    """
    Abstract base class for broker connectors.
    All broker implementations must inherit from this class.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.connected = False
        self.logger = TradingLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to broker."""
        pass
    
    @abstractmethod
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """Place an order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> BrokerOrder:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> List[AccountBalance]:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get open positions."""
        pass
    
    @abstractmethod
    async def get_symbol_price(self, symbol: str) -> float:
        """Get current symbol price."""
        pass
    
    # Common methods
    async def health_check(self) -> bool:
        """Check if broker connection is healthy."""
        try:
            await self.get_balance()
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# ============================================================================
# BINANCE CONNECTOR
# ============================================================================

class BinanceConnector(BrokerConnector):
    """
    Binance broker connector.
    Supports spot, futures, and margin trading.
    Uses aiohttp for async HTTP requests with HMAC-SHA256 signing.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        super().__init__(api_key, secret_key, testnet)
        self.broker = Broker.BINANCE
        self.base_url = (
            "https://testnet.binance.vision/api" if testnet
            else "https://api.binance.com/api"
        )
        self._session = None
    
    # ---- HTTP helpers ----
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
    
    async def _signed_request(self, method: str, endpoint: str, params: Dict = None) -> Any:
        """Make a signed request to Binance API."""
        import hmac
        import hashlib
        import time
        from urllib.parse import urlencode
        
        await self._ensure_session()
        params = params or {}
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}/{endpoint}"
        
        if method == 'GET':
            async with self._session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise Exception(f"Binance API error {resp.status}: {data}")
                return data
        elif method == 'POST':
            async with self._session.post(url, params=params, headers=headers) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise Exception(f"Binance API error {resp.status}: {data}")
                return data
        elif method == 'DELETE':
            async with self._session.delete(url, params=params, headers=headers) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise Exception(f"Binance API error {resp.status}: {data}")
                return data
    
    async def _public_request(self, endpoint: str, params: Dict = None) -> Any:
        """Make a public (unsigned) request."""
        await self._ensure_session()
        url = f"{self.base_url}/{endpoint}"
        async with self._session.get(url, params=params or {}) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise Exception(f"Binance API error {resp.status}: {data}")
            return data
    
    # ---- BrokerConnector implementation ----
    
    async def connect(self) -> bool:
        """Connect to Binance and verify credentials."""
        self.logger.info(f"Connecting to Binance (testnet={self.testnet})...")
        try:
            await self._ensure_session()
            # Test public connectivity
            await self._public_request('v3/ping')
            
            # If API keys are set, verify account access
            if self.api_key and self.secret_key:
                await self._signed_request('GET', 'v3/account')
                self.logger.info("Binance account verified")
            
            self.connected = True
            self.logger.info("Connected to Binance")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Binance and close session."""
        self.logger.info("Disconnecting from Binance...")
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self.connected = False
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """Place order on Binance via REST API."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        self.logger.info(
            f"Placing order: {order.side} {order.quantity} {order.symbol} "
            f"@ {order.price or 'MARKET'}"
        )
        
        params: Dict[str, Any] = {
            'symbol': order.symbol,
            'side': order.side,
            'type': order.order_type,
            'quantity': str(order.quantity),
            'newClientOrderId': order.order_id,
        }
        
        # Add price for LIMIT orders
        if order.order_type in ('LIMIT', 'STOP_LIMIT'):
            if order.price is None:
                raise ValueError(f"Price required for {order.order_type} orders")
            params['price'] = str(order.price)
            params['timeInForce'] = order.time_in_force
        
        # Add stop price
        if order.stop_price is not None:
            params['stopPrice'] = str(order.stop_price)
        
        try:
            result = await self._signed_request('POST', 'v3/order', params)
            
            order.broker_order_id = str(result.get('orderId', ''))
            order.status = result.get('status', 'NEW')
            order.filled_quantity = float(result.get('executedQty', 0))
            
            # Calculate average fill price from fills array
            fills = result.get('fills', [])
            if fills:
                total_qty = sum(float(f['qty']) for f in fills)
                total_cost = sum(float(f['qty']) * float(f['price']) for f in fills)
                order.average_price = total_cost / total_qty if total_qty > 0 else None
            elif result.get('price') and float(result['price']) > 0:
                order.average_price = float(result['price'])
            
            order.updated_at = datetime.utcnow()
            
            self.logger.info(
                f"Order placed: ID={order.broker_order_id} status={order.status}"
            )
            return order
            
        except Exception as e:
            order.status = 'REJECTED'
            order.error_message = str(e)
            self.logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('DELETE', 'v3/order', {
                'symbol': symbol,
                'origClientOrderId': order_id
            })
            self.logger.info(f"Order cancelled: {order_id} -> {result.get('status')}")
            return result.get('status') == 'CANCELED'
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> BrokerOrder:
        """Get order status from Binance."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        result = await self._signed_request('GET', 'v3/order', {
            'symbol': symbol,
            'origClientOrderId': order_id
        })
        
        return BrokerOrder(
            order_id=order_id,
            broker_order_id=str(result.get('orderId', '')),
            symbol=result['symbol'],
            side=result['side'],
            order_type=result['type'],
            quantity=float(result['origQty']),
            price=float(result['price']) if float(result.get('price', 0)) > 0 else None,
            status=result['status'],
            filled_quantity=float(result.get('executedQty', 0)),
            average_price=float(result['price']) if float(result.get('price', 0)) > 0 else None,
            broker=self.broker.value,
        )
    
    async def get_balance(self) -> List[AccountBalance]:
        """Get account balances from Binance."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        result = await self._signed_request('GET', 'v3/account')
        balances = []
        
        for b in result.get('balances', []):
            free = float(b.get('free', 0))
            locked = float(b.get('locked', 0))
            total = free + locked
            if total > 0:
                balances.append(AccountBalance(
                    asset=b['asset'],
                    free=free,
                    locked=locked,
                    total=total
                ))
        
        return balances
    
    async def get_positions(self) -> List[Position]:
        """Get open positions from Binance (spot balances with value)."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        balances = await self.get_balance()
        positions = []
        
        for b in balances:
            if b.asset in ('USDT', 'BUSD', 'USDC', 'USD'):
                continue
            if b.total <= 0:
                continue
            
            symbol = f"{b.asset}USDT"
            try:
                current_price = await self.get_symbol_price(symbol)
            except Exception:
                continue
            
            positions.append(Position(
                symbol=symbol,
                side="LONG",
                quantity=b.total,
                entry_price=0.0,  # Binance spot doesn't track entry
                current_price=current_price,
                unrealized_pnl=0.0,
                leverage=1.0,
                margin=b.total * current_price
            ))
        
        return positions
    
    async def get_symbol_price(self, symbol: str) -> float:
        """Get current symbol price from Binance."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        result = await self._public_request('v3/ticker/price', {'symbol': symbol})
        return float(result['price'])


# ============================================================================
# BYBIT CONNECTOR
# ============================================================================

class BybitConnector(BrokerConnector):
    """
    Bybit broker connector.
    Uses Bybit V5 API for unified trading.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        super().__init__(api_key, secret_key, testnet)
        self.broker = Broker.BYBIT
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet
            else "https://api.bybit.com"
        )
        self._session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
    
    async def _signed_request(self, method: str, endpoint: str, params: Dict = None) -> Any:
        """Make a signed request to Bybit V5 API."""
        import hmac
        import hashlib
        import time
        import json as json_mod
        
        await self._ensure_session()
        timestamp = str(int(time.time() * 1000))
        recv_window = '5000'
        params = params or {}
        
        if method == 'GET':
            from urllib.parse import urlencode
            query_string = urlencode(params)
            sign_payload = f"{timestamp}{self.api_key}{recv_window}{query_string}"
        else:
            body = json_mod.dumps(params)
            sign_payload = f"{timestamp}{self.api_key}{recv_window}{body}"
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            sign_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        if method == 'GET':
            async with self._session.get(url, params=params, headers=headers) as resp:
                data = await resp.json()
                if data.get('retCode', -1) != 0:
                    raise Exception(f"Bybit API error: {data}")
                return data.get('result', {})
        else:
            async with self._session.post(url, json=params, headers=headers) as resp:
                data = await resp.json()
                if data.get('retCode', -1) != 0:
                    raise Exception(f"Bybit API error: {data}")
                return data.get('result', {})
    
    async def _public_request(self, endpoint: str, params: Dict = None) -> Any:
        """Make a public request to Bybit."""
        await self._ensure_session()
        url = f"{self.base_url}/{endpoint}"
        async with self._session.get(url, params=params or {}) as resp:
            data = await resp.json()
            if data.get('retCode', -1) != 0:
                raise Exception(f"Bybit API error: {data}")
            return data.get('result', {})
    
    async def connect(self) -> bool:
        """Connect to Bybit."""
        self.logger.info(f"Connecting to Bybit (testnet={self.testnet})...")
        try:
            await self._ensure_session()
            # Test server time
            url = f"{self.base_url}/v5/market/time"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    self.connected = True
                    self.logger.info("Connected to Bybit")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Bybit."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self.connected = False
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """Place order on Bybit V5."""
        if not self.connected:
            raise ConnectionError("Not connected to Bybit")
        
        params = {
            'category': 'spot',
            'symbol': order.symbol,
            'side': order.side.capitalize(),
            'orderType': 'Market' if order.order_type == 'MARKET' else 'Limit',
            'qty': str(order.quantity),
            'orderLinkId': order.order_id,
        }
        if order.price and order.order_type != 'MARKET':
            params['price'] = str(order.price)
            params['timeInForce'] = order.time_in_force or 'GTC'
        
        try:
            result = await self._signed_request('POST', 'v5/order/create', params)
            order.broker_order_id = result.get('orderId', '')
            order.status = 'NEW'
            order.updated_at = datetime.utcnow()
            return order
        except Exception as e:
            order.status = 'REJECTED'
            order.error_message = str(e)
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Bybit."""
        try:
            await self._signed_request('POST', 'v5/order/cancel', {
                'category': 'spot',
                'symbol': symbol,
                'orderLinkId': order_id
            })
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> BrokerOrder:
        """Get order status from Bybit."""
        result = await self._signed_request('GET', 'v5/order/realtime', {
            'category': 'spot',
            'symbol': symbol,
            'orderLinkId': order_id
        })
        orders = result.get('list', [])
        if not orders:
            raise Exception(f"Order {order_id} not found")
        o = orders[0]
        return BrokerOrder(
            order_id=order_id,
            broker_order_id=o.get('orderId', ''),
            symbol=o['symbol'],
            side=o['side'].upper(),
            order_type=o['orderType'].upper(),
            quantity=float(o['qty']),
            price=float(o.get('price', 0)) or None,
            status=o['orderStatus'],
            filled_quantity=float(o.get('cumExecQty', 0)),
            average_price=float(o.get('avgPrice', 0)) or None,
            broker=self.broker.value,
        )
    
    async def get_balance(self) -> List[AccountBalance]:
        """Get account balance from Bybit."""
        result = await self._signed_request('GET', 'v5/account/wallet-balance', {
            'accountType': 'UNIFIED'
        })
        balances = []
        for account in result.get('list', []):
            for coin in account.get('coin', []):
                free = float(coin.get('availableToWithdraw', 0))
                locked = float(coin.get('locked', 0))
                total = float(coin.get('walletBalance', 0))
                if total > 0:
                    balances.append(AccountBalance(
                        asset=coin['coin'],
                        free=free,
                        locked=locked,
                        total=total
                    ))
        return balances
    
    async def get_positions(self) -> List[Position]:
        """Get open positions from Bybit."""
        result = await self._signed_request('GET', 'v5/position/list', {
            'category': 'linear',
            'settleCoin': 'USDT'
        })
        positions = []
        for p in result.get('list', []):
            qty = float(p.get('size', 0))
            if qty > 0:
                positions.append(Position(
                    symbol=p['symbol'],
                    side=p.get('side', 'Buy').upper(),
                    quantity=qty,
                    entry_price=float(p.get('avgPrice', 0)),
                    current_price=float(p.get('markPrice', 0)),
                    unrealized_pnl=float(p.get('unrealisedPnl', 0)),
                    leverage=float(p.get('leverage', 1)),
                    margin=float(p.get('positionIM', 0))
                ))
        return positions
    
    async def get_symbol_price(self, symbol: str) -> float:
        """Get current symbol price from Bybit."""
        result = await self._public_request('v5/market/tickers', {
            'category': 'spot',
            'symbol': symbol
        })
        tickers = result.get('list', [])
        if tickers:
            return float(tickers[0].get('lastPrice', 0))
        raise Exception(f"No ticker data for {symbol}")


# ============================================================================
# PAPER TRADING CONNECTOR
# ============================================================================

class PaperTradingConnector(BrokerConnector):
    """
    Paper trading connector for backtesting and simulation.
    """
    
    def __init__(self, initial_balance: float = 1000000.0):
        super().__init__("", "", True)
        self.broker = Broker.PAPER
        self.balance = {"USDT": initial_balance}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, BrokerOrder] = {}
    
    async def connect(self) -> bool:
        self.connected = True
        self.logger.info(f"Paper trading initialized with balance: {self.balance}")
        return True
    
    async def disconnect(self) -> None:
        self.connected = False
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        # Simulate order execution at market price
        price = order.price or await self.get_symbol_price(order.symbol)
        
        # Update balance
        if order.side == "BUY":
            cost = order.quantity * price
            self.balance["USDT"] -= cost
        else:
            revenue = order.quantity * price
            self.balance["USDT"] += revenue
        
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.average_price = price
        
        # Update position
        if order.side == "BUY":
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity += order.quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side="LONG",
                    quantity=order.quantity,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                )
        
        self.orders[order.order_id] = order
        return order
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = "CANCELLED"
            return True
        return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> BrokerOrder:
        return self.orders.get(order_id, BrokerOrder(order_id=order_id, symbol=symbol,
                          side="BUY", order_type="MARKET", quantity=1.0, status="NEW"))
    
    async def get_balance(self) -> List[AccountBalance]:
        return [AccountBalance(asset=k, free=v, locked=0.0, total=v) 
                for k, v in self.balance.items()]
    
    async def get_positions(self) -> List[Position]:
        return list(self.positions.values())
    
    async def get_symbol_price(self, symbol: str) -> float:
        prices = {"BTCUSDT": 43500.0, "ETHUSDT": 2350.0, "SOLUSDT": 95.0}
        return prices.get(symbol.upper(), 100.0)


# ============================================================================
# BROKER FACTORY
# ============================================================================

class BrokerFactory:
    """Factory for creating broker connectors."""
    
    @staticmethod
    def create_broker(
        broker: str,
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True,
    ) -> BrokerConnector:
        """Create a broker connector instance."""
        broker = broker.lower()
        
        if broker == "binance":
            return BinanceConnector(api_key, secret_key, testnet)
        elif broker == "bybit":
            return BybitConnector(api_key, secret_key, testnet)
        elif broker == "paper":
            return PaperTradingConnector()
        else:
            raise ValueError(f"Unsupported broker: {broker}")
