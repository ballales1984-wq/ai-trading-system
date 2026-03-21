'''
Bybit Connector
===============
Live trading connector for Bybit exchange.
Uses Bybit API v5 for trading.

NOTE: This is a skeleton implementation. Full functionality requires:
- Bybit API keys with trading permissions
- Proper error handling for API rate limits
- WebSocket integration for real-time updates

Requirements:
    pip install requests aiohttp

Author: AI Trading System
Data: 2026-03-21
'''

import asyncio
import hmac
import hashlib
import time
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlencode
import uuid

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    import requests

from app.execution.broker_connector import (
    BrokerConnector,
    BrokerOrder,
    Position,
    AccountBalance,
    OrderStatus
)


logger = logging.getLogger(__name__)


class BybitConnector(BrokerConnector):
    """
    Bybit exchange connector.
    Connects to Bybit API for live trading.
    
    Features:
    - Spot, Linear, Inverse trading
    - Limit, Market, Conditional orders
    - Account balance queries
    - Position tracking
    
    TODO:
    - Add WebSocket support for real-time order updates
    - Implement derivative trading (USDT perpetual, inverse)
    - Add unified margin account support
    """
    
    def __init__(self, config: Dict):
        """Initialize Bybit connector."""
        super().__init__(config)
        
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.testnet = config.get("testnet", True)
        
        # API URLs
        self.base_url = (
            "https://api-testnet.bybit.com"
            if self.testnet
            else "https://api.bybit.com"
        )
        
        self._session = None
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, BrokerOrder] = {}
        
        logger.info(f"BybitConnector initialized (testnet: {self.testnet})")
    
    def _sign(self, params: Dict, timestamp: int) -> str:
        """Generate HMAC-SHA256 signature for Bybit API."""
        param_str = urlencode(sorted(params.items()))
        message = f"{timestamp}{self.api_key}{param_str}"
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_headers(self, timestamp: int, params: Dict = None) -> Dict[str, str]:
        """Get headers for authenticated request."""
        params = params or {}
        signature = self._sign(params, timestamp)
        
        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': str(timestamp),
            'Content-Type': 'application/json'
        }
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        authenticated: bool = True
    ) -> Any:
        """Make HTTP request to Bybit API."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        timestamp = int(time.time() * 1000)
        
        headers = {}
        if authenticated:
            headers = self._get_headers(timestamp, params)
        else:
            headers = {'Content-Type': 'application/json'}
        
        try:
            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    if method == 'GET':
                        async with session.get(url, params=params, headers=headers) as resp:
                            data = await resp.json()
                    elif method == 'POST':
                        async with session.post(url, json=params, headers=headers) as resp:
                            data = await resp.json()
            else:
                if method == 'GET':
                    resp = requests.get(url, params=params, headers=headers, timeout=30)
                elif method == 'POST':
                    resp = requests.post(url, json=params, headers=headers, timeout=30)
                data = resp.json()
            
            if data.get('retCode') == 0:
                return data.get('result')
            else:
                logger.error(f"Bybit API error: {data.get('retMsg')}")
                return None
                
        except Exception as e:
            logger.error(f"Bybit request failed: {e}")
            return None
    
    async def connect(self) -> bool:
        """Connect to Bybit and verify credentials."""
        try:
            logger.info(f"Connecting to Bybit (testnet: {self.testnet})")
            
            result = await self._request('GET', '/v5/market/time', authenticated=False)
            if result is None:
                logger.error("Failed to connect to Bybit public API")
                return False
            
            if self.api_key and self.api_secret:
                wallet = await self._request('GET', '/v5/account/wallet-balance', 
                                           params={'accountType': 'UNIFIED'})
                if wallet is None:
                    logger.error("Failed to verify Bybit account")
                    return False
                logger.info("Bybit account verified")
            
            self._connected = True
            logger.info("Connected to Bybit")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Bybit: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Bybit."""
        self._connected = False
        logger.info("Disconnected from Bybit")
    
    async def get_balance(self) -> AccountBalance:
        """Get account balance from Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        result = await self._request(
            'GET',
            '/v5/account/wallet-balance',
            params={'accountType': 'UNIFIED'}
        )
        
        if not result or not result.get('list'):
            return AccountBalance(
                total_equity=0.0,
                available_balance=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        
        account_info = result['list'][0]
        total_equity = float(account_info.get('totalEquity', 0))
        available = float(account_info.get('availableToWithdraw', 0))
        
        return AccountBalance(
            total_equity=total_equity,
            available_balance=available,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all positions from Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        return []
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """Place order on Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        if not order.order_id:
            order.order_id = f"BB_{uuid.uuid4().hex[:12]}"
        
        category = 'linear'
        
        params: Dict[str, Any] = {
            'category': category,
            'symbol': order.symbol,
            'side': 'Buy' if order.side.upper() == 'BUY' else 'Sell',
            'orderType': order.order_type.upper(),
            'qty': str(order.quantity),
            'clOrdId': order.order_id,
        }
        
        if order.order_type.upper() in ('LIMIT', 'STOP_LIMIT'):
            if order.price is None:
                raise ValueError(f"Price required for {order.order_type} orders")
            params['price'] = str(order.price)
            params['timeInForce'] = order.time_in_force or 'GTC'
        
        if order.stop_price is not None:
            params['stopPrice'] = str(order.stop_price)
        
        try:
            logger.info(
                f"Placing Bybit order: {order.side} {order.quantity} {order.symbol} "
                f"@ {order.price or 'MARKET'}"
            )
            
            result = await self._request('POST', '/v5/order/create', params)
            
            if result:
                order_id = result.get('orderId', '')
                order.broker_order_id = order_id
                
                status_map = {
                    'Created': OrderStatus.NEW,
                    'New': OrderStatus.NEW,
                    'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
                    'Filled': OrderStatus.FILLED,
                    'Cancelled': OrderStatus.CANCELLED,
                    'Rejected': OrderStatus.REJECTED,
                }
                
                order.status = status_map.get(
                    result.get('orderStatus', ''), OrderStatus.NEW
                )
                order.filled_quantity = float(result.get('avgPrice', 0)) * order.quantity
                
                if result.get('avgPrice'):
                    order.avg_fill_price = float(result.get('avgPrice'))
                
                order.filled_at = datetime.now()
                self._orders[order.order_id] = order
                
                logger.info(
                    f"Bybit order placed: ID={order.broker_order_id} "
                    f"status={order.status.value}"
                )
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = "Failed to place order"
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Failed to place Bybit order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        try:
            broker_order_id = None
            if order_id in self._orders:
                broker_order_id = self._orders[order_id].broker_order_id
            
            if not broker_order_id:
                logger.warning(f"Order {order_id} not found")
                return False
            
            params = {
                'category': 'linear',
                'orderId': broker_order_id
            }
            
            result = await self._request('POST', '/v5/order/cancel', params)
            
            if result:
                if order_id in self._orders:
                    self._orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order from Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        if order_id in self._orders:
            return self._orders[order_id]
        
        return None
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price from Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        try:
            result = await self._request(
                'GET',
                '/v5/market/ticker',
                params={'symbol': symbol},
                authenticated=False
            )
            if result and result.get('list'):
                return float(result['list'][0].get('lastPrice', 0))
        except Exception as e:
            logger.error(f"Failed to get market price for {symbol}: {e}")
        
        return 0.0
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book from Bybit."""
        if not self._connected:
            raise ConnectionError("Not connected to Bybit")
        
        return await self._request(
            'GET',
            '/v5/market/orderbook',
            params={'symbol': symbol, 'category': 'linear'},
            authenticated=False
        ) or {}
    
    async def get_open_orders(self) -> List[BrokerOrder]:
        """Get all open orders."""
        if not self._connected:
            return []
        
        result = await self._request(
            'GET',
            '/v5/order/realtime',
            params={'category': 'linear', 'openOnly': 0}
        )
        
        if not result or not result.get('list'):
            return []
        
        orders = []
        for order_data in result['list']:
            order = BrokerOrder(
                order_id=order_data.get('orderId', ''),
                broker_order_id=order_data.get('orderId', ''),
                symbol=order_data.get('symbol', ''),
                side=order_data.get('side', ''),
                order_type=order_data.get('orderType', ''),
                quantity=float(order_data.get('qty', 0)),
                status=OrderStatus.NEW,
                filled_quantity=float(order_data.get('cumExecQty', 0))
            )
            orders.append(order)
        
        return orders
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        return {}
    
    async def get_symbol_price(self, symbol: str) -> Optional[float]:
        """Get current symbol price"""
        return await self.get_market_price(symbol)
