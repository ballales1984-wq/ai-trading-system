"""
Coinbase Connector
==================
Live trading connector for Coinbase exchange.
Uses Coinbase Pro API for trading.

NOTE: This is a skeleton implementation. Full functionality requires:
- Coinbase Pro API keys with trading permissions
- Proper error handling for API rate limits
- WebSocket integration for real-time updates

Requirements:
    pip install requests

Author: AI Trading System
Data: 2026-03-21
"""

import asyncio
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlencode
import uuid

import requests

from app.execution.broker_connector import (
    BrokerConnector,
    BrokerOrder,
    Position,
    AccountBalance,
    OrderStatus
)


logger = logging.getLogger(__name__)


class CoinbaseConnector(BrokerConnector):
    """
    Coinbase exchange connector.
    Connects to Coinbase Pro API for live trading.
    
    Features:
    - Spot trading (BUY/SELL)
    - Limit and market orders
    - Account balance queries
    - Position tracking
    
    TODO:
    - Add WebSocket support for real-time order updates
    - Implement OCO (One Cancels Other) orders
    - Add margin trading support
    """
    
    def __init__(self, config: Dict):
        """Initialize Coinbase connector."""
        super().__init__(config)
        
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.passphrase = config.get("passphrase", "")
        self.sandbox = config.get("sandbox", True)
        
        # API URLs
        self.base_url = (
            "https://api-public.sandbox.pro.coinbase.com"
            if self.sandbox
            else "https://api.pro.coinbase.com"
        )
        
        self._session: Optional[requests.Session] = None
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, BrokerOrder] = {}
        
        logger.info(f"CoinbaseConnector initialized (sandbox: {self.sandbox})")
    
    # ---- HTTP helpers ----
    
    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authenticated headers for Coinbase API."""
        timestamp = str(int(time.time()))
        message = timestamp + method + path + body
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        authenticated: bool = True
    ) -> Any:
        """Make HTTP request to Coinbase API."""
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(params) if params else ""
        
        headers = {}
        if authenticated:
            headers = self._get_headers(method, endpoint, body)
        else:
            headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, data=body, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Coinbase API: Resource not found")
                return None
            else:
                logger.error(f"Coinbase API error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Coinbase request failed: {e}")
            return None
    
    # ---- BrokerConnector implementation ----
    
    async def connect(self) -> bool:
        """Connect to Coinbase and verify credentials."""
        try:
            logger.info(f"Connecting to Coinbase (sandbox: {self.sandbox})")
            
            # Test public API
            products = self._request('GET', '/products', authenticated=False)
            if products is None:
                logger.error("Failed to connect to Coinbase public API")
                return False
            
            # If API keys provided, verify account access
            if self.api_key and self.api_secret and self.passphrase:
                accounts = self._request('GET', '/accounts')
                if accounts is None:
                    logger.error("Failed to verify Coinbase account")
                    return False
                logger.info("Coinbase account verified")
            
            self._connected = True
            logger.info("Connected to Coinbase")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Coinbase."""
        self._connected = False
        logger.info("Disconnected from Coinbase")
    
    async def get_balance(self) -> AccountBalance:
        """Get account balance from Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        accounts = self._request('GET', '/accounts')
        
        if not accounts:
            return AccountBalance(
                asset="USD",
                total=0.0,
                free=0.0,
                locked=0.0
            )
        
        # Sum all USD balances
        total_equity = 0.0
        available = 0.0
        
        for account in accounts:
            if account.get('currency') in ('USD', 'USDC', 'USDT'):
                total_equity += float(account.get('balance', 0))
                available += float(account.get('available', 0))
        
        return AccountBalance(
            total_equity=total_equity,
            available_balance=available,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all positions from Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        positions = []
        accounts = self._request('GET', '/accounts')
        
        if not accounts:
            return positions
        
        for account in accounts:
            balance = float(account.get('balance', 0))
            if balance > 0 and account.get('currency') not in ('USD', 'USDC', 'USDT'):
                symbol = f"{account['currency']}-USD"
                
                # Get current price
                try:
                    price_data = self._request(
                        'GET',
                        f"/products/{symbol}/ticker",
                        authenticated=False
                    )
                    current_price = float(price_data.get('price', 0)) if price_data else 0
                except Exception:
                    current_price = 0
                
                pos = Position(
                    symbol=symbol,
                    quantity=balance,
                    entry_price=0.0,
                    current_price=current_price,
                    unrealized_pnl=0.0,
                    leverage=1.0,
                    opened_at=None
                )
                positions.append(pos)
                self._positions[symbol] = pos
        
        return positions
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """Place order on Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        if not order.order_id:
            order.order_id = f"CB_{uuid.uuid4().hex[:12]}"
        
        # Build order payload
        params: Dict[str, Any] = {
            'product_id': order.symbol,
            'side': order.side.lower(),
            'type': order.order_type.lower(),
            'size': str(order.quantity),
        }
        
        # Add price for LIMIT orders
        if order.order_type.upper() in ('LIMIT', 'STOP_LIMIT'):
            if order.price is None:
                raise ValueError(f"Price required for {order.order_type} orders")
            params['price'] = str(order.price)
            params['time_in_force'] = order.time_in_force or 'GTC'
        
        # Add stop price for STOP orders
        if order.stop_price is not None:
            params['stop'] = 'price'
            params['stop_price'] = str(order.stop_price)
        
        try:
            logger.info(
                f"Placing Coinbase order: {order.side} {order.quantity} {order.symbol} "
                f"@ {order.price or 'MARKET'}"
            )
            
            result = self._request('POST', '/orders', params)
            
            if result:
                order.broker_order_id = result.get('id', '')
                
                status_map = {
                    'pending': OrderStatus.NEW,
                    'open': OrderStatus.NEW,
                    'filled': OrderStatus.FILLED,
                    'done': OrderStatus.FILLED,
                    'cancelled': OrderStatus.CANCELLED,
                    'rejected': OrderStatus.REJECTED,
                }
                
                order.status = status_map.get(
                    result.get('status', ''), OrderStatus.NEW
                )
                order.filled_quantity = float(result.get('filled_size', 0))
                
                if result.get('price'):
                    order.avg_fill_price = float(result.get('price'))
                
                order.filled_at = datetime.now()
                self._orders[order.order_id] = order
                
                logger.info(
                    f"Coinbase order placed: ID={order.broker_order_id} "
                    f"status={order.status.value}"
                )
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = "Failed to place order"
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Failed to place Coinbase order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        try:
            # Find the broker order ID
            broker_order_id = None
            if order_id in self._orders:
                broker_order_id = self._orders[order_id].broker_order_id
            
            if not broker_order_id:
                logger.warning(f"Order {order_id} not found")
                return False
            
            result = self._request('DELETE', f'/orders/{broker_order_id}')
            
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
        """Get order from Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        # Check local cache first
        if order_id in self._orders:
            return self._orders[order_id]
        
        # Try to fetch from API
        try:
            result = self._request('GET', f'/orders/{order_id}')
            
            if result:
                order = BrokerOrder(
                    order_id=order_id,
                    broker_order_id=result.get('id', ''),
                    symbol=result.get('product_id', ''),
                    side=result.get('side', ''),
                    order_type=result.get('type', ''),
                    quantity=float(result.get('size', 0)),
                    status=result.get('status', 'NEW'),
                    filled_quantity=float(result.get('filled_size', 0))
                )
                
                self._orders[order_id] = order
                return order
                
        except Exception as e:
            logger.error(f"Failed to get order: {e}")
        
        return None
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price from Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        try:
            result = self._request('GET', f"/products/{symbol}/ticker", authenticated=False)
            if result:
                return float(result.get('price', 0))
        except Exception as e:
            logger.error(f"Failed to get market price for {symbol}: {e}")
        
        return 0.0
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book from Coinbase."""
        if not self._connected:
            raise ConnectionError("Not connected to Coinbase")
        
        return self._request(
            'GET',
            f"/products/{symbol}/book",
            params={'level': min(depth, 3)},
            authenticated=False
        ) or {}
    
    async def get_open_orders(self) -> List[BrokerOrder]:
        """Get all open orders."""
        if not self._connected:
            return []
        
        result = self._request('GET', '/orders', params={'status': 'open'})
        
        if not result:
            return []
        
        orders = []
        for order_data in result:
            order = BrokerOrder(
                order_id=order_data.get('client_order_id', ''),
                broker_order_id=order_data.get('id', ''),
                symbol=order_data.get('product_id', ''),
                side=order_data.get('side', ''),
                order_type=order_data.get('type', ''),
                quantity=float(order_data.get('size', 0)),
                status=OrderStatus.NEW,
                filled_quantity=float(order_data.get('filled_size', 0))
            )
            orders.append(order)
        
        return orders

    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        return {}

    async def get_symbol_price(self, symbol: str) -> Optional[float]:
        """Get current symbol price"""
        return await self.get_market_price(symbol)


# Import missing json
import json
