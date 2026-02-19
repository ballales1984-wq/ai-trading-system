"""
Binance Connector
==============
Live trading connector for Binance exchange.
Uses aiohttp for async HTTP requests with HMAC-SHA256 signing.
"""

import asyncio
import hmac
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlencode
import uuid

import aiohttp

from app.execution.broker_connector import (
    BrokerConnector, BrokerOrder, Position, AccountBalance,
    OrderStatus
)


logger = logging.getLogger(__name__)


class BinanceConnector(BrokerConnector):
    """
    Binance exchange connector.
    Connects to Binance Spot API for live trading.
    """
    
    def __init__(self, config: Dict):
        """Initialize Binance connector."""
        super().__init__(config)
        
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.testnet = config.get("testnet", True)
        
        self.base_url = (
            "https://testnet.binance.vision/api"
            if self.testnet
            else "https://api.binance.com/api"
        )
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, BrokerOrder] = {}
    
    # ---- HTTP helpers ----
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    def _sign(self, params: Dict) -> Dict:
        """Add timestamp and HMAC-SHA256 signature to params."""
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params
    
    async def _signed_request(
        self, method: str, endpoint: str, params: Dict = None
    ) -> Any:
        """Make a signed request to Binance API."""
        await self._ensure_session()
        params = self._sign(params or {})
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}/{endpoint}"
        
        async with getattr(self._session, method.lower())(
            url, params=params, headers=headers
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise Exception(f"Binance API error {resp.status}: {data}")
            return data
    
    async def _public_request(
        self, endpoint: str, params: Dict = None
    ) -> Any:
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
        try:
            logger.info(f"Connecting to Binance (testnet: {self.testnet})")
            await self._ensure_session()
            
            # Test public connectivity
            await self._public_request('v3/ping')
            
            # If API keys are set, verify account access
            if self.api_key and self.api_secret:
                await self._signed_request('GET', 'v3/account')
                logger.info("Binance account verified")
            
            self._connected = True
            logger.info("Connected to Binance")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Binance."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False
        logger.info("Disconnected from Binance")
    
    async def get_balance(self) -> AccountBalance:
        """Get account balance from Binance."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        result = await self._signed_request('GET', 'v3/account')
        
        # Sum stablecoin balances
        total_equity = 0.0
        available = 0.0
        
        for b in result.get('balances', []):
            free = float(b.get('free', 0))
            locked = float(b.get('locked', 0))
            if b['asset'] in ('USDT', 'BUSD', 'USDC'):
                total_equity += free + locked
                available += free
        
        return AccountBalance(
            total_equity=total_equity,
            available_balance=available,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all positions from Binance (spot balances with value)."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        result = await self._signed_request('GET', 'v3/account')
        positions = []
        
        for b in result.get('balances', []):
            free = float(b.get('free', 0))
            locked = float(b.get('locked', 0))
            total = free + locked
            
            if total > 0 and b['asset'] not in ('USDT', 'BUSD', 'USDC', 'USD'):
                symbol = f"{b['asset']}USDT"
                try:
                    current_price = await self.get_market_price(symbol)
                except Exception:
                    continue
                
                pos = Position(
                    symbol=symbol,
                    quantity=total,
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
        """Place order on Binance."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        if not order.order_id:
            order.order_id = f"BIN_{uuid.uuid4().hex[:12]}"
        
        params: Dict[str, Any] = {
            'symbol': order.symbol,
            'side': order.side.upper(),
            'type': order.order_type.upper(),
            'quantity': str(order.quantity),
            'newClientOrderId': order.order_id,
        }
        
        # Add price for LIMIT orders
        if order.order_type.upper() in ('LIMIT', 'STOP_LIMIT'):
            if order.price is None:
                raise ValueError(f"Price required for {order.order_type} orders")
            params['price'] = str(order.price)
            params['timeInForce'] = order.time_in_force or 'GTC'
        
        # Add stop price
        if order.stop_price is not None:
            params['stopPrice'] = str(order.stop_price)
        
        try:
            logger.info(
                f"Placing order: {order.side} {order.quantity} {order.symbol} "
                f"@ {order.price or 'MARKET'}"
            )
            
            result = await self._signed_request('POST', 'v3/order', params)
            
            # Map response
            order.broker_order_id = str(result.get('orderId', ''))
            
            status_map = {
                'NEW': OrderStatus.NEW,
                'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED,
            }
            order.status = status_map.get(
                result.get('status', ''), OrderStatus.NEW
            )
            order.filled_quantity = float(result.get('executedQty', 0))
            
            # Calculate average fill price from fills
            fills = result.get('fills', [])
            if fills:
                total_qty = sum(float(f['qty']) for f in fills)
                total_cost = sum(float(f['qty']) * float(f['price']) for f in fills)
                order.avg_fill_price = total_cost / total_qty if total_qty > 0 else 0
                order.commission = sum(float(f.get('commission', 0)) for f in fills)
            elif result.get('price') and float(result['price']) > 0:
                order.avg_fill_price = float(result['price'])
            
            order.filled_at = datetime.now()
            self._orders[order.order_id] = order
            
            logger.info(
                f"Order placed: ID={order.broker_order_id} "
                f"status={order.status.value}"
            )
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Binance."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('DELETE', 'v3/order', {
                'origClientOrderId': order_id
            })
            
            if result.get('status') == 'CANCELED':
                if order_id in self._orders:
                    self._orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order from Binance."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('GET', 'v3/order', {
                'origClientOrderId': order_id
            })
            
            status_map = {
                'NEW': OrderStatus.NEW,
                'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED,
            }
            
            order = BrokerOrder(
                order_id=order_id,
                broker_order_id=str(result.get('orderId', '')),
                symbol=result['symbol'],
                side=result['side'],
                order_type=result['type'],
                quantity=float(result['origQty']),
                price=float(result['price']) if float(result.get('price', 0)) > 0 else None,
                status=status_map.get(result['status'], OrderStatus.NEW),
                filled_quantity=float(result.get('executedQty', 0)),
                avg_fill_price=float(result.get('price', 0)),
            )
            
            self._orders[order_id] = order
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order: {e}")
            return self._orders.get(order_id)
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price from Binance."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        result = await self._public_request('v3/ticker/price', {'symbol': symbol})
        return float(result['price'])
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book from Binance."""
        if not self._connected:
            raise ConnectionError("Not connected to Binance")
        
        return await self._public_request('v3/depth', {
            'symbol': symbol,
            'limit': depth
        })
    
    async def get_open_orders(self) -> List[BrokerOrder]:
        """Get all open orders."""
        if not self._connected:
            return []
        
        return [
            o for o in self._orders.values()
            if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        ]
