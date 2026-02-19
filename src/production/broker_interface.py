# src/production/broker_interface.py
"""
Production Broker Interface
========================
Abstract interface for broker connections with support for
multiple exchanges. Built for real money trading.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"   # Immediate or Cancel
    FOK = "FOK"   # Fill or Kill


@dataclass
class Order:
    """
    Represents a trading order.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    client_order_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'order_id': self.order_id,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'client_order_id': self.client_order_id
        }


@dataclass
class Position:
    """
    Represents a trading position.
    """
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    leverage: float = 1.0
    margin: float = 0.0
    liquidation_price: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'leverage': self.leverage,
            'margin': self.margin,
            'liquidation_price': self.liquidation_price
        }
    
    @property
    def notional_value(self) -> float:
        """Get notional value of position."""
        return abs(self.quantity * self.current_price)
    
    @property
    def pnl_percent(self) -> float:
        """Get PnL as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * abs(self.quantity))) * 100


@dataclass
class AccountBalance:
    """
    Represents account balance information.
    """
    total_equity: float = 0.0
    available_balance: float = 0.0
    total_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert balance to dictionary."""
        return {
            'total_equity': self.total_equity,
            'available_balance': self.available_balance,
            'total_margin': self.total_margin,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


@dataclass
class MarketTicker:
    """
    Represents market ticker data.
    """
    symbol: str
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_price: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def mid_price(self) -> float:
        """Get mid price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_percent(self) -> float:
        """Get spread as percentage."""
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 100


class BrokerInterface(ABC):
    """
    Abstract broker interface for trading.
    All broker implementations must inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker interface.
        
        Args:
            config: Broker configuration dictionary
        """
        self.config = config
        self.is_connected = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Rate limiting
        self._rate_limit_calls = 0
        self._rate_limit_reset = datetime.now()
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from broker.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """
        Place an order.
        
        Args:
            order: Order to place
            
        Returns:
            Updated order with broker order ID
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            Updated order status
        """
        pass
    
    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """
        Get account balance.
        
        Returns:
            Account balance information
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> MarketTicker:
        """
        Get market ticker.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market ticker data
        """
        pass
    
    async def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [o for o in self._orders.values() 
                if o.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]]
    
    async def get_closed_orders(self) -> List[Order]:
        """Get all closed orders."""
        return [o for o in self._orders.values() 
                if o.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                               OrderStatus.REJECTED, OrderStatus.EXPIRED]]
    
    def _generate_client_order_id(self) -> str:
        """Generate unique client order ID."""
        import uuid
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = datetime.now()
        
        # Reset counter every minute
        if (now - self._rate_limit_reset).total_seconds() >= 60:
            self._rate_limit_calls = 0
            self._rate_limit_reset = now
        
        # Check limit (default: 1200 calls per minute for Binance)
        max_calls = self.config.get('rate_limit_per_minute', 1200)
        
        if self._rate_limit_calls >= max_calls:
            import asyncio
            wait_time = 60 - (now - self._rate_limit_reset).total_seconds()
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                asyncio.sleep(wait_time)
        
        self._rate_limit_calls += 1
    
    def _log_order(self, order: Order, action: str):
        """Log order action to file."""
        log_dir = Path("logs/orders")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"orders_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'order': order.to_dict()
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class PaperTradingBroker(BrokerInterface):
    """
    Paper trading broker for testing.
    Simulates real trading without actual money.
    """
    
    def __init__(self, config: Dict[str, Any], initial_balance: float = 100000):
        super().__init__(config)
        self.initial_balance = initial_balance
        self._balance = AccountBalance(
            total_equity=initial_balance,
            available_balance=initial_balance,
            total_margin=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        self._tickers: Dict[str, MarketTicker] = {}
        
    async def connect(self) -> bool:
        """Connect to paper trading."""
        self.is_connected = True
        self.logger.info("Connected to paper trading broker")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from paper trading."""
        self.is_connected = False
        self.logger.info("Disconnected from paper trading broker")
        return True
    
    async def place_order(self, order: Order) -> Order:
        """Place a paper trade order."""
        if not self.is_connected:
            raise ConnectionError("Not connected to broker")
        
        # Generate order ID
        order.order_id = self._generate_client_order_id()
        order.client_order_id = order.client_order_id or order.order_id
        
        # Simulate execution
        ticker = await self.get_ticker(order.symbol)
        fill_price = ticker.ask_price if order.side == OrderSide.BUY else ticker.bid_price
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = fill_price * order.quantity * 0.001  # 0.1% fee
        order.updated_at = datetime.now()
        
        # Update balance
        if order.side == OrderSide.BUY:
            self._balance.available_balance -= fill_price * order.quantity + order.commission
        else:
            self._balance.available_balance += fill_price * order.quantity - order.commission
        
        # Update position
        self._update_position(order)
        
        # Store order
        self._orders[order.order_id] = order
        
        # Log order
        self._log_order(order, "FILLED")
        
        self.logger.info(f"Paper trade executed: {order.side.value} {order.quantity} {order.symbol} @ {fill_price}")
        
        return order
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a paper trade order."""
        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self._log_order(order, "CANCELLED")
            return True
        return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        return self._orders.get(order_id)
    
    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        return self._balance
    
    async def get_positions(self) -> List[Position]:
        """Get open positions."""
        return [p for p in self._positions.values() if p.quantity != 0]
    
    async def get_ticker(self, symbol: str) -> MarketTicker:
        """Get mock ticker."""
        if symbol not in self._tickers:
            # Create mock ticker
            base_price = 50000 if 'BTC' in symbol else 3000
            self._tickers[symbol] = MarketTicker(
                symbol=symbol,
                bid_price=base_price * 0.9995,
                ask_price=base_price * 1.0005,
                last_price=base_price,
                volume=1000,
                quote_volume=base_price * 1000
            )
        return self._tickers[symbol]
    
    def _update_position(self, order: Order):
        """Update position after trade."""
        symbol = order.symbol
        
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        
        position = self._positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Update long position
            old_qty = position.quantity
            new_qty = old_qty + order.filled_quantity
            
            if old_qty > 0:
                position.entry_price = (
                    (position.entry_price * old_qty + order.avg_fill_price * order.filled_quantity) / new_qty
                )
            else:
                position.entry_price = order.avg_fill_price
            
            position.quantity = new_qty
        else:
            # Update short position or reduce long
            position.quantity -= order.filled_quantity
            
            if position.quantity < 0:
                # Flipped position
                position.entry_price = order.avg_fill_price
                position.quantity = abs(position.quantity)
        
        position.commission += order.commission
        self._balance.total_margin = sum(p.margin for p in self._positions.values())


class BinanceBroker(BrokerInterface):
    """
    Binance broker implementation.
    For real trading with Binance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        self.base_url = config.get('testnet_url', 'https://testnet.binance.vision/api')
        
    async def connect(self) -> bool:
        """Connect to Binance."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Test connection
                async with session.get(f"{self.base_url}/ping") as resp:
                    if resp.status == 200:
                        self.is_connected = True
                        self.logger.info("Connected to Binance")
                        return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance."""
        self.is_connected = False
        self.logger.info("Disconnected from Binance")
        return True
    
    async def _signed_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make a signed request to Binance API."""
        import aiohttp
        import hmac
        import hashlib
        import time
        from urllib.parse import urlencode
        
        params = params or {}
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            if method == 'GET':
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        raise Exception(f"Binance API error {resp.status}: {data}")
                    return data
            elif method == 'POST':
                async with session.post(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        raise Exception(f"Binance API error {resp.status}: {data}")
                    return data
            elif method == 'DELETE':
                async with session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        raise Exception(f"Binance API error {resp.status}: {data}")
                    return data
    
    async def _public_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a public (unsigned) request to Binance API."""
        import aiohttp
        
        url = f"{self.base_url}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params or {}) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise Exception(f"Binance API error {resp.status}: {data}")
                return data
    
    async def place_order(self, order: Order) -> Order:
        """Place order on Binance."""
        self._check_rate_limit()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")
        
        # Generate client order ID
        order.client_order_id = order.client_order_id or self._generate_client_order_id()
        
        params = {
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': str(order.quantity),
            'newClientOrderId': order.client_order_id,
            'timeInForce': order.time_in_force.value
        }
        
        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if order.price is None:
                raise ValueError(f"Price required for {order.order_type.value} orders")
            params['price'] = str(order.price)
        
        # Add stop price
        if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT,
                                 OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
            if order.stop_price is None:
                raise ValueError(f"Stop price required for {order.order_type.value} orders")
            params['stopPrice'] = str(order.stop_price)
        
        # Market orders don't need timeInForce
        if order.order_type == OrderType.MARKET:
            params.pop('timeInForce', None)
        
        try:
            result = await self._signed_request('POST', 'v3/order', params)
            
            # Map Binance response to Order
            order.order_id = str(result.get('orderId', ''))
            status_map = {
                'NEW': OrderStatus.OPEN,
                'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED
            }
            order.status = status_map.get(result.get('status', ''), OrderStatus.PENDING)
            order.filled_quantity = float(result.get('executedQty', 0))
            
            # Calculate average fill price from fills
            fills = result.get('fills', [])
            if fills:
                total_qty = sum(float(f['qty']) for f in fills)
                total_cost = sum(float(f['qty']) * float(f['price']) for f in fills)
                order.avg_fill_price = total_cost / total_qty if total_qty > 0 else 0
                order.commission = sum(float(f.get('commission', 0)) for f in fills)
            elif result.get('price'):
                order.avg_fill_price = float(result['price'])
            
            order.updated_at = datetime.now()
            
            # Store order
            self._orders[order.order_id] = order
            self._log_order(order, "PLACED")
            
            self.logger.info(
                f"Order placed on Binance: {order.side.value} {order.quantity} {order.symbol} "
                f"-> ID={order.order_id} status={order.status.value}"
            )
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            self._log_order(order, f"REJECTED: {e}")
            self.logger.error(f"Failed to place order on Binance: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance."""
        self._check_rate_limit()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('DELETE', 'v3/order', {
                'symbol': symbol,
                'orderId': int(order_id)
            })
            
            if result.get('status') == 'CANCELED':
                if order_id in self._orders:
                    self._orders[order_id].status = OrderStatus.CANCELLED
                    self._orders[order_id].updated_at = datetime.now()
                    self._log_order(self._orders[order_id], "CANCELLED")
                self.logger.info(f"Order cancelled on Binance: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status from Binance."""
        self._check_rate_limit()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('GET', 'v3/order', {
                'symbol': symbol,
                'orderId': int(order_id)
            })
            
            status_map = {
                'NEW': OrderStatus.OPEN,
                'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED
            }
            
            order = Order(
                symbol=result['symbol'],
                side=OrderSide(result['side']),
                order_type=OrderType(result['type']),
                quantity=float(result['origQty']),
                price=float(result['price']) if float(result['price']) > 0 else None,
                stop_price=float(result.get('stopPrice', 0)) or None,
                order_id=str(result['orderId']),
                status=status_map.get(result['status'], OrderStatus.PENDING),
                filled_quantity=float(result['executedQty']),
                avg_fill_price=float(result.get('avgPrice', 0) or result.get('price', 0)),
                client_order_id=result.get('clientOrderId', '')
            )
            
            # Update cache
            self._orders[order.order_id] = order
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to get order status {order_id}: {e}")
            raise
    
    async def get_balance(self) -> AccountBalance:
        """Get balance from Binance."""
        self._check_rate_limit()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('GET', 'v3/account')
            
            # Calculate totals from all balances
            total_equity = 0.0
            available_balance = 0.0
            
            for asset_balance in result.get('balances', []):
                free = float(asset_balance.get('free', 0))
                locked = float(asset_balance.get('locked', 0))
                
                # For simplicity, sum USDT-equivalent (only count stablecoins directly)
                asset = asset_balance['asset']
                if asset in ['USDT', 'BUSD', 'USDC']:
                    total_equity += free + locked
                    available_balance += free
            
            return AccountBalance(
                total_equity=total_equity,
                available_balance=available_balance,
                total_margin=total_equity - available_balance,
                unrealized_pnl=0.0,  # Spot doesn't have unrealized PnL
                realized_pnl=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get balance from Binance: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get positions from Binance (spot balances as positions)."""
        self._check_rate_limit()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            result = await self._signed_request('GET', 'v3/account')
            positions = []
            
            for asset_balance in result.get('balances', []):
                free = float(asset_balance.get('free', 0))
                locked = float(asset_balance.get('locked', 0))
                total = free + locked
                
                if total > 0 and asset_balance['asset'] not in ['USDT', 'BUSD', 'USDC']:
                    symbol = f"{asset_balance['asset']}USDT"
                    
                    # Get current price
                    try:
                        ticker = await self.get_ticker(symbol)
                        current_price = ticker.last_price
                    except Exception:
                        current_price = 0.0
                    
                    position = Position(
                        symbol=symbol,
                        quantity=total,
                        current_price=current_price,
                        entry_price=0.0,  # Binance spot doesn't track entry price
                        unrealized_pnl=0.0
                    )
                    positions.append(position)
                    self._positions[symbol] = position
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions from Binance: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> MarketTicker:
        """Get ticker from Binance."""
        self._check_rate_limit()
        
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance")
        
        try:
            # Get 24hr ticker + order book top
            ticker_data = await self._public_request('v3/ticker/24hr', {'symbol': symbol})
            book_data = await self._public_request('v3/ticker/bookTicker', {'symbol': symbol})
            
            return MarketTicker(
                symbol=symbol,
                bid_price=float(book_data.get('bidPrice', 0)),
                ask_price=float(book_data.get('askPrice', 0)),
                last_price=float(ticker_data.get('lastPrice', 0)),
                volume=float(ticker_data.get('volume', 0)),
                quote_volume=float(ticker_data.get('quoteVolume', 0)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise


def create_broker(broker_type: str, config: Dict[str, Any]) -> BrokerInterface:
    """
    Factory function to create broker instance.
    
    Args:
        broker_type: Type of broker ('paper', 'binance')
        config: Broker configuration
        
    Returns:
        Broker interface instance
    """
    brokers = {
        'paper': PaperTradingBroker,
        'binance': BinanceBroker
    }
    
    if broker_type not in brokers:
        raise ValueError(f"Unknown broker type: {broker_type}")
    
    return brokers[broker_type](config)
