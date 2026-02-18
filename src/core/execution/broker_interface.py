# src/core/execution/broker_interface.py
"""
Broker Interface - Execution Layer
=================================
Abstract broker interface for order execution.
Supports Paper Trading and Live Trading modes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import random


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Fill info
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Metadata
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    reduce_only: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'time_in_force': self.time_in_force,
            'reduce_only': self.reduce_only
        }


@dataclass
class Position:
    """Position information."""
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    leverage: float = 1.0
    opened_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'leverage': self.leverage,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class AccountBalance:
    """Account balance information."""
    total_equity: float = 0.0
    available_balance: float = 0.0
    reserved_balance: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'total_equity': self.total_equity,
            'available_balance': self.available_balance,
            'reserved_balance': self.reserved_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'timestamp': self.timestamp.isoformat()
        }


class Broker(ABC):
    """Abstract broker interface."""
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        pass
    
    @abstractmethod
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price."""
        pass


class PaperBroker(Broker):
    """Paper trading broker for backtesting/simulation."""
    
    def __init__(self, initial_balance: float = 100000, commission_pct: float = 0.001):
        """
        Initialize paper broker.
        
        Args:
            initial_balance: Starting balance
            commission_pct: Commission percentage per trade
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_pct = commission_pct
        
        # State
        self._connected = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        
        # Market prices (simulated)
        self._market_prices: Dict[str, float] = {}
        
        # PnL tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        logger.info(f"Paper broker initialized with balance: {initial_balance}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Connect (simulated)."""
        await asyncio.sleep(0.1)  # Simulate connection
        self._connected = True
        logger.info("Paper broker connected")
        return True
    
    async def disconnect(self):
        """Disconnect (simulated)."""
        self._connected = False
        logger.info("Paper broker disconnected")
    
    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        self._update_unrealized_pnl()
        
        return AccountBalance(
            total_equity=self.balance + self.unrealized_pnl,
            available_balance=self.balance,
            reserved_balance=0,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        self._update_unrealized_pnl()
        return list(self.positions.values())
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    async def place_order(self, order: Order) -> Order:
        """Place paper trade order."""
        if not self._connected:
            raise ConnectionError("Broker not connected")
        
        # Generate order ID
        if not order.order_id:
            order.order_id = f"PAPER_{uuid.uuid4().hex[:12]}"
        
        # Get current price
        current_price = await self.get_market_price(order.symbol)
        
        # Determine fill price
        if order.order_type == OrderType.MARKET:
            fill_price = current_price
        elif order.price:
            fill_price = order.price
        else:
            fill_price = current_price
        
        # Simulate fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = fill_price * order.quantity * self.commission_pct
        order.filled_at = datetime.now()
        
        # Update balance
        self.balance -= order.commission
        
        # Update position
        await self._update_position_from_fill(order)
        
        # Store order
        self.orders[order.order_id] = order
        
        logger.info(
            f"Paper order filled: {order.symbol} {order.side.value} "
            f"{order.quantity} @ {fill_price} (commission: {order.commission:.2f})"
        )
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        order = self.orders.get(order_id)
        
        if order and order.status in [OrderStatus.PENDING, OrderStatus.OPEN]:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            logger.info(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    async def get_market_price(self, symbol: str) -> float:
        """Get simulated market price."""
        if symbol not in self._market_prices:
            # Generate initial price based on symbol
            base_prices = {
                'BTCUSDT': 45000,
                'ETHUSDT': 2500,
                'BNBUSDT': 300,
                'SOLUSDT': 100,
                'XRPUSDT': 0.5
            }
            self._market_prices[symbol] = base_prices.get(symbol, 100)
        
        # Add small random variation
        variation = random.uniform(-0.001, 0.001)
        self._market_prices[symbol] *= (1 + variation)
        
        return self._market_prices[symbol]
    
    def set_market_price(self, symbol: str, price: float):
        """Set market price (for testing)."""
        self._market_prices[symbol] = price
    
    async def _update_position_from_fill(self, order: Order):
        """Update position after order fill."""
        symbol = order.symbol
        quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
        
        position = self.positions.get(symbol)
        
        if position is None:
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=order.avg_fill_price,
                current_price=order.avg_fill_price,
                commission=order.commission,
                opened_at=datetime.now()
            )
            self.positions[symbol] = position
        else:
            # Average in to position
            old_qty = position.quantity
            new_qty = old_qty + quantity
            
            if old_qty * new_qty >= 0:
                # Same direction - average entry
                total_cost = (old_qty * position.entry_price) + (quantity * order.avg_fill_price)
                position.entry_price = total_cost / new_qty if new_qty != 0 else 0
            else:
                # Opposite direction - reduce/flip
                if abs(quantity) >= abs(old_qty):
                    # Position flipped
                    position.entry_price = order.avg_fill_price
                    position.commission = 0
            
            position.quantity = new_qty
            position.current_price = order.avg_fill_price
            position.commission += order.commission
            position.updated_at = datetime.now()
        
        # Remove if closed
        if position.quantity == 0:
            self.realized_pnl += position.realized_pnl
            del self.positions[symbol]
    
    def _update_unrealized_pnl(self):
        """Update unrealized PnL for all positions."""
        self.unrealized_pnl = 0
        
        for position in self.positions.values():
            if position.quantity != 0:
                price_diff = position.current_price - position.entry_price
                position.unrealized_pnl = price_diff * abs(position.quantity) - position.commission
                self.unrealized_pnl += position.unrealized_pnl
    
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history."""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return sorted(orders, key=lambda x: x.created_at, reverse=True)


class LiveBroker(Broker):
    """Live trading broker (Binance implementation)."""
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True
    ):
        """
        Initialize live broker.
        
        Args:
            api_key: API key
            api_secret: API secret
            testnet: Use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        self._connected = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        
        # Will be initialized with actual client
        self.client = None
        
        logger.info(f"Live broker initialized (testnet: {testnet})")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> bool:
        """Connect to exchange."""
        try:
            # Import binance client
            from binance.client import Client
            
            # Connect to Binance
            if self.testnet:
                self.client = Client(self.api_key, self.api_secret, testnet=True)
            else:
                self.client = Client(self.api_key, self.api_secret)
            
            # Test connection
            self.client.get_account()
            
            self._connected = True
            logger.info("Live broker connected to Binance")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to broker: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from exchange."""
        self._connected = False
        self.client = None
        logger.info("Live broker disconnected")
    
    async def get_balance(self) -> AccountBalance:
        """Get account balance from exchange."""
        if not self.client:
            raise ConnectionError("Not connected")
        
        try:
            account = self.client.get_account()
            
            # Find USDT balance
            usdt_balance = 0.0
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            return AccountBalance(
                total_equity=usdt_balance,
                available_balance=usdt_balance
            )
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        # For spot trading, get all assets with balance
        if not self.client:
            raise ConnectionError("Not connected")
        
        positions = []
        
        try:
            account = self.client.get_account()
            
            for balance in account['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if free + locked > 0:
                    # Get current price
                    try:
                        ticker = self.client.get_symbol_ticker(
                            symbol=f"{balance['asset']}USDT"
                        )
                        current_price = float(ticker['price'])
                    except:
                        current_price = 0
                    
                    if current_price > 0:
                        position = Position(
                            symbol=f"{balance['asset']}USDT",
                            quantity=free + locked,
                            current_price=current_price
                        )
                        positions.append(position)
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        
        return positions
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = await self.get_positions()
        
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        
        return None
    
    async def place_order(self, order: Order) -> Order:
        """Place live order."""
        if not self.client:
            raise ConnectionError("Not connected")
        
        # Generate order ID if not provided
        if not order.order_id:
            order.order_id = uuid.uuid4().hex[:12]
        
        try:
            # Build order parameters
            params = {
                'symbol': order.symbol,
                'side': order.side.value.upper(),
                'type': order.order_type.value.upper(),
                'quantity': order.quantity
            }
            
            if order.order_type == OrderType.LIMIT and order.price:
                params['price'] = order.price
                params['timeInForce'] = order.time_in_force
            
            if order.stop_price:
                params['stopPrice'] = order.stop_price
            
            if order.reduce_only:
                params['reduceOnly'] = True
            
            # Place order
            result = self.client.create_order(**params)
            
            # Update order with exchange response
            order.order_id = result['orderId']
            order.status = OrderStatus(result['status'])
            
            if result.get('fills'):
                fills = result['fills']
                total_qty = sum(float(f['qty']) for f in fills)
                total_price = sum(float(f['qty']) * float(f['price']) for f in fills)
                
                order.avg_fill_price = total_price / total_qty if total_qty > 0 else 0
                order.filled_quantity = total_qty
                order.commission = sum(float(f.get('commission', 0)) for f in fills)
            
            self.orders[order.order_id] = order
            
            logger.info(f"Live order placed: {order.order_id}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if not self.client:
            raise ConnectionError("Not connected")
        
        try:
            self.client.cancel_order(orderId=order_id)
            
            order = self.orders.get(order_id)
            if order:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        if not self.client:
            raise ConnectionError("Not connected")
        
        try:
            result = self.client.get_order(orderId=order_id)
            
            order = Order(
                order_id=result['orderId'],
                symbol=result['symbol'],
                side=OrderSide(result['side'].lower()),
                order_type=OrderType(result['type'].lower()),
                quantity=float(result['origQty']),
                filled_quantity=float(result['executedQty']),
                price=float(result['price']),
                status=OrderStatus(result['status'].lower()),
                created_at=datetime.fromtimestamp(result['time'] / 1000)
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order: {e}")
            return None
    
    async def get_market_price(self, symbol: str) -> float:
        """Get current market price."""
        if not self.client:
            raise ConnectionError("Not connected")
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            raise


def create_broker(
    mode: str = 'paper',
    **kwargs
) -> Broker:
    """
    Factory function to create broker.
    
    Args:
        mode: 'paper' or 'live'
        **kwargs: Broker configuration
        
    Returns:
        Broker instance
    """
    if mode.lower() == 'paper':
        return PaperBroker(
            initial_balance=kwargs.get('initial_balance', 100000),
            commission_pct=kwargs.get('commission_pct', 0.001)
        )
    elif mode.lower() == 'live':
        return LiveBroker(
            api_key=kwargs.get('api_key', ''),
            api_secret=kwargs.get('api_secret', ''),
            testnet=kwargs.get('testnet', True)
        )
    else:
        raise ValueError(f"Unknown broker mode: {mode}")
