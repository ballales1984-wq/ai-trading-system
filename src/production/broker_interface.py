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
    
    async def place_order(self, order: Order) -> Order:
        """Place order on Binance."""
        self._check_rate_limit()
        
        # Implementation would use Binance API
        # For now, raise NotImplementedError
        raise NotImplementedError("Binance integration requires API keys")
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance."""
        raise NotImplementedError("Binance integration requires API keys")
    
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status from Binance."""
        raise NotImplementedError("Binance integration requires API keys")
    
    async def get_balance(self) -> AccountBalance:
        """Get balance from Binance."""
        raise NotImplementedError("Binance integration requires API keys")
    
    async def get_positions(self) -> List[Position]:
        """Get positions from Binance."""
        raise NotImplementedError("Binance integration requires API keys")
    
    async def get_ticker(self, symbol: str) -> MarketTicker:
        """Get ticker from Binance."""
        raise NotImplementedError("Binance integration requires API keys")


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
