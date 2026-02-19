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
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        super().__init__(api_key, secret_key, testnet)
        self.broker = Broker.BINANCE
        self.base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com"
    
    async def connect(self) -> bool:
        """Connect to Binance."""
        self.logger.info("Connecting to Binance...")
        # In production, initialize Binance client here
        self.connected = True
        self.logger.info("Connected to Binance")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        self.logger.info("Disconnecting from Binance...")
        self.connected = False
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        """Place order on Binance."""
        if not self.connected:
            raise ConnectionError("Not connected to Binance")
        
        self.logger.info(
            f"Placing order: {order.side} {order.quantity} {order.symbol} "
            f"@ {order.price or 'MARKET'}"
        )
        
        # In production, call Binance API here
        # Simulate order placement
        order.broker_order_id = f"binance_{order.order_id[:8]}"
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        order.average_price = order.price or 43500.0  # Simulated price
        
        return order
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance."""
        self.logger.info(f"Cancelling order {order_id}")
        return True
    
    async def get_order_status(self, order_id: str, symbol: str) -> BrokerOrder:
        """Get order status from Binance."""
        # Simulated response
        return BrokerOrder(
            order_id=order_id,
            broker_order_id=f"binance_{order_id[:8]}",
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quantity=1.0,
            status="FILLED",
            filled_quantity=1.0,
            average_price=43500.0,
        )
    
    async def get_balance(self) -> List[AccountBalance]:
        """Get account balance from Binance."""
        return [
            AccountBalance(asset="USDT", free=500000.0, locked=0.0, total=500000.0),
            AccountBalance(asset="BTC", free=1.5, locked=0.0, total=1.5),
            AccountBalance(asset="ETH", free=15.0, locked=0.0, total=15.0),
        ]
    
    async def get_positions(self) -> List[Position]:
        """Get open positions from Binance."""
        return [
            Position(
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.5,
                entry_price=42000.0,
                current_price=43500.0,
                unrealized_pnl=2250.0,
                leverage=1.0,
                margin=31500.0,
            ),
        ]
    
    async def get_symbol_price(self, symbol: str) -> float:
        """Get symbol price from Binance."""
        prices = {"BTCUSDT": 43500.0, "ETHUSDT": 2350.0, "SOLUSDT": 95.0}
        return prices.get(symbol.upper(), 100.0)


# ============================================================================
# BYBIT CONNECTOR
# ============================================================================

class BybitConnector(BrokerConnector):
    """Bybit broker connector."""
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        super().__init__(api_key, secret_key, testnet)
        self.broker = Broker.BYBIT
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    
    async def connect(self) -> bool:
        self.logger.info("Connecting to Bybit...")
        self.connected = True
        return True
    
    async def disconnect(self) -> None:
        self.connected = False
    
    async def place_order(self, order: BrokerOrder) -> BrokerOrder:
        order.broker_order_id = f"bybit_{order.order_id[:8]}"
        order.status = "FILLED"
        order.filled_quantity = order.quantity
        return order
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        return True
    
    async def get_order_status(self, order_id: str, symbol: str) -> BrokerOrder:
        return BrokerOrder(order_id=order_id, symbol=symbol, side="BUY", 
                          order_type="MARKET", quantity=1.0, status="FILLED")
    
    async def get_balance(self) -> List[AccountBalance]:
        return [AccountBalance(asset="USDT", free=500000.0, locked=0.0, total=500000.0)]
    
    async def get_positions(self) -> List[Position]:
        return []
    
    async def get_symbol_price(self, symbol: str) -> float:
        return 43500.0


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
