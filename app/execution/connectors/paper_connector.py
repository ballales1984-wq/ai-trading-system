"""
Paper Connector
=============
Paper trading broker for backtesting and simulation.
"""

import logging
import random
import uuid
from typing import Dict, List, Optional

from app.execution.broker_connector import (
    BrokerConnector, Order, Position, AccountBalance,
    OrderSide, OrderType, OrderStatus
)


logger = logging.getLogger(__name__)


class PaperConnector(BrokerConnector):
    """
    Paper trading broker.
    
    Simulates order execution without real money.
    """
    
    def __init__(self, config: Dict):
        """Initialize paper broker."""
        super().__init__(config)
        
        self.initial_balance = config.get("initial_balance", 100000)
        self.balance = self.initial_balance
        self.commission_pct = config.get("commission_pct", 0.001)
        
        # State
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        
        # Market prices (simulated)
        self._market_prices: Dict[str, float] = {}
        
        # PnL
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Initialize default prices
        self._init_default_prices()
    
    def _init_default_prices(self):
        """Initialize default market prices."""
        self._market_prices = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 2500.0,
            "BNBUSDT": 300.0,
            "SOLUSDT": 100.0,
            "XRPUSDT": 0.5,
            "ADAUSDT": 0.5,
            "DOGEUSDT": 0.1,
            "DOTUSDT": 10.0,
            "MATICUSDT": 1.0,
            "LTCUSDT": 100.0
        }
    
    async def connect(self) -> bool:
        """Connect (simulated)."""
        await asyncio.sleep(0.1)
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
            # Simulate slippage
            slippage = random.uniform(-0.001, 0.001)
            fill_price = current_price * (1 + slippage)
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
        if order.side == OrderSide.BUY:
            self.balance -= (order.quantity * fill_price + order.commission)
        else:
            self.balance += (order.quantity * fill_price - order.commission)
        
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
            logger.info(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    async def get_market_price(self, symbol: str) -> float:
        """Get simulated market price."""
        if symbol not in self._market_prices:
            self._market_prices[symbol] = 100.0
        
        # Add small random variation
        variation = random.uniform(-0.0005, 0.0005)
        self._market_prices[symbol] *= (1 + variation)
        
        return self._market_prices[symbol]
    
    def set_market_price(self, symbol: str, price: float):
        """Set market price (for testing)."""
        self._market_prices[symbol] = price
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """Get simulated order book."""
        current_price = await self.get_market_price(symbol)
        
        bids = []
        asks = []
        
        for i in range(1, depth + 1):
            bids.append({
                "price": current_price * (1 - i * 0.0001),
                "quantity": random.uniform(1, 10)
            })
            asks.append({
                "price": current_price * (1 + i * 0.0001),
                "quantity": random.uniform(1, 10)
            })
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks
        }
    
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


# Import asyncio
import asyncio
from datetime import datetime

