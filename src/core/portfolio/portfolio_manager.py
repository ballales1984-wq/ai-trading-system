# src/core/portfolio/portfolio_manager.py
"""
Portfolio Manager
================
Multi-asset portfolio management with risk-adjusted position sizing.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Position information."""
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    leverage: float = 1.0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Position market value."""
        return abs(self.quantity) * self.current_price
    
    @property
    def notional_value(self) -> float:
        """Position notional value."""
        return abs(self.quantity) * self.entry_price
    
    @property
    def exposure_pct(self) -> float:
        """Exposure as percentage of notional."""
        return self.notional_value
    
    def update_price(self, price: float):
        """Update current price and recalculate PnL."""
        self.current_price = price
        self.updated_at = datetime.now()
        
        if self.quantity != 0:
            price_diff = price - self.entry_price
            if self.side == PositionSide.SHORT:
                price_diff = -price_diff
            
            self.unrealized_pnl = price_diff * abs(self.quantity) - self.commission
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'leverage': self.leverage,
            'market_value': self.market_value,
            'notional_value': self.notional_value,
            'opened_at': self.opened_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_equity: float = 0.0
    available_balance: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    
    # Risk metrics
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    
    # Counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_equity': self.total_equity,
            'available_balance': self.available_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_commission': self.total_commission,
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }


class PortfolioManager:
    """
    Manages portfolio positions and calculates optimal position sizes.
    """
    
    def __init__(
        self,
        initial_balance: float = 100000,
        max_position_pct: float = 0.3,
        max_leverage: float = 1.0,
        base_currency: str = "USDT"
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_balance: Starting balance
            max_position_pct: Maximum position as % of portfolio
            max_leverage: Maximum leverage allowed
            base_currency: Base currency for calculations
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.available_balance = initial_balance
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.base_currency = base_currency
        
        # Positions
        self.positions: Dict[str, Position] = {}
        
        # Trade history
        self.trades: List[Dict] = []
        
        # Performance tracking
        self.equity_history: List[float] = [initial_balance]
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        
        # Risk metrics
        self.total_commission = 0.0
        self.realized_pnl = 0.0
        
        logger.info(f"Portfolio manager initialized: balance={initial_balance}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())
    
    def get_open_positions(self) -> List[Position]:
        """Get open positions (non-zero)."""
        return [p for p in self.positions.values() if p.quantity != 0]
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure."""
        return sum(p.market_value for p in self.get_open_positions())
    
    def get_net_exposure(self) -> float:
        """Get net exposure (long - short)."""
        long_exposure = sum(
            p.market_value for p in self.get_open_positions()
            if p.side == PositionSide.LONG
        )
        short_exposure = sum(
            p.market_value for p in self.get_open_positions()
            if p.side == PositionSide.SHORT
        )
        return long_exposure - short_exposure
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_pct: float = 0.02,
        confidence: float = 1.0,
        risk_pct: float = 0.02
    ) -> float:
        """
        Calculate optimal position size using risk management.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage
            confidence: Signal confidence (0-1)
            risk_pct: Risk per trade as % of portfolio
            
        Returns:
            Recommended position size
        """
        # Maximum position size based on risk
        risk_amount = self.balance * risk_pct * confidence
        
        # Position size based on stop loss
        price_risk = entry_price * stop_loss_pct
        position_size = risk_amount / price_risk
        
        # Apply leverage
        position_size *= self.max_leverage
        
        # Ensure within limits
        max_size = (self.balance * self.max_position_pct) / entry_price
        position_size = min(position_size, max_size)
        
        # Ensure we have enough balance
        position_cost = position_size * entry_price
        if position_cost > self.available_balance:
            position_size = self.available_balance / entry_price
        
        logger.debug(
            f"Position size for {symbol}: {position_size:.4f} "
            f"(risk: {risk_amount:.2f}, stop: {stop_loss_pct:.2%})"
        )
        
        return position_size
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        leverage: float = 1.0
    ) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            quantity: Position quantity
            price: Entry price
            commission: Commission paid
            leverage: Position leverage
            
        Returns:
            Opened position
        """
        # Validate quantity
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Check balance
        position_cost = quantity * price
        if position_cost > self.available_balance:
            raise ValueError(f"Insufficient balance: {self.available_balance} < {position_cost}")
        
        # Get or create position
        position = self.positions.get(symbol)
        
        if position is None:
            position = Position(
                symbol=symbol,
                entry_price=price,
                current_price=price,
                commission=commission,
                leverage=leverage
            )
            self.positions[symbol] = position
        
        # Update position
        position.side = PositionSide.LONG if side.upper() == 'LONG' else PositionSide.SHORT
        position.quantity = quantity
        position.entry_price = price
        position.current_price = price
        position.commission = commission
        position.leverage = leverage
        position.opened_at = datetime.now()
        position.updated_at = datetime.now()
        
        # Update balance
        self.available_balance -= (position_cost + commission)
        self.total_commission += commission
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'type': 'open',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(
            f"Position opened: {symbol} {side} {quantity} @ {price} "
            f"(cost: {position_cost:.2f})"
        )
        
        return position
    
    def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        price: float = 0.0,
        commission: float = 0.0
    ) -> Dict:
        """
        Close position partially or fully.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to close (None = full close)
            price: Exit price
            commission: Commission paid
            
        Returns:
            Close result with PnL
        """
        position = self.positions.get(symbol)
        
        if position is None or position.quantity == 0:
            raise ValueError(f"No open position for {symbol}")
        
        # Determine close quantity
        close_qty = quantity if quantity else position.quantity
        close_qty = min(close_qty, abs(position.quantity))
        
        # Calculate PnL
        price_diff = price - position.entry_price
        if position.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        pnl = price_diff * close_qty - commission
        
        # Update position
        position.quantity -= close_qty
        position.realized_pnl += pnl
        position.commission += commission
        position.updated_at = datetime.now()
        
        # Update balance
        exit_value = close_qty * price
        self.available_balance += (exit_value - commission)
        self.realized_pnl += pnl
        self.total_commission += commission
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'side': position.side.value,
            'quantity': close_qty,
            'price': price,
            'commission': commission,
            'pnl': pnl,
            'type': 'close',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(
            f"Position closed: {symbol} {close_qty} @ {price} "
            f"PnL: {pnl:.2f}"
        )
        
        # Remove if fully closed
        if position.quantity == 0:
            del self.positions[symbol]
        
        return {
            'symbol': symbol,
            'quantity': close_qty,
            'price': price,
            'pnl': pnl,
            'realized_pnl': self.realized_pnl,
            'commission': commission
        }
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update all position prices and recalculate PnL.
        
        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, price in prices.items():
            position = self.positions.get(symbol)
            if position:
                old_pnl = position.unrealized_pnl
                position.update_price(price)
                
                logger.debug(
                    f"{symbol} price updated: {price} "
                    f"PnL: {old_pnl:.2f} -> {position.unrealized_pnl:.2f}"
                )
        
        # Update equity
        self._update_equity()
    
    def _update_equity(self):
        """Update total equity."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.total_equity = self.balance + unrealized
        
        # Track peak equity and drawdown
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity
        
        drawdown = (self.peak_equity - self.total_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        self.equity_history.append(self.total_equity)
    
    def get_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics."""
        self._update_equity()
        
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        
        # Calculate win rate
        closed_trades = [t for t in self.trades if t.get('type') == 'close']
        winning = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        win_rate = winning / len(closed_trades) if closed_trades else 0.0
        
        return PortfolioMetrics(
            total_equity=self.total_equity,
            available_balance=self.available_balance,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            total_commission=self.total_commission,
            gross_exposure=self.get_total_exposure(),
            net_exposure=self.get_net_exposure(),
            max_drawdown=self.max_drawdown,
            total_trades=len(closed_trades),
            winning_trades=winning,
            losing_trades=len(closed_trades) - winning,
            win_rate=win_rate
        )
    
    def get_metrics_dict(self) -> Dict:
        """Get metrics as dictionary."""
        return self.get_metrics().to_dict()
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from equity history."""
        if len(self.equity_history) < 2:
            return 0.0
        
        returns = np.diff(self.equity_history) / self.equity_history[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def to_dict(self) -> Dict:
        """Export portfolio state."""
        return {
            'balance': self.balance,
            'available_balance': self.available_balance,
            'total_equity': self.total_equity,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'realized_pnl': self.realized_pnl,
            'positions': [p.to_dict() for p in self.positions.values()],
            'metrics': self.get_metrics_dict()
        }
    
    def save_state(self, path: str):
        """Save portfolio state to file."""
        state = self.to_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Portfolio state saved to {path}")
    
    def load_state(self, path: str):
        """Load portfolio state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.balance = state.get('balance', self.initial_balance)
        self.available_balance = state.get('available_balance', self.balance)
        self.realized_pnl = state.get('realized_pnl', 0.0)
        
        # Load positions
        self.positions = {}
        for pos_data in state.get('positions', []):
            position = Position(
                symbol=pos_data['symbol'],
                side=PositionSide[pos_data['side'].upper()],
                quantity=pos_data['quantity'],
                entry_price=pos_data['entry_price'],
                current_price=pos_data['current_price'],
                unrealized_pnl=pos_data['unrealized_pnl'],
                realized_pnl=pos_data['realized_pnl'],
                commission=pos_data['commission'],
                leverage=pos_data['leverage'],
                opened_at=datetime.fromisoformat(pos_data['opened_at']),
                updated_at=datetime.fromisoformat(pos_data['updated_at'])
            )
            self.positions[position.symbol] = position
        
        self._update_equity()
        
        logger.info(f"Portfolio state loaded from {path}")
