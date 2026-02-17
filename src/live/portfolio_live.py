"""
Live Portfolio Management Module
Dynamic portfolio allocation and management for multi-asset trading
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LivePosition:
    """Represents a single position in the portfolio."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        side: str = "long"
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.side = side  # 'long' or 'short'
        self.entry_time = datetime.now()
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
    
    def update(self, current_price: float):
        """Update position with current price."""
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.entry_price + (self.unrealized_pnl / self.quantity) if self.quantity > 0 else 0,
            'side': self.side,
            'unrealized_pnl': self.unrealized_pnl,
            'entry_time': self.entry_time.isoformat()
        }


class LivePortfolio:
    """
    Live portfolio manager for multi-asset trading.
    """
    
    def __init__(
        self,
        initial_capital: float,
        fee_rate: float = 0.001
    ):
        """
        Initialize the live portfolio.
        
        Args:
            initial_capital: Initial capital
            fee_rate: Trading fee rate (default 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, LivePosition] = {}
        self.trades: List[dict] = []
        self.fee_rate = fee_rate
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"Initialized LivePortfolio with capital: ${initial_capital:,.2f}")
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Current prices for all assets
            
        Returns:
            Total portfolio value
        """
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update(current_prices[symbol])
                positions_value += position.quantity * current_prices[symbol]
        
        return self.cash + positions_value
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Get total unrealized PnL."""
        total = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update(current_prices[symbol])
                total += position.unrealized_pnl
        return total
    
    def can_open_position(
        self,
        symbol: str,
        price: float,
        quantity: float
    ) -> bool:
        """Check if a position can be opened."""
        required_capital = price * quantity * (1 + self.fee_rate)
        return self.cash >= required_capital
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            quantity: Number of units
            price: Entry price
            
        Returns:
            True if position opened successfully
        """
        if not self.can_open_position(symbol, price, quantity):
            logger.warning(f"Insufficient capital to open position: {symbol}")
            return False
        
        # Calculate fees
        trade_value = price * quantity
        fees = trade_value * self.fee_rate
        
        # Deduct from cash
        self.cash -= (trade_value + fees)
        
        # Create position
        position = LivePosition(symbol, quantity, price, side)
        self.positions[symbol] = position
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'fees': fees,
            'type': 'open',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Opened {side} position: {symbol} x {quantity} @ ${price:,.2f}")
        
        return True
    
    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "signal"
    ) -> Optional[float]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            reason: Reason for closing
            
        Returns:
            Realized PnL or None if position didn't exist
        """
        if symbol not in self.positions:
            logger.warning(f"Position not found: {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate fees
        trade_value = price * position.quantity
        fees = trade_value * self.fee_rate
        
        # Calculate PnL
        position.update(price)
        realized_pnl = position.unrealized_pnl - fees
        
        # Update cash
        self.cash += (trade_value - fees)
        
        # Update performance
        self.total_realized_pnl += realized_pnl
        self.total_trades += 1
        if realized_pnl > 0:
            self.winning_trades += 1
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'side': position.side,
            'quantity': position.quantity,
            'price': price,
            'fees': fees,
            'pnl': realized_pnl,
            'type': 'close',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ ${price:,.2f}, PnL: ${realized_pnl:,.2f}")
        
        return realized_pnl
    
    def get_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Get total value of all positions."""
        total = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position.quantity * current_prices[symbol]
        return total
    
    def get_win_rate(self) -> float:
        """Get win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def get_stats(self, current_prices: Dict[str, float]) -> dict:
        """Get portfolio statistics."""
        total_value = self.get_total_value(current_prices)
        unrealized_pnl = self.get_unrealized_pnl(current_prices)
        
        return {
            'cash': self.cash,
            'positions_value': self.get_positions_value(current_prices),
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_realized_pnl,
            'total_pnl': unrealized_pnl + self.total_realized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.get_win_rate(),
            'num_positions': len(self.positions),
            'initial_capital': self.initial_capital,
            'return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100
        }
    
    def to_dict(self, current_prices: Dict[str, float]) -> dict:
        """Export portfolio state."""
        return {
            'stats': self.get_stats(current_prices),
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'recent_trades': self.trades[-10:] if len(self.trades) > 0 else []
        }


class BaseAllocator:
    """Base class for portfolio allocation strategies."""
    
    def allocate(
        self,
        signals: Dict[str, int],
        capital: float,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate allocation weights.
        
        Args:
            signals: Trading signals per asset (-1, 0, 1)
            capital: Total capital
            prices: Current prices
            
        Returns:
            Allocation per asset in units
        """
        raise NotImplementedError


class EqualWeightAllocator(BaseAllocator):
    """Equal weight allocation strategy."""
    
    def __init__(self, max_positions: int = 5):
        self.max_positions = max_positions
    
    def allocate(
        self,
        signals: Dict[str, int],
        capital: float,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Allocate capital equally among assets with non-zero signals."""
        # Filter for assets with signals
        active_signals = {s: sig for s, sig in signals.items() if sig != 0}
        
        if not active_signals:
            return {}
        
        # Limit to max positions
        if len(active_signals) > self.max_positions:
            # Sort by absolute signal strength if available
            active_signals = dict(list(active_signals.items())[:self.max_positions])
        
        # Equal weight
        n_assets = len(active_signals)
        weight_per_asset = 1.0 / n_assets
        
        allocations = {}
        capital_per_asset = capital * weight_per_asset
        
        for symbol in active_signals.keys():
            if symbol in prices:
                allocations[symbol] = capital_per_asset / prices[symbol]
        
        return allocations


class VolatilityParityAllocator(BaseAllocator):
    """Volatility parity allocation strategy."""
    
    def __init__(
        self,
        target_vol: float = 0.02,
        lookback: int = 20,
        max_positions: int = 5
    ):
        self.target_vol = target_vol
        self.lookback = lookback
        self.max_positions = max_positions
    
    def allocate(
        self,
        signals: Dict[str, int],
        capital: float,
        prices: Dict[str, float],
        data_frames: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """Allocate capital based on inverse volatility."""
        active_signals = {s: sig for s, sig in signals.items() if sig != 0}
        
        if not active_signals:
            return {}
        
        # Limit to max positions
        if len(active_signals) > self.max_positions:
            active_signals = dict(list(active_signals.items())[:self.max_positions])
        
        # Calculate volatilities
        volatilities = {}
        
        if data_frames:
            for symbol in active_signals.keys():
                if symbol in data_frames and len(data_frames[symbol]) >= self.lookback:
                    returns = data_frames[symbol]['close'].pct_change().tail(self.lookback)
                    volatilities[symbol] = returns.std() if len(returns) > 0 else 0.02
                else:
                    volatilities[symbol] = 0.02
        else:
            # Default volatility
            volatilities = {s: 0.02 for s in active_signals.keys()}
        
        # Calculate inverse volatility weights
        inv_vols = {s: 1.0 / max(v, 0.001) for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        allocations = {}
        
        for symbol in active_signals.keys():
            if symbol in prices and symbol in inv_vols:
                weight = inv_vols[symbol] / total_inv_vol
                allocations[symbol] = (capital * weight) / prices[symbol]
        
        return allocations


class RiskParityAllocator(BaseAllocator):
    """Risk parity allocation strategy."""
    
    def __init__(
        self,
        lookback: int = 20,
        max_positions: int = 5
    ):
        self.lookback = lookback
        self.max_positions = max_positions
    
    def allocate(
        self,
        signals: Dict[str, int],
        capital: float,
        prices: Dict[str, float],
        data_frames: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """Allocate capital using risk parity."""
        active_signals = {s: sig for s, sig in signals.items() if sig != 0}
        
        if not active_signals:
            return {}
        
        # Limit to max positions
        if len(active_signals) > self.max_positions:
            active_signals = dict(list(active_signals.items())[:self.max_positions])
        
        # Calculate volatilities
        volatilities = {}
        
        if data_frames:
            for symbol in active_signals.keys():
                if symbol in data_frames and len(data_frames[symbol]) >= self.lookback:
                    returns = data_frames[symbol]['close'].pct_change().tail(self.lookback)
                    volatilities[symbol] = returns.std() if len(returns) > 0 else 0.02
                else:
                    volatilities[symbol] = 0.02
        else:
            volatilities = {s: 0.02 for s in active_signals.keys()}
        
        # Risk parity: each asset contributes equally to portfolio risk
        # Using inverse volatility as a proxy
        vol_array = np.array([volatilities[s] for s in active_signals.keys()])
        
        # Normalize to get weights
        vol_array = vol_array / vol_array.sum()
        
        allocations = {}
        
        for i, symbol in enumerate(active_signals.keys()):
            if symbol in prices:
                weight = vol_array[i]
                allocations[symbol] = (capital * weight) / prices[symbol]
        
        return allocations


class MomentumAllocator(BaseAllocator):
    """Momentum-based allocation strategy."""
    
    def __init__(
        self,
        lookback: int = 20,
        max_positions: int = 5
    ):
        self.lookback = lookback
        self.max_positions = max_positions
    
    def allocate(
        self,
        signals: Dict[str, int],
        capital: float,
        prices: Dict[str, float],
        data_frames: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """Allocate capital based on momentum."""
        active_signals = {s: sig for s, sig in signals.items() if sig != 0}
        
        if not active_signals:
            return {}
        
        # Calculate momentum scores
        momenta = {}
        
        if data_frames:
            for symbol in active_signals.keys():
                if symbol in data_frames and len(data_frames[symbol]) >= self.lookback:
                    prices_df = data_frames[symbol]['close']
                    momentum = (prices_df.iloc[-1] - prices_df.iloc[-self.lookback]) / prices_df.iloc[-self.lookback]
                    momenta[symbol] = momentum
                else:
                    momenta[symbol] = 0
        else:
            momenta = {s: 0 for s in active_signals.keys()}
        
        # Filter for positive momentum
        positive_momentum = {s: m for s, m in momenta.items() if m > 0}
        
        if not positive_momentum:
            return {}
        
        # Sort by momentum and limit
        sorted_momentum = sorted(positive_momentum.items(), key=lambda x: x[1], reverse=True)
        top_assets = dict(sorted_momentum[:self.max_positions])
        
        # Weight by momentum
        total_momentum = sum(top_assets.values())
        
        allocations = {}
        
        for symbol in top_assets.keys():
            if symbol in prices:
                weight = top_assets[symbol] / total_momentum
                allocations[symbol] = (capital * weight) / prices[symbol]
        
        return allocations
