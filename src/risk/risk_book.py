"""
Risk Book Module
================
Centralized risk management for tracking positions, exposures, and limits.

This module provides a centralized RiskBook class that tracks:
- Current positions and their exposures
- Equity history for drawdown calculations
- Position limits and drawdown limits

Usage:
    from src.risk.risk_book import RiskBook, RiskLimits, Position
    
    limits = RiskLimits(
        max_position_pct=0.10,
        max_daily_drawdown_pct=0.05,
        var_95_limit=0.08,
        cvar_95_limit=0.10,
    )
    risk_book = RiskBook(limits)
    
    # Update positions
    pos = Position(symbol="BTCUSDT", quantity=0.1, avg_price=50000, side="long")
    risk_book.update_position(pos)
    
    # Check limits
    prices = {"BTCUSDT": 50000}
    equity = 100000
    if risk_book.check_position_limit("BTCUSDT", prices, equity):
        print("Position within limits")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Represents a trading position.
    
    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT")
        quantity: Position size (positive for long, negative for short)
        avg_price: Average entry price
        side: Position side ("long" or "short")
        entry_time: Optional entry timestamp
    """
    symbol: str
    quantity: float
    avg_price: float
    side: str = "long"
    entry_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.side not in ["long", "short"]:
            raise ValueError(f"Invalid side: {self.side}")
        if self.entry_time is None:
            self.entry_time = datetime.utcnow()
    
    @property
    def value(self) -> float:
        """Current position value (not mark-to-market)."""
        return abs(self.quantity) * self.avg_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == "long"
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == "short"


@dataclass
class RiskLimits:
    """
    Risk limits configuration.
    
    Attributes:
        max_position_pct: Maximum position size as percentage of equity
        max_daily_drawdown_pct: Maximum allowed daily drawdown
        var_95_limit: VaR 95% limit as percentage
        cvar_95_limit: CVaR 95% limit as percentage
        max_leverage: Maximum allowed leverage
        max_correlation: Maximum allowed correlation between positions
    """
    max_position_pct: float = 0.10
    max_daily_drawdown_pct: float = 0.05
    var_95_limit: float = 0.08
    cvar_95_limit: float = 0.10
    max_leverage: float = 1.0
    max_correlation: float = 0.80
    
    def __post_init__(self):
        """Validate limits."""
        if not 0 < self.max_position_pct <= 1.0:
            raise ValueError("max_position_pct must be between 0 and 1")
        if not 0 < self.max_daily_drawdown_pct <= 1.0:
            raise ValueError("max_daily_drawdown_pct must be between 0 and 1")


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    total_exposure: float
    equity: float
    exposure_pct: float
    daily_drawdown_pct: float
    var_95: float
    cvar_95: float
    leverage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskBook:
    """
    Centralized risk management book.
    
    Tracks all positions, monitors limits, and calculates risk metrics.
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize RiskBook with specified limits.
        
        Args:
            limits: RiskLimits configuration
        """
        self.limits = limits
        self.positions: Dict[str, Position] = {}
        self.equity_history: List[float] = []
        self.last_update: Optional[datetime] = None
        self._equity: float = 0.0
    
    def update_position(self, pos: Position) -> None:
        """
        Update or add a position.
        
        Args:
            pos: Position to add/update
        """
        self.positions[pos.symbol] = pos
        self.last_update = datetime.utcnow()
    
    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from the book.
        
        Args:
            symbol: Symbol to remove
        """
        self.positions.pop(symbol, None)
        self.last_update = datetime.utcnow()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        return self.positions.get(symbol)
    
    def total_exposure(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio exposure.
        
        Args:
            prices: Current prices for all symbols
            
        Returns:
            Total exposure in USD
        """
        total = 0.0
        for pos in self.positions.values():
            price = prices.get(pos.symbol, pos.avg_price)
            total += abs(pos.quantity * price)
        return total
    
    def exposure_pct(self, prices: Dict[str, float], equity: float) -> float:
        """
        Calculate exposure as percentage of equity.
        
        Args:
            prices: Current prices
            equity: Current equity value
            
        Returns:
            Exposure percentage
        """
        if equity <= 0:
            return 0.0
        return self.total_exposure(prices) / equity
    
    def check_position_limit(
        self, 
        symbol: str, 
        prices: Dict[str, float], 
        equity: float
    ) -> bool:
        """
        Check if adding a position would exceed limits.
        
        Args:
            symbol: Trading symbol
            prices: Current prices
            equity: Current equity
            
        Returns:
            True if within limits, False otherwise
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            exposure = abs(pos.quantity * prices.get(symbol, pos.avg_price))
        else:
            exposure = 0.0
        
        exposure_ratio = exposure / equity if equity > 0 else 0.0
        return exposure_ratio <= self.limits.max_position_pct
    
    def register_equity(self, equity: float) -> None:
        """
        Register current equity value for drawdown tracking.
        
        Args:
            equity: Current equity value
        """
        self._equity = equity
        self.equity_history.append(equity)
        self.last_update = datetime.utcnow()
    
    @property
    def equity(self) -> float:
        """Get current equity."""
        return self._equity
    
    def daily_drawdown_pct(self) -> float:
        """
        Calculate current daily drawdown percentage.
        
        Returns:
            Drawdown as percentage (0.0 = no drawdown)
        """
        if len(self.equity_history) < 2:
            return 0.0
        
        peak = max(self.equity_history)
        current = self.equity_history[-1]
        
        if peak <= 0:
            return 0.0
        
        return (peak - current) / peak
    
    def daily_drawdown_ok(self) -> bool:
        """
        Check if daily drawdown is within limits.
        
        Returns:
            True if drawdown is within limits
        """
        dd = self.daily_drawdown_pct()
        return dd <= self.limits.max_daily_drawdown_pct
    
    def get_metrics(self, prices: Dict[str, float]) -> RiskMetrics:
        """
        Calculate current risk metrics.
        
        Args:
            prices: Current market prices
            
        Returns:
            RiskMetrics snapshot
        """
        equity = self._equity if self._equity > 0 else 1.0
        exposure = self.total_exposure(prices)
        
        return RiskMetrics(
            total_exposure=exposure,
            equity=equity,
            exposure_pct=exposure / equity if equity > 0 else 0,
            daily_drawdown_pct=self.daily_drawdown_pct(),
            var_95=0.0,  # To be calculated by Monte Carlo
            cvar_95=0.0,  # To be calculated by Monte Carlo
            leverage=exposure / equity if equity > 0 else 0,
        )
    
    def get_positions_summary(self) -> List[Dict]:
        """
        Get summary of all positions.
        
        Returns:
            List of position dictionaries
        """
        return [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "side": pos.side,
                "value": pos.value,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
            }
            for pos in self.positions.values()
        ]
    
    def reset(self) -> None:
        """Reset all positions and equity history."""
        self.positions.clear()
        self.equity_history.clear()
        self._equity = 0.0
        self.last_update = None
    
    def __repr__(self) -> str:
        return (
            f"RiskBook(positions={len(self.positions)}, "
            f"equity={self._equity:.2f}, "
            f"limits={self.limits.max_position_pct:.1%})"
        )
