# src/core/risk/risk_engine.py
"""
Risk Engine
===========
Unified risk management engine with position limits, 
drawdown protection, and emergency stop mechanisms.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCheckResult:
    """Result of risk check."""
    
    def __init__(self, passed: bool, reason: str = "", level: RiskLevel = RiskLevel.LOW):
        self.passed = passed
        self.reason = reason
        self.level = level
    
    def __bool__(self):
        return self.passed
    
    def to_dict(self) -> Dict:
        return {
            'passed': self.passed,
            'reason': self.reason,
            'level': self.level.value
        }


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    # Position limits
    max_position_pct: float = 0.3  # Max 30% in single position
    max_total_exposure_pct: float = 1.0  # Max 100% total exposure
    
    # Order limits
    max_order_pct: float = 0.1  # Max 10% per order
    max_orders_per_day: int = 50
    
    # Loss limits
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_weekly_loss_pct: float = 0.15  # Max 15% weekly loss
    max_drawdown_pct: float = 0.20  # Max 20% drawdown
    
    # Leverage
    max_leverage: float = 3.0
    
    # Trading hours
    allowed_start_hour: int = 0  # 00:00 UTC
    allowed_end_hour: int = 23  # 23:00 UTC
    
    # Emergency
    emergency_stop_pct: float = 0.25  # Stop at 25% loss


@dataclass
class RiskState:
    """Current risk state."""
    current_exposure: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    drawdown: float = 0.0
    orders_today: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'current_exposure': self.current_exposure,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'drawdown': self.drawdown,
            'orders_today': self.orders_today,
            'last_reset': self.last_reset.isoformat()
        }


class RiskEngine:
    """
    Risk management engine for trading.
    Validates orders, monitors exposure, and triggers emergency stops.
    """
    
    def __init__(
        self,
        initial_balance: float = 100000,
        limits: Optional[RiskLimits] = None
    ):
        """
        Initialize risk engine.
        
        Args:
            initial_balance: Starting balance
            limits: Risk limits configuration
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.limits = limits or RiskLimits()
        
        # State tracking
        self.state = RiskState()
        self.daily_trades: List[Dict] = []
        self.weekly_trades: List[Dict] = []
        
        # Emergency state
        self.emergency_stop = False
        self.pause_trading = False
        
        # Callbacks
        self.on_risk_alert: Optional[callable] = None
        self.on_emergency_stop: Optional[callable] = None
        
        logger.info(f"Risk engine initialized with balance: {initial_balance}")
    
    def check_signal(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if signal passes risk validation.
        
        Args:
            signal: Signal dictionary with symbol, action, quantity, price
            
        Returns:
            (passed, reason)
        """
        if self.emergency_stop:
            return False, "Emergency stop active"
        
        if self.pause_trading:
            return False, "Trading paused"
        
        symbol = signal.get('symbol')
        action = signal.get('action', '').upper()
        quantity = signal.get('quantity', 0)
        price = signal.get('price', 0)
        
        # Check trading hours
        if not self._check_trading_hours():
            return False, "Outside trading hours"
        
        # Check daily order limit
        if self.state.orders_today >= self.limits.max_orders_per_day:
            return False, f"Daily order limit reached ({self.limits.max_orders_per_day})"
        
        # Calculate order value
        order_value = quantity * price
        order_pct = order_value / self.balance
        
        # Check order size
        if order_pct > self.limits.max_order_pct:
            return False, f"Order too large: {order_pct:.2%} > {self.limits.max_order_pct:.2%}"
        
        # Check position limit
        position = signal.get('position', 0)
        new_position = position + quantity if action == 'BUY' else position - quantity
        position_pct = (new_position * price) / self.balance
        
        if position_pct > self.limits.max_position_pct:
            return False, f"Position limit exceeded: {position_pct:.2%} > {self.limits.max_position_pct:.2%}"
        
        # Check total exposure
        total_exposure = self._calculate_total_exposure()
        new_exposure = total_exposure + (quantity * price if action == 'BUY' else 0)
        exposure_pct = new_exposure / self.balance
        
        if exposure_pct > self.limits.max_total_exposure_pct:
            return False, f"Total exposure exceeded: {exposure_pct:.2%}"
        
        # Check leverage
        leverage = signal.get('leverage', 1.0)
        if leverage > self.limits.max_leverage:
            return False, f"Leverage too high: {leverage}x > {self.limits.max_leverage}x"
        
        return True, ""
    
    def check_order(self, order: Dict) -> RiskCheckResult:
        """
        Comprehensive order risk check.
        
        Args:
            order: Order dictionary
            
        Returns:
            RiskCheckResult
        """
        # Check emergency
        if self.emergency_stop:
            return RiskCheckResult(
                False,
                "Emergency stop active",
                RiskLevel.CRITICAL
            )
        
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        side = order.get('side', '').upper()
        
        # Validate order parameters
        if quantity <= 0:
            return RiskCheckResult(False, "Invalid quantity", RiskLevel.HIGH)
        
        if price <= 0:
            return RiskCheckResult(False, "Invalid price", RiskLevel.HIGH)
        
        # Check order value
        order_value = quantity * price
        order_pct = order_value / self.balance
        
        if order_pct > self.limits.max_order_pct:
            return RiskCheckResult(
                False,
                f"Order size {order_pct:.2%} exceeds limit {self.limits.max_order_pct:.2%}",
                RiskLevel.HIGH
            )
        
        # Check daily loss
        if abs(self.state.daily_pnl) / self.balance >= self.limits.max_daily_loss_pct:
            return RiskCheckResult(
                False,
                f"Daily loss {abs(self.state.daily_pnl)/self.balance:.2%} exceeds limit",
                RiskLevel.CRITICAL
            )
        
        # Check drawdown
        if self.state.drawdown >= self.limits.max_drawdown_pct:
            return RiskCheckResult(
                False,
                f"Drawdown {self.state.drawdown:.2%} exceeds limit",
                RiskLevel.CRITICAL
            )
        
        return RiskCheckResult(True, "Order approved", RiskLevel.LOW)
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ):
        """Update position after trade."""
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'side': side,
            'value': quantity * price,
            'timestamp': datetime.now()
        }
        
        self.daily_trades.append(trade)
        self.weekly_trades.append(trade)
        
        # Update order count
        self.state.orders_today += 1
        
        logger.debug(f"Position updated: {symbol} {side} {quantity}")
    
    def update_pnl(self, pnl: float):
        """Update PnL and check limits."""
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        
        # Check daily loss
        daily_loss_pct = abs(self.state.daily_pnl) / self.balance
        
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            self._trigger_alert(
                f"Daily loss limit reached: {daily_loss_pct:.2%}",
                RiskLevel.HIGH
            )
        
        # Check emergency stop
        if daily_loss_pct >= self.limits.emergency_stop_pct:
            self._trigger_emergency_stop(
                f"Emergency stop: {daily_loss_pct:.2%} daily loss"
            )
    
    def update_balance(self, new_balance: float):
        """Update balance and recalculate drawdown."""
        old_balance = self.balance
        self.balance = new_balance
        
        # Update drawdown
        if old_balance > 0:
            self.state.drawdown = max(
                0,
                (self.initial_balance - new_balance) / self.initial_balance
            )
            
            if self.state.drawdown >= self.limits.max_drawdown_pct:
                self._trigger_alert(
                    f"Max drawdown reached: {self.state.drawdown:.2%}",
                    RiskLevel.CRITICAL
                )
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total exposure from daily trades."""
        return sum(t['value'] for t in self.daily_trades if t['side'] == 'BUY')
    
    def _check_trading_hours(self) -> bool:
        """Check if within allowed trading hours."""
        now = datetime.utcnow()
        current_hour = now.hour
        
        return self.limits.allowed_start_hour <= current_hour <= self.limits.allowed_end_hour
    
    def _trigger_alert(self, message: str, level: RiskLevel):
        """Trigger risk alert."""
        logger.warning(f"Risk Alert [{level.value}]: {message}")
        
        if self.on_risk_alert:
            self.on_risk_alert({
                'message': message,
                'level': level.value,
                'state': self.state.to_dict()
            })
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop."""
        logger.critical(f"EMERGENCY STOP: {reason}")
        
        self.emergency_stop = True
        
        if self.on_emergency_stop:
            self.on_emergency_stop({
                'reason': reason,
                'state': self.state.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
    
    def reset_daily(self):
        """Reset daily counters."""
        self.state.daily_pnl = 0
        self.state.orders_today = 0
        self.daily_trades.clear()
        self.state.last_reset = datetime.now()
        
        logger.info("Daily risk counters reset")
    
    def reset_weekly(self):
        """Reset weekly counters."""
        self.state.weekly_pnl = 0
        self.weekly_trades.clear()
        
        logger.info("Weekly risk counters reset")
    
    def reset_emergency(self):
        """Reset emergency stop (manual override)."""
        self.emergency_stop = False
        logger.info("Emergency stop reset (manual override)")
    
    def pause(self):
        """Pause trading."""
        self.pause_trading = True
        logger.warning("Trading paused")
    
    def resume(self):
        """Resume trading."""
        self.pause_trading = False
        logger.info("Trading resumed")
    
    def get_status(self) -> Dict:
        """Get current risk status."""
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'state': self.state.to_dict(),
            'limits': {
                'max_position_pct': self.limits.max_position_pct,
                'max_daily_loss_pct': self.limits.max_daily_loss_pct,
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'max_leverage': self.limits.max_leverage
            },
            'emergency_stop': self.emergency_stop,
            'paused': self.pause_trading,
            'risk_level': self._calculate_current_risk_level().value
        }
    
    def _calculate_current_risk_level(self) -> RiskLevel:
        """Calculate current risk level based on state."""
        # Check drawdown
        if self.state.drawdown >= self.limits.max_drawdown_pct * 0.9:
            return RiskLevel.CRITICAL
        
        if self.state.drawdown >= self.limits.max_drawdown_pct * 0.7:
            return RiskLevel.HIGH
        
        # Check daily loss
        daily_loss_pct = abs(self.state.daily_pnl) / self.balance
        
        if daily_loss_pct >= self.limits.max_daily_loss_pct * 0.9:
            return RiskLevel.CRITICAL
        
        if daily_loss_pct >= self.limits.max_daily_loss_pct * 0.7:
            return RiskLevel.HIGH
        
        # Check exposure
        exposure_pct = self._calculate_total_exposure() / self.balance
        
        if exposure_pct >= self.limits.max_total_exposure_pct * 0.9:
            return RiskLevel.HIGH
        
        if exposure_pct >= self.limits.max_total_exposure_pct * 0.7:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def get_available_balance(self) -> float:
        """Get available balance for trading."""
        # Subtract reserved margin
        exposure = self._calculate_total_exposure()
        return max(0, self.balance - exposure)
    
    def get_max_position_size(self, price: float, existing: float = 0) -> float:
        """Get maximum position size for symbol."""
        max_by_position = (self.balance * self.limits.max_position_pct - existing * price) / price
        max_by_exposure = (self.balance * self.limits.max_total_exposure_pct) / price
        max_by_balance = self.get_available_balance() / price
        
        return min(max_by_position, max_by_exposure, max_by_balance)
    
    def to_dict(self) -> Dict:
        """Export risk engine state."""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'state': self.state.to_dict(),
            'emergency_stop': self.emergency_stop,
            'paused': self.pause_trading,
            'status': self.get_status()
        }
    
    def save_config(self, path: str):
        """Save risk configuration."""
        config = {
            'initial_balance': self.initial_balance,
            'limits': {
                'max_position_pct': self.limits.max_position_pct,
                'max_total_exposure_pct': self.limits.max_total_exposure_pct,
                'max_order_pct': self.limits.max_order_pct,
                'max_orders_per_day': self.limits.max_orders_per_day,
                'max_daily_loss_pct': self.limits.max_daily_loss_pct,
                'max_weekly_loss_pct': self.limits.max_weekly_loss_pct,
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'max_leverage': self.limits.max_leverage,
                'emergency_stop_pct': self.limits.emergency_stop_pct
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Risk config saved to {path}")
    
    @classmethod
    def from_config(cls, config: Dict) -> 'RiskEngine':
        """Create risk engine from config."""
        limits = RiskLimits(**config.get('limits', {}))
        return cls(
            initial_balance=config.get('initial_balance', 100000),
            limits=limits
        )
