"""
Risk Guard Module for AI Trading System
========================================
Automatic risk monitoring and trading halt system.

Features:
- Real-time risk monitoring
- Automatic trading halt on threshold breach
- Multiple risk levels (Warning, Critical, Emergency)
- Configurable thresholds
- Integration with Performance Monitor

Author: AI Trading System
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for trading."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3
    
    def __str__(self):
        return self.name


class TradingStatus(Enum):
    """Trading status."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    LOCKED = "LOCKED"


@dataclass
class RiskThresholds:
    """Risk threshold configuration."""
    # Drawdown thresholds
    max_drawdown_percent: float = 0.20  # 20% max drawdown
    warning_drawdown_percent: float = 0.10  # 10% warning
    critical_drawdown_percent: float = 0.15  # 15% critical
    
    # Daily loss limits
    max_daily_loss_percent: float = 0.05  # 5% max daily loss
    warning_daily_loss_percent: float = 0.03  # 3% warning
    
    # Consecutive losses
    max_consecutive_losses: int = 10
    warning_consecutive_losses: int = 5
    
    # Win rate minimum
    min_win_rate: float = 0.30  # 30% minimum win rate
    
    # Sharpe ratio minimum
    min_sharpe_ratio: float = 0.0
    
    # Position limits
    max_position_size_percent: float = 0.10  # 10% per position
    max_total_exposure_percent: float = 0.80  # 80% total capital
    
    # Time-based limits
    max_trades_per_hour: int = 20
    max_trades_per_day: int = 100


@dataclass
class RiskAlert:
    """Risk alert data."""
    timestamp: datetime
    level: RiskLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    action_taken: str = ""
    acknowledged: bool = False


class RiskGuard:
    """
    Risk Guard for automatic trading protection.
    
    Monitors trading performance and automatically halts trading
    when risk thresholds are breached.
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskThresholds] = None,
        performance_monitor: Optional[Any] = None
    ):
        """
        Initialize Risk Guard.
        
        Args:
            thresholds: Risk threshold configuration
            performance_monitor: PerformanceMonitor instance
        """
        self.thresholds = thresholds or RiskThresholds()
        self.performance_monitor = performance_monitor
        
        # State
        self.status = TradingStatus.ACTIVE
        self.risk_level = RiskLevel.NORMAL
        self.alerts: List[RiskAlert] = []
        self.active_alerts: List[RiskAlert] = []
        
        # Callbacks
        self._halt_callbacks: List[Callable] = []
        self._resume_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Tracking
        self._trades_today: int = 0
        self._trades_this_hour: int = 0
        self._last_hour_reset: datetime = datetime.now()
        self._last_day_reset: datetime = datetime.now()
        self._halt_reason: str = ""
        self._halt_time: Optional[datetime] = None
        
        # Cooldown
        self._cooldown_until: Optional[datetime] = None
        
        logger.info("RiskGuard initialized with thresholds")
    
    def register_halt_callback(self, callback: Callable):
        """Register callback for trading halt."""
        self._halt_callbacks.append(callback)
    
    def register_resume_callback(self, callback: Callable):
        """Register callback for trading resume."""
        self._resume_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for risk alerts."""
        self._alert_callbacks.append(callback)
    
    def check_risk(self, metrics: Optional[Dict] = None) -> RiskLevel:
        """
        Check current risk level based on metrics.
        
        Args:
            metrics: Performance metrics dict (optional, uses monitor if not provided)
            
        Returns:
            Current RiskLevel
        """
        # Reset time-based counters
        self._reset_counters()
        
        # Get metrics from performance monitor if not provided
        if metrics is None and self.performance_monitor:
            metrics = self.performance_monitor.get_summary()
        
        if not metrics:
            return RiskLevel.NORMAL
        
        risk_level = RiskLevel.NORMAL
        alerts_to_fire = []
        
        # Check drawdown
        dd_risk, dd_alerts = self._check_drawdown(metrics)
        if dd_risk.value > risk_level.value:
            risk_level = dd_risk
        alerts_to_fire.extend(dd_alerts)
        
        # Check daily loss
        daily_risk, daily_alerts = self._check_daily_loss(metrics)
        if daily_risk.value > risk_level.value:
            risk_level = daily_risk
        alerts_to_fire.extend(daily_alerts)
        
        # Check consecutive losses
        streak_risk, streak_alerts = self._check_consecutive_losses(metrics)
        if streak_risk.value > risk_level.value:
            risk_level = streak_risk
        alerts_to_fire.extend(streak_alerts)
        
        # Check win rate
        wr_risk, wr_alerts = self._check_win_rate(metrics)
        if wr_risk.value > risk_level.value:
            risk_level = wr_risk
        alerts_to_fire.extend(wr_alerts)
        
        # Check Sharpe ratio
        sharpe_risk, sharpe_alerts = self._check_sharpe_ratio(metrics)
        if sharpe_risk.value > risk_level.value:
            risk_level = sharpe_risk
        alerts_to_fire.extend(sharpe_alerts)
        
        # Fire alerts
        for alert in alerts_to_fire:
            self._fire_alert(alert)
        
        # Update risk level
        self.risk_level = risk_level
        
        # Take action based on risk level
        self._take_action(risk_level)
        
        # Return the highest risk level found
        return risk_level
    
    def _reset_counters(self):
        """Reset time-based counters."""
        now = datetime.now()
        
        # Reset hourly counter
        if (now - self._last_hour_reset) >= timedelta(hours=1):
            self._trades_this_hour = 0
            self._last_hour_reset = now
        
        # Reset daily counter
        if now.date() != self._last_day_reset.date():
            self._trades_today = 0
            self._last_day_reset = now
    
    def _check_drawdown(self, metrics: Dict) -> tuple:
        """Check drawdown risk."""
        alerts = []
        level = RiskLevel.NORMAL
        
        try:
            dd_str = metrics.get('risk', {}).get('max_drawdown_percent', '0%')
            dd_percent = float(dd_str.replace('%', '')) / 100
        except (ValueError, TypeError):
            dd_percent = 0.0
        
        if dd_percent >= self.thresholds.max_drawdown_percent:
            level = RiskLevel.EMERGENCY
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.EMERGENCY,
                message=f"Maximum drawdown exceeded: {dd_percent:.2%} >= {self.thresholds.max_drawdown_percent:.2%}",
                metric_name="drawdown_percent",
                current_value=dd_percent,
                threshold_value=self.thresholds.max_drawdown_percent
            ))
        elif dd_percent >= self.thresholds.critical_drawdown_percent:
            level = RiskLevel.CRITICAL
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.CRITICAL,
                message=f"Critical drawdown: {dd_percent:.2%}",
                metric_name="drawdown_percent",
                current_value=dd_percent,
                threshold_value=self.thresholds.critical_drawdown_percent
            ))
        elif dd_percent >= self.thresholds.warning_drawdown_percent:
            level = RiskLevel.WARNING
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.WARNING,
                message=f"Warning: Drawdown at {dd_percent:.2%}",
                metric_name="drawdown_percent",
                current_value=dd_percent,
                threshold_value=self.thresholds.warning_drawdown_percent
            ))
        
        return level, alerts
    
    def _check_daily_loss(self, metrics: Dict) -> tuple:
        """Check daily loss risk."""
        alerts = []
        level = RiskLevel.NORMAL
        
        try:
            daily_pnl_str = metrics.get('daily', {}).get('avg_daily_pnl', '0')
            daily_pnl = float(daily_pnl_str.replace(',', ''))
        except (ValueError, TypeError):
            daily_pnl = 0.0
        
        # Calculate daily loss percentage
        if self.performance_monitor:
            initial_capital = self.performance_monitor.initial_capital
        else:
            initial_capital = 100000.0
        
        daily_loss_percent = abs(daily_pnl) / initial_capital if daily_pnl < 0 else 0.0
        
        if daily_loss_percent >= self.thresholds.max_daily_loss_percent:
            level = RiskLevel.EMERGENCY
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.EMERGENCY,
                message=f"Max daily loss exceeded: {daily_loss_percent:.2%}",
                metric_name="daily_loss_percent",
                current_value=daily_loss_percent,
                threshold_value=self.thresholds.max_daily_loss_percent
            ))
        elif daily_loss_percent >= self.thresholds.warning_daily_loss_percent:
            level = RiskLevel.WARNING
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.WARNING,
                message=f"Warning: Daily loss at {daily_loss_percent:.2%}",
                metric_name="daily_loss_percent",
                current_value=daily_loss_percent,
                threshold_value=self.thresholds.warning_daily_loss_percent
            ))
        
        return level, alerts
    
    def _check_consecutive_losses(self, metrics: Dict) -> tuple:
        """Check consecutive losses risk."""
        alerts = []
        level = RiskLevel.NORMAL
        
        streaks = metrics.get('streaks', {})
        consecutive_losses = abs(streaks.get('current_streak', 0)) if streaks.get('current_streak', 0) < 0 else 0
        max_consecutive_losses = streaks.get('max_consecutive_losses', 0)
        
        if consecutive_losses >= self.thresholds.max_consecutive_losses:
            level = RiskLevel.EMERGENCY
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.EMERGENCY,
                message=f"Max consecutive losses: {consecutive_losses}",
                metric_name="consecutive_losses",
                current_value=consecutive_losses,
                threshold_value=self.thresholds.max_consecutive_losses
            ))
        elif consecutive_losses >= self.thresholds.warning_consecutive_losses:
            level = RiskLevel.WARNING
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.WARNING,
                message=f"Warning: {consecutive_losses} consecutive losses",
                metric_name="consecutive_losses",
                current_value=consecutive_losses,
                threshold_value=self.thresholds.warning_consecutive_losses
            ))
        
        return level, alerts
    
    def _check_win_rate(self, metrics: Dict) -> tuple:
        """Check win rate risk."""
        alerts = []
        level = RiskLevel.NORMAL
        
        try:
            wr_str = metrics.get('trades', {}).get('win_rate', '100%')
            win_rate = float(wr_str.replace('%', '')) / 100
        except (ValueError, TypeError):
            win_rate = 1.0
        
        total_trades = metrics.get('trades', {}).get('total', 0)
        
        # Only check if we have enough trades
        if total_trades < 10:
            return level, alerts
        
        if win_rate < self.thresholds.min_win_rate:
            level = RiskLevel.CRITICAL
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.CRITICAL,
                message=f"Win rate below minimum: {win_rate:.2%}",
                metric_name="win_rate",
                current_value=win_rate,
                threshold_value=self.thresholds.min_win_rate
            ))
        
        return level, alerts
    
    def _check_sharpe_ratio(self, metrics: Dict) -> tuple:
        """Check Sharpe ratio risk."""
        alerts = []
        level = RiskLevel.NORMAL
        
        try:
            sharpe_str = metrics.get('risk', {}).get('sharpe_ratio', '0')
            sharpe_ratio = float(sharpe_str)
        except (ValueError, TypeError):
            sharpe_ratio = 0.0
        
        trading_days = metrics.get('daily', {}).get('trading_days', 0)
        
        # Only check if we have enough data
        if trading_days < 20:
            return level, alerts
        
        if sharpe_ratio < self.thresholds.min_sharpe_ratio:
            level = RiskLevel.WARNING
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                level=RiskLevel.WARNING,
                message=f"Sharpe ratio below minimum: {sharpe_ratio:.2f}",
                metric_name="sharpe_ratio",
                current_value=sharpe_ratio,
                threshold_value=self.thresholds.min_sharpe_ratio
            ))
        
        return level, alerts
    
    def _fire_alert(self, alert: RiskAlert):
        """Fire a risk alert."""
        self.alerts.append(alert)
        self.active_alerts.append(alert)
        
        logger.warning(f"RISK ALERT [{alert.level.name}]: {alert.message}")
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _take_action(self, risk_level: RiskLevel):
        """Take action based on risk level."""
        if risk_level == RiskLevel.EMERGENCY:
            self._halt_trading("Emergency risk level reached")
        elif risk_level == RiskLevel.CRITICAL:
            self._pause_trading("Critical risk level reached")
        elif risk_level == RiskLevel.WARNING:
            logger.warning("Risk warning - monitoring closely")
        else:
            # Resume if was paused
            if self.status == TradingStatus.PAUSED:
                self._resume_trading()
    
    def _halt_trading(self, reason: str):
        """Halt all trading."""
        if self.status == TradingStatus.HALTED:
            return
        
        logger.critical(f"HALTING TRADING: {reason}")
        
        self.status = TradingStatus.HALTED
        self._halt_reason = reason
        self._halt_time = datetime.now()
        
        # Call halt callbacks
        for callback in self._halt_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Halt callback error: {e}")
    
    def _pause_trading(self, reason: str):
        """Pause trading temporarily."""
        if self.status in [TradingStatus.HALTED, TradingStatus.PAUSED]:
            return
        
        logger.warning(f"PAUSING TRADING: {reason}")
        
        self.status = TradingStatus.PAUSED
        self._halt_reason = reason
        self._halt_time = datetime.now()
        
        # Call halt callbacks
        for callback in self._halt_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Pause callback error: {e}")
    
    def _resume_trading(self):
        """Resume trading."""
        if self.status == TradingStatus.ACTIVE:
            return
        
        logger.info("RESUMING TRADING")
        
        self.status = TradingStatus.ACTIVE
        self._halt_reason = ""
        self._halt_time = None
        
        # Call resume callbacks
        for callback in self._resume_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Resume callback error: {e}")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self.status in [TradingStatus.HALTED, TradingStatus.LOCKED]:
            return False
        
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            return False
        
        # Check trade frequency
        if self._trades_this_hour >= self.thresholds.max_trades_per_hour:
            return False
        
        if self._trades_today >= self.thresholds.max_trades_per_day:
            return False
        
        return True
    
    def record_trade(self):
        """Record a trade for frequency tracking."""
        self._trades_this_hour += 1
        self._trades_today += 1
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert."""
        if 0 <= alert_index < len(self.active_alerts):
            self.active_alerts[alert_index].acknowledged = True
            return True
        return False
    
    def clear_acknowledged_alerts(self):
        """Clear acknowledged alerts."""
        self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]
    
    def manual_halt(self, reason: str = "Manual halt"):
        """Manually halt trading."""
        self._halt_trading(reason)
    
    def manual_resume(self):
        """Manually resume trading."""
        if self.status == TradingStatus.HALTED:
            logger.warning("Cannot resume from HALTED state manually - use unlock()")
            return
        self._resume_trading()
    
    def unlock(self):
        """Unlock from HALTED state."""
        logger.info("Unlocking trading system")
        self.status = TradingStatus.ACTIVE
        self._halt_reason = ""
        self._halt_time = None
        self.risk_level = RiskLevel.NORMAL
        
        # Call resume callbacks
        for callback in self._resume_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Resume callback error: {e}")
    
    def set_cooldown(self, minutes: int):
        """Set a cooldown period."""
        self._cooldown_until = datetime.now() + timedelta(minutes=minutes)
        logger.info(f"Cooldown set for {minutes} minutes")
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            'status': self.status.name,
            'risk_level': self.risk_level.name,
            'can_trade': self.can_trade(),
            'halt_reason': self._halt_reason,
            'halt_time': self._halt_time.isoformat() if self._halt_time else None,
            'cooldown_until': self._cooldown_until.isoformat() if self._cooldown_until else None,
            'trades_today': self._trades_today,
            'trades_this_hour': self._trades_this_hour,
            'active_alerts': len(self.active_alerts),
            'total_alerts': len(self.alerts)
        }
    
    def get_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        alerts = self.alerts[-limit:]
        return [{
            'timestamp': a.timestamp.isoformat(),
            'level': a.level.name,
            'message': a.message,
            'metric': a.metric_name,
            'current_value': a.current_value,
            'threshold': a.threshold_value,
            'acknowledged': a.acknowledged
        } for a in alerts]
    
    def to_json(self) -> str:
        """Export status as JSON."""
        return json.dumps(self.get_status(), indent=2)


# Singleton instance
_risk_guard: Optional[RiskGuard] = None


def get_risk_guard(
    thresholds: Optional[RiskThresholds] = None,
    performance_monitor: Optional[Any] = None
) -> RiskGuard:
    """Get or create RiskGuard singleton."""
    global _risk_guard
    
    if _risk_guard is None:
        _risk_guard = RiskGuard(thresholds=thresholds, performance_monitor=performance_monitor)
    
    return _risk_guard
