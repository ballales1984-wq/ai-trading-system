"""
Performance Monitor Module for AI Trading System
=================================================
Real-time tracking of trading performance metrics:
- P&L (Profit and Loss)
- Drawdown
- Win/Loss rate
- Sharpe/Sortino ratios
- Max consecutive losses
- Risk-adjusted returns

Author: AI Trading System
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'LONG' or 'SHORT'
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    status: str = 'OPEN'  # 'OPEN', 'CLOSED', 'CANCELLED'


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Win rate
    win_rate: float = 0.0
    loss_rate: float = 0.0
    
    # Average trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    current_drawdown: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0  # Positive for wins, negative for losses
    
    # Time-based
    trading_days: int = 0
    avg_daily_pnl: float = 0.0
    avg_weekly_pnl: float = 0.0
    avg_monthly_pnl: float = 0.0
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """
    Real-time Performance Monitor for Trading System.
    
    Tracks all trades and calculates comprehensive performance metrics.
    Supports:
    - Real-time P&L tracking
    - Drawdown monitoring
    - Risk-adjusted return calculations
    - Trade statistics
    - Performance alerts
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.02,
        max_history_days: int = 365
    ):
        """
        Initialize Performance Monitor.
        
        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            max_history_days: Maximum days of history to keep
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.max_history_days = max_history_days
        
        # Trade storage
        self.trades: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        
        # Daily returns for Sharpe/Sortino
        self.daily_returns: deque = deque(maxlen=max_history_days)
        self.daily_pnl: deque = deque(maxlen=max_history_days)
        self.equity_curve: deque = deque(maxlen=max_history_days * 24)  # Hourly
        
        # Current positions
        self.open_positions: Dict[str, TradeRecord] = {}
        
        # Metrics cache
        self._metrics: Optional[PerformanceMetrics] = None
        self._last_calculation: Optional[datetime] = None
        
        logger.info(f"PerformanceMonitor initialized with capital: {initial_capital:,.2f}")
    
    def open_trade(
        self,
        trade_id: str,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str,
        fees: float = 0.0
    ) -> TradeRecord:
        """
        Record a new trade opening.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position size
            side: 'LONG' or 'SHORT'
            fees: Entry fees
            
        Returns:
            TradeRecord
        """
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            side=side.upper(),
            fees=fees,
            status='OPEN'
        )
        
        self.trades[trade_id] = trade
        self.open_positions[trade_id] = trade
        
        logger.info(f"Trade opened: {trade_id} {side} {quantity} {symbol} @ {entry_price}")
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        fees: float = 0.0
    ) -> Optional[TradeRecord]:
        """
        Record a trade closing.
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            fees: Exit fees
            
        Returns:
            Updated TradeRecord or None if not found
        """
        if trade_id not in self.trades:
            logger.warning(f"Trade not found: {trade_id}")
            return None
        
        trade = self.trades[trade_id]
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.fees += fees
        trade.status = 'CLOSED'
        
        # Calculate P&L
        if trade.side == 'LONG':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity - trade.fees
        else:  # SHORT
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity - trade.fees
        
        trade.pnl_percent = trade.pnl / (trade.entry_price * trade.quantity)
        
        # Update capital
        self.current_capital += trade.pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Move to closed trades
        self.closed_trades.append(trade)
        if trade_id in self.open_positions:
            del self.open_positions[trade_id]
        
        # Record daily P&L
        self._record_daily_pnl(trade.pnl)
        
        logger.info(f"Trade closed: {trade_id} P&L: {trade.pnl:,.2f} ({trade.pnl_percent:.2%})")
        
        # Invalidate cache
        self._metrics = None
        
        return trade
    
    def update_unrealized_pnl(self, positions: Dict[str, Dict]):
        """
        Update unrealized P&L for open positions.
        
        Args:
            positions: Dict of {symbol: {'current_price': float, 'quantity': float}}
        """
        unrealized = 0.0
        
        for trade_id, trade in self.open_positions.items():
            if trade.symbol in positions:
                current_price = positions[trade.symbol].get('current_price', trade.entry_price)
                
                if trade.side == 'LONG':
                    pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    pnl = (trade.entry_price - current_price) * trade.quantity
                
                unrealized += pnl
        
        return unrealized
    
    def _record_daily_pnl(self, pnl: float):
        """Record daily P&L for metrics calculation."""
        today = datetime.now().date()
        
        if self.daily_pnl and self.daily_pnl[-1][0] == today:
            # Add to existing day
            self.daily_pnl[-1] = (today, self.daily_pnl[-1][1] + pnl)
        else:
            # New day
            self.daily_pnl.append((today, pnl))
        
        # Record equity
        self.equity_curve.append((datetime.now(), self.current_capital))
    
    def calculate_metrics(self, force: bool = False) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            force: Force recalculation even if cached
            
        Returns:
            PerformanceMetrics
        """
        if self._metrics and not force:
            if self._last_calculation and (datetime.now() - self._last_calculation).seconds < 60:
                return self._metrics
        
        metrics = PerformanceMetrics()
        
        # Basic counts
        metrics.total_trades = len(self.closed_trades)
        metrics.winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        metrics.losing_trades = sum(1 for t in self.closed_trades if t.pnl < 0)
        
        # P&L
        metrics.realized_pnl = sum(t.pnl for t in self.closed_trades)
        metrics.total_pnl = metrics.realized_pnl
        metrics.total_pnl_percent = metrics.total_pnl / self.initial_capital
        
        # Win/Loss rates
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
            metrics.loss_rate = metrics.losing_trades / metrics.total_trades
        
        # Average trade
        wins = [t.pnl for t in self.closed_trades if t.pnl > 0]
        losses = [t.pnl for t in self.closed_trades if t.pnl < 0]
        
        if wins:
            metrics.avg_win = np.mean(wins)
        if losses:
            metrics.avg_loss = np.mean(losses)
        if self.closed_trades:
            metrics.avg_trade = np.mean([t.pnl for t in self.closed_trades])
        
        # Drawdown
        metrics.max_drawdown, metrics.max_drawdown_percent = self._calculate_max_drawdown()
        metrics.current_drawdown = self.peak_capital - self.current_capital
        
        # Risk-adjusted returns
        metrics.sharpe_ratio = self._calculate_sharpe_ratio()
        metrics.sortino_ratio = self._calculate_sortino_ratio()
        metrics.calmar_ratio = self._calculate_calmar_ratio(metrics.max_drawdown_percent)
        
        # Streaks
        metrics.max_consecutive_wins, metrics.max_consecutive_losses, metrics.current_streak = \
            self._calculate_streaks()
        
        # Time-based metrics
        if self.daily_pnl:
            metrics.trading_days = len(set(date for date, _ in self.daily_pnl))
            total_daily_pnl = sum(pnl for _, pnl in self.daily_pnl)
            metrics.avg_daily_pnl = total_daily_pnl / max(metrics.trading_days, 1)
            metrics.avg_weekly_pnl = metrics.avg_daily_pnl * 5
            metrics.avg_monthly_pnl = metrics.avg_daily_pnl * 20
        
        metrics.last_updated = datetime.now()
        
        self._metrics = metrics
        self._last_calculation = datetime.now()
        
        return metrics
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0
        
        equity_values = [e for _, e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        max_dd_percent = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            
            dd = peak - equity
            dd_percent = dd / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_percent = dd_percent
        
        return max_dd, max_dd_percent
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.daily_pnl) < 2:
            return 0.0
        
        returns = [pnl / self.initial_capital for _, pnl in self.daily_pnl]
        
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (252 trading days)
        daily_rf = self.risk_free_rate / 252
        sharpe = (avg_return - daily_rf) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate annualized Sortino ratio."""
        if len(self.daily_pnl) < 2:
            return 0.0
        
        returns = [pnl / self.initial_capital for _, pnl in self.daily_pnl]
        
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        
        # Downside deviation
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        daily_rf = self.risk_free_rate / 252
        sortino = (avg_return - daily_rf) / downside_std * np.sqrt(252)
        
        return sortino
    
    def _calculate_calmar_ratio(self, max_dd_percent: float) -> float:
        """Calculate Calmar ratio."""
        if max_dd_percent == 0:
            return 0.0
        
        annual_return = self._calculate_annual_return()
        calmar = annual_return / max_dd_percent
        
        return calmar
    
    def _calculate_annual_return(self) -> float:
        """Calculate annualized return."""
        if not self.daily_pnl:
            return 0.0
        
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Annualize based on trading days
        trading_days = len(set(date for date, _ in self.daily_pnl))
        
        if trading_days == 0:
            return 0.0
        
        annual_return = total_return * (252 / trading_days)
        
        return annual_return
    
    def _calculate_streaks(self) -> Tuple[int, int, int]:
        """Calculate win/loss streaks."""
        if not self.closed_trades:
            return 0, 0, 0
        
        max_wins = 0
        max_losses = 0
        current_streak = 0
        
        for trade in self.closed_trades:
            if trade.pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_wins = max(max_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_losses = max(max_losses, abs(current_streak))
        
        return max_wins, max_losses, current_streak
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        data = [(t, e) for t, e in self.equity_curve]
        df = pd.DataFrame(data, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        trades = self.closed_trades[-limit:]
        
        if not trades:
            return pd.DataFrame()
        
        data = [{
            'trade_id': t.trade_id,
            'symbol': t.symbol,
            'side': t.side,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'status': t.status
        } for t in trades]
        
        return pd.DataFrame(data)
    
    def get_summary(self) -> Dict:
        """Get performance summary as dictionary."""
        metrics = self.calculate_metrics()
        
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'total_pnl': metrics.total_pnl,
                'total_pnl_percent': f"{metrics.total_pnl_percent:.2%}"
            },
            'trades': {
                'total': metrics.total_trades,
                'winning': metrics.winning_trades,
                'losing': metrics.losing_trades,
                'win_rate': f"{metrics.win_rate:.2%}",
                'open_positions': len(self.open_positions)
            },
            'averages': {
                'avg_win': f"{metrics.avg_win:,.2f}",
                'avg_loss': f"{metrics.avg_loss:,.2f}",
                'avg_trade': f"{metrics.avg_trade:,.2f}"
            },
            'risk': {
                'max_drawdown': f"{metrics.max_drawdown:,.2f}",
                'max_drawdown_percent': f"{metrics.max_drawdown_percent:.2%}",
                'current_drawdown': f"{metrics.current_drawdown:,.2f}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}"
            },
            'streaks': {
                'max_consecutive_wins': metrics.max_consecutive_wins,
                'max_consecutive_losses': metrics.max_consecutive_losses,
                'current_streak': metrics.current_streak
            },
            'daily': {
                'trading_days': metrics.trading_days,
                'avg_daily_pnl': f"{metrics.avg_daily_pnl:,.2f}",
                'avg_weekly_pnl': f"{metrics.avg_weekly_pnl:,.2f}",
                'avg_monthly_pnl': f"{metrics.avg_monthly_pnl:,.2f}"
            }
        }
    
    def to_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.get_summary(), indent=2, default=str)


# Singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(initial_capital: float = 100000.0) -> PerformanceMonitor:
    """Get or create PerformanceMonitor singleton."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(initial_capital=initial_capital)
    
    return _performance_monitor
