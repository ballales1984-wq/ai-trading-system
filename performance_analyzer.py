"""
Performance Analysis Module
=========================
Comprehensive performance metrics for trading systems.

Provides:
- Sharpe ratio calculation
- Maximum drawdown tracking
- Sortino ratio
- Trade logging to CSV
- Performance summary

Usage:
    from performance_analyzer import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer(initial_balance=1000)
    
    # Log each trade
    analyzer.log_trade(timestamp, symbol, action, price, quantity, pnl)
    
    # Log equity
    analyzer.log_equity(timestamp, equity)
    
    # Get final metrics
    metrics = analyzer.get_metrics()
    
    # Save to CSV
    analyzer.save_to_csv("trades_log.csv")
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TradeRecord:
    """Single trade record for logging."""
    timestamp: datetime
    symbol: str
    action: str  # BUY/SELL
    price: float
    quantity: float
    pnl: float
    fees: float
    balance: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    initial_balance: float
    final_balance: float
    total_pnl: float
    pnl_percent: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_percent: float
    
    # Return metrics
    sharpe_ratio: float
    sortino_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Average metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Timing
    start_time: datetime
    end_time: datetime
    duration_days: float


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading systems.
    
    Tracks all metrics needed for hedge fund-level evaluation.
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
    ):
        """
        Initialize the analyzer.
        
        Args:
            initial_balance: Starting balance
            risk_free_rate: Annual risk-free rate for Sharpe calc
            trading_days: Trading days per year (default 252)
        """
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        
        # Data storage
        self.trades: List[TradeRecord] = []
        self.equity_history: List[float] = []
        self.timestamps: List[datetime] = []
        
        # State
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.peak_balance: float = initial_balance
        self.min_balance: float = initial_balance
    
    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        pnl: float,
        fees: float = 0.0,
        balance: float = 0.0,
    ) -> None:
        """
        Log a trade.
        
        Args:
            timestamp: Trade timestamp
            symbol: Trading symbol
            action: BUY or SELL
            price: Execution price
            quantity: Trade quantity
            pnl: Profit/loss (for sells)
            fees: Total fees paid
            balance: Balance after trade
        """
        if self.start_time is None:
            self.start_time = timestamp
        
        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            pnl=pnl,
            fees=fees,
            balance=balance,
        )
        
        self.trades.append(trade)
        self.end_time = timestamp
    
    def log_equity(self, timestamp: datetime, equity: float) -> None:
        """
        Log equity value at a point in time.
        
        Args:
            timestamp: Current timestamp
            equity: Current total equity value
        """
        self.timestamps.append(timestamp)
        self.equity_history.append(equity)
        
        # Track peak for drawdown
        if equity > self.peak_balance:
            self.peak_balance = equity
        if equity < self.min_balance:
            self.min_balance = equity
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        sharpe = (mean_return - self.risk_free_rate / self.trading_days) / std_return
        sharpe *= np.sqrt(self.trading_days)
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: Array of returns
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        sortino = (mean_return - self.risk_free_rate / self.trading_days) / downside_std
        sortino *= np.sqrt(self.trading_days)
        
        return sortino
    
    def calculate_max_drawdown(self) -> tuple[float, float]:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_drawdown, max_drawdown_percent)
        """
        if not self.equity_history:
            return 0.0, 0.0
        
        peak = self.initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0
        
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            
            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        return max_dd, max_dd_pct
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Returns:
            PerformanceMetrics object with all calculations
        """
        # Basic stats
        if not self.trades:
            return PerformanceMetrics(
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                total_pnl=0.0,
                pnl_percent=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_days=0.0,
            )
        
        # Calculate totals
        final_balance = self.trades[-1].balance if self.trades else self.initial_balance
        total_pnl = final_balance - self.initial_balance
        pnl_percent = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Trade stats
        closed_trades = [t for t in self.trades if t.action == "SELL"]
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        losing_trades = len([t for t in closed_trades if t.pnl <= 0])
        
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        
        # Average win/loss
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [t.pnl for t in closed_trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Calculate returns from equity history
        if len(self.equity_history) > 1:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
        else:
            sharpe = 0.0
            sortino = 0.0
        
        # Drawdown
        max_dd, max_dd_pct = self.calculate_max_drawdown()
        
        # Duration
        duration_days = 0.0
        if self.start_time and self.end_time:
            duration_days = (self.end_time - self.start_time).total_seconds() / 86400
        
        return PerformanceMetrics(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_pnl=total_pnl,
            pnl_percent=pnl_percent,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_trades=len(closed_trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            start_time=self.start_time or datetime.now(),
            end_time=self.end_time or datetime.now(),
            duration_days=duration_days,
        )
    
    def save_to_csv(self, filepath: str = "trades_log.csv") -> None:
        """
        Save all trades to CSV file.
        
        Args:
            filepath: Output file path
        """
        if not self.trades:
            print("No trades to save")
            return
        
        fieldnames = [
            "timestamp",
            "symbol",
            "action",
            "price",
            "quantity",
            "pnl",
            "fees",
            "balance",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in self.trades:
                writer.writerow({
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "fees": trade.fees,
                    "balance": trade.balance,
                })
        
        print(f"Trades saved to {filepath}")
    
    def save_equity_history(self, filepath: str = "equity_history.csv") -> None:
        """
        Save equity history to CSV.
        
        Args:
            filepath: Output file path
        """
        if not self.equity_history:
            print("No equity history to save")
            return
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity"])
            
            for ts, equity in zip(self.timestamps, self.equity_history):
                writer.writerow([ts.isoformat(), equity])
        
        print(f"Equity history saved to {filepath}")
    
    def print_summary(self) -> None:
        """Print a formatted summary of metrics."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("  PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"\n  BALANCE:")
        print(f"    Initial:    ${metrics.initial_balance:,.2f}")
        print(f"    Final:      ${metrics.final_balance:,.2f}")
        print(f"    PnL:        ${metrics.total_pnl:,.2f} ({metrics.pnl_percent:.2f}%)")
        
        print(f"\n  RISK:")
        print(f"    Max Drawdown:    ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
        print(f"    Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
        print(f"    Sortino Ratio:  {metrics.sortino_ratio:.2f}")
        
        print(f"\n  TRADING:")
        print(f"    Total Trades:   {metrics.total_trades}")
        print(f"    Winning:         {metrics.winning_trades}")
        print(f"    Losing:          {metrics.losing_trades}")
        print(f"    Win Rate:        {metrics.win_rate:.1f}%")
        
        if metrics.avg_win != 0 or metrics.avg_loss != 0:
            print(f"\n  AVERAGES:")
            print(f"    Avg Win:        ${metrics.avg_win:.2f}")
            print(f"    Avg Loss:       ${metrics.avg_loss:.2f}")
            print(f"    Profit Factor:  {metrics.profit_factor:.2f}")
        
        print(f"\n  DURATION:")
        print(f"    Start:    {metrics.start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"    End:      {metrics.end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"    Days:     {metrics.duration_days:.1f}")
        
        print("\n" + "=" * 60)
        
        # Rating
        rating = self._get_rating(metrics)
        print(f"  RATING: {rating}")
        print("=" * 60 + "\n")
    
    def _get_rating(self, metrics: PerformanceMetrics) -> str:
        """Get a rating based on metrics."""
        score = 0
        
        # PnL rating
        if metrics.pnl_percent > 10:
            score += 2
        elif metrics.pnl_percent > 5:
            score += 1
        
        # Drawdown rating
        if metrics.max_drawdown_percent < 5:
            score += 2
        elif metrics.max_drawdown_percent < 10:
            score += 1
        
        # Sharpe rating
        if metrics.sharpe_ratio > 1.5:
            score += 2
        elif metrics.sharpe_ratio > 1.0:
            score += 1
        
        # Win rate rating
        if metrics.win_rate > 60:
            score += 1
        elif metrics.win_rate > 50:
            score += 0.5
        
        if score >= 6:
            return "A - EXCELLENT"
        elif score >= 4:
            return "B - GOOD"
        elif score >= 2:
            return "C - AVERAGE"
        else:
            return "D - NEEDS IMPROVEMENT"


# Convenience function for quick analysis
def analyze_trades(
    trades: List[Dict],
    initial_balance: float = 1000.0,
) -> PerformanceMetrics:
    """
    Quick analysis from a list of trade dictionaries.
    
    Args:
        trades: List of trade dicts with keys: timestamp, symbol, action, price, quantity, pnl
        initial_balance: Starting balance
        
    Returns:
        PerformanceMetrics
    """
    analyzer = PerformanceAnalyzer(initial_balance)
    
    for trade in trades:
        analyzer.log_trade(
            timestamp=trade["timestamp"],
            symbol=trade["symbol"],
            action=trade["action"],
            price=trade["price"],
            quantity=trade["quantity"],
            pnl=trade.get("pnl", 0),
            fees=trade.get("fees", 0),
            balance=trade.get("balance", initial_balance),
        )
    
    return analyzer.get_metrics()
