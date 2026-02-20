"""
Tests for Performance Monitor and Risk Guard modules.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from performance_monitor import (
    PerformanceMonitor, TradeRecord, PerformanceMetrics, get_performance_monitor
)
from risk_guard import (
    RiskGuard, RiskLevel, TradingStatus, RiskThresholds, RiskAlert, get_risk_guard
)


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor for each test."""
        return PerformanceMonitor(initial_capital=100000.0)
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.initial_capital == 100000.0
        assert monitor.current_capital == 100000.0
        assert monitor.peak_capital == 100000.0
        assert len(monitor.trades) == 0
        assert len(monitor.closed_trades) == 0
    
    def test_open_trade(self, monitor):
        """Test opening a trade."""
        trade = monitor.open_trade(
            trade_id="TEST001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.1,
            side="LONG"
        )
        
        assert trade.trade_id == "TEST001"
        assert trade.symbol == "BTCUSDT"
        assert trade.entry_price == 50000.0
        assert trade.quantity == 0.1
        assert trade.side == "LONG"
        assert trade.status == "OPEN"
        assert "TEST001" in monitor.open_positions
    
    def test_close_trade_long_profit(self, monitor):
        """Test closing a long trade with profit."""
        monitor.open_trade(
            trade_id="TEST001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.1,
            side="LONG"
        )
        
        trade = monitor.close_trade(
            trade_id="TEST001",
            exit_price=55000.0
        )
        
        assert trade is not None
        assert trade.status == "CLOSED"
        assert trade.pnl == pytest.approx(500.0, rel=0.01)  # (55000-50000)*0.1
        assert "TEST001" not in monitor.open_positions
    
    def test_close_trade_long_loss(self, monitor):
        """Test closing a long trade with loss."""
        monitor.open_trade(
            trade_id="TEST001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.1,
            side="LONG"
        )
        
        trade = monitor.close_trade(
            trade_id="TEST001",
            exit_price=45000.0
        )
        
        assert trade is not None
        assert trade.pnl == pytest.approx(-500.0, rel=0.01)
    
    def test_close_trade_short_profit(self, monitor):
        """Test closing a short trade with profit."""
        monitor.open_trade(
            trade_id="TEST001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.1,
            side="SHORT"
        )
        
        trade = monitor.close_trade(
            trade_id="TEST001",
            exit_price=45000.0
        )
        
        assert trade is not None
        assert trade.pnl == pytest.approx(500.0, rel=0.01)  # (50000-45000)*0.1
    
    def test_close_trade_short_loss(self, monitor):
        """Test closing a short trade with loss."""
        monitor.open_trade(
            trade_id="TEST001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.1,
            side="SHORT"
        )
        
        trade = monitor.close_trade(
            trade_id="TEST001",
            exit_price=55000.0
        )
        
        assert trade is not None
        assert trade.pnl == pytest.approx(-500.0, rel=0.01)
    
    def test_close_nonexistent_trade(self, monitor):
        """Test closing a trade that doesn't exist."""
        trade = monitor.close_trade("NONEXISTENT", 50000.0)
        assert trade is None
    
    def test_calculate_metrics_empty(self, monitor):
        """Test metrics calculation with no trades."""
        metrics = monitor.calculate_metrics()
        
        assert metrics.total_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.win_rate == 0.0
    
    def test_calculate_metrics_with_trades(self, monitor):
        """Test metrics calculation with trades."""
        # Open and close multiple trades
        for i in range(5):
            monitor.open_trade(
                trade_id=f"WIN{i}",
                symbol="BTCUSDT",
                entry_price=50000.0,
                quantity=0.1,
                side="LONG"
            )
            monitor.close_trade(f"WIN{i}", exit_price=51000.0)  # Profit
        
        for i in range(3):
            monitor.open_trade(
                trade_id=f"LOSS{i}",
                symbol="BTCUSDT",
                entry_price=50000.0,
                quantity=0.1,
                side="LONG"
            )
            monitor.close_trade(f"LOSS{i}", exit_price=49000.0)  # Loss
        
        metrics = monitor.calculate_metrics()
        
        assert metrics.total_trades == 8
        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 3
        assert metrics.win_rate == pytest.approx(0.625, rel=0.01)
    
    def test_drawdown_calculation(self, monitor):
        """Test drawdown calculation."""
        # Profitable trade
        monitor.open_trade("P1", "BTC", 50000, 0.1, "LONG")
        monitor.close_trade("P1", 60000)  # +1000
        
        # Losing trade
        monitor.open_trade("L1", "BTC", 60000, 0.1, "LONG")
        monitor.close_trade("L1", 40000)  # -2000
        
        metrics = monitor.calculate_metrics()
        
        # Total P&L should be -1000
        assert metrics.total_pnl == pytest.approx(-1000.0, rel=0.01)
    
    def test_streak_calculation(self, monitor):
        """Test win/loss streak calculation."""
        # 3 wins
        for i in range(3):
            monitor.open_trade(f"W{i}", "BTC", 50000, 0.1, "LONG")
            monitor.close_trade(f"W{i}", 51000)
        
        # 2 losses
        for i in range(2):
            monitor.open_trade(f"L{i}", "BTC", 50000, 0.1, "LONG")
            monitor.close_trade(f"L{i}", 49000)
        
        metrics = monitor.calculate_metrics()
        
        assert metrics.max_consecutive_wins == 3
        assert metrics.max_consecutive_losses == 2
    
    def test_get_summary(self, monitor):
        """Test summary generation."""
        monitor.open_trade("T1", "BTC", 50000, 0.1, "LONG")
        monitor.close_trade("T1", 51000)
        
        summary = monitor.get_summary()
        
        assert 'capital' in summary
        assert 'trades' in summary
        assert 'risk' in summary
        assert summary['trades']['total'] == 1
    
    def test_get_equity_curve(self, monitor):
        """Test equity curve generation."""
        monitor.open_trade("T1", "BTC", 50000, 0.1, "LONG")
        monitor.close_trade("T1", 51000)
        
        df = monitor.get_equity_curve()
        
        assert not df.empty
        assert 'equity' in df.columns
    
    def test_get_trade_history(self, monitor):
        """Test trade history generation."""
        monitor.open_trade("T1", "BTC", 50000, 0.1, "LONG")
        monitor.close_trade("T1", 51000)
        
        df = monitor.get_trade_history()
        
        assert not df.empty
        assert len(df) == 1
        assert 'trade_id' in df.columns
        assert 'pnl' in df.columns
    
    def test_singleton(self, monitor):
        """Test singleton pattern."""
        m1 = get_performance_monitor()
        m2 = get_performance_monitor()
        
        assert m1 is m2


class TestRiskGuard:
    """Tests for RiskGuard class."""
    
    @pytest.fixture
    def guard(self):
        """Create a fresh guard for each test."""
        return RiskGuard()
    
    @pytest.fixture
    def guard_with_monitor(self):
        """Create guard with performance monitor."""
        monitor = PerformanceMonitor(initial_capital=100000.0)
        return RiskGuard(performance_monitor=monitor)
    
    def test_initialization(self, guard):
        """Test guard initialization."""
        assert guard.status == TradingStatus.ACTIVE
        assert guard.risk_level == RiskLevel.NORMAL
        assert len(guard.alerts) == 0
    
    def test_default_thresholds(self, guard):
        """Test default threshold values."""
        assert guard.thresholds.max_drawdown_percent == 0.20
        assert guard.thresholds.max_daily_loss_percent == 0.05
        assert guard.thresholds.max_consecutive_losses == 10
    
    def test_check_risk_normal(self, guard):
        """Test risk check with normal metrics."""
        metrics = {
            'risk': {'max_drawdown_percent': '5%', 'sharpe_ratio': '1.5'},
            'trades': {'win_rate': '50%', 'total': 20},
            'daily': {'avg_daily_pnl': '1000', 'trading_days': 30},
            'streaks': {'current_streak': 2, 'max_consecutive_losses': 3}
        }
        
        level = guard.check_risk(metrics)
        
        assert level == RiskLevel.NORMAL
        assert guard.can_trade()
    
    def test_check_risk_warning_drawdown(self, guard):
        """Test risk check with warning drawdown."""
        metrics = {
            'risk': {'max_drawdown_percent': '12%', 'sharpe_ratio': '1.0'},
            'trades': {'win_rate': '50%', 'total': 20},
            'daily': {'avg_daily_pnl': '0', 'trading_days': 30},
            'streaks': {'current_streak': 0, 'max_consecutive_losses': 0}
        }
        
        level = guard.check_risk(metrics)
        
        assert level == RiskLevel.WARNING
    
    def test_check_risk_critical_drawdown(self, guard):
        """Test risk check with critical drawdown."""
        metrics = {
            'risk': {'max_drawdown_percent': '16%', 'sharpe_ratio': '0.5'},
            'trades': {'win_rate': '50%', 'total': 20},
            'daily': {'avg_daily_pnl': '0', 'trading_days': 30},
            'streaks': {'current_streak': 0, 'max_consecutive_losses': 0}
        }
        
        level = guard.check_risk(metrics)
        
        assert level == RiskLevel.CRITICAL
        assert guard.status == TradingStatus.PAUSED or guard.status == TradingStatus.HALTED
    
    def test_check_risk_emergency_drawdown(self, guard):
        """Test risk check with emergency drawdown."""
        metrics = {
            'risk': {'max_drawdown_percent': '25%', 'sharpe_ratio': '-0.5'},
            'trades': {'win_rate': '20%', 'total': 20},
            'daily': {'avg_daily_pnl': '-5000', 'trading_days': 30},
            'streaks': {'current_streak': -12, 'max_consecutive_losses': 12}
        }
        
        level = guard.check_risk(metrics)
        
        assert level == RiskLevel.EMERGENCY
        assert guard.status == TradingStatus.HALTED
        assert not guard.can_trade()
    
    def test_can_trade_frequency_limit(self, guard):
        """Test trading frequency limits."""
        # Should be able to trade initially
        assert guard.can_trade()
        
        # Simulate hitting hourly limit
        guard._trades_this_hour = guard.thresholds.max_trades_per_hour
        assert not guard.can_trade()
    
    def test_record_trade(self, guard):
        """Test trade recording."""
        initial_hour = guard._trades_this_hour
        initial_day = guard._trades_today
        
        guard.record_trade()
        
        assert guard._trades_this_hour == initial_hour + 1
        assert guard._trades_today == initial_day + 1
    
    def test_manual_halt(self, guard):
        """Test manual halt."""
        guard.manual_halt("Test halt")
        
        assert guard.status == TradingStatus.HALTED
        assert guard._halt_reason == "Test halt"
        assert not guard.can_trade()
    
    def test_unlock(self, guard):
        """Test unlock from halted state."""
        guard.manual_halt("Test halt")
        assert guard.status == TradingStatus.HALTED
        
        guard.unlock()
        
        assert guard.status == TradingStatus.ACTIVE
        assert guard.can_trade()
    
    def test_cooldown(self, guard):
        """Test cooldown period."""
        guard.set_cooldown(5)
        
        assert guard._cooldown_until is not None
        assert not guard.can_trade()
    
    def test_callbacks(self, guard):
        """Test halt and resume callbacks."""
        halt_called = []
        resume_called = []
        
        def on_halt(reason):
            halt_called.append(reason)
        
        def on_resume():
            resume_called.append(True)
        
        guard.register_halt_callback(on_halt)
        guard.register_resume_callback(on_resume)
        
        guard.manual_halt("Test")
        assert len(halt_called) == 1
        
        guard.unlock()
        assert len(resume_called) == 1
    
    def test_alert_callbacks(self, guard):
        """Test alert callbacks."""
        alerts_received = []
        
        def on_alert(alert):
            alerts_received.append(alert)
        
        guard.register_alert_callback(on_alert)
        
        metrics = {
            'risk': {'max_drawdown_percent': '12%', 'sharpe_ratio': '1.0'},
            'trades': {'win_rate': '50%', 'total': 20},
            'daily': {'avg_daily_pnl': '0', 'trading_days': 30},
            'streaks': {'current_streak': 0, 'max_consecutive_losses': 0}
        }
        
        guard.check_risk(metrics)
        
        assert len(alerts_received) > 0
    
    def test_get_status(self, guard):
        """Test status retrieval."""
        status = guard.get_status()
        
        assert 'status' in status
        assert 'risk_level' in status
        assert 'can_trade' in status
        assert status['status'] == 'ACTIVE'
    
    def test_get_alerts(self, guard):
        """Test alerts retrieval."""
        metrics = {
            'risk': {'max_drawdown_percent': '12%', 'sharpe_ratio': '1.0'},
            'trades': {'win_rate': '50%', 'total': 20},
            'daily': {'avg_daily_pnl': '0', 'trading_days': 30},
            'streaks': {'current_streak': 0, 'max_consecutive_losses': 0}
        }
        
        guard.check_risk(metrics)
        alerts = guard.get_alerts()
        
        assert len(alerts) > 0
        assert 'level' in alerts[0]
        assert 'message' in alerts[0]
    
    def test_acknowledge_alert(self, guard):
        """Test alert acknowledgment."""
        metrics = {
            'risk': {'max_drawdown_percent': '12%', 'sharpe_ratio': '1.0'},
            'trades': {'win_rate': '50%', 'total': 20},
            'daily': {'avg_daily_pnl': '0', 'trading_days': 30},
            'streaks': {'current_streak': 0, 'max_consecutive_losses': 0}
        }
        
        guard.check_risk(metrics)
        
        result = guard.acknowledge_alert(0)
        assert result is True
        assert guard.active_alerts[0].acknowledged
    
    def test_singleton(self, guard):
        """Test singleton pattern."""
        g1 = get_risk_guard()
        g2 = get_risk_guard()
        
        assert g1 is g2


class TestIntegration:
    """Integration tests for Performance Monitor and Risk Guard."""
    
    def test_full_workflow(self):
        """Test complete workflow with monitor and guard."""
        # Create monitor
        monitor = PerformanceMonitor(initial_capital=100000.0)
        
        # Create guard with monitor
        guard = RiskGuard(performance_monitor=monitor)
        
        # Simulate some losing trades
        for i in range(8):
            monitor.open_trade(f"L{i}", "BTC", 50000, 0.1, "LONG")
            monitor.close_trade(f"L{i}", 48000)  # Loss
        
        # Check risk
        metrics = monitor.get_summary()
        level = guard.check_risk(metrics)
        
        # Should be warning due to consecutive losses
        assert level in [RiskLevel.WARNING, RiskLevel.CRITICAL]
        
        # Simulate more losses to trigger emergency
        for i in range(5):
            monitor.open_trade(f"L2_{i}", "BTC", 50000, 0.1, "LONG")
            monitor.close_trade(f"L2_{i}", 45000)  # Bigger loss
        
        metrics = monitor.get_summary()
        level = guard.check_risk(metrics)
        
        # Should trigger emergency halt
        assert guard.status in [TradingStatus.PAUSED, TradingStatus.HALTED]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
