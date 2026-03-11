"""
Test Coverage for Portfolio Module
===================================
Comprehensive tests to improve coverage for app/portfolio/* modules.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPortfolioOptimization:
    """Test app.portfolio.optimization module."""
    
    def test_optimization_module_import(self):
        """Test optimization module can be imported."""
        from app.portfolio import optimization
        assert optimization is not None
    
    def test_portfolio_optimizer_class(self):
        """Test PortfolioOptimizer class exists."""
        from app.portfolio.optimization import PortfolioOptimizer
        assert PortfolioOptimizer is not None
    
    def test_optimization_result_class(self):
        """Test OptimizationResult class exists."""
        from app.portfolio.optimization import OptimizationResult
        assert OptimizationResult is not None
    
    def test_optimization_constraints_class(self):
        """Test OptimizationConstraints class exists."""
        from app.portfolio.optimization import OptimizationConstraints
        assert OptimizationConstraints is not None
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation with correct fields."""
        from app.portfolio.optimization import OptimizationResult
        result = OptimizationResult(
            weights={"BTC": 0.5, "ETH": 0.5},
            expected_return=0.1
        )
        assert result.weights is not None
    
    def test_optimization_constraints_creation(self):
        """Test OptimizationConstraints creation."""
        from app.portfolio.optimization import OptimizationConstraints
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.5
        )
        assert constraints.min_weight == 0.05


class TestPortfolioPerformance:
    """Test app.portfolio.performance module."""
    
    def test_performance_module_import(self):
        """Test performance module can be imported."""
        from app.portfolio import performance
        assert performance is not None
    
    def test_performance_metrics_class(self):
        """Test PerformanceMetrics class exists."""
        from app.portfolio.performance import PerformanceMetrics
        assert PerformanceMetrics is not None
    
    def test_trade_record_class(self):
        """Test TradeRecord class exists."""
        from app.portfolio.performance import TradeRecord
        assert TradeRecord is not None
    
    def test_portfolio_performance_class(self):
        """Test PortfolioPerformance class exists."""
        from app.portfolio.performance import PortfolioPerformance
        assert PortfolioPerformance is not None
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        from app.portfolio.performance import PerformanceMetrics
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.6
        )
        assert metrics.total_return == 0.15
    
    def test_trade_record_creation(self):
        """Test TradeRecord creation with correct fields."""
        from app.portfolio.performance import TradeRecord
        record = TradeRecord(
            symbol="BTC",
            side="BUY",
            entry_price=45000.0,
            exit_price=50000.0,
            quantity=0.5,
            pnl=2500.0
        )
        assert record.symbol == "BTC"
        assert record.side == "BUY"
    
    def test_portfolio_performance_creation(self):
        """Test PortfolioPerformance creation."""
        from app.portfolio.performance import PortfolioPerformance
        perf = PortfolioPerformance(initial_capital=100000.0)
        assert perf.initial_capital == 100000.0
    
    def test_portfolio_performance_record_equity(self):
        """Test PortfolioPerformance record_equity method."""
        from app.portfolio.performance import PortfolioPerformance
        perf = PortfolioPerformance(initial_capital=100000.0)
        perf.record_equity(105000.0)
        curve = perf.get_equity_curve()
        assert len(curve) > 0
    
    def test_portfolio_performance_record_trade(self):
        """Test PortfolioPerformance record_trade method."""
        from app.portfolio.performance import PortfolioPerformance, TradeRecord
        perf = PortfolioPerformance(initial_capital=100000.0)
        trade = TradeRecord(
            symbol="BTC",
            side="BUY",
            entry_price=45000.0,
            exit_price=50000.0,
            quantity=0.5,
            pnl=2500.0
        )
        perf.record_trade(trade)
        trades = perf.get_trades()
        assert len(trades) == 1
    
    def test_portfolio_performance_compute_metrics(self):
        """Test PortfolioPerformance compute_metrics method."""
        from app.portfolio.performance import PortfolioPerformance
        perf = PortfolioPerformance(initial_capital=100000.0)
        perf.record_equity(100000.0)
        perf.record_equity(105000.0)
        metrics = perf.compute_metrics()
        assert metrics is not None
        assert hasattr(metrics, 'total_return')


class TestPortfolioModuleIntegration:
    """Integration tests for portfolio module."""
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        from app.portfolio.performance import PortfolioPerformance
        
        # Create portfolio
        perf = PortfolioPerformance(initial_capital=100000.0)
        
        # Record some equity values
        perf.record_equity(100000.0)
        perf.record_equity(105000.0)
        perf.record_equity(110000.0)
        
        # Get equity curve
        curve = perf.get_equity_curve()
        assert len(curve) >= 3
    
    def test_portfolio_with_multiple_trades(self):
        """Test portfolio with multiple trades."""
        from app.portfolio.performance import PortfolioPerformance, TradeRecord
        
        perf = PortfolioPerformance(initial_capital=100000.0)
        
        # Add multiple trades
        trades = [
            TradeRecord(symbol="BTC", side="BUY", entry_price=45000, exit_price=50000, quantity=0.5, pnl=2500),
            TradeRecord(symbol="ETH", side="BUY", entry_price=2800, exit_price=3200, quantity=10, pnl=4000),
            TradeRecord(symbol="SOL", side="BUY", entry_price=80, exit_price=75, quantity=100, pnl=-500)
        ]
        
        for trade in trades:
            perf.record_trade(trade)
        
        assert len(perf.get_trades()) == 3

