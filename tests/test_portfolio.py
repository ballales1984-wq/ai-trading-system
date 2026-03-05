"""
Tests for app/portfolio modules - Portfolio Optimization and Performance
"""

import pytest
import numpy as np
from datetime import datetime


class TestOptimizationDataClasses:
    """Test suite for optimization data classes."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult dataclass."""
        from app.portfolio.optimization import OptimizationResult
        
        result = OptimizationResult(
            weights={"BTC": 0.5, "ETH": 0.5},
            expected_return=0.1,
            expected_volatility=0.2,
            sharpe_ratio=0.5,
            method="max_sharpe",
            converged=True,
            iterations=100,
            constraints_satisfied=True
        )
        
        assert result.weights == {"BTC": 0.5, "ETH": 0.5}
        assert result.expected_return == 0.1
        assert result.expected_volatility == 0.2
        assert result.sharpe_ratio == 0.5
        assert result.method == "max_sharpe"
        assert result.converged is True
        assert result.iterations == 100
        assert result.constraints_satisfied is True

    def test_optimization_result_defaults(self):
        """Test OptimizationResult default values."""
        from app.portfolio.optimization import OptimizationResult
        
        result = OptimizationResult()
        
        assert result.weights == {}
        assert result.expected_return == 0.0
        assert result.expected_volatility == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.method == ""
        assert result.converged is True
        assert result.iterations == 0
        assert result.constraints_satisfied is True

    def test_optimization_constraints_creation(self):
        """Test OptimizationConstraints dataclass."""
        from app.portfolio.optimization import OptimizationConstraints
        
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.4,
            max_positions=10,
            max_sector_weight=0.3,
            max_turnover=0.25,
            long_only=False
        )
        
        assert constraints.min_weight == 0.05
        assert constraints.max_weight == 0.4
        assert constraints.max_positions == 10
        assert constraints.max_sector_weight == 0.3
        assert constraints.max_turnover == 0.25
        assert constraints.long_only is False

    def test_optimization_constraints_defaults(self):
        """Test OptimizationConstraints default values."""
        from app.portfolio.optimization import OptimizationConstraints
        
        constraints = OptimizationConstraints()
        
        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.max_positions == 20
        assert constraints.max_sector_weight == 0.4
        assert constraints.max_turnover == 0.5
        assert constraints.long_only is True


class TestPortfolioOptimizer:
    """Test suite for PortfolioOptimizer class."""

    def test_optimizer_creation(self):
        """Test PortfolioOptimizer creation."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        symbols = ["BTC", "ETH", "SOL"]
        returns = np.random.randn(100, 3) * 0.02
        
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        
        assert optimizer.symbols == symbols
        assert optimizer.returns.shape == (100, 3)
        assert optimizer.risk_free_rate == 0.04

    def test_optimizer_custom_risk_free(self):
        """Test PortfolioOptimizer with custom risk-free rate."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        symbols = ["BTC", "ETH"]
        returns = np.array([[0.01], [-0.01]])
        
        optimizer = PortfolioOptimizer(
            symbols=symbols,
            returns=returns,
            risk_free_rate=0.02
        )
        
        assert optimizer.risk_free_rate == 0.02

    def test_optimizer_with_constraints(self):
        """Test PortfolioOptimizer with constraints."""
        from app.portfolio.optimization import PortfolioOptimizer, OptimizationConstraints
        
        symbols = ["BTC", "ETH", "SOL"]
        returns = np.random.randn(50, 3) * 0.02
        
        constraints = OptimizationConstraints(
            min_weight=0.1,
            max_weight=0.5,
            max_positions=3
        )
        
        optimizer = PortfolioOptimizer(
            symbols=symbols,
            returns=returns,
            constraints=constraints
        )
        
        assert optimizer.constraints == constraints


class TestPerformanceDataClasses:
    """Test suite for performance data classes."""

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics dataclass."""
        from app.portfolio.performance import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            total_return=0.25,
            annualized_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.1,
            win_rate=0.6,
            profit_factor=1.5,
            total_trades=100,
            winning_trades=60,
            losing_trades=40
        )
        
        assert metrics.total_return == 0.25
        assert metrics.annualized_return == 0.15
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == 0.1
        assert metrics.win_rate == 0.6
        assert metrics.profit_factor == 1.5
        assert metrics.total_trades == 100
        assert metrics.winning_trades == 60
        assert metrics.losing_trades == 40

    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics default values."""
        from app.portfolio.performance import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.daily_returns_mean == 0.0
        assert metrics.daily_returns_std == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0


class TestPortfolioImports:
    """Test suite for portfolio module imports."""

    def test_optimization_imports(self):
        """Test optimization module can be imported."""
        from app.portfolio import optimization
        
        assert optimization is not None

    def test_performance_imports(self):
        """Test performance module can be imported."""
        from app.portfolio import performance
        
        assert performance is not None

    def test_portfolio_optimizer_import(self):
        """Test PortfolioOptimizer can be imported."""
        from app.portfolio.optimization import PortfolioOptimizer
        
        assert PortfolioOptimizer is not None

    def test_optimization_result_import(self):
        """Test OptimizationResult can be imported."""
        from app.portfolio.optimization import OptimizationResult
        
        assert OptimizationResult is not None

    def test_optimization_constraints_import(self):
        """Test OptimizationConstraints can be imported."""
        from app.portfolio.optimization import OptimizationConstraints
        
        assert OptimizationConstraints is not None

    def test_performance_metrics_import(self):
        """Test PerformanceMetrics can be imported."""
        from app.portfolio.performance import PerformanceMetrics
        
        assert PerformanceMetrics is not None
