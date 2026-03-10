"""
Test Suite for Portfolio Optimization Module
============================================
Comprehensive tests for portfolio optimization.
"""

import pytest
import numpy as np
from app.portfolio.optimization import (
    OptimizationResult,
    OptimizationConstraints,
    PortfolioOptimizer,
)


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            weights={"BTC": 0.5, "ETH": 0.5},
            expected_return=0.1,
            expected_volatility=0.2,
            sharpe_ratio=0.5,
            method="max_sharpe",
        )
        assert result.weights["BTC"] == 0.5
        assert result.expected_return == 0.1
        assert result.sharpe_ratio == 0.5
        assert result.converged is True
    
    def test_optimization_result_defaults(self):
        """Test optimization result default values."""
        result = OptimizationResult()
        assert result.weights == {}
        assert result.expected_return == 0.0
        assert result.expected_volatility == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.method == ""
        assert result.converged is True


class TestOptimizationConstraints:
    """Tests for OptimizationConstraints dataclass."""
    
    def test_constraints_creation(self):
        """Test constraints creation."""
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=1.0,
            max_positions=10,
        )
        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.max_positions == 10
    
    def test_constraints_defaults(self):
        """Test constraints default values."""
        constraints = OptimizationConstraints()
        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.max_positions == 20
        assert constraints.max_sector_weight == 0.4
        assert constraints.max_turnover == 0.5
        assert constraints.long_only is True


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""
    
    def test_optimizer_creation(self):
        """Test portfolio optimizer creation."""
        symbols = ["BTC", "ETH", "SOL"]
        returns = np.random.randn(100, 3) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        assert optimizer.symbols == symbols
        assert optimizer.returns.shape == (100, 3)
        assert optimizer.risk_free_rate == 0.04
    
    def test_optimizer_custom_risk_free_rate(self):
        """Test portfolio optimizer with custom risk free rate."""
        symbols = ["BTC", "ETH"]
        returns = np.random.randn(50, 2) * 0.02
        optimizer = PortfolioOptimizer(
            symbols=symbols,
            returns=returns,
            risk_free_rate=0.05,
        )
        assert optimizer.risk_free_rate == 0.05
    
    def test_optimizer_with_constraints(self):
        """Test portfolio optimizer with constraints."""
        symbols = ["BTC", "ETH", "SOL"]
        returns = np.random.randn(100, 3) * 0.02
        constraints = OptimizationConstraints(
            max_weight=0.5,
            max_positions=2,
        )
        optimizer = PortfolioOptimizer(
            symbols=symbols,
            returns=returns,
            constraints=constraints,
        )
        assert optimizer.constraints.max_weight == 0.5
        assert optimizer.constraints.max_positions == 2
    
    def test_calculate_returns(self):
        """Test return calculation."""
        symbols = ["BTC", "ETH"]
        # Create simple returns: 10 days, 2 assets
        returns = np.array([
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
        ])
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        mean_returns = optimizer._calculate_mean_returns()
        assert mean_returns.shape == (2,)
        assert np.all(mean_returns > 0)
    
    def test_calculate_covariance(self):
        """Test covariance matrix calculation."""
        symbols = ["BTC", "ETH"]
        returns = np.array([
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
        ])
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        cov = optimizer._calculate_covariance()
        assert cov.shape == (2, 2)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        symbols = ["BTC", "ETH"]
        returns = np.array([
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
        ])
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        sharpe = optimizer._calculate_sharpe_ratio(np.array([0.5, 0.5]))
        assert isinstance(sharpe, (int, float, np.floating))
    
    def test_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        symbols = ["BTC", "ETH"]
        returns = np.random.randn(100, 2) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        weights = np.array([0.5, 0.5])
        vol = optimizer._portfolio_volatility(weights)
        assert vol >= 0
    
    def test_portfolio_return(self):
        """Test portfolio return calculation."""
        symbols = ["BTC", "ETH"]
        returns = np.random.randn(100, 2) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        weights = np.array([0.5, 0.5])
        ret = optimizer._portfolio_return(weights)
        assert isinstance(ret, (int, float, np.floating))
    
    def test_optimize_min_variance(self):
        """Test minimum variance optimization."""
        symbols = ["BTC", "ETH", "SOL"]
        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        result = optimizer.optimize(method="min_variance")
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 3
    
    def test_optimize_max_sharpe(self):
        """Test maximum Sharpe ratio optimization."""
        symbols = ["BTC", "ETH", "SOL"]
        np.random.seed(42)
        returns = np.random.randn(100, 3) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        result = optimizer.optimize(method="max_sharpe")
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 3
    
    def test_optimize_equal_weight(self):
        """Test equal weight optimization."""
        symbols = ["BTC", "ETH", "SOL"]
        returns = np.random.randn(100, 3) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        result = optimizer.optimize(method="equal_weight")
        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 3
    
    def test_optimize_risk_parity(self):
        """Test risk parity optimization."""
        symbols = ["BTC", "ETH", "SOL"]
        returns = np.random.randn(100, 3) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        result = optimizer.optimize(method="risk_parity")
        assert isinstance(result, OptimizationResult)
    
    def test_optimize_invalid_method(self):
        """Test optimization with invalid method."""
        symbols = ["BTC", "ETH"]
        returns = np.random.randn(50, 2) * 0.02
        optimizer = PortfolioOptimizer(symbols=symbols, returns=returns)
        result = optimizer.optimize(method="invalid_method")
        assert result is None
