"""
Tests for Risk Parameter Optimization Module
=============================================
Tests risk parameter optimization functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
import warnings

warnings.filterwarnings('ignore')


class TestRiskOptimizerImports:
    """Test module imports."""
    
    def test_import_risk_optimizer(self):
        """Test that risk_optimizer module can be imported."""
        from src.risk_optimizer import RiskOptimizer, RiskParameters, OptimizationResult
        assert RiskOptimizer is not None
        assert RiskParameters is not None
        assert OptimizationResult is not None
    
    def test_param_ranges_defined(self):
        """Test that parameter ranges are defined."""
        from src.risk_optimizer import RiskOptimizer
        
        assert 'var_confidence' in RiskOptimizer.PARAM_RANGES
        assert 'max_drawdown' in RiskOptimizer.PARAM_RANGES
        assert 'stop_loss_pct' in RiskOptimizer.PARAM_RANGES


class TestRiskParameters:
    """Test RiskParameters dataclass."""
    
    def test_risk_parameters_creation(self):
        """Test creating RiskParameters."""
        from src.risk_optimizer import RiskParameters
        
        params = RiskParameters(
            var_confidence=0.95,
            max_drawdown=0.20,
            max_position_pct=0.10,
            stop_loss_pct=0.05
        )
        
        assert params.var_confidence == 0.95
        assert params.max_drawdown == 0.20
        assert params.max_position_pct == 0.10
        assert params.stop_loss_pct == 0.05
    
    def test_risk_parameters_defaults(self):
        """Test RiskParameters default values."""
        from src.risk_optimizer import RiskParameters
        
        params = RiskParameters()
        
        assert params.var_confidence == 0.95
        assert params.max_drawdown == 0.20
        assert params.take_profit_pct == 0.10
        assert params.kelly_fraction == 0.25
    
    def test_risk_parameters_to_dict(self):
        """Test to_dict method."""
        from src.risk_optimizer import RiskParameters
        
        params = RiskParameters(var_confidence=0.99)
        d = params.to_dict()
        
        assert isinstance(d, dict)
        assert 'var_confidence' in d
        assert d['var_confidence'] == 0.99


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test creating OptimizationResult."""
        from src.risk_optimizer import OptimizationResult, RiskParameters
        
        result = OptimizationResult(
            best_params=RiskParameters(),
            best_score=1.5,
            all_results=[],
            metrics={'sharpe': 1.5},
            optimization_time=10.5
        )
        
        assert result.best_score == 1.5
        assert result.timestamp is not None
    
    def test_optimization_result_with_timestamp(self):
        """Test OptimizationResult with explicit timestamp."""
        from src.risk_optimizer import OptimizationResult, RiskParameters
        
        ts = datetime(2026, 1, 1)
        result = OptimizationResult(
            best_params=RiskParameters(),
            best_score=1.0,
            all_results=[],
            metrics={},
            optimization_time=1.0,
            timestamp=ts
        )
        
        assert result.timestamp == ts


class TestRiskOptimizer:
    """Test RiskOptimizer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 300))
        return returns
    
    @pytest.fixture
    def optimizer(self):
        """Create RiskOptimizer instance."""
        from src.risk_optimizer import RiskOptimizer
        return RiskOptimizer(n_splits=3, min_trades=5)
    
    def test_optimizer_initialization(self, optimizer):
        """Test RiskOptimizer initialization."""
        assert optimizer.n_splits == 3
        assert optimizer.min_trades == 5
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.results == []
    
    def test_calculate_sharpe_ratio(self, optimizer, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = optimizer.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        # Sharpe should be reasonable for random data
        assert -5 < sharpe < 5
    
    def test_calculate_sortino_ratio(self, optimizer, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = optimizer.calculate_sortino_ratio(sample_returns)
        
        assert isinstance(sortino, float)
    
    def test_calculate_max_drawdown(self, optimizer, sample_returns):
        """Test max drawdown calculation."""
        equity_curve = (1 + sample_returns).cumprod()
        max_dd = optimizer.calculate_max_drawdown(equity_curve)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1
    
    def test_calculate_calmar_ratio(self, optimizer, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = optimizer.calculate_calmar_ratio(sample_returns)
        
        assert isinstance(calmar, float)
    
    def test_simulate_trades(self, optimizer, sample_returns):
        """Test trade simulation."""
        from src.risk_optimizer import RiskParameters
        
        params = RiskParameters()
        adjusted, trades = optimizer.simulate_trades(sample_returns, params)
        
        assert isinstance(adjusted, pd.Series)
        assert isinstance(trades, int)
        assert trades > 0
    
    def test_evaluate_params(self, optimizer, sample_returns):
        """Test parameter evaluation."""
        from src.risk_optimizer import RiskParameters
        
        params = RiskParameters()
        metrics = optimizer.evaluate_params(sample_returns, params)
        
        assert 'sharpe' in metrics
        assert 'sortino' in metrics
        assert 'calmar' in metrics
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'trades' in metrics
    
    def test_optimize(self, optimizer, sample_returns):
        """Test optimization."""
        # Use small param ranges for speed
        param_ranges = {
            'var_confidence': [0.95],
            'max_drawdown': [0.20],
            'max_position_pct': [0.10],
            'stop_loss_pct': [0.03, 0.05],
            'take_profit_pct': [0.10],
            'trailing_stop_pct': [0.03],
            'kelly_fraction': [0.25]
        }
        
        result = optimizer.optimize(sample_returns, param_ranges, metric='sharpe')
        
        assert result.best_params is not None
        assert result.best_score != -float('inf')
        assert len(result.all_results) == 2  # 2 stop_loss values
        assert result.optimization_time > 0
    
    def test_optimize_with_custom_metric(self, optimizer, sample_returns):
        """Test optimization with different metric."""
        param_ranges = {
            'var_confidence': [0.95],
            'max_drawdown': [0.20],
            'max_position_pct': [0.10],
            'stop_loss_pct': [0.05],
            'take_profit_pct': [0.10],
            'trailing_stop_pct': [0.03],
            'kelly_fraction': [0.25]
        }
        
        result = optimizer.optimize(sample_returns, param_ranges, metric='sortino')
        
        assert result.best_params is not None


class TestMonteCarloStressTest:
    """Test Monte Carlo stress testing."""
    
    def test_monte_carlo_stress_test(self):
        """Test Monte Carlo simulation."""
        from src.risk_optimizer import RiskOptimizer, RiskParameters
        
        optimizer = RiskOptimizer()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        params = RiskParameters()
        
        results = optimizer.monte_carlo_stress_test(
            returns, params,
            n_simulations=100,
            n_days=50
        )
        
        assert 'mean_final_value' in results
        assert 'median_final_value' in results
        assert 'worst_case' in results
        assert 'best_case' in results
        assert 'prob_loss' in results
        assert 'avg_max_drawdown' in results
        
        # Prob loss should be between 0 and 1
        assert 0 <= results['prob_loss'] <= 1


class TestWalkForwardOptimization:
    """Test walk-forward optimization."""
    
    def test_walk_forward_optimize(self):
        """Test walk-forward optimization."""
        from src.risk_optimizer import RiskOptimizer
        
        optimizer = RiskOptimizer(min_trades=2)
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 400))
        
        param_ranges = {
            'var_confidence': [0.95],
            'max_drawdown': [0.20],
            'max_position_pct': [0.10],
            'stop_loss_pct': [0.05],
            'take_profit_pct': [0.10],
            'trailing_stop_pct': [0.03],
            'kelly_fraction': [0.25]
        }
        
        results = optimizer.walk_forward_optimize(
            returns,
            train_size=100,
            test_size=50,
            param_ranges=param_ranges
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for r in results:
            assert 'fold' in r
            assert 'train_metrics' in r
            assert 'test_metrics' in r
            assert 'best_params' in r


class TestOptimizeRiskForStrategy:
    """Test convenience function."""
    
    def test_optimize_risk_for_strategy(self):
        """Test optimize_risk_for_strategy function."""
        from src.risk_optimizer import optimize_risk_for_strategy
        
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 200))
        
        params, metrics = optimize_risk_for_strategy(returns, metric='sharpe')
        
        assert params is not None
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
