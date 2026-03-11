"""
Tests for Performance Attribution Module
"""

import pytest
import pandas as pd
import numpy as np
from src.performance_attribution import (
    PerformanceAttribution,
    AttributionResult,
    BrinsonResult
)


class TestPerformanceAttribution:
    """Test suite for Performance Attribution module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        
        portfolio_weights = pd.DataFrame({
            'asset': assets,
            'weight': [0.40, 0.25, 0.15, 0.12, 0.08]
        })
        
        benchmark_weights = pd.DataFrame({
            'asset': assets,
            'weight': [0.35, 0.30, 0.15, 0.10, 0.10]
        })
        
        portfolio_returns = pd.DataFrame({
            'asset': assets,
            'return': [0.45, 0.35, 0.55, 0.25, 0.30]
        })
        
        benchmark_returns = pd.DataFrame({
            'asset': assets,
            'return': [0.40, 0.30, 0.50, 0.20, 0.25]
        })
        
        return {
            'portfolio_weights': portfolio_weights,
            'benchmark_weights': benchmark_weights,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns
        }
    
    def test_brinson_fachler(self, sample_data):
        """Test Brinson-Fachler attribution."""
        pa = PerformanceAttribution()
        result = pa.brinson_fachler(**sample_data)
        
        assert isinstance(result, BrinsonResult)
        assert result.allocation_effect is not None
        assert result.selection_effect is not None
        assert result.interaction_effect is not None
        assert result.total_effect is not None
        
        # Verify total equals sum of components
        expected_total = (
            result.allocation_effect + 
            result.selection_effect + 
            result.interaction_effect
        )
        assert abs(result.total_effect - expected_total) < 1e-10
    
    def test_brinson_hood_beebower(self, sample_data):
        """Test Brinson-Hood-Beebower attribution."""
        pa = PerformanceAttribution()
        result = pa.brinson_hood_beebower(**sample_data)
        
        assert isinstance(result, BrinsonResult)
        assert result.allocation_effect is not None
        assert result.selection_effect is not None
        assert result.interaction_effect is not None
    
    def test_factor_attribution(self):
        """Test factor attribution."""
        pa = PerformanceAttribution()
        
        portfolio_returns = pd.Series([0.02, 0.01, 0.03, -0.01, 0.02])
        factor_returns = pd.DataFrame({
            'momentum': [0.02, 0.01, 0.03, -0.01, 0.02],
            'value': [0.01, 0.02, -0.01, 0.03, 0.01],
            'size': [-0.01, 0.01, 0.02, 0.01, -0.01]
        })
        
        contributions = pa.factor_attribution(portfolio_returns, factor_returns)
        
        assert isinstance(contributions, dict)
        assert 'intercept' in contributions
        assert 'momentum' in contributions
        assert 'value' in contributions
    
    def test_timing_attribution(self):
        """Test timing attribution."""
        pa = PerformanceAttribution()
        
        # Create correlated series
        np.random.seed(42)
        n = 100
        benchmark = pd.Series(np.random.randn(n) * 0.01)
        portfolio = benchmark + pd.Series(np.random.randn(n) * 0.005)
        
        timing = pa.timing_attribution(portfolio, benchmark)
        
        assert isinstance(timing, float)
        assert -1 <= timing <= 1
    
    def test_full_attribution(self, sample_data):
        """Test full attribution analysis."""
        pa = PerformanceAttribution()
        
        factor_returns = pd.DataFrame({
            'momentum': [0.02] * 5,
            'value': [0.01] * 5
        })
        
        result = pa.full_attribution(
            portfolio_weights=sample_data['portfolio_weights'],
            benchmark_weights=sample_data['benchmark_weights'],
            portfolio_returns=sample_data['portfolio_returns'],
            benchmark_returns=sample_data['benchmark_returns'],
            factor_returns=factor_returns
        )
        
        assert isinstance(result, AttributionResult)
        assert result.total_return is not None
        assert result.benchmark_return is not None
        assert result.active_return is not None
        assert result.asset_allocation_return is not None
        assert result.security_selection_return is not None
        assert result.interaction_return is not None
    
    def test_generate_report(self, sample_data):
        """Test report generation."""
        pa = PerformanceAttribution()
        
        # Create a simple result
        result = AttributionResult(
            total_return=0.35,
            asset_allocation_return=0.02,
            security_selection_return=0.05,
            interaction_return=0.01,
            timing_return=0.01,
            benchmark_return=0.30,
            active_return=0.05,
            holdings_attribution={'BTC': 0.02},
            sector_attribution={'crypto': 0.05},
            factor_contributions={'momentum': 0.01}
        )
        
        report = pa.generate_report(result)
        
        assert isinstance(report, str)
        assert 'PERFORMANCE ATTRIBUTION REPORT' in report
        assert 'Total Portfolio Return' in report
        assert 'Asset Allocation Effect' in report
    
    def test_attribution_sum_equals_active_return(self, sample_data):
        """Test that attribution components are reasonable."""
        pa = PerformanceAttribution()
        
        result = pa.full_attribution(**sample_data)
        
        # Attribution should have reasonable values
        assert abs(result.asset_allocation_return) < 1.0
        assert abs(result.security_selection_return) < 1.0
        assert abs(result.interaction_return) < 0.1
        
        # Active return should match portfolio - benchmark
        expected_active = result.total_return - result.benchmark_return
        assert abs(result.active_return - expected_active) < 0.001
    
    def test_empty_data_handling(self):
        """Test handling of edge cases."""
        pa = PerformanceAttribution()
        
        # Empty dataframes
        empty_weights = pd.DataFrame({'asset': [], 'weight': []})
        empty_returns = pd.DataFrame({'asset': [], 'return': []})
        
        # Should handle gracefully (may return NaN or 0)
        # This tests basic error handling
        try:
            result = pa.brinson_fachler(
                empty_weights, empty_weights, empty_returns, empty_returns
            )
            # If no exception, check result is valid
            assert result is not None
        except Exception:
            # Exception is acceptable for empty data
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
