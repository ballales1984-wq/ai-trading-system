"""Extended tests for portfolio module"""
import pytest

class TestPortfolioModuleExtended:
    def test_portfolio_module_exists(self):
        from app.portfolio import optimization
        assert optimization is not None
    
    def test_performance_import(self):
        from app.portfolio import performance
        assert performance is not None
    
    def test_optimization_class_exists(self):
        from app.portfolio.optimization import PortfolioOptimizer
        assert PortfolioOptimizer is not None
    
    def test_performance_class_exists(self):
        from app.portfolio.performance import PerformanceTracker
        assert PerformanceTracker is not None
