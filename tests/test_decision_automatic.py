"""
Tests for Decision Automatic Module
=================================
Tests for src/decision/decision_automatic.py
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""
    
    def test_monte_carlo_simulator_creation(self):
        """Test MonteCarloSimulator creation."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(
            n_simulations=1000,
            confidence_level=0.95
        )
        
        assert simulator.n_simulations == 1000
        assert simulator.confidence_level == 0.95
    
    def test_monte_carlo_simulator_defaults(self):
        """Test MonteCarloSimulator default values."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator()
        
        assert simulator.n_simulations == 1000
        assert simulator.confidence_level == 0.95
    
    def test_simulate_price_path(self):
        """Test simulate_price_path method."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(n_simulations=100)
        
        price_paths = simulator.simulate_price_path(
            current_price=50000.0,
            volatility=0.3,
            drift=0.0,
            time_horizon=30
        )
        
        # n_simulations rows + initial price = n_simulations rows, time_horizon+1 columns
        assert price_paths.shape[0] == 100  # n_simulations
        assert price_paths.shape[1] == 31  # time_horizon + 1
        assert price_paths[0, 0] == 50000.0  # Initial price
    
    def test_simulate_price_path_returns_array(self):
        """Test simulate_price_path returns numpy array."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(n_simulations=50)
        
        price_paths = simulator.simulate_price_path(
            current_price=100.0,
            volatility=0.2,
            time_horizon=10
        )
        
        assert isinstance(price_paths, np.ndarray)
        assert price_paths.shape[0] == 50  # n_simulations
        assert price_paths.shape[1] == 11  # 10 days + initial
    
    def test_calculate_var(self):
        """Test calculate_var method."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(n_simulations=1000)
        
        # Generate sample returns
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var = simulator.calculate_var(returns)
        
        assert isinstance(var, (float, np.floating))
    
    def test_calculate_var_with_custom_confidence(self):
        """Test calculate_var with custom confidence level."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(n_simulations=500)
        
        returns = np.random.normal(0, 0.01, 500)
        
        var = simulator.calculate_var(returns, confidence_level=0.99)
        
        assert isinstance(var, (float, np.floating))
    
    def test_calculate_var_default_confidence(self):
        """Test calculate_var uses default confidence level."""
        from src.decision.decision_automatic import MonteCarloSimulator
        
        simulator = MonteCarloSimulator(n_simulations=500, confidence_level=0.90)
        
        returns = np.random.normal(0, 0.01, 500)
        
        var = simulator.calculate_var(returns)  # Uses default from __init__
        
        assert isinstance(var, (float, np.floating))


class TestOpportunityFilter:
    """Tests for OpportunityFilter class."""
    
    def test_opportunity_filter_creation(self):
        """Test OpportunityFilter creation."""
        from src.decision.filtro_opportunita import OpportunityFilter
        
        # Just test that it can be created without arguments
        filter_obj = OpportunityFilter()
        
        assert filter_obj is not None
    
    def test_opportunity_filter_is_class(self):
        """Test OpportunityFilter is a class."""
        from src.decision.filtro_opportunita import OpportunityFilter
        
        assert isinstance(OpportunityFilter, type)


class TestDecisionMonteCarlo:
    """Tests for decision_montecarlo module."""
    
    def test_decision_montecarlo_import(self):
        """Test importing decision_montecarlo module."""
        from src.decision import decision_montecarlo
        
        assert decision_montecarlo is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
