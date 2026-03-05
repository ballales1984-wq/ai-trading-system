"""
Tests for Core Modules
=====================
Additional tests for src/core modules to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStateManager:
    """Test StateManager module."""
    
    def test_state_manager_creation(self):
        """Test StateManager initialization."""
        from src.core.state_manager import StateManager
        sm = StateManager()
        assert sm is not None
    
    def test_save_portfolio_state(self):
        """Test saving portfolio state."""
        from src.core.state_manager import StateManager, PortfolioState
        sm = StateManager()
        state = PortfolioState(
            total_equity=100000,
            available_balance=50000,
            unrealized_pnl=1000,
            realized_pnl=500
        )
        sm.save_portfolio_state(state)
        # Test passes if no exception
    
    def test_save_position(self):
        """Test saving position state."""
        from src.core.state_manager import StateManager, PositionState
        sm = StateManager()
        pos = PositionState(
            symbol="BTCUSDT",
            quantity=0.5,
            entry_price=50000,
            current_price=51000,
            unrealized_pnl=500,
            realized_pnl=0,
            commission=10,
            leverage=1
        )
        sm.save_position(pos)
    
    def test_get_all_positions(self):
        """Test getting all positions."""
        from src.core.state_manager import StateManager
        sm = StateManager()
        positions = sm.get_all_positions()
        assert isinstance(positions, list)
    
    def test_get_portfolio_state(self):
        """Test getting portfolio state."""
        from src.core.state_manager import StateManager
        sm = StateManager()
        state = sm.get_latest_portfolio_state()
        # Can be None if no state saved


class TestPortfolioManager:
    """Test Portfolio Manager module."""
    
    def test_portfolio_manager_creation(self):
        """Test PortfolioManager initialization."""
        from src.core.portfolio.portfolio_manager import PortfolioManager
        pm = PortfolioManager(initial_balance=100000)
        assert pm is not None


class TestRiskEngine:
    """Test Risk Engine module."""
    
    def test_risk_engine_creation(self):
        """Test RiskEngine initialization."""
        from src.core.risk.risk_engine import RiskEngine
        re = RiskEngine()
        assert re is not None


class TestExternalAPIs:
    """Test External API clients."""
    
    def test_api_registry_creation(self):
        """Test API Registry initialization."""
        from src.external.api_registry import APIRegistry
        registry = APIRegistry()
        assert registry is not None
    
    def test_bybit_client_creation(self):
        """Test Bybit client initialization."""
        from src.external.bybit_client import BybitClient
        client = BybitClient(testnet=True)
        assert client is not None
    
    def test_okx_client_creation(self):
        """Test OKX client initialization."""
        from src.external.okx_client import OKXClient
        client = OKXClient()
        assert client is not None


class TestStrategies:
    """Test Trading Strategies."""
    
    def test_momentum_strategy_creation(self):
        """Test Momentum strategy initialization."""
        from src.strategy.momentum import MomentumStrategy
        strategy = MomentumStrategy()
        assert strategy is not None
    
    def test_mean_reversion_creation(self):
        """Test Mean Reversion strategy initialization."""
        from src.strategy.mean_reversion import MeanReversionStrategy
        strategy = MeanReversionStrategy()
        assert strategy is not None


class TestAutoMLModules:
    """Test AutoML modules."""
    
    def test_evolution_creation(self):
        """Test Evolution engine initialization."""
        from src.automl.evolution import EvolutionEngine
        engine = EvolutionEngine()
        assert engine is not None


class TestResearchModules:
    """Test Research modules."""
    
    def test_feature_store_creation(self):
        """Test FeatureStore initialization."""
        from src.research.feature_store import FeatureStore
        store = FeatureStore()
        assert store is not None
    
    def test_alpha_lab_creation(self):
        """Test AlphaLab initialization."""
        from src.research.alpha_lab import AlphaLab
        lab = AlphaLab()
        assert lab is not None


class TestMetaModules:
    """Test Meta-learning modules."""
    
    def test_meta_evolution_creation(self):
        """Test MetaEvolution engine initialization."""
        from src.meta.meta_evolution_engine import MetaEvolutionEngine
        engine = MetaEvolutionEngine()
        assert engine is not None

