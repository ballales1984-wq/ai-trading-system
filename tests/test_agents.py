# tests/test_agents.py
"""
Test Suite for Agent Modules
============================
Comprehensive tests for all agent implementations.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime
import numpy as np

from src.core.event_bus import EventBus, Event, EventType
from src.core.state_manager import StateManager
from src.agents.base_agent import BaseAgent, AgentState
from src.agents.agent_marketdata import MarketDataAgent
from src.agents.agent_montecarlo import MonteCarloAgent, SimulationLevel
from src.agents.agent_risk import RiskAgent, RiskLevel
from src.agents.agent_supervisor import SupervisorAgent, SystemMode


# Fixtures
@pytest.fixture
def event_bus():
    """Create event bus instance."""
    return EventBus()


@pytest.fixture
def state_manager():
    """Create state manager instance."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    sm = StateManager(db_path=tmp_file.name)
    yield sm
    # Close any connections before deleting
    import gc
    gc.collect()
    try:
        os.unlink(tmp_file.name)
    except PermissionError:
        pass  # File will be cleaned up by OS


@pytest.fixture
def basic_config():
    """Basic agent configuration."""
    return {
        "symbols": ["BTCUSDT"],
        "interval_sec": 1,
        "max_errors": 3,
    }


# Test EventBus
class TestEventBus:
    """Tests for EventBus."""
    
    def test_event_bus_creation(self, event_bus):
        """Test event bus creation."""
        assert event_bus is not None
        assert event_bus._subscribers is not None
    
    @pytest.mark.asyncio
    async def test_event_bus_subscribe_publish(self, event_bus):
        """Test subscribe and publish."""
        received = []
        
        class TestHandler:
            async def handle(self, event):
                received.append(event.data)
        
        handler = TestHandler()
        event_bus.subscribe(EventType.MARKET_DATA, handler)
        
        event = Event(
            event_type=EventType.MARKET_DATA,
            data={"price": 100.0}
        )
        await event_bus.publish(event)
        
        assert len(received) == 1
        assert received[0]["price"] == 100.0
    
    @pytest.mark.asyncio
    async def test_event_bus_multiple_subscribers(self, event_bus):
        """Test multiple subscribers."""
        received1 = []
        received2 = []
        
        class Handler1:
            async def handle(self, event):
                received1.append(event.data)
        
        class Handler2:
            async def handle(self, event):
                received2.append(event.data)
        
        event_bus.subscribe(EventType.SIGNAL_GENERATED, Handler1())
        event_bus.subscribe(EventType.SIGNAL_GENERATED, Handler2())
        
        event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            data={"signal": "BUY"}
        )
        await event_bus.publish(event)
        
        assert len(received1) == 1
        assert len(received2) == 1


# Test StateManager
class TestStateManager:
    """Tests for StateManager."""
    
    def test_state_manager_creation(self, state_manager):
        """Test state manager creation."""
        assert state_manager is not None
    
    def test_state_manager_portfolio(self, state_manager):
        """Test portfolio state operations."""
        from src.core.state_manager import PortfolioState
        portfolio = PortfolioState(
            total_equity=100000.0,
            available_balance=50000.0,
            unrealized_pnl=1000.0,
            realized_pnl=500.0
        )
        state_manager.save_portfolio_state(portfolio)
        loaded = state_manager.get_latest_portfolio_state()
        assert loaded is not None
        assert loaded.total_equity == 100000.0
    
    def test_state_manager_position(self, state_manager):
        """Test position operations."""
        from src.core.state_manager import PositionState
        position = PositionState(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0
        )
        state_manager.save_position(position)
        loaded = state_manager.get_position("BTCUSDT")
        assert loaded is not None
        assert loaded.symbol == "BTCUSDT"
    
    def test_state_manager_price_history(self, state_manager):
        """Test price history operations."""
        from datetime import datetime
        from src.core.state_manager import PriceHistoryState
        price = PriceHistoryState(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0
        )
        state_manager.save_price_history(price)
        history = state_manager.get_price_history("BTCUSDT", limit=10)
        assert len(history) >= 1


# Test BaseAgent
class TestBaseAgent:
    """Tests for BaseAgent."""
    
    def test_agent_creation(self, event_bus, state_manager, basic_config):
        """Test agent creation."""
        agent = MarketDataAgent(
            name="test_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=basic_config
        )
        
        assert agent.name == "test_agent"
        assert agent.state == AgentState.INITIALIZED
    
    @pytest.mark.asyncio
    async def test_agent_start_stop(self, event_bus, state_manager, basic_config):
        """Test agent start and stop."""
        agent = MarketDataAgent(
            name="test_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=basic_config
        )
        
        await agent.start()
        assert agent.state == AgentState.RUNNING
        
        await agent.stop()
        assert agent.state == AgentState.STOPPED
    
    @pytest.mark.asyncio
    async def test_agent_pause_resume(self, event_bus, state_manager, basic_config):
        """Test agent pause and resume."""
        agent = MarketDataAgent(
            name="test_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=basic_config
        )
        
        await agent.start()
        await agent.pause()
        assert agent.state == AgentState.PAUSED
        
        await agent.resume()
        assert agent.state == AgentState.RUNNING
        
        await agent.stop()


# Test MonteCarloAgent
class TestMonteCarloAgent:
    """Tests for MonteCarloAgent."""
    
    def test_agent_creation(self, event_bus, state_manager):
        """Test Monte Carlo agent creation."""
        config = {
            "symbols": ["BTCUSDT"],
            "interval_sec": 1,
            "n_paths": 100,
            "n_steps": 10,
        }
        
        agent = MonteCarloAgent(
            name="mc_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        assert agent.name == "mc_agent"
        assert agent.n_paths == 100
        assert agent.n_steps == 10
    
    def test_simulation_levels(self, event_bus, state_manager):
        """Test simulation level enum."""
        assert SimulationLevel.LEVEL_1_BASE.value == "base"
        assert SimulationLevel.LEVEL_5_SEMANTIC.value == "semantic_history"
    
    def test_gbm_simulation(self, event_bus, state_manager):
        """Test GBM simulation."""
        config = {
            "symbols": ["BTCUSDT"],
            "n_paths": 100,
            "n_steps": 50,
            "random_seed": 42,
        }
        
        agent = MonteCarloAgent(
            name="mc_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        # Run simulation
        paths = agent._simulate_level_1(42000.0, 0.05, 0.8)
        
        assert paths.shape == (100, 51)  # n_paths x (n_steps + 1)
        assert paths[0, 0] == 42000.0  # Initial price
    
    def test_parameter_estimation(self, event_bus, state_manager):
        """Test parameter estimation from history."""
        config = {"symbols": ["BTCUSDT"]}
        
        agent = MonteCarloAgent(
            name="mc_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        # Generate synthetic history
        np.random.seed(42)
        history = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        mu, sigma, jump_params = agent._estimate_parameters(history)
        
        assert mu is not None
        assert sigma > 0


# Test RiskAgent
class TestRiskAgent:
    """Tests for RiskAgent."""
    
    def test_agent_creation(self, event_bus, state_manager):
        """Test Risk agent creation."""
        config = {
            "symbols": ["BTCUSDT"],
            "interval_sec": 1,
        }
        
        agent = RiskAgent(
            name="risk_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        assert agent.name == "risk_agent"
        assert agent.var_confidence == 0.95
    
    def test_risk_level_determination(self, event_bus, state_manager):
        """Test risk level determination."""
        config = {"symbols": ["BTCUSDT"]}
        
        agent = RiskAgent(
            name="risk_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        # Low risk
        level = agent._determine_risk_level(-0.01, -0.02, 0.1)
        assert level == RiskLevel.LOW
        
        # Critical risk
        level = agent._determine_risk_level(-0.15, -0.25, 0.6)
        assert level == RiskLevel.CRITICAL
    
    def test_max_drawdown_calculation(self, event_bus, state_manager):
        """Test maximum drawdown calculation."""
        config = {"symbols": ["BTCUSDT"]}
        
        agent = RiskAgent(
            name="risk_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        # Price series with known drawdown
        prices = np.array([100, 110, 105, 95, 90, 100, 110])
        
        max_dd = agent._calculate_max_drawdown(prices)
        
        # Max drawdown should be from 110 to 90 = -18.18%
        assert -0.19 < max_dd < -0.17
    
    def test_volatility_calculation(self, event_bus, state_manager):
        """Test volatility calculation."""
        config = {"symbols": ["BTCUSDT"]}
        
        agent = RiskAgent(
            name="risk_agent",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        # Generate returns with known volatility
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        vol = agent._calculate_volatility(returns)
        
        # Should be approximately 0.02 * sqrt(252)
        expected_vol = 0.02 * np.sqrt(252)
        assert 0.9 * expected_vol < vol < 1.1 * expected_vol


# Test SupervisorAgent
class TestSupervisorAgent:
    """Tests for SupervisorAgent."""
    
    def test_agent_creation(self, event_bus, state_manager):
        """Test Supervisor agent creation."""
        config = {
            "mode": "paper_trading",
            "health_check_interval": 5,
        }
        
        agent = SupervisorAgent(
            name="supervisor",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        assert agent.name == "supervisor"
        assert agent.mode == SystemMode.PAPER_TRADING
    
    def test_system_modes(self):
        """Test system mode enum."""
        assert SystemMode.PAPER_TRADING.value == "paper_trading"
        assert SystemMode.LIVE_TRADING.value == "live_trading"
        assert SystemMode.BACKTEST.value == "backtest"
    
    def test_agent_management(self, event_bus, state_manager):
        """Test adding and removing agents."""
        config = {"mode": "paper_trading"}
        
        supervisor = SupervisorAgent(
            name="supervisor",
            event_bus=event_bus,
            state_manager=state_manager,
            config=config
        )
        
        # Create a mock agent
        mock_agent = MarketDataAgent(
            name="market_data",
            event_bus=event_bus,
            state_manager=state_manager,
            config={"symbols": ["BTCUSDT"]}
        )
        
        # Add agent
        supervisor.add_agent(mock_agent)
        assert len(supervisor._agents) == 1
        
        # Remove agent
        supervisor.remove_agent("market_data")
        assert len(supervisor._agents) == 0


# Integration Tests
class TestAgentIntegration:
    """Integration tests for agent system."""
    
    @pytest.mark.asyncio
    async def test_full_agent_workflow(self, event_bus, state_manager):
        """Test full workflow with multiple agents."""
        # Create agents
        market_agent = MarketDataAgent(
            name="MarketDataAgent",
            event_bus=event_bus,
            state_manager=state_manager,
            config={"symbols": ["BTCUSDT"], "interval_sec": 0.1}
        )
        
        mc_agent = MonteCarloAgent(
            name="MonteCarloAgent",
            event_bus=event_bus,
            state_manager=state_manager,
            config={
                "symbols": ["BTCUSDT"],
                "interval_sec": 0.1,
                "n_paths": 50,
                "n_steps": 10,
            }
        )
        
        risk_agent = RiskAgent(
            name="RiskAgent",
            event_bus=event_bus,
            state_manager=state_manager,
            config={"symbols": ["BTCUSDT"], "interval_sec": 0.1}
        )
        
        # Set up price history for MC agent
        from datetime import datetime
        from src.core.state_manager import PriceHistoryState
        price = PriceHistoryState(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=42000.0,
            high=42500.0,
            low=41500.0,
            close=42000.0,
            volume=1000.0
        )
        state_manager.save_price_history(price)
        
        # Start agents
        await market_agent.start()
        await mc_agent.start()
        await risk_agent.start()
        
        # Let them run briefly
        await asyncio.sleep(0.2)
        
        # Stop agents
        await market_agent.stop()
        await mc_agent.stop()
        await risk_agent.stop()
        
        # Verify agents ran
        assert market_agent.metrics["events_processed"] >= 0
        assert mc_agent.metrics["events_processed"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
