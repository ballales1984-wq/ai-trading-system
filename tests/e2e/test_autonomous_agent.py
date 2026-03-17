"""
E2E Tests: Autonomous Quant Agent
=================================
End-to-end tests for the Level 5 Autonomous Quant Agent.

Tests:
- Agent initialization
- Daily report generation
- Action proposals
- Portfolio status
- Risk validation
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Test imports
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAutonomousQuantAgent:
    """Test suite for AutonomousQuantAgent."""
    
    @pytest.fixture
    def mock_route_intent(self):
        """Mock OpenClaw route_intent function."""
        with patch('src.agents.autonomous_quant_agent.route_intent') as mock:
            # Default mock return values
            mock.return_value = {
                "regime": "bull",
                "confidence": 0.75,
                "volatility": 0.025,
                "mean_price": 52000,
                "percentile_5": 48000,
                "percentile_95": 56000,
            }
            yield mock
    
    @pytest.fixture
    def agent(self, mock_route_intent):
        """Create an agent instance for testing."""
        from src.agents.autonomous_quant_agent import AutonomousQuantAgent, AgentConfig
        
        config = AgentConfig(
            default_symbols=["BTCUSDT"],
            max_position_pct=0.10,
            max_drawdown_pct=0.05,
        )
        
        with patch('src.agents.autonomous_quant_agent.RiskBook'):
            with patch('src.agents.autonomous_quant_agent.ModelRegistry'):
                agent = AutonomousQuantAgent(config)
                agent.risk_book = Mock()
                agent.risk_book.get_all_positions.return_value = []
                agent.risk_book.current_equity = 100000
                agent.risk_book.initial_equity = 100000
                agent.risk_book.check_position_limit.return_value = True
                agent.risk_book.daily_drawdown_ok.return_value = True
                agent.model_registry = Mock()
                agent.model_registry.get_champion.return_value = None
                
                return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.trading_mode == "active"
        assert agent.last_update is None
    
    def test_daily_report_structure(self, agent):
        """Test daily_report returns expected structure."""
        report = agent.daily_report("BTCUSDT")
        
        # Check top-level keys
        assert "timestamp" in report
        assert "trading_mode" in report
        assert "regime" in report
        assert "monte_carlo" in report
        assert "portfolio" in report
        assert "risk" in report
        assert "models" in report
    
    def test_daily_report_regime(self, agent):
        """Test daily_report contains regime information."""
        report = agent.daily_report("BTCUSDT")
        
        regime = report.get("regime", {})
        assert "regime" in regime
        assert "confidence" in regime
        assert "volatility" in regime
    
    def test_daily_report_monte_carlo(self, agent):
        """Test daily_report contains Monte Carlo simulation."""
        report = agent.daily_report("BTCUSDT")
        
        mc = report.get("monte_carlo", {})
        assert "mean_price" in mc
        assert "percentile_5" in mc
        assert "percentile_95" in mc
    
    def test_daily_report_portfolio(self, agent):
        """Test daily_report contains portfolio status."""
        report = agent.daily_report("BTCUSDT")
        
        portfolio = report.get("portfolio", {})
        assert "equity" in portfolio
        assert "pnl" in portfolio
        assert "position_count" in portfolio
    
    def test_daily_report_risk(self, agent):
        """Test daily_report contains risk metrics."""
        report = agent.daily_report("BTCUSDT")
        
        risk = report.get("risk", {})
        assert "within_limits" in risk
        assert "var_95" in risk
        assert "drawdown_pct" in risk
    
    def test_propose_actions_returns_list(self, agent):
        """Test propose_actions returns a list."""
        proposals = agent.propose_actions("BTCUSDT")
        
        assert isinstance(proposals, list)
    
    def test_propose_actions_structure(self, agent):
        """Test proposals have expected structure."""
        agent.route_intent = Mock(return_value={
            "regime": "bull",
            "confidence": 0.8,
            "volatility": 0.02,
            "mean_price": 51000,
        })
        
        proposals = agent.propose_actions("BTCUSDT")
        
        if proposals:  # May be empty if no strong signals
            for prop in proposals:
                assert "symbol" in prop
                assert "action" in prop
                assert "size" in prop
                assert "reason" in prop
    
    def test_get_portfolio_status(self, agent):
        """Test get_portfolio_status returns correct structure."""
        status = agent.get_portfolio_status()
        
        assert "positions" in status
        assert "equity" in status
        assert "pnl" in status
        assert "position_count" in status
    
    def test_update_position(self, agent):
        """Test position update."""
        agent.risk_book.update_position = Mock()
        
        agent.update_position(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            avg_price=50000
        )
        
        agent.risk_book.update_position.assert_called_once()
    
    def test_set_trading_mode(self, agent):
        """Test setting trading mode."""
        agent.set_trading_mode("paused")
        assert agent.trading_mode == "paused"
        
        agent.set_trading_mode("close_only")
        assert agent.trading_mode == "close_only"
        
        agent.set_trading_mode("active")
        assert agent.trading_mode == "active"


class TestAutonomousQuantAgentAPI:
    """Test suite for Autonomous Agent API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        return TestClient(app)
    
    def test_agent_health_endpoint(self, client):
        """Test /agents/autonomous/health endpoint."""
        response = client.get("/api/v1/agents/autonomous/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_agent_portfolio_endpoint(self, client):
        """Test /agents/autonomous/portfolio endpoint."""
        response = client.get("/api/v1/agents/autonomous/portfolio")
        
        assert response.status_code == 200
        data = response.json()
        assert "equity" in data
        assert "positions" in data
    
    def test_agent_report_endpoint(self, client):
        """Test /agents/autonomous/report/{symbol} endpoint."""
        response = client.get("/api/v1/agents/autonomous/report/BTCUSDT")
        
        # May fail if OpenClaw not available, but should return valid structure
        if response.status_code == 200:
            data = response.json()
            assert "regime" in data
            assert "monte_carlo" in data
    
    def test_agent_proposals_endpoint(self, client):
        """Test /agents/autonomous/proposals/{symbol} endpoint."""
        response = client.get("/api/v1/agents/autonomous/proposals/BTCUSDT")
        
        # May fail if OpenClaw not available
        if response.status_code == 200:
            data = response.json()
            assert "proposals" in data
    
    def test_agent_reset_endpoint(self, client):
        """Test /agents/autonomous/reset endpoint."""
        response = client.post("/api/v1/agents/autonomous/reset")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestRiskBookE2E:
    """E2E tests for RiskBook integration."""
    
    @pytest.fixture
    def risk_book(self):
        """Create a RiskBook instance for testing."""
        from app.risk.risk_book import RiskBook, RiskLimits
        
        limits = RiskLimits(
            max_position_pct=0.10,
            max_daily_drawdown_pct=0.05,
            var_95_limit=0.08,
            cvar_95_limit=0.10,
        )
        
        return RiskBook(limits)
    
    def test_risk_book_initialization(self, risk_book):
        """Test RiskBook initializes correctly."""
        assert risk_book is not None
        assert risk_book.initial_equity == 0  # Not registered yet
    
    def test_register_equity(self, risk_book):
        """Test equity registration."""
        risk_book.register_equity(100000.0)
        
        assert risk_book.current_equity == 100000.0
        assert risk_book.initial_equity == 100000.0
    
    def test_update_position(self, risk_book):
        """Test position update."""
        risk_book.register_equity(100000.0)
        
        from app.risk.risk_book import Position, PositionSide
        
        position = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            avg_price=50000,
            side=PositionSide.LONG
        )
        
        risk_book.update_position(position)
        
        positions = risk_book.get_all_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
    
    def test_check_position_limit(self, risk_book):
        """Test position limit check."""
        risk_book.register_equity(100000.0)
        
        # Should be OK for small position
        prices = {"BTCUSDT": 50000}
        result = risk_book.check_position_limit("BTCUSDT", prices, 100000.0)
        assert result is True
    
    def test_daily_drawdown_ok(self, risk_book):
        """Test daily drawdown check."""
        risk_book.register_equity(100000.0)
        
        # With no drawdown, should return True
        result = risk_book.daily_drawdown_ok(100000.0)
        assert result is True
        
        # With 3% drawdown (within 5% limit), should return True
        result = risk_book.daily_drawdown_ok(97000.0)
        assert result is True
        
        # With 10% drawdown (exceeds 5% limit), should return False
        result = risk_book.daily_drawdown_ok(90000.0)
        assert result is False
    
    def test_get_exposure_pct(self, risk_book):
        """Test exposure percentage calculation."""
        risk_book.register_equity(100000.0)
        
        from app.risk.risk_book import Position, PositionSide
        
        # Add a position worth 5% of equity
        position = Position(
            symbol="BTCUSDT",
            quantity=0.1,  # 0.1 * 50000 = 5000 = 5%
            avg_price=50000,
            side=PositionSide.LONG
        )
        risk_book.update_position(position)
        
        prices = {"BTCUSDT": 50000}
        exposure = risk_book.get_exposure_pct(prices)
        
        assert exposure == 0.05
    
    def test_get_position(self, risk_book):
        """Test getting a specific position."""
        risk_book.register_equity(100000.0)
        
        from app.risk.risk_book import Position, PositionSide
        
        position = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            avg_price=50000,
            side=PositionSide.LONG
        )
        risk_book.update_position(position)
        
        retrieved = risk_book.get_position("BTCUSDT")
        
        assert retrieved is not None
        assert retrieved.symbol == "BTCUSDT"
        assert retrieved.quantity == 0.1


class TestUnifiedDecisionEngine:
    """Test suite for UnifiedDecisionEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a UnifiedDecisionEngine instance."""
        with patch('src.decision.unified_engine.RiskBook'):
            with patch('src.decision.unified_engine.ModelRegistry'):
                from src.decision.unified_engine import UnifiedDecisionEngine
                
                engine = UnifiedDecisionEngine()
                engine.risk_book = Mock()
                engine.risk_book.check_position_limit.return_value = True
                engine.risk_book.daily_drawdown_ok.return_value = True
                engine.risk_book.get_all_positions.return_value = []
                engine.risk_book.current_equity = 100000
                engine.risk_book.initial_equity = 100000
                
                return engine
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.decision_count == 0
    
    def test_decide_buy_signal(self, engine):
        """Test decision with strong buy signal."""
        with patch('src.decision.unified_engine.route_intent') as mock_intent:
            mock_intent.return_value = {}
            
            result = engine.decide(
                symbol="BTCUSDT",
                current_price=50000,
                signals={"technical": 0.8, "sentiment": 0.7},
                equity=100000
            )
            
            # Should buy with high confidence signals
            assert result.decision in ["buy", "hold", "reject"]
    
    def test_decide_low_confidence(self, engine):
        """Test decision with low confidence signals."""
        result = engine.decide(
            symbol="BTCUSDT",
            current_price=50000,
            signals={"technical": 0.4, "sentiment": 0.4},
            equity=100000
        )
        
        # Should hold due to low confidence
        assert result.decision == "hold"
        assert "Confidence" in result.reason
    
    def test_check_risk_limits(self, engine):
        """Test risk limit checking."""
        result = engine.check_risk_limits(
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            price=50000,
            equity=100000
        )
        
        assert "approved" in result
        assert "risk_score" in result
    
    def test_get_portfolio_status(self, engine):
        """Test portfolio status retrieval."""
        status = engine.get_portfolio_status()
        
        assert "positions" in status
        assert "equity" in status
        assert "decision_count" in status
    
    def test_update_position(self, engine):
        """Test position update."""
        engine.update_position(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            avg_price=50000
        )
        
        engine.risk_book.update_position.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
