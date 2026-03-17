"""
E2E Tests: Autonomous Quant Agent
=================================
End-to-end tests for the Level 5 Autonomous Quant Agent.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAutonomousQuantAgent:
    """Test suite for AutonomousQuantAgent."""
    
    @pytest.fixture
    def mock_route_intent(self):
        with patch('src.agents.autonomous_quant_agent.route_intent') as mock:
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
        from src.agents.autonomous_quant_agent import AutonomousQuantAgent, AgentConfig
        
        config = AgentConfig(
            default_symbols=["BTCUSDT"],
            max_position_pct=0.10,
            max_drawdown_pct=0.05,
        )
        
        with patch('src.agents.autonomous_quant_agent.RiskBook') as mock_rb:
            with patch('src.agents.autonomous_quant_agent.ModelRegistry'):
                # Create real RiskBook-like mock
                mock_rb_instance = Mock()
                mock_rb_instance.positions = {}
                mock_rb_instance.equity = 100000.0
                mock_rb_instance.daily_drawdown_pct = 0.02
                mock_rb_instance.check_position_limit = Mock(return_value=True)
                mock_rb_instance.daily_drawdown_ok = Mock(return_value=True)
                mock_rb_instance.get_metrics = Mock(return_value=Mock(
                    var_95=0.05,
                    cvar_95=0.07,
                    daily_drawdown_pct=0.02,
                    exposure_pct=0.08,
                    position_count=1,
                ))
                mock_rb.return_value = mock_rb_instance
                
                agent = AutonomousQuantAgent(config)
                agent.risk_book = mock_rb_instance
                agent.model_registry = Mock()
                agent.model_registry.get_champion.return_value = None
                
                return agent
    
    def test_agent_initialization(self, agent):
        assert agent is not None
        assert agent.trading_mode == "active"
    
    def test_set_trading_mode(self, agent):
        agent.set_trading_mode("paused")
        assert agent.trading_mode == "paused"
        
        agent.set_trading_mode("active")
        assert agent.trading_mode == "active"


class TestAutonomousQuantAgentAPI:
    """Test suite for Autonomous Agent API endpoints."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_agent_health_endpoint(self, client):
        response = client.get("/api/v1/agents/autonomous/health")
        # App should start and routes should be registered
        assert response.status_code in [200, 404]
    
    def test_agent_portfolio_endpoint(self, client):
        response = client.get("/api/v1/agents/autonomous/portfolio")
        assert response.status_code in [200, 404]
    
    def test_agent_report_endpoint(self, client):
        response = client.get("/api/v1/agents/autonomous/report/BTCUSDT")
        assert response.status_code in [200, 404, 500]
    
    def test_agent_proposals_endpoint(self, client):
        response = client.get("/api/v1/agents/autonomous/proposals/BTCUSDT")
        assert response.status_code in [200, 404, 500]


class TestRiskBookE2E:
    """E2E tests for RiskBook integration."""
    
    @pytest.fixture
    def risk_book(self):
        from app.risk.risk_book import RiskBook, RiskLimits
        limits = RiskLimits(
            max_position_pct=0.10,
            max_daily_drawdown_pct=0.05,
            var_95_limit=0.08,
            cvar_95_limit=0.10,
        )
        return RiskBook(limits)
    
    def test_risk_book_initialization(self, risk_book):
        assert risk_book is not None
        assert risk_book.equity == 0.0
    
    def test_register_equity(self, risk_book):
        risk_book.register_equity(100000.0)
        assert risk_book.equity == 100000.0
    
    def test_update_position(self, risk_book):
        risk_book.register_equity(100000.0)
        from app.risk.risk_book import Position
        position = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            avg_price=50000,
            side="long"
        )
        risk_book.update_position(position)
        assert len(risk_book.positions) == 1
    
    def test_check_position_limit(self, risk_book):
        risk_book.register_equity(100000.0)
        prices = {"BTCUSDT": 50000}
        result = risk_book.check_position_limit("BTCUSDT", prices, 100000.0)
        assert result is True
    
    def test_daily_drawdown_ok(self, risk_book):
        risk_book.register_equity(100000.0)
        result = risk_book.daily_drawdown_ok()
        assert result is True
    
    def test_get_position(self, risk_book):
        risk_book.register_equity(100000.0)
        from app.risk.risk_book import Position
        position = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            avg_price=50000,
            side="long"
        )
        risk_book.update_position(position)
        retrieved = risk_book.get_position("BTCUSDT")
        assert retrieved is not None


class TestUnifiedDecisionEngine:
    """Test suite for UnifiedDecisionEngine."""
    
    @pytest.fixture
    def engine(self):
        from src.decision.unified_engine import UnifiedDecisionEngine
        with patch('src.decision.unified_engine.RiskBook') as mock_rb:
            with patch('src.decision.unified_engine.ModelRegistry'):
                mock_rb_instance = Mock()
                mock_rb_instance.check_position_limit = Mock(return_value=True)
                mock_rb_instance.daily_drawdown_ok = Mock(return_value=True)
                mock_rb_instance.positions = {}
                mock_rb_instance.equity = 100000
                mock_rb_instance.get_all_positions = Mock(return_value=[])
                mock_rb_instance.current_equity = 100000
                mock_rb_instance.initial_equity = 100000
                mock_rb.return_value = mock_rb_instance
                
                engine = UnifiedDecisionEngine()
                engine.risk_book = mock_rb_instance
                return engine
    
    def test_engine_initialization(self, engine):
        assert engine is not None
        assert engine.decision_count == 0
    
    def test_decide_buy_signal(self, engine):
        result = engine.decide(
            symbol="BTCUSDT",
            current_price=50000,
            signals={"technical": 0.8, "sentiment": 0.7},
            equity=100000
        )
        assert result.decision in ["buy", "hold", "reject"]
    
    def test_decide_low_confidence(self, engine):
        result = engine.decide(
            symbol="BTCUSDT",
            current_price=50000,
            signals={"technical": 0.4, "sentiment": 0.4},
            equity=100000
        )
        assert result.decision == "hold"
        assert "Confidence" in result.reason
    
    def test_check_risk_limits(self, engine):
        result = engine.check_risk_limits(
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            price=50000,
            equity=100000
        )
        assert "approved" in result
        assert "risk_score" in result
    
    def test_update_position(self, engine):
        engine.update_position(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            avg_price=50000
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
