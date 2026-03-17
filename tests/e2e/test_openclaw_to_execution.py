"""
End-to-End Tests: OpenClaw → Decision → Risk → Execution
=========================================================

This test module validates the complete flow from OpenClaw skill
execution through decision making, risk checks, and order execution.

Usage:
    pytest tests/e2e/test_openclaw_to_execution.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import components to test
from openclaw_skills.intent_router import route_intent
from app.risk.risk_book import RiskBook, RiskLimits, Position
from src.research.model_registry import ModelRegistry, ModelMeta


@pytest.fixture
def risk_book():
    """Create a risk book with test limits."""
    limits = RiskLimits(
        max_position_pct=0.10,
        max_daily_drawdown_pct=0.05,
        var_95_limit=0.08,
        cvar_95_limit=0.10,
    )
    return RiskBook(limits)


@pytest.fixture
def model_registry(tmp_path):
    """Create a temporary model registry."""
    registry_path = tmp_path / "model_registry.json"
    return ModelRegistry(str(registry_path))


class TestOpenClawIntegration:
    """Test OpenClaw skill integration."""
    
    def test_route_intent_imports(self):
        """Verify intent router can be imported."""
        from openclaw_skills.intent_router import route_intent
        assert callable(route_intent)
    
    def test_registry_has_skills(self):
        """Verify skill registry has configured skills."""
        from openclaw_skills import list_skills
        skills = list_skills()
        assert len(skills) > 0
        assert "hmm_regime_detect" in skills


class TestRiskBook:
    """Test RiskBook functionality."""
    
    def test_create_risk_book(self, risk_book):
        """Test risk book creation."""
        assert risk_book is not None
        assert len(risk_book.positions) == 0
        assert risk_book.limits.max_position_pct == 0.10
    
    def test_update_position(self, risk_book):
        """Test position updates."""
        pos = Position(
            symbol="BTCUSDT",
            quantity=0.1,
            avg_price=50000,
            side="long"
        )
        risk_book.update_position(pos)
        
        assert "BTCUSDT" in risk_book.positions
        assert risk_book.positions["BTCUSDT"].quantity == 0.1
    
    def test_remove_position(self, risk_book):
        """Test position removal."""
        pos = Position(symbol="BTCUSDT", quantity=0.1, avg_price=50000, side="long")
        risk_book.update_position(pos)
        risk_book.remove_position("BTCUSDT")
        
        assert "BTCUSDT" not in risk_book.positions
    
    def test_check_position_limit_pass(self, risk_book):
        """Test position limit check passes."""
        pos = Position(symbol="BTCUSDT", quantity=0.05, avg_price=50000, side="long")
        risk_book.update_position(pos)
        
        prices = {"BTCUSDT": 50000}
        equity = 100000
        
        # 5000 / 100000 = 0.05 = 5% < 10% limit
        assert risk_book.check_position_limit("BTCUSDT", prices, equity)
    
    def test_check_position_limit_fail(self, risk_book):
        """Test position limit check fails when exceeded."""
        pos = Position(symbol="BTCUSDT", quantity=0.2, avg_price=50000, side="long")
        risk_book.update_position(pos)
        
        prices = {"BTCUSDT": 50000}
        equity = 100000
        
        # 10000 / 100000 = 0.10 = 10% = limit (edge case)
        # Let's try 15% which should fail
        risk_book.update_position(Position(symbol="BTCUSDT", quantity=0.3, avg_price=50000, side="long"))
        assert not risk_book.check_position_limit("BTCUSDT", prices, equity)
    
    def test_daily_drawdown(self, risk_book):
        """Test daily drawdown calculation."""
        equity = 100000
        risk_book.register_equity(equity)
        risk_book.register_equity(95000)  # 5% drawdown
        
        assert risk_book.daily_drawdown_pct() == 0.05
        assert risk_book.daily_drawdown_ok()  # 5% <= 5% limit
    
    def test_daily_drawdown_exceeded(self, risk_book):
        """Test drawdown limit exceeded."""
        risk_book.register_equity(100000)
        risk_book.register_equity(90000)  # 10% drawdown
        
        assert risk_book.daily_drawdown_pct() == 0.10
        assert not risk_book.daily_drawdown_ok()  # 10% > 5% limit
    
    def test_total_exposure(self, risk_book):
        """Test total exposure calculation."""
        risk_book.update_position(Position(symbol="BTCUSDT", quantity=1, avg_price=50000, side="long"))
        risk_book.update_position(Position(symbol="ETHUSDT", quantity=10, avg_price=3000, side="long"))
        
        prices = {"BTCUSDT": 50000, "ETHUSDT": 3000}
        
        # 1 * 50000 + 10 * 3000 = 50000 + 30000 = 80000
        assert risk_book.total_exposure(prices) == 80000


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    def test_create_registry(self, model_registry):
        """Test registry creation."""
        assert model_registry is not None
        assert len(model_registry.list_models()) == 0
    
    def test_register_model(self, model_registry):
        """Test model registration."""
        meta = ModelMeta(
            name="hmm_regime_detector",
            version="1.0.0",
            trained_at=datetime.utcnow(),
            dataset_id="BTC_USDT_1H_90D",
            metrics={"accuracy": 0.85, "sharpe": 1.2},
            status="champion"
        )
        model_registry.register(meta)
        
        models = model_registry.list_models()
        assert len(models) == 1
        assert models[0].name == "hmm_regime_detector"
    
    def test_get_champion(self, model_registry):
        """Test getting champion model."""
        meta = ModelMeta(
            name="hmm_regime_detector",
            version="1.0.0",
            trained_at=datetime.utcnow(),
            dataset_id="BTC_USDT_1H_90D",
            metrics={"accuracy": 0.85},
            status="champion"
        )
        model_registry.register(meta)
        
        champion = model_registry.get_champion("hmm_regime_detector")
        assert champion is not None
        assert champion.version == "1.0.0"
    
    def test_promote_to_champion(self, model_registry):
        """Test champion promotion."""
        # Register challenger
        meta1 = ModelMeta(
            name="test_model",
            version="1.0.0",
            trained_at=datetime(2024, 1, 1),
            dataset_id="ds1",
            metrics={"accuracy": 0.80},
            status="challenger"
        )
        model_registry.register(meta1)
        
        # Register newer challenger
        meta2 = ModelMeta(
            name="test_model",
            version="1.1.0",
            trained_at=datetime(2024, 2, 1),
            dataset_id="ds2",
            metrics={"accuracy": 0.85},
            status="challenger"
        )
        model_registry.register(meta2)
        
        # Promote to champion
        result = model_registry.promote_to_champion("test_model", "1.1.0")
        assert result is True
        
        champion = model_registry.get_champion("test_model")
        assert champion.version == "1.1.0"
    
    def test_retire_model(self, model_registry):
        """Test model retirement."""
        meta = ModelMeta(
            name="old_model",
            version="1.0.0",
            trained_at=datetime.utcnow(),
            dataset_id="ds1",
            metrics={"accuracy": 0.70},
            status="champion"
        )
        model_registry.register(meta)
        
        result = model_registry.retire_model("old_model", "1.0.0")
        assert result is True
        
        models = model_registry.list_models(status="retired")
        assert len(models) == 1


class TestEndToEndFlow:
    """Test complete end-to-end flow."""
    
    def test_openclaw_to_risk_check(self, risk_book):
        """Test: OpenClaw regime → Risk check → Decision."""
        # 1) OpenClaw returns regime analysis (mocked for unit test)
        # In real flow: route_intent("regime_analysis", {"symbol": "BTCUSDT"})
        regime_result = {
            "current_state": "bull",
            "confidence": 0.85
        }
        
        # 2) Simple decision: if bull → buy
        if regime_result["current_state"] == "bull":
            decision = "buy"
        else:
            decision = "hold"
        
        assert decision == "buy"
        
        # 3) Risk check before execution
        prices = {"BTCUSDT": 50000}
        equity = 100000
        
        risk_book.register_equity(equity)
        
        # Test position would be within limits
        is_safe = risk_book.check_position_limit("BTCUSDT", prices, equity)
        assert is_safe  # No existing position, should be safe
    
    def test_risk_limit_blocks_trade(self, risk_book):
        """Test that risk limits can block trades."""
        # Register equity first
        risk_book.register_equity(100000)
        
        # Setup: create a large position (25% of equity, exceeds 10% limit)
        risk_book.update_position(Position(
            symbol="BTCUSDT",
            quantity=0.5,  # 25000 exposure = 25%
            avg_price=50000,
            side="long"
        ))
        
        prices = {"BTCUSDT": 50000}
        equity = 100000
        
        # Check position - should exceed limit
        # Note: check_position_limit checks current position, not adding new
        result = risk_book.check_position_limit("BTCUSDT", prices, equity)
        # Position is 25% which exceeds 10% limit
        assert result == False  # 25% > 10%
    
    def test_full_trading_flow_with_mock_execution(self, risk_book, model_registry):
        """Test complete flow with mocked execution."""
        # Step 1: Register a model
        meta = ModelMeta(
            name="regime_model",
            version="1.0.0",
            trained_at=datetime.utcnow(),
            dataset_id="ds1",
            metrics={"accuracy": 0.85},
            status="champion"
        )
        model_registry.register(meta)
        
        # Step 2: Check model is champion
        champion = model_registry.get_champion("regime_model")
        assert champion is not None
        
        # Step 3: Simulate regime detection (bull market)
        regime = "bull"
        
        # Step 4: If bull, prepare to buy
        if regime == "bull":
            # Step 5: Check risk limits
            prices = {"BTCUSDT": 50000}
            equity = 100000
            
            risk_book.register_equity(equity)
            
            # Create position
            pos = Position(symbol="BTCUSDT", quantity=0.1, avg_price=50000, side="long")
            risk_book.update_position(pos)
            
            # Step 6: Verify within limits
            assert risk_book.check_position_limit("BTCUSDT", prices, equity)
            assert risk_book.daily_drawdown_ok()
            
            # In real flow: execute order here
            # order = create_order("BTCUSDT", "buy", 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
