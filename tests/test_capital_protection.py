"""
Test Capital Protection Layer
=============================
Tests for institutional-grade capital protection system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.capital_protection import (
    CapitalProtectionEngine,
    ProtectionConfig,
    ProtectionLevel,
    TriggerType,
    PositionRisk,
    create_default_protection_engine
)


class TestProtectionConfig:
    """Test protection configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProtectionConfig()
        
        assert config.max_daily_loss_pct == 0.03
        assert config.max_drawdown_pct == 0.10
        assert config.max_single_position_pct == 0.15
        assert config.max_sector_exposure_pct == 0.30
        assert config.max_correlated_assets_pct == 0.40
        assert config.correlation_threshold == 0.70
        assert config.max_api_failures_per_hour == 10
        assert config.max_api_failures_per_day == 50
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProtectionConfig(
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
            max_single_position_pct=0.20
        )
        
        assert config.max_daily_loss_pct == 0.05
        assert config.max_drawdown_pct == 0.15
        assert config.max_single_position_pct == 0.20


class TestCapitalProtectionEngine:
    """Test capital protection engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a protection engine for testing."""
        config = ProtectionConfig(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_single_position_pct=0.15,
            max_api_failures_per_hour=5
        )
        return CapitalProtectionEngine(
            config=config,
            initial_capital=100000.0,
            state_file="data/test_protection_state.json"
        )
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.initial_capital == 100000.0
        assert engine.state.level == ProtectionLevel.NORMAL
        assert len(engine.state.active_triggers) == 0
    
    def test_normal_operation(self, engine):
        """Test normal operation without triggers."""
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=100000.0,
            daily_pnl=1000.0
        )
        
        assert level == ProtectionLevel.NORMAL
        assert len(triggers) == 0
        assert len(warnings) == 0
    
    def test_daily_loss_warning(self, engine):
        """Test daily loss warning threshold."""
        # 2% loss (warning at 2%)
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=98000.0,
            daily_pnl=-2000.0
        )
        
        assert level == ProtectionLevel.NORMAL
        assert len(warnings) > 0
        assert any("daily loss" in w.lower() for w in warnings)
    
    def test_daily_loss_limit_breach(self, engine):
        """Test daily loss limit breach."""
        # 3.5% loss (limit at 3%)
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=96500.0,
            daily_pnl=-3500.0
        )
        
        assert level == ProtectionLevel.FROZEN
        assert TriggerType.DAILY_LOSS in triggers
    
    def test_drawdown_warning(self, engine):
        """Test drawdown warning threshold."""
        # Set peak capital
        engine.state.peak_capital = 100000.0
        
        # 7.5% drawdown (warning at 7%)
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=92500.0,
            daily_pnl=0.0
        )
        
        assert len(warnings) > 0
        assert any("drawdown" in w.lower() for w in warnings)
    
    def test_drawdown_limit_breach(self, engine):
        """Test drawdown limit breach."""
        # Set peak capital
        engine.state.peak_capital = 100000.0
        
        # 11% drawdown (limit at 10%)
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=89000.0,
            daily_pnl=0.0
        )
        
        assert level == ProtectionLevel.FROZEN
        assert TriggerType.DRAWDOWN in triggers
    
    def test_position_size_limit(self, engine):
        """Test position size limit breach."""
        # Create position that exceeds limit
        positions = {
            "BTCUSDT": PositionRisk(
                symbol="BTCUSDT",
                market_value=20000.0,  # 20% of portfolio
                weight_pct=0.20,  # 20% > 15% limit
                sector="crypto",
                correlation_cluster=1,
                var_contribution=0.02,
                leverage=1.0
            )
        }
        
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=100000.0,
            positions=positions,
            daily_pnl=0.0
        )
        
        assert TriggerType.POSITION_SIZE in triggers
    
    def test_api_failure_tracking(self, engine):
        """Test API failure tracking."""
        # Record failures
        for i in range(5):
            engine.record_api_failure("binance", f"Error {i}")
        
        assert engine.state.api_failures_hour == 5
    
    def test_api_failure_limit_breach(self, engine):
        """Test API failure limit breach."""
        # Record failures to exceed limit
        for i in range(6):
            engine.record_api_failure("binance", f"Error {i}")
        
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=100000.0,
            daily_pnl=0.0
        )
        
        assert TriggerType.API_FAILURE in triggers
        assert level == ProtectionLevel.FROZEN
    
    def test_manual_kill_switch(self, engine):
        """Test manual kill switch activation."""
        engine.activate_manual_kill_switch("Test emergency")
        
        assert engine.state.level == ProtectionLevel.FROZEN
        assert TriggerType.MANUAL in engine.state.active_triggers
        assert "Test emergency" in engine.state.frozen_reason
    
    def test_kill_switch_deactivation(self, engine):
        """Test kill switch deactivation."""
        engine.activate_manual_kill_switch("Test")
        
        # Force deactivation
        result = engine.deactivate_kill_switch(force=True)
        
        assert result is True
        assert TriggerType.MANUAL not in engine.state.active_triggers
    
    def test_trading_permission_normal(self, engine):
        """Test trading permission in normal mode."""
        allowed, reason = engine.request_trading_permission()
        
        assert allowed is True
        assert "allowed" in reason.lower()
    
    def test_trading_permission_frozen(self, engine):
        """Test trading permission when frozen."""
        engine.activate_emergency_shutdown(
            TriggerType.DAILY_LOSS,
            "Daily loss limit reached"
        )
        
        allowed, reason = engine.request_trading_permission()
        
        assert allowed is False
        assert "frozen" in reason.lower()
    
    def test_get_status(self, engine):
        """Test status retrieval."""
        status = engine.get_status()
        
        assert "level" in status
        assert "active_triggers" in status
        assert "daily_pnl" in status
        assert "current_drawdown" in status
        assert "config" in status
    
    def test_correlation_clusters(self, engine):
        """Test correlation cluster detection."""
        # Create sample returns
        np.random.seed(42)
        returns = np.random.randn(5, 30)  # 5 assets, 30 days
        
        # Make some assets correlated
        returns[1] = returns[0] * 0.9 + np.random.randn(30) * 0.1
        returns[3] = returns[2] * 0.85 + np.random.randn(30) * 0.15
        
        symbols = ["BTC", "ETH", "SOL", "AVAX", "DOT"]
        
        engine.update_correlation_matrix(returns, symbols)
        clusters = engine.get_correlation_clusters()
        
        assert len(clusters) > 0
        # BTC and ETH should be in same cluster
        # SOL and AVAX should be in same cluster


class TestProtectionCallbacks:
    """Test protection callbacks."""
    
    def test_callback_triggered(self):
        """Test that callbacks are triggered on protection events."""
        engine = CapitalProtectionEngine(
            initial_capital=100000.0,
            state_file="data/test_protection_state.json"
        )
        
        callback_mock = Mock()
        engine.register_protection_callback(callback_mock)
        
        # Trigger protection
        engine.check_all_protections(
            portfolio_value=96000.0,  # 4% loss > 3% limit
            daily_pnl=-4000.0
        )
        
        # Callback should have been called
        assert callback_mock.called
        
        # Check event data
        call_args = callback_mock.call_args[0][0]
        assert "level" in call_args
        assert "triggers" in call_args


class TestStatePersistence:
    """Test state persistence."""
    
    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        state_file = tmp_path / "test_state.json"
        
        # Create engine and modify state
        engine = CapitalProtectionEngine(
            initial_capital=100000.0,
            state_file=str(state_file)
        )
        
        engine.state.peak_capital = 105000.0
        engine.state.current_drawdown = 0.05
        engine._save_state()
        
        # Create new engine to load state
        engine2 = CapitalProtectionEngine(
            initial_capital=100000.0,
            state_file=str(state_file)
        )
        
        assert engine2.state.peak_capital == 105000.0
        assert engine2.state.current_drawdown == 0.05


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_default_protection_engine(self):
        """Test default engine creation."""
        engine = create_default_protection_engine(
            initial_capital=50000.0,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.08
        )
        
        assert engine.initial_capital == 50000.0
        assert engine.config.max_daily_loss_pct == 0.02
        assert engine.config.max_drawdown_pct == 0.08


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_capital(self):
        """Test with zero capital."""
        engine = CapitalProtectionEngine(initial_capital=100000.0)
        
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=0.0,
            daily_pnl=-100000.0
        )
        
        # Should trigger drawdown
        assert level == ProtectionLevel.FROZEN
    
    def test_negative_pnl_recovery(self):
        """Test recovery from negative PnL."""
        engine = CapitalProtectionEngine(initial_capital=100000.0)
        
        # First, trigger a loss
        engine.check_all_protections(
            portfolio_value=97000.0,
            daily_pnl=-3000.0
        )
        
        # Then recover
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=100000.0,
            daily_pnl=0.0
        )
        
        # Should be back to normal after recovery
        assert engine.state.daily_pnl_pct == 0.0
    
    def test_multiple_triggers(self):
        """Test multiple simultaneous triggers."""
        engine = CapitalProtectionEngine(
            config=ProtectionConfig(
                max_daily_loss_pct=0.03,
                max_drawdown_pct=0.10,
                max_single_position_pct=0.15
            ),
            initial_capital=100000.0
        )
        
        # Set peak for drawdown calculation
        engine.state.peak_capital = 100000.0
        
        # Create oversized position
        positions = {
            "BTCUSDT": PositionRisk(
                symbol="BTCUSDT",
                market_value=20000.0,
                weight_pct=0.20,
                sector="crypto",
                correlation_cluster=1,
                var_contribution=0.02,
                leverage=1.0
            )
        }
        
        # Trigger both daily loss and position size
        level, triggers, warnings = engine.check_all_protections(
            portfolio_value=89000.0,  # 11% drawdown
            positions=positions,
            daily_pnl=-4000.0  # 4% daily loss
        )
        
        # Should have multiple triggers
        assert len(triggers) >= 2
        assert TriggerType.DAILY_LOSS in triggers
        assert TriggerType.POSITION_SIZE in triggers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
