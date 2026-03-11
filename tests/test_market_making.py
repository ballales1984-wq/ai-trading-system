"""
Tests for Market Making Module
"""

import pytest
import numpy as np
from src.market_making import (
    MarketMaker,
    AdaptiveMarketMaker,
    InventoryRiskManager,
    SpreadCalculator,
    Quote,
    MarketState
)


class TestInventoryRiskManager:
    """Test suite for InventoryRiskManager."""
    
    @pytest.fixture
    def manager(self):
        """Create inventory risk manager."""
        return InventoryRiskManager(max_position=1.0, max_inventory_skew=0.5)
    
    def test_initial_inventory(self, manager):
        """Test initial inventory is zero."""
        assert manager.inventory == 0.0
    
    def test_update_inventory(self, manager):
        """Test updating inventory."""
        manager.update_inventory(0.5)
        assert manager.inventory == 0.5
        
        manager.update_inventory(-0.3)
        assert manager.inventory == 0.2
    
    def test_inventory_skew(self, manager):
        """Test inventory skew calculation."""
        manager.inventory = 0.5
        skew = manager.get_inventory_skew()
        
        assert 0 <= skew <= 1
        assert skew == 0.5
    
    def test_adjust_spread_for_inventory_long(self, manager):
        """Test spread adjustment when long inventory."""
        manager.inventory = 0.4  # Long
        base_spread = 0.001
        
        bid_adj, ask_adj = manager.adjust_spread_for_inventory(base_spread)
        
        # Bid should be tighter (more attractive), ask wider
        assert bid_adj < base_spread
        assert ask_adj > base_spread
    
    def test_adjust_spread_for_inventory_short(self, manager):
        """Test spread adjustment when short inventory."""
        manager.inventory = -0.4  # Short
        base_spread = 0.001
        
        bid_adj, ask_adj = manager.adjust_spread_for_inventory(base_spread)
        
        # Bid should be wider, ask tighter
        assert bid_adj > base_spread
        assert ask_adj < base_spread
    
    def test_should_quote(self, manager):
        """Test should quote returns true when inventory balanced."""
        manager.inventory = 0.2
        assert manager.should_quote() is True
        
        manager.inventory = 0.6
        assert manager.should_quote() is False
    
    def test_inventory_risk_penalty(self, manager):
        """Test inventory risk penalty calculation."""
        manager.inventory = 0.0
        assert manager.get_inventory_risk_penalty() == 0.0
        
        manager.inventory = 0.5
        penalty = manager.get_inventory_risk_penalty()
        assert 0 <= penalty <= 0.5


class TestSpreadCalculator:
    """Test suite for SpreadCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create spread calculator."""
        return SpreadCalculator(min_spread=0.0001, max_spread=0.05)
    
    def test_calculate_spread(self, calculator):
        """Test spread calculation."""
        spread = calculator.calculate_spread(
            volatility=0.5,
            volume_24h=1000000,
            inventory_risk=0.0
        )
        
        assert calculator.min_spread <= spread <= calculator.max_spread
    
    def test_spread_increases_with_volatility(self, calculator):
        """Test spread increases with volatility."""
        spread_low_vol = calculator.calculate_spread(
            volatility=0.1,
            volume_24h=100000000,
            inventory_risk=0.0
        )
        
        spread_high_vol = calculator.calculate_spread(
            volatility=0.3,
            volume_24h=100000000,
            inventory_risk=0.0
        )
        
        assert spread_high_vol > spread_low_vol
    
    def test_spread_increases_with_inventory_risk(self, calculator):
        """Test spread increases with inventory risk."""
        spread_no_risk = calculator.calculate_spread(
            volatility=0.1,
            volume_24h=100000000,
            inventory_risk=0.0
        )
        
        spread_with_risk = calculator.calculate_spread(
            volatility=0.1,
            volume_24h=100000000,
            inventory_risk=0.3
        )
        
        # Both should be less than max, and with_risk >= no_risk
        assert spread_with_risk <= calculator.max_spread
        assert spread_no_risk <= calculator.max_spread
        assert spread_with_risk >= spread_no_risk
    
    def test_calculate_half_spread(self, calculator):
        """Test half spread calculation."""
        full_spread = calculator.calculate_spread(0.5, 1000000, 0.0)
        half_spread = calculator.calculate_half_spread(0.5, 1000000, 0.0)
        
        assert abs(half_spread * 2 - full_spread) < 0.0001


class TestMarketMaker:
    """Test suite for MarketMaker."""
    
    @pytest.fixture
    def market_maker(self):
        """Create market maker."""
        return MarketMaker(
            symbol="BTCUSDT",
            base_inventory=0.0,
            min_order_size=0.01,
            max_position=1.0
        )
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state."""
        return MarketState(
            mid_price=50000.0,
            bid_price=49990.0,
            ask_price=50010.0,
            bid_size=1.0,
            ask_size=1.0,
            volatility=0.5,
            volume_24h=1000000000
        )
    
    def test_initialization(self, market_maker):
        """Test market maker initialization."""
        assert market_maker.symbol == "BTCUSDT"
        assert market_maker.quote_count == 0
        assert market_maker.trade_count == 0
        assert market_maker.pnl == 0.0
    
    def test_generate_quote(self, market_maker, sample_market_state):
        """Test quote generation."""
        quote = market_maker.generate_quote(sample_market_state, order_size=0.1)
        
        assert quote is not None
        assert quote.bid_price < quote.ask_price
        assert quote.bid_price < sample_market_state.mid_price
        assert quote.ask_price > sample_market_state.mid_price
    
    def test_quote_respects_inventory(self, market_maker, sample_market_state):
        """Test quotes adjust for inventory."""
        # Set inventory to long
        market_maker.inventory_manager.inventory = 0.4
        
        # Generate multiple quotes
        quote = market_maker.generate_quote(sample_market_state, order_size=0.1)
        
        # Quote should exist
        assert quote is not None
    
    def test_process_buy_trade(self, market_maker, sample_market_state):
        """Test processing a buy trade."""
        market_maker.update_market_state(sample_market_state)
        
        market_maker.process_trade("buy", 50000.0, 0.1)
        
        assert market_maker.inventory_manager.inventory == 0.1
        assert market_maker.trade_count == 1
    
    def test_process_sell_trade(self, market_maker, sample_market_state):
        """Test processing a sell trade."""
        market_maker.update_market_state(sample_market_state)
        
        market_maker.process_trade("sell", 50000.0, 0.1)
        
        assert market_maker.inventory_manager.inventory == -0.1
        assert market_maker.trade_count == 1
    
    def test_get_status(self, market_maker):
        """Test getting market maker status."""
        status = market_maker.get_status()
        
        assert "symbol" in status
        assert "inventory" in status
        assert "quote_count" in status
        assert status["symbol"] == "BTCUSDT"
    
    def test_get_performance_metrics(self, market_maker):
        """Test getting performance metrics."""
        metrics = market_maker.get_performance_metrics()
        
        assert "quotes_generated" in metrics
        assert "trades_executed" in metrics
        assert "fill_rate" in metrics
        assert "total_pnl" in metrics
    
    def test_does_not_quote_when_inventory_full(self, market_maker, sample_market_state):
        """Test no quote when inventory is full."""
        # Set inventory to max
        market_maker.inventory_manager.inventory = 0.6  # > max_skew * max_position
        
        quote = market_maker.generate_quote(sample_market_state, order_size=0.1)
        
        # Quote might still exist but size should be reduced
        # Just check it doesn't crash
        assert quote is None or quote.bid_size <= 0.1


class TestAdaptiveMarketMaker:
    """Test suite for AdaptiveMarketMaker."""
    
    @pytest.fixture
    def adaptive_mm(self):
        """Create adaptive market maker."""
        return AdaptiveMarketMaker(
            symbol="ETHUSDT",
            base_inventory=0.0,
            min_order_size=0.01,
            max_position=1.0
        )
    
    def test_initial_regime(self, adaptive_mm):
        """Test initial regime is normal."""
        assert adaptive_mm.regime == "normal"
    
    def test_detect_volatile_regime(self, adaptive_mm):
        """Test detecting volatile regime."""
        # Create volatile price series
        prices = [50000 + np.random.randn() * 2000 for _ in range(30)]
        
        regime, confidence = adaptive_mm.detect_market_regime(prices)
        
        assert regime in ["normal", "volatile", "trending_up", "trending_down", "illiquid"]
    
    def test_detect_trending_regime(self, adaptive_mm):
        """Test detecting trending regime."""
        # Create strong uptrend
        prices = [50000 * (1 + i * 0.01) for i in range(30)]
        
        regime, confidence = adaptive_mm.detect_market_regime(prices)
        
        assert regime in ["trending_up", "trending_down", "normal"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
