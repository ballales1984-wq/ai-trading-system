"""
Tests for Options Pricing Module
"""

import pytest
import math
from src.options_pricing import (
    OptionsPricer,
    OptionsStrategies,
    OptionGreeks
)


class TestOptionsPricer:
    """Test suite for OptionsPricer."""
    
    @pytest.fixture
    def pricer(self):
        """Create pricer instance."""
        return OptionsPricer(risk_free_rate=0.05)
    
    def test_black_scholes_call_atm(self, pricer):
        """Test Black-Scholes for ATM call option."""
        S = 50000  # ATM
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        price = pricer.black_scholes(S, K, T, r, sigma, "call")
        
        assert price > 0
        assert price < S  # Option price less than underlying
        assert price > S - K * math.exp(-r * T)  # Greater than intrinsic value
    
    def test_black_scholes_put(self, pricer):
        """Test Black-Scholes for put option."""
        S = 50000
        K = 55000  # ITM
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        price = pricer.black_scholes(S, K, T, r, sigma, "put")
        
        assert price > 0
        assert price < K  # Put price less than strike
    
    def test_black_scholes_at_expiration(self, pricer):
        """Test Black-Scholes at expiration."""
        S = 50000
        K = 55000
        T = 0  # At expiration
        r = 0.05
        sigma = 0.60
        
        call_price = pricer.black_scholes(S, K, T, r, sigma, "call")
        put_price = pricer.black_scholes(S, K, T, r, sigma, "put")
        
        # At expiration, option worth intrinsic value only
        assert call_price == max(S - K, 0)
        assert put_price == max(K - S, 0)
    
    def test_calculate_greeks_call(self, pricer):
        """Test Greeks calculation for call option."""
        S = 50000
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")
        
        assert isinstance(greeks, OptionGreeks)
        assert 0 <= greeks.delta <= 1  # Call delta between 0 and 1
        assert greeks.gamma >= 0
        assert greeks.theta <= 0  # Theta is negative (time decay)
        assert greeks.vega >= 0
    
    def test_calculate_greeks_put(self, pricer):
        """Test Greeks calculation for put option."""
        S = 50000
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        greeks = pricer.calculate_greeks(S, K, T, r, sigma, "put")
        
        assert -1 <= greeks.delta <= 0  # Put delta between -1 and 0
    
    def test_delta_atm(self, pricer):
        """Test delta is 0.5 for ATM option."""
        S = 50000
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")
        
        # ATM delta should be close to 0.5
        assert 0.4 < greeks.delta < 0.6
    
    def test_delta_itm_call(self, pricer):
        """Test delta is high for ITM call."""
        S = 55000
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")
        
        # ITM call delta should be close to 1
        assert greeks.delta > 0.7
    
    def test_delta_otm_call(self, pricer):
        """Test delta is low for OTM call."""
        S = 45000
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")
        
        # OTM call delta should be close to 0
        assert greeks.delta < 0.35
    
    def test_calculate_breakeven_call(self, pricer):
        """Test breakeven calculation for call."""
        K = 50000
        premium = 2000
        
        breakeven = pricer.calculate_breakeven(K, premium, "call")
        
        assert breakeven == K + premium
    
    def test_calculate_breakeven_put(self, pricer):
        """Test breakeven calculation for put."""
        K = 50000
        premium = 2000
        
        breakeven = pricer.calculate_breakeven(K, premium, "put")
        
        assert breakeven == K - premium
    
    def test_max_profit_loss_long_call(self, pricer):
        """Test max profit/loss for long call."""
        K = 50000
        premium = 2000
        
        max_profit, max_loss = pricer.calculate_max_profit_loss(
            K, premium, "call", "long"
        )
        
        assert max_profit == float('inf')
        assert max_loss == -premium
    
    def test_max_profit_loss_short_call(self, pricer):
        """Test max profit/loss for short call."""
        K = 50000
        premium = 2000
        
        max_profit, max_loss = pricer.calculate_max_profit_loss(
            K, premium, "call", "short"
        )
        
        assert max_profit == premium
        assert max_loss == float('-inf')
    
    def test_max_profit_loss_long_put(self, pricer):
        """Test max profit/loss for long put."""
        K = 50000
        premium = 2000
        
        max_profit, max_loss = pricer.calculate_max_profit_loss(
            K, premium, "put", "long"
        )
        
        assert max_profit == K - premium
        assert max_loss == -premium


class TestOptionsStrategies:
    """Test suite for options strategies."""
    
    @pytest.fixture
    def pricer(self):
        """Create pricer instance."""
        return OptionsPricer(risk_free_rate=0.05)
    
    def test_straddle(self, pricer):
        """Test straddle strategy."""
        S = 50000
        K = 50000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        call_price = pricer.black_scholes(S, K, T, r, sigma, "call")
        put_price = pricer.black_scholes(S, K, T, r, sigma, "put")
        
        strategies = OptionsStrategies()
        result = strategies.straddle(S, K, T, r, sigma, call_price, put_price)
        
        assert result["type"] == "straddle"
        assert result["strike"] == K
        assert result["cost"] > 0
        assert result["breakeven_up"] > result["breakeven_down"]
    
    def test_bull_call_spread(self, pricer):
        """Test bull call spread."""
        K_long = 48000
        K_short = 52000
        premium_long = pricer.black_scholes(50000, K_long, 30/365, 0.05, 0.60, "call")
        premium_short = pricer.black_scholes(50000, K_short, 30/365, 0.05, 0.60, "call")
        
        strategies = OptionsStrategies()
        result = strategies.bull_call_spread(50000, K_long, K_short, premium_long, premium_short)
        
        assert result["type"] == "bull_call_spread"
        assert result["net_debit"] > 0
        assert result["max_profit"] > 0
        assert result["max_loss"] > 0
        assert result["breakeven"] == K_long + result["net_debit"]
    
    def test_strangle(self, pricer):
        """Test strangle strategy."""
        S = 50000
        K_call = 55000
        K_put = 45000
        T = 30/365
        r = 0.05
        sigma = 0.60
        
        call_price = pricer.black_scholes(S, K_call, T, r, sigma, "call")
        put_price = pricer.black_scholes(S, K_put, T, r, sigma, "put")
        
        strategies = OptionsStrategies()
        result = strategies.strangle(S, K_call, K_put, T, r, sigma, call_price, put_price)
        
        assert result["type"] == "strangle"
        assert result["cost"] > 0
    
    def test_protective_put(self, pricer):
        """Test protective put strategy."""
        S = 50000
        K = 48000
        premium_put = pricer.black_scholes(S, K, 30/365, 0.05, 0.60, "put")
        
        strategies = OptionsStrategies()
        result = strategies.protective_put(S, K, premium_put, S)
        
        assert result["type"] == "protective_put"
        assert result["breakeven"] == S - premium_put
        assert result["max_loss"] == premium_put


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
