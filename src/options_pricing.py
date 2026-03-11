"""
Options Pricing Module
=====================
Black-Scholes and Greeks calculation for crypto options trading.

Author: AI Trading System
Version: 1.0.0
"""

import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.stats import norm
import numpy as np


@dataclass
class OptionGreeks:
    """Option Greeks container"""
    delta: float  # Rate of change of option price with underlying price
    gamma: float  # Rate of change of delta with underlying price
    theta: float  # Time decay (per day)
    vega: float  # Sensitivity to volatility
    rho: float   # Sensitivity to interest rate


class OptionsPricer:
    """
    Options pricing using Black-Scholes model.
    Supports European-style options on crypto assets.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize options pricer.
        
        Args:
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    def black_scholes(
        self,
        S: float,      # Current underlying price
        K: float,      # Strike price
        T: float,      # Time to expiration (in years)
        r: float,      # Risk-free rate
        sigma: float,  # Volatility (annualized)
        option_type: str = "call"
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Annualized volatility
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        if T <= 0:
            # At expiration
            if option_type == "call":
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Handle edge case
        if sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower() == "call":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0.0)
    
    def calculate_greeks(
        self,
        S: float,      # Current underlying price
        K: float,      # Strike price
        T: float,      # Time to expiration (in years)
        r: float,      # Risk-free rate
        sigma: float,  # Volatility
        option_type: str = "call"
    ) -> OptionGreeks:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Annualized volatility
            option_type: "call" or "put"
            
        Returns:
            OptionGreeks object
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return OptionGreeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
        
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        sqrt_T = math.sqrt(T)
        
        # Delta
        if option_type.lower() == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        
        # Theta (per day = divide by 365)
        if option_type.lower() == "call":
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T) 
                    - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T) 
                    + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf(d1) * sqrt_T / 100
        
        # Rho (per 1% change in interest rate)
        if option_type.lower() == "call":
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )
    
    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call"
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Current market price of option
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: "call" or "put"
            
        Returns:
            Implied volatility or None if not found
        """
        if T <= 0 or market_price <= 0:
            return None
        
        # Initial guess
        sigma = 0.5
        max_iterations = 100
        tolerance = 0.0001
        
        for _ in range(max_iterations):
            # Calculate price and vega
            price = self.black_scholes(S, K, T, r, sigma, option_type)
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            
            # Check convergence
            diff = market_price - price
            if abs(diff) < tolerance:
                return sigma
            
            # Update volatility
            if greeks.vega == 0:
                break
            sigma = sigma + diff / (greeks.vega * 100)
            
            # Keep sigma in reasonable range
            sigma = max(0.01, min(sigma, 5.0))
        
        return sigma if 0.01 < sigma < 5.0 else None
    
    def calculate_breakeven(
        self,
        K: float,
        premium: float,
        option_type: str = "call"
    ) -> float:
        """
        Calculate breakeven price at expiration.
        
        Args:
            K: Strike price
            premium: Option premium paid
            option_type: "call" or "put"
            
        Returns:
            Breakeven price
        """
        if option_type.lower() == "call":
            return K + premium
        else:
            return K - premium
    
    def calculate_max_profit_loss(
        self,
        K: float,
        premium: float,
        option_type: str = "call",
        position: str = "long"
    ) -> Tuple[float, float]:
        """
        Calculate max profit and loss for an option position.
        
        Args:
            K: Strike price
            premium: Option premium
            option_type: "call" or "put"
            position: "long" or "short"
            
        Returns:
            (max_profit, max_loss)
        """
        if option_type.lower() == "call":
            if position.lower() == "long":
                # Long call: unlimited upside, premium is max loss
                return float('inf'), -premium
            else:
                # Short call: premium is max profit, unlimited loss
                return premium, float('-inf')
        else:  # put
            if position.lower() == "long":
                # Long put: max profit is K - premium, max loss is premium
                return K - premium, -premium
            else:
                # Short put: max profit is premium, max loss is K - premium
                return premium, -(K - premium)


class OptionsStrategies:
    """
    Common options strategies implementation.
    """
    
    @staticmethod
    def straddle(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        premium_call: float,
        premium_put: float
    ) -> Dict:
        """
        Straddle strategy (long call + long put at same strike).
        
        Returns dict with payoffs at expiration for different underlying prices.
        """
        payoff_at_expiry = {}
        price_range = np.linspace(S * 0.5, S * 1.5, 100)
        
        for price in price_range:
            # Long call payoff
            call_payoff = max(price - K, 0) - premium_call
            # Long put payoff
            put_payoff = max(K - price, 0) - premium_put
            # Total straddle payoff
            payoff_at_expiry[price] = call_payoff + put_payoff
        
        return {
            "type": "straddle",
            "strike": K,
            "cost": premium_call + premium_put,
            "breakeven_up": K + premium_call + premium_put,
            "breakeven_down": K - premium_call - premium_put,
            "payoff_at_expiry": payoff_at_expiry
        }
    
    @staticmethod
    def strangle(
        S: float,
        K_call: float,
        K_put: float,
        T: float,
        r: float,
        sigma: float,
        premium_call: float,
        premium_put: float
    ) -> Dict:
        """
        Strangle strategy (long call + long put at different strikes).
        """
        payoff_at_expiry = {}
        price_range = np.linspace(S * 0.5, S * 1.5, 100)
        
        for price in price_range:
            call_payoff = max(price - K_call, 0) - premium_call
            put_payoff = max(K_put - price, 0) - premium_put
            payoff_at_expiry[price] = call_payoff + put_payoff
        
        total_premium = premium_call + premium_put
        
        return {
            "type": "strangle",
            "strike_call": K_call,
            "strike_put": K_put,
            "cost": total_premium,
            "breakeven_up": K_call + total_premium,
            "breakeven_down": K_put - total_premium,
            "payoff_at_expiry": payoff_at_expiry
        }
    
    @staticmethod
    def iron_condor(
        S: float,
        K_put_short: float,
        K_put_long: float,
        K_call_long: float,
        K_call_short: float,
        premium_put_short: float,
        premium_put_long: float,
        premium_call_long: float,
        premium_call_short: float
    ) -> Dict:
        """
        Iron Condor strategy (short put + long put + long call + short call).
        """
        payoff_at_expiry = {}
        price_range = np.linspace(S * 0.5, S * 1.5, 100)
        
        for price in price_range:
            # Short put
            put_short_payoff = premium_put_short - max(K_put_short - price, 0)
            # Long put
            put_long_payoff = max(K_put_long - price, 0) - premium_put_long
            # Long call
            call_long_payoff = max(price - K_call_long, 0) - premium_call_long
            # Short call
            call_short_payoff = premium_call_short - max(price - K_call_short, 0)
            
            payoff_at_expiry[price] = (
                put_short_payoff + put_long_payoff + 
                call_long_payoff + call_short_payoff
            )
        
        net_credit = (
            premium_put_short + premium_call_short - 
            premium_put_long - premium_call_long
        )
        
        return {
            "type": "iron_condor",
            "strikes": {
                "put_short": K_put_short,
                "put_long": K_put_long,
                "call_long": K_call_long,
                "call_short": K_call_short
            },
            "net_credit": net_credit,
            "max_loss": (
                (K_put_short - K_put_long) - net_credit 
                if K_put_short - K_put_long > K_call_short - K_call_long
                else (K_call_short - K_call_long) - net_credit
            ),
            "payoff_at_expiry": payoff_at_expiry
        }
    
    @staticmethod
    def bull_call_spread(
        S: float,
        K_long: float,
        K_short: float,
        premium_long: float,
        premium_short: float
    ) -> Dict:
        """
        Bull Call Spread (long call at lower strike + short call at higher strike).
        """
        net_debit = premium_long - premium_short
        
        return {
            "type": "bull_call_spread",
            "strike_long": K_long,
            "strike_short": K_short,
            "net_debit": net_debit,
            "max_profit": (K_short - K_long) - net_debit,
            "max_loss": net_debit,
            "breakeven": K_long + net_debit
        }
    
    @staticmethod
    def protective_put(
        S: float,
        K: float,
        premium_put: float,
        stock_price_at_expiry: float
    ) -> Dict:
        """
        Protective Put (long stock + long put).
        """
        stock_payoff = stock_price_at_expiry - S
        put_payoff = max(K - stock_price_at_expiry, 0) - premium_put
        total_payoff = stock_payoff + put_payoff
        
        # Breakeven is stock price - put premium
        breakeven = S - premium_put
        
        return {
            "type": "protective_put",
            "strike": K,
            "cost": S + premium_put,
            "breakeven": breakeven,
            "max_loss": premium_put,
            "unlimited_upside": True,
            "payoff_at_expiry": stock_price_at_expiry + max(K - stock_price_at_expiry, 0) - premium_put
        }


def run_demo():
    """Demo function showing options pricing."""
    pricer = OptionsPricer(risk_free_rate=0.05)
    
    # Example: Bitcoin call option
    S = 50000      # BTC price
    K = 55000      # Strike price
    T = 30/365     # 30 days to expiration
    r = 0.05       # Risk-free rate
    sigma = 0.60   # 60% annualized volatility
    
    # Calculate price
    call_price = pricer.black_scholes(S, K, T, r, sigma, "call")
    put_price = pricer.black_scholes(S, K, T, r, sigma, "put")
    
    print("=" * 50)
    print("OPTIONS PRICING DEMO")
    print("=" * 50)
    print(f"Underlying Price: ${S:,.2f}")
    print(f"Strike Price: ${K:,.2f}")
    print(f"Time to Expiry: {T*365:.0f} days")
    print(f"Volatility: {sigma*100:.0f}%")
    print("-" * 50)
    print(f"Call Option Price: ${call_price:,.2f}")
    print(f"Put Option Price: ${put_price:,.2f}")
    print("-" * 50)
    
    # Calculate Greeks
    greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.4f}")
    print(f"Theta: {greeks.theta:.4f}")
    print(f"Vega: {greeks.vega:.4f}")
    print(f"Rho: {greeks.rho:.4f}")
    
    # Breakeven
    be = pricer.calculate_breakeven(K, call_price, "call")
    print("-" * 50)
    print(f"Breakeven Price: ${be:,.2f}")
    
    # Straddle example
    strategies = OptionsStrategies()
    straddle = strategies.straddle(S, K, T, r, sigma, call_price, put_price)
    print("-" * 50)
    print(f"Straddle Cost: ${straddle['cost']:,.2f}")
    print(f"Straddle Breakeven Up: ${straddle['breakeven_up']:,.2f}")
    print(f"Straddle Breakeven Down: ${straddle['breakeven_down']:,.2f}")


if __name__ == "__main__":
    run_demo()
