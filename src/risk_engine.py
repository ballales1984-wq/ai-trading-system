"""
Risk Engine Istituzionale
========================
Advanced risk management for institutional-level trading.

Features:
- VaR (Value at Risk)
- CVaR (Conditional VaR / Expected Shortfall)
- Kelly Criterion
- Stress Testing
- Risk Budgeting
- Kill Switch
- Position Sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Institutional-level risk management engine.
    """
    
    def __init__(
        self,
        var_confidence: float = 0.95,
        max_drawdown: float = 0.20,
        max_position_pct: float = 0.10,
    ):
        """
        Initialize risk engine.
        
        Args:
            var_confidence: VaR confidence level (e.g., 0.95)
            max_drawdown: Maximum allowed drawdown (e.g., 0.20 = 20%)
            max_position_pct: Maximum position size as % of portfolio
        """
        self.var_confidence = var_confidence
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        
        # State
        self.peak_equity = 0
        self.kill_switch_triggered = False
    
    # ======================
    # VALUE AT RISK (VaR)
    # ======================
    
    def var_historical(
        self,
        returns: pd.Series,
        confidence: float = None,
    ) -> float:
        """
        Historical VaR.
        
        Args:
            returns: Historical returns
            confidence: Confidence level (default: self.var_confidence)
            
        Returns:
            VaR (positive value representing loss)
        """
        if confidence is None:
            confidence = self.var_confidence
            
        if len(returns) == 0:
            return 0.0
        
        var = -np.percentile(returns, (1 - confidence) * 100)
        return max(var, 0)
    
    def var_parametric(
        self,
        returns: pd.Series,
        confidence: float = None,
    ) -> float:
        """
        Parametric (Gaussian) VaR.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            
        Returns:
            VaR
        """
        if confidence is None:
            confidence = self.var_confidence
            
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf(1 - confidence)
        
        mu = returns.mean()
        sigma = returns.std()
        
        var = -(mu + z * sigma)
        return max(var, 0)
    
    def var_monte_carlo(
        self,
        returns: pd.Series,
        n_simulations: int = 10000,
        confidence: float = None,
    ) -> float:
        """
        Monte Carlo VaR.
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            confidence: Confidence level
            
        Returns:
            VaR
        """
        if confidence is None:
            confidence = self.var_confidence
            
        if len(returns) < 2:
            return 0.0
        
        # Fit distribution
        mu = returns.mean()
        sigma = returns.std()
        
        # Simulate
        simulated = np.random.normal(mu, sigma, n_simulations)
        
        var = -np.percentile(simulated, (1 - confidence) * 100)
        return max(var, 0)
    
    def var(self, returns: pd.Series) -> float:
        """Get VaR using default method."""
        return self.var_parametric(returns)
    
    # ======================
    # CONDITIONAL VaR (Expected Shortfall)
    # ======================
    
    def cvar(
        self,
        returns: pd.Series,
        confidence: float = None,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            
        Returns:
            CVaR (average loss in worst cases)
        """
        if confidence is None:
            confidence = self.var_confidence
            
        if len(returns) == 0:
            return 0.0
        
        var = -self.var(returns, confidence)
        cvar = -returns[returns <= -var].mean()
        
        return max(cvar, 0) if not np.isnan(cvar) else 0.0
    
    # ======================
    # KELLY CRITERION
    # ======================
    
    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Kelly Criterion for position sizing.
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive value)
            fraction: Kelly fraction to use (0.25 = half-Kelly)
            
        Returns:
            Optimal Kelly fraction
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fraction and ensure positive
        kelly = max(kelly * fraction, 0)
        
        return min(kelly, 0.25)  # Cap at 25%
    
    def kelly_optimal(
        self,
        returns: pd.Series,
        fraction: float = 0.25,
    ) -> float:
        """
        Calculate Kelly from returns series.
        
        Args:
            returns: Historical returns
            fraction: Fraction of Kelly to use
            
        Returns:
            Optimal position size
        """
        if len(returns) < 10:
            return 0.0
        
        # Calculate win rate and avg win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        
        return self.kelly_fraction(win_rate, avg_win, avg_loss, fraction)
    
    # ======================
    # STRESS TESTING
    # ======================
    
    def stress_test(
        self,
        current_price: float,
        shocks: List[float] = None,
    ) -> Dict[str, float]:
        """
        Stress test with price shocks.
        
        Args:
            current_price: Current asset price
            shocks: List of shock percentages (e.g., [-0.1, -0.2, -0.3])
            
        Returns:
            Dict of {shock_percentage: new_price}
        """
        if shocks is None:
            shocks = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.50]
        
        results = {}
        for shock in shocks:
            results[f"{int(shock*100)}%"] = current_price * (1 + shock)
        
        return results
    
    def stress_test_portfolio(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        shocks: Dict[str, float] = None,
    ) -> float:
        """
        Stress test entire portfolio.
        
        Args:
            positions: {symbol: position_value}
            prices: {symbol: current_price}
            shocks: {symbol: shock_pct}
            
        Returns:
            Total portfolio value after stress
        """
        total = 0
        
        for symbol, pos_value in positions.items():
            price = prices.get(symbol, 0)
            shock = shocks.get(symbol, 0) if shocks else 0
            
            total += pos_value * (1 + shock)
        
        return total
    
    # ======================
    # DRAWDOWN
    # ======================
    
    def calculate_drawdown(
        self,
        equity: pd.Series,
    ) -> pd.Series:
        """
        Calculate drawdown series.
        
        Args:
            equity: Equity curve
            
        Returns:
            Drawdown series
        """
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown
    
    def max_drawdown(
        self,
        equity: pd.Series,
    ) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity: Equity curve
            
        Returns:
            Max drawdown (negative value)
        """
        if len(equity) == 0:
            return 0.0
        
        drawdown = self.calculate_drawdown(equity)
        return drawdown.min()
    
    def current_drawdown(
        self,
        current_equity: float,
    ) -> float:
        """
        Calculate current drawdown.
        
        Args:
            current_equity: Current equity value
            
        Returns:
            Current drawdown (negative value)
        """
        if self.peak_equity == 0:
            self.peak_equity = current_equity
            return 0.0
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        return (current_equity - self.peak_equity) / self.peak_equity
    
    # ======================
    # POSITION SIZING
    # ======================
    
    def position_size_fixed(
        self,
        equity: float,
        risk_pct: float,
        stop_loss_pct: float,
    ) -> float:
        """
        Fixed risk position sizing.
        
        Args:
            equity: Total equity
            risk_pct: Risk per trade (e.g., 0.01 = 1%)
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Position size
        """
        risk_amount = equity * risk_pct
        size = risk_amount / stop_loss_pct
        return size
    
    def position_size_kelly(
        self,
        equity: float,
        returns: pd.Series,
        fraction: float = 0.25,
    ) -> float:
        """
        Kelly-based position sizing.
        
        Args:
            equity: Total equity
            returns: Historical returns
            fraction: Kelly fraction
            
        Returns:
            Position size
        """
        kelly = self.kelly_optimal(returns, fraction)
        return equity * kelly
    
    def position_size_volatility(
        self,
        equity: float,
        target_vol: float,
        current_vol: float,
        max_pct: float = None,
    ) -> float:
        """
        Volatility-based position sizing.
        
        Args:
            equity: Total equity
            target_vol: Target volatility
            current_vol: Current volatility
            max_pct: Maximum position as % of equity
            
        Returns:
            Position size
        """
        if current_vol <= 0:
            return 0.0
        
        # Scale to target volatility
        vol_scalar = target_vol / current_vol
        
        size = equity * vol_scalar
        
        # Apply max cap
        if max_pct is not None:
            max_size = equity * max_pct
            size = min(size, max_size)
        
        return size
    
    # ======================
    # RISK BUDGETING
    # ======================
    
    def risk_budget(
        self,
        volatilities: Dict[str, float],
        total_risk_budget: float = 1.0,
    ) -> Dict[str, float]:
        """
        Risk budgeting (inverse volatility weighting).
        
        Args:
            volatilities: {asset: volatility}
            total_risk_budget: Total risk budget
            
        Returns:
            {asset: risk_allocation}
        """
        if not volatilities:
            return {}
        
        # Inverse volatility
        inv_vol = {k: 1/v if v > 0 else 0 for k, v in volatilities.items()}
        total = sum(inv_vol.values())
        
        if total == 0:
            return {k: 0 for k in volatilities}
        
        return {
            k: (inv_vol[k] / total) * total_risk_budget
            for k in volatilities
        }
    
    # ======================
    # KILL SWITCH
    # ======================
    
    def check_kill_switch(
        self,
        current_equity: float,
        threshold: float = None,
    ) -> Tuple[bool, str]:
        """
        Check if kill switch should be triggered.
        
        Args:
            current_equity: Current equity
            threshold: Override threshold
            
        Returns:
            (should_stop, reason)
        """
        if threshold is None:
            threshold = self.max_drawdown
        
        dd = self.current_drawdown(current_equity)
        
        if dd <= -threshold:
            self.kill_switch_triggered = True
            return True, f"Max drawdown exceeded: {dd:.2%}"
        
        return False, ""
    
    def reset_kill_switch(self):
        """Reset kill switch."""
        self.kill_switch_triggered = False
        self.peak_equity = 0
        logger.info("Kill switch reset")
    
    # ======================
    # RISK METRICS SUMMARY
    # ======================
    
    def get_risk_summary(
        self,
        returns: pd.Series,
        equity: float,
    ) -> Dict[str, float]:
        """
        Get comprehensive risk summary.
        
        Args:
            returns: Historical returns
            equity: Current equity
            
        Returns:
            Dict of risk metrics
        """
        return {
            'VaR_95': self.var(returns) * equity,
            'CVaR_95': self.cvar(returns) * equity,
            'Max_Drawdown': self.max_drawdown(equity) if isinstance(equity, pd.Series) else self.current_drawdown(equity),
            'Volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'Kelly': self.kelly_optimal(returns),
            'Skewness': returns.skew() if len(returns) > 0 else 0,
            'Kurtosis': returns.kurtosis() if len(returns) > 0 else 0,
        }


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    # Create risk engine
    re = RiskEngine(max_drawdown=0.20)
    
    # Simula returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.02)
    equity = pd.Series(10000 * (1 + returns).cumprod())
    
    print("📊 Risk Metrics:")
    print("=" * 50)
    
    print(f"\nVaR (95%): ${re.var(returns) * 10000:.2f}")
    print(f"CVaR (95%): ${re.cvar(returns) * 10000:.2f}")
    print(f"Max Drawdown: {re.max_drawdown(equity)*100:.2f}%")
    print(f"Volatility (ann.): {returns.std() * np.sqrt(252)*100:.2f}%")
    print(f"Kelly: {re.kelly_optimal(returns)*100:.2f}%")
    
    # Stress test
    print("\n📉 Stress Test (BTC at $50,000):")
    stress = re.stress_test(50000)
    for shock, price in stress.items():
        print(f"  {shock}: ${price:,.0f}")
    
    # Kill switch
    should_stop, reason = re.check_kill_switch(8000)
    print(f"\n🛡️ Kill Switch: {should_stop} ({reason})")
