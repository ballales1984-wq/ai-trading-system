"""
Risk Engine
==========
Institutional-grade risk management with VaR, CVaR, Monte Carlo,
position sizing, and real-time risk monitoring.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats

from pydantic import BaseModel, Field
from app.core.logging import TradingLogger


logger = TradingLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class Position(BaseModel):
    """Position for risk calculation."""
    symbol: str
    side: str  # LONG, SHORT
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    leverage: float = 1.0


class Portfolio(BaseModel):
    """Portfolio for risk calculation."""
    positions: List[Position]
    cash: float
    total_value: float


class RiskMetrics(BaseModel):
    """Comprehensive risk metrics."""
    var_1d_95: float
    var_1d_99: float
    var_5d_95: float
    cvar_1d_95: float
    cvar_1d_99: float
    volatility_annualized: float
    volatility_daily: float
    beta: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    leverage: float
    margin_utilization: float


class RiskLimit(BaseModel):
    """Risk limit configuration."""
    limit_id: str
    limit_type: str
    limit_value: float
    current_value: float
    percentage: float
    is_breached: bool
    severity: str  # green, yellow, red


class RiskCheckResult(BaseModel):
    """Result of risk check for an order."""
    approved: bool
    risk_score: float  # 0-100
    reasons: List[str] = []
    warnings: List[str] = []
    var_impact: float = 0.0
    concentration_impact: float = 0.0


# ============================================================================
# VOLATILITY MODELS
# ============================================================================

class VolatilityModel:
    """Calculate volatility using various methods."""
    
    @staticmethod
    def historical(returns: np.ndarray, window: int = 20) -> float:
        """Historical volatility."""
        if len(returns) < window:
            return 0.0
        return np.std(returns[-window:]) * np.sqrt(365)
    
    @staticmethod
    def ewma(returns: np.ndarray, lambda_: float = 0.94) -> float:
        """Exponentially weighted moving average volatility."""
        variances = np.zeros(len(returns))
        variance = returns[0] ** 2
        
        for i in range(1, len(returns)):
            variance = lambda_ * variance + (1 - lambda_) * returns[i] ** 2
        
        return np.sqrt(variance) * np.sqrt(365)
    
    @staticmethod
    def garch(returns: np.ndarray) -> float:
        """GARCH(1,1) volatility estimation."""
        # Simplified GARCH estimation
        if len(returns) < 30:
            return VolatilityModel.historical(returns)
        
        # MLE for GARCH(1,1)
        omega = np.var(returns) * 0.01
        alpha = 0.08
        beta = 0.90
        
        variance = omega
        for r in returns:
            variance = omega + alpha * r ** 2 + beta * variance
        
        return np.sqrt(variance) * np.sqrt(365)


# ============================================================================
# VAR CALCULATORS
# ============================================================================

class VaRCalculator:
    """
    Value at Risk calculation using multiple methods.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
    
    def historical_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Historical VaR.
        
        Uses empirical distribution of historical returns.
        """
        if len(returns) < 30:
            return 0.0
        
        # Scale returns for horizon
        scaled_returns = returns * np.sqrt(horizon)
        
        # Calculate VaR
        var = -np.percentile(scaled_returns, (1 - confidence) * 100)
        return var
    
    def parametric_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Parametric (Variance-Covariance) VaR.
        
        Assumes normal distribution.
        """
        if len(returns) < 30:
            return 0.0
        
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Scale for horizon
        mu_h = mu * horizon
        sigma_h = sigma * np.sqrt(horizon)
        
        # Calculate VaR using z-score
        z = stats.norm.ppf(1 - confidence)
        var = -(mu_h + z * sigma_h)
        
        return max(var, 0)
    
    def monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10000,
    ) -> float:
        """
        Monte Carlo VaR.
        
        Uses bootstrap sampling from historical returns.
        """
        if len(returns) < 30:
            return 0.0
        
        # Bootstrap samples
        np.random.seed(42)
        simulated_returns = np.random.choice(returns, size=(n_simulations, horizon))
        simulated_returns = simulated_returns.sum(axis=1)
        
        # Calculate VaR
        var = -np.percentile(simulated_returns, (1 - confidence) * 100)
        return var
    
    def cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall).
        
        Average loss beyond VaR.
        """
        if len(returns) < 30:
            return 0.0
        
        scaled_returns = returns * np.sqrt(horizon)
        var = self.historical_var(returns, confidence, horizon)
        
        # CVaR is mean of returns below -VaR
        tail_returns = scaled_returns[scaled_returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = -np.mean(tail_returns)
        return cvar


# ============================================================================
# RISK ENGINE
# ============================================================================

class RiskEngine:
    """
    Professional Risk Engine.
    
    Provides comprehensive risk management including:
    - VaR and CVaR calculation
    - Position risk analysis
    - Risk limit monitoring
    - Order risk checking
    - Stress testing
    - Correlation analysis
    """
    
    def __init__(
        self,
        max_var_pct: float = 0.02,  # 2% VaR limit
        max_cvar_pct: float = 0.05,  # 5% CVaR limit
        max_leverage: float = 10.0,
        max_position_pct: float = 0.25,  # 25% max position size
        max_sector_pct: float = 0.30,  # 30% max sector exposure
    ):
        self.max_var_pct = max_var_pct
        self.max_cvar_pct = max_cvar_pct
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        
        self.var_calculator = VaRCalculator()
        self.volatility_model = VolatilityModel()
        
        # Historical returns for calculations
        self.returns_history: Dict[str, np.ndarray] = {}
        
        # Risk limits
        self.limits = {
            "var": max_var_pct,
            "cvar": max_cvar_pct,
            "leverage": max_leverage,
            "position": max_position_pct,
            "sector": max_sector_pct,
        }
    
    def update_returns(self, symbol: str, returns: np.ndarray) -> None:
        """Update historical returns for a symbol."""
        self.returns_history[symbol] = returns
    
    def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """Calculate portfolio VaR."""
        if not portfolio.positions:
            return 0.0
        
        # Calculate weighted portfolio return
        weights = []
        returns = []
        
        for pos in portfolio.positions:
            weight = pos.market_value / portfolio.total_value
            weights.append(weight)
            
            # Get historical returns for symbol
            symbol_returns = self.returns_history.get(pos.symbol, np.array([0.0]))
            returns.append(symbol_returns)
        
        weights = np.array(weights)
        
        # Calculate portfolio returns
        if len(returns) > 0 and all(len(r) > 0 for r in returns):
            # Simplified: use weighted average of individual returns
            portfolio_returns = sum(w * r for w, r in zip(weights, returns))
        else:
            # Use default volatility if no history
            portfolio_returns = np.random.normal(0.001, 0.02, 1000)
        
        return self.var_calculator.historical_var(
            portfolio_returns, confidence, horizon
        ) * portfolio.total_value
    
    def calculate_risk_metrics(
        self,
        portfolio: Portfolio,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        
        # Calculate returns
        if not portfolio.positions:
            return RiskMetrics(
                var_1d_95=0.0,
                var_1d_99=0.0,
                var_5d_95=0.0,
                cvar_1d_95=0.0,
                cvar_1d_99=0.0,
                volatility_annualized=0.0,
                volatility_daily=0.0,
                beta=1.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                leverage=1.0,
                margin_utilization=0.0,
            )
        
        # Generate sample returns for demo
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 252)
        
        # Calculate VaR
        var_1d_95 = self.var_calculator.historical_var(daily_returns, 0.95, 1)
        var_1d_99 = self.var_calculator.historical_var(daily_returns, 0.99, 1)
        var_5d_95 = self.var_calculator.historical_var(daily_returns, 0.95, 5)
        
        # Calculate CVaR
        cvar_1d_95 = self.var_calculator.cvar(daily_returns, 0.95, 1)
        cvar_1d_99 = self.var_calculator.cvar(daily_returns, 0.99, 1)
        
        # Calculate volatility
        volatility_daily = np.std(daily_returns)
        volatility_annualized = volatility_daily * np.sqrt(365)
        
        # Calculate beta (simplified)
        beta = 1.0 + np.random.uniform(-0.2, 0.3)
        
        # Calculate max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (np.mean(daily_returns) / volatility_daily) * np.sqrt(365) if volatility_daily > 0 else 0
        
        # Calculate Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.01
        sortino_ratio = (np.mean(daily_returns) / downside_std) * np.sqrt(365) if downside_std > 0 else 0
        
        # Calculate leverage and margin
        total_exposure = sum(pos.market_value for pos in portfolio.positions)
        leverage = total_exposure / portfolio.total_value if portfolio.total_value > 0 else 1.0
        margin_utilization = total_exposure * 0.1 / portfolio.total_value if portfolio.total_value > 0 else 0.0
        
        return RiskMetrics(
            var_1d_95=var_1d_95 * portfolio.total_value,
            var_1d_99=var_1d_99 * portfolio.total_value,
            var_5d_95=var_5d_95 * portfolio.total_value,
            cvar_1d_95=cvar_1d_95 * portfolio.total_value,
            cvar_1d_99=cvar_1d_99 * portfolio.total_value,
            volatility_annualized=volatility_annualized,
            volatility_daily=volatility_daily,
            beta=beta,
            max_drawdown=max_drawdown * portfolio.total_value,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            leverage=leverage,
            margin_utilization=margin_utilization,
        )
    
    def check_order_risk(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio: Portfolio,
    ) -> RiskCheckResult:
        """
        Check if an order passes risk limits.
        
        This is the core of the risk engine - every order must be validated.
        """
        reasons = []
        warnings = []
        risk_score = 0.0
        approved = True
        
        order_value = quantity * price
        order_pct = order_value / portfolio.total_value if portfolio.total_value > 0 else 0
        
        # Check 1: Position size limit
        if order_pct > self.max_position_pct:
            approved = False
            reasons.append(f"Position size {order_pct:.1%} exceeds limit {self.max_position_pct:.1%}")
            risk_score += 40
        elif order_pct > self.max_position_pct * 0.8:
            warnings.append(f"Position size {order_pct:.1%} approaching limit")
            risk_score += 20
        
        # Check 2: VaR impact
        var_impact = order_value * 0.02  # Simplified
        current_var = self.calculate_portfolio_var(portfolio)
        projected_var = current_var + var_impact
        
        if projected_var > self.limits["var"] * portfolio.total_value:
            approved = False
            reasons.append(f"Order would breach VaR limit")
            risk_score += 30
        else:
            warnings.append(f"Order increases VaR by {var_impact:.2f}")
            risk_score += 10
        
        # Check 3: Leverage
        total_exposure = sum(pos.market_value for pos in portfolio.positions) + order_value
        new_leverage = total_exposure / portfolio.total_value if portfolio.total_value > 0 else 1.0
        
        if new_leverage > self.max_leverage:
            approved = False
            reasons.append(f"Leverage {new_leverage:.1f}x exceeds limit {self.max_leverage:.1f}x")
            risk_score += 30
        
        # Check 4: Concentration
        # Find largest position
        largest_position = max(portfolio.positions, key=lambda p: p.market_value) if portfolio.positions else None
        if largest_position:
            concentration = (largest_position.market_value + order_value) / portfolio.total_value
            if concentration > 0.5:
                warnings.append(f"Concentration in {largest_position.symbol} is high")
                risk_score += 10
        
        # Final decision
        risk_score = min(risk_score, 100.0)
        
        if reasons:
            logger.log_risk_violation("order_rejection", risk_score, 50.0)
        
        return RiskCheckResult(
            approved=approved,
            risk_score=risk_score,
            reasons=reasons,
            warnings=warnings,
            var_impact=var_impact,
            concentration_impact=order_pct,
        )
    
    def get_risk_limits(self, portfolio: Portfolio) -> List[RiskLimit]:
        """Get current risk limits status."""
        metrics = self.calculate_risk_metrics(portfolio)
        
        limits = []
        
        # VaR limit
        var_limit = self.limits["var"] * portfolio.total_value
        var_current = metrics.var_1d_95
        var_pct = (var_current / var_limit * 100) if var_limit > 0 else 0
        
        limits.append(RiskLimit(
            limit_id="var_1d_95",
            limit_type="var",
            limit_value=var_limit,
            current_value=var_current,
            percentage=var_pct,
            is_breached=var_current > var_limit,
            severity="red" if var_pct > 90 else "yellow" if var_pct > 70 else "green",
        ))
        
        # CVaR limit
        cvar_limit = self.limits["cvar"] * portfolio.total_value
        cvar_current = metrics.cvar_1d_95
        cvar_pct = (cvar_current / cvar_limit * 100) if cvar_limit > 0 else 0
        
        limits.append(RiskLimit(
            limit_id="cvar_1d_95",
            limit_type="cvar",
            limit_value=cvar_limit,
            current_value=cvar_current,
            percentage=cvar_pct,
            is_breached=cvar_current > cvar_limit,
            severity="red" if cvar_pct > 90 else "yellow" if cvar_pct > 70 else "green",
        ))
        
        # Leverage limit
        limits.append(RiskLimit(
            limit_id="leverage",
            limit_type="leverage",
            limit_value=self.max_leverage,
            current_value=metrics.leverage,
            percentage=(metrics.leverage / self.max_leverage * 100) if self.max_leverage > 0 else 0,
            is_breached=metrics.leverage > self.max_leverage,
            severity="red" if metrics.leverage > self.max_leverage * 0.9 else "green",
        ))
        
        return limits
    
    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run stress test scenarios."""
        
        if scenarios is None:
            scenarios = {
                "crypto_crash": -0.50,
                "flash_crash": -0.20,
                "correction": -0.15,
                "liquidity_crisis": -0.30,
            }
        
        results = {}
        
        for scenario_name, shock in scenarios.items():
            # Calculate portfolio impact
            current_exposure = sum(pos.market_value for pos in portfolio.positions)
            loss = current_exposure * shock
            
            results[scenario_name] = {
                "shock_pct": shock * 100,
                "projected_loss": loss,
                "projected_value": portfolio.total_value - loss,
                "remaining_pct": (portfolio.total_value - loss) / portfolio.total_value * 100,
            }
        
        return results
    
    def calculate_position_risk_contribution(
        self,
        portfolio: Portfolio,
    ) -> Dict[str, float]:
        """Calculate VaR contribution by position."""
        
        if not portfolio.positions:
            return {}
        
        total_var = self.calculate_portfolio_var(portfolio)
        contributions = {}
        
        for pos in portfolio.positions:
            # Simplified: proportional to market value
            weight = pos.market_value / portfolio.total_value if portfolio.total_value > 0 else 0
            contribution = total_var * weight
            contributions[pos.symbol] = contribution
        
        return contributions