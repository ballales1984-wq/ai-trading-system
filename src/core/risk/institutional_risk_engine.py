"""
Institutional Risk Engine Module
Professional-grade risk management with VaR, CVaR, Monte Carlo, and Volatility Targeting
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class InstitutionalRiskEngine:
    """
    Professional risk engine with institutional-grade metrics:
    - Historical VaR
    - Parametric VaR
    - Expected Shortfall (CVaR)
    - Portfolio VaR with correlation
    - Monte Carlo simulation
    - Volatility targeting
    - Risk of Ruin
    - Volatility shock scenarios
    """
    
    def __init__(self, confidence: float = 0.95, target_vol: float = 0.15):
        """
        Initialize the institutional risk engine.
        
        Args:
            confidence: Confidence level for VaR (default 95%)
            target_vol: Annual target volatility (default 15%)
        """
        self.confidence = confidence
        self.target_vol = target_vol
        self.z_score = norm.ppf(1 - confidence)
        
    # =========================================================================
    # 1️⃣ HISTORICAL VAR
    # =========================================================================
    def historical_var(self, returns: pd.Series) -> float:
        """
        Calculate Historical VaR.
        
        VaR = -Percentile(returns, (1 - confidence) * 100)
        
        Args:
            returns: Series of returns
            
        Returns:
            VaR as positive number (loss)
        """
        if len(returns) == 0:
            return 0.0
            
        percentile = np.percentile(returns, (1 - self.confidence) * 100)
        var = -percentile
        
        logger.debug(f"Historical VaR ({self.confidence}): {var:.4f}")
        return var
    
    # =========================================================================
    # 2️⃣ PARAMETRIC VAR (GAUSSIAN)
    # =========================================================================
    def parametric_var(self, returns: pd.Series) -> float:
        """
        Calculate Parametric (Gaussian) VaR.
        
        VaR = -(mean + z * std)
        
        Args:
            returns: Series of returns
            
        Returns:
            VaR as positive number (loss)
        """
        if len(returns) == 0:
            return 0.0
            
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0.0
            
        var = -(mean + self.z_score * std)
        
        logger.debug(f"Parametric VaR ({self.confidence}): {var:.4f}")
        return var
    
    # =========================================================================
    # 3️⃣ STUDENT-T VAR (FAT TAILS)
    # =========================================================================
    def student_t_var(self, returns: pd.Series, df: int = 5) -> float:
        """
        Calculate VaR using Student-t distribution (fat tails).
        
        Args:
            returns: Series of returns
            df: Degrees of freedom (lower = fatter tails)
            
        Returns:
            VaR as positive number (loss)
        """
        if len(returns) < 10:
            return self.parametric_var(returns)
        
        # Fit student-t to returns
        mu, std = returns.mean(), returns.std()
        
        # Calculate VaR using student-t
        t_var = -(
            mu + std * student_t.ppf(1 - self.confidence, df)
        )
        
        logger.debug(f"Student-t VaR (df={df}): {t_var:.4f}")
        return t_var
    
    # =========================================================================
    # 4️⃣ EXPECTED SHORTFALL (CVAR)
    # =========================================================================
    def expected_shortfall(self, returns: pd.Series) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        
        CVaR = Mean of losses beyond VaR threshold
        
        Args:
            returns: Series of returns
            
        Returns:
            CVaR as positive number (loss)
        """
        if len(returns) == 0:
            return 0.0
            
        var_threshold = np.percentile(
            returns, (1 - self.confidence) * 100
        )
        
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return self.historical_var(returns)
        
        cvar = -tail_losses.mean()
        
        logger.debug(f"Expected Shortfall ({self.confidence}): {cvar:.4f}")
        return cvar
    
    # =========================================================================
    # 5️⃣ PORTFOLIO VAR (CORRELATION-AWARE)
    # =========================================================================
    def portfolio_var(
        self, 
        returns_df: pd.DataFrame, 
        weights: np.ndarray
    ) -> float:
        """
        Calculate portfolio-level VaR with correlation matrix.
        
        VaR_p = z * sqrt(w^T * Σ * w)
        
        Args:
            returns_df: DataFrame of asset returns
            weights: Array of portfolio weights
            
        Returns:
            Portfolio VaR as positive number (loss)
        """
        if len(returns_df) == 0 or len(weights) == 0:
            return 0.0
            
        # Ensure weights sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Portfolio variance: w^T * Σ * w
        portfolio_var = np.dot(
            weights.T, 
            np.dot(cov_matrix, weights)
        )
        portfolio_std = np.sqrt(portfolio_var)
        
        # VaR
        var = self.z_score * portfolio_std
        
        logger.debug(f"Portfolio VaR ({self.confidence}): {var:.4f}")
        return var
    
    # =========================================================================
    # 6️⃣ MONTE CARLO SIMULATION
    # =========================================================================
    def monte_carlo_simulation(
        self, 
        returns: pd.Series, 
        simulations: int = 5000, 
        horizon: int = 252
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on returns.
        
        Args:
            returns: Series of returns
            simulations: Number of simulations
            horizon: Time horizon (days)
            
        Returns:
            Dict with various percentile outcomes
        """
        if len(returns) == 0:
            return {"mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
        
        mean = returns.mean()
        std = returns.std()
        
        results = []
        
        for _ in range(simulations):
            # Bootstrap approach
            simulated = np.random.choice(returns, horizon, replace=True)
            cumulative = (1 + simulated).prod() - 1
            results.append(cumulative)
        
        results = np.array(results)
        
        outcomes = {
            "mean": results.mean(),
            "std": results.std(),
            "p5": np.percentile(results, 5),
            "p10": np.percentile(results, 10),
            "p25": np.percentile(results, 25),
            "p50": np.percentile(results, 50),
            "p75": np.percentile(results, 75),
            "p90": np.percentile(results, 90),
            "p95": np.percentile(results, 95),
        }
        
        logger.debug(f"Monte Carlo - 5th percentile: {outcomes['p5']:.4f}")
        logger.debug(f"Monte Carlo - 95th percentile: {outcomes['p95']:.4f}")
        
        return outcomes
    
    # =========================================================================
    # 7️⃣ VOLATILITY TARGETING
    # =========================================================================
    def volatility_scaling_factor(self, returns: pd.Series) -> float:
        """
        Calculate volatility scaling factor for position sizing.
        
        scale = target_vol / realized_vol
        
        Args:
            returns: Series of returns
            
        Returns:
            Scaling factor (1.0 = no adjustment)
        """
        if len(returns) < 2:
            return 1.0
            
        # Annualized volatility
        realized_vol = returns.std() * np.sqrt(252)
        
        if realized_vol == 0:
            return 1.0
            
        scale = self.target_vol / realized_vol
        
        # Cap scaling to prevent extreme leverage
        scale = max(0.25, min(4.0, scale))
        
        logger.debug(f"Vol scaling factor: {scale:.2f} (realized: {realized_vol:.2%})")
        return scale
    
    # =========================================================================
    # 8️⃣ RISK OF RUIN
    # =========================================================================
    def risk_of_ruin(
        self, 
        win_rate: float, 
        reward_risk_ratio: float,
        num_trades: int = 100
    ) -> float:
        """
        Approximate risk of ruin calculation.
        
        Uses simplified Kelly criterion approach.
        
        Args:
            win_rate: Probability of winning (0-1)
            reward_risk_ratio: Average win / average loss
            num_trades: Number of trades to simulate
            
        Returns:
            Probability of ruin (0-1)
        """
        if win_rate <= 0 or reward_risk_ratio <= 0:
            return 1.0
        
        # Edge per trade
        edge = win_rate * reward_risk_ratio - (1 - win_rate)
        
        if edge <= 0:
            return 1.0  # Negative edge = certain ruin
        
        # Simplified risk of ruin approximation
        # Based on Kelly fraction
        kelly = edge / reward_risk_ratio
        
        # Conservative fraction (half-Kelly)
        conservative_kelly = kelly / 2
        
        # Probability of ruin decreases with edge
        ruin_prob = (1 - conservative_kelly) ** num_trades
        
        logger.debug(f"Risk of Ruin: {ruin_prob:.4f} (Kelly: {kelly:.4f})")
        return ruin_prob
    
    # =========================================================================
    # 9️⃣ VOLATILITY SHOCK SCENARIO
    # =========================================================================
    def volatility_shock(
        self, 
        returns: pd.Series, 
        multiplier: float = 3.0
    ) -> Dict[str, float]:
        """
        Simulate extreme volatility expansion scenario.
        
        Args:
            returns: Series of returns
            multiplier: Volatility multiplier (3x = ~3 sigma)
            
        Returns:
            Dict with shocked return statistics
        """
        mean = returns.mean()
        std = returns.std() * multiplier
        
        # Simulate shocked distribution
        shocked = np.random.normal(mean, std, 10000)
        
        outcomes = {
            "mean": shocked.mean(),
            "std": shocked.std(),
            "p5": np.percentile(shocked, 5),
            "p1": np.percentile(shocked, 1),
            "worst": np.min(shocked),
        }
        
        logger.debug(f"Vol shock ({multiplier}x) - 5th percentile: {outcomes['p5']:.4f}")
        
        return outcomes
    
    # =========================================================================
    # 🔟 CRASH SCENARIO
    # =========================================================================
    def crash_scenario(
        self, 
        returns: pd.Series, 
        crash_size: float = -0.20
    ) -> float:
        """
        Simulate a specific crash scenario.
        
        Args:
            returns: Series of returns
            crash_size: Size of crash (negative)
            
        Returns:
            Portfolio value change
        """
        shocked_returns = returns.copy()
        shocked_returns.iloc[-1] = crash_size
        
        cumulative = (1 + shocked_returns).prod() - 1
        
        logger.debug(f"Crash scenario ({crash_size:.1%}): {cumulative:.4f}")
        return cumulative
    
    # =========================================================================
    # 1️⃣1️⃣ FULL RISK REPORT
    # =========================================================================
    def full_risk_report(
        self, 
        returns: pd.Series,
        weights: Optional[np.ndarray] = None,
        returns_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate comprehensive risk report.
        
        Args:
            returns: Series of portfolio returns
            weights: Optional weights for portfolio VaR
            returns_df: Optional DataFrame for multi-asset
            
        Returns:
            Complete risk metrics dictionary
        """
        report = {
            "confidence_level": self.confidence,
            "target_volatility": self.target_vol,
        }
        
        # Basic metrics
        report["mean_return"] = returns.mean()
        report["volatility"] = returns.std()
        report["annual_volatility"] = returns.std() * np.sqrt(252)
        report["sharpe_approx"] = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # VaR metrics
        report["historical_var"] = self.historical_var(returns)
        report["parametric_var"] = self.parametric_var(returns)
        report["student_t_var"] = self.student_t_var(returns)
        report["expected_shortfall"] = self.expected_shortfall(returns)
        
        # Monte Carlo
        mc_results = self.monte_carlo_simulation(returns)
        report["monte_carlo"] = mc_results
        
        # Volatility targeting
        report["vol_scaling_factor"] = self.volatility_scaling_factor(returns)
        
        # Shock scenarios
        report["vol_shock_2x"] = self.volatility_shock(returns, 2.0)
        report["vol_shock_3x"] = self.volatility_shock(returns, 3.0)
        report["crash_scenario"] = self.crash_scenario(returns, -0.20)
        
        # Portfolio VaR if multi-asset
        if weights is not None and returns_df is not None:
            report["portfolio_var"] = self.portfolio_var(returns_df, weights)
        
        return report


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Max drawdown as positive number
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return abs(drawdown.min())


def calculate_sortino_ratio(
    returns: pd.Series, 
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    
    Args:
        returns: Series of returns
        target_return: Minimum acceptable return
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf
    
    return excess_returns.mean() / downside_std


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        returns: Series of returns
        
    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * 252
    equity = (1 + returns).cumprod()
    max_dd = calculate_max_drawdown(equity)
    
    if max_dd == 0:
        return np.inf
    
    return annual_return / max_dd
