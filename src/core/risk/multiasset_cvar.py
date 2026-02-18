"""
Multi-Asset CVaR Optimization and Monte Carlo Stress Testing
Portfolio optimization using Conditional Value-at-Risk (Expected Shortfall)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiAssetCVaR:
    """
    Multi-Asset CVaR Optimization and Monte Carlo Stress Testing.
    
    Features:
    - Monte Carlo simulation for portfolio returns
    - CVaR optimization
    - Multi-asset stress testing
    - Dynamic correlation handling
    """
    
    def __init__(
        self, 
        returns: pd.DataFrame, 
        vol: Optional[pd.DataFrame] = None,
        correlations: Optional[pd.DataFrame] = None
    ):
        """
        Initialize Multi-Asset CVaR engine.
        
        Args:
            returns: DataFrame of asset returns
            vol: Optional conditional volatility from GARCH
            correlations: Optional correlation matrix
        """
        self.returns = returns
        self.vol = vol if vol is not None else returns.rolling(20).std()
        self.correlations = correlations if correlations is not None else returns.corr()
        
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)
        
        # Start with equal weights
        self.weights = np.ones(self.n_assets) / self.n_assets
        
    # =========================================================================
    # 1️⃣ Simulate Portfolio Returns
    # =========================================================================
    def simulate_portfolio(
        self, 
        n_simulations: int = 50000,
        weights: Optional[np.ndarray] = None,
        use_garch: bool = True,
        use_dcc: bool = True
    ) -> np.ndarray:
        """
        Simulate portfolio returns using Monte Carlo.
        
        Args:
            n_simulations: Number of simulations
            weights: Portfolio weights (uses current if None)
            use_garch: Use conditional volatility
            use_dcc: Use dynamic correlations
            
        Returns:
            Array of simulated portfolio returns
        """
        if weights is None:
            weights = self.weights
            
        weights = np.array(weights)
        
        n_samples = len(self.returns)
        
        # Get current volatility
        if use_garch and self.vol is not None:
            current_vol = self.vol.iloc[-1].values
        else:
            current_vol = self.returns.std().values
        
        # Get correlation matrix
        if use_dcc:
            corr_matrix = self.correlations.iloc[-1].values if hasattr(self.correlations, 'iloc') else self.correlations.values
        else:
            corr_matrix = self.returns.corr().values
        
        # Cholesky decomposition for correlated random numbers
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use identity
            L = np.eye(self.n_assets)
        
        # Generate uncorrelated random numbers
        z = np.random.standard_normal((n_simulations, self.n_assets))
        
        # Apply correlation
        correlated_z = z @ L.T
        
        # Scale by volatility
        simulated_returns = correlated_z * current_vol
        
        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        return portfolio_returns
    
    # =========================================================================
    # 2️⃣ Calculate Portfolio VaR
    # =========================================================================
    def portfolio_var(
        self, 
        alpha: float = 0.01,
        n_simulations: int = 50000,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate portfolio VaR.
        
        Args:
            alpha: Significance level
            n_simulations: Number of Monte Carlo simulations
            weights: Portfolio weights
            
        Returns:
            VaR as positive number (loss)
        """
        simulated = self.simulate_portfolio(n_simulations, weights)
        var = -np.percentile(simulated, alpha * 100)
        return var
    
    # =========================================================================
    # 3️⃣ Calculate Portfolio CVaR (Expected Shortfall)
    # =========================================================================
    def portfolio_cvar(
        self, 
        alpha: float = 0.01,
        n_simulations: int = 50000,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate portfolio CVaR (Expected Shortfall).
        
        CVaR = Mean of losses beyond VaR threshold
        
        Args:
            alpha: Significance level
            n_simulations: Number of Monte Carlo simulations
            weights: Portfolio weights
            
        Returns:
            CVaR as positive number (loss)
        """
        simulated = self.simulate_portfolio(n_simulations, weights)
        
        # VaR threshold
        var_threshold = np.percentile(simulated, alpha * 100)
        
        # CVaR: mean of losses beyond VaR
        worst_losses = simulated[simulated <= var_threshold]
        
        if len(worst_losses) == 0:
            return self.portfolio_var(alpha, n_simulations, weights)
        
        cvar = -worst_losses.mean()
        return cvar
    
    # =========================================================================
    # 4️⃣ Full Risk Metrics
    # =========================================================================
    def risk_metrics(
        self, 
        alpha: float = 0.01,
        n_simulations: int = 50000,
        weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            alpha: Significance level
            n_simulations: Number of simulations
            weights: Portfolio weights
            
        Returns:
            Dictionary with all risk metrics
        """
        simulated = self.simulate_portfolio(n_simulations, weights)
        
        metrics = {
            'var_1pct': self.portfolio_var(0.01, n_simulations, weights),
            'var_5pct': self.portfolio_var(0.05, n_simulations, weights),
            'cvar_1pct': self.portfolio_cvar(0.01, n_simulations, weights),
            'cvar_5pct': self.portfolio_cvar(0.05, n_simulations, weights),
            'expected_return': simulated.mean(),
            'volatility': simulated.std(),
            'sharpe_approx': simulated.mean() / simulated.std() if simulated.std() > 0 else 0,
            'worst_case': simulated.min(),
            'best_case': simulated.max(),
            'p5': np.percentile(simulated, 5),
            'p10': np.percentile(simulated, 10),
            'p25': np.percentile(simulated, 25),
            'median': np.percentile(simulated, 50),
            'p75': np.percentile(simulated, 75),
            'p90': np.percentile(simulated, 90),
            'p95': np.percentile(simulated, 95),
        }
        
        return metrics
    
    # =========================================================================
    # 5️⃣ CVaR Optimization
    # =========================================================================
    def optimize_cvar(
        self, 
        alpha: float = 0.01,
        n_simulations: int = 50000,
        bounds: Optional list = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Optimize portfolio to minimize CVaR.
        
        Args:
            alpha: Significance level for CVaR
            n_simulations: Number of Monte Carlo simulations
            bounds: List of (min, max) tuples for each weight
            constraints: Additional optimization constraints
            
        Returns:
            Optimization result dictionary
        """
        if bounds is None:
            # Long-only, no leverage
            bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Default constraint: weights sum to 1
        default_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        if constraints:
            default_constraints.extend(constraints)
        
        # Objective function: minimize CVaR
        def objective(weights):
            return self.portfolio_cvar(alpha, n_simulations, weights)
        
        # Optimize
        result = minimize(
            objective,
            self.weights,
            method='SLSQP',
            bounds=bounds,
            constraints=default_constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        self.weights = result.x
        
        return {
            'optimal_weights': dict(zip(self.asset_names, self.weights)),
            'optimal_cvar': result.fun,
            'var_at_optimal': self.portfolio_var(alpha, n_simulations, self.weights),
            'expected_return': self.simulate_portfolio(n_simulations, self.weights).mean(),
            'optimizer_message': result.message,
            'optimizer_success': result.success
        }
    
    # =========================================================================
    # 6️⃣ Risk Parity Optimization
    # =========================================================================
    def optimize_risk_parity(
        self, 
        target_vol: float = 0.15
    ) -> Dict:
        """
        Optimize for risk parity (equal risk contribution).
        
        Args:
            target_vol: Target annual volatility
            
        Returns:
            Optimal weights dictionary
        """
        def risk_contribution(weights):
            vol = self.returns.std().values
            port_vol = np.sqrt(weights @ self.returns.corr().values @ weights) * vol
            
            # Risk contributions
            marginal_risk = (self.returns.corr().values @ weights) * vol**2
            risk_contrib = weights * marginal_risk / port_vol
            
            # Target: equal risk contribution
            target_rc = port_vol / self.n_assets
            return np.sum((risk_contrib - target_rc)**2)
        
        # Optimize
        result = minimize(
            risk_contribution,
            self.weights,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(self.n_assets)],
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        )
        
        self.weights = result.x
        
        return {
            'optimal_weights': dict(zip(self.asset_names, self.weights)),
            'target_volatility': target_vol
        }
    
    # =========================================================================
    # 7️⃣ Mean-Variance Optimization (Sharpe)
    # =========================================================================
    def optimize_sharpe(
        self, 
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Optimize for maximum Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Optimal weights dictionary
        """
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()
        
        def negative_sharpe(weights):
            port_return = weights @ mean_returns
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            sharpe = (port_return - risk_free_rate) / port_vol
            return -sharpe
        
        result = minimize(
            negative_sharpe,
            self.weights,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(self.n_assets)],
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        )
        
        self.weights = result.x
        
        port_return = self.weights @ mean_returns
        port_vol = np.sqrt(self.weights @ cov_matrix.values @ self.weights)
        
        return {
            'optimal_weights': dict(zip(self.asset_names, self.weights)),
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': (port_return - risk_free_rate) / port_vol
        }
    
    # =========================================================================
    # 8️⃣ Monte Carlo Stress Test
    # =========================================================================
    def stress_test(
        self, 
        scenarios: Dict[str, Dict] = None,
        n_simulations: int = 10000
    ) -> Dict:
        """
        Run Monte Carlo stress tests with custom scenarios.
        
        Args:
            scenarios: Dict of scenario name -> asset returns multiplier
            n_simulations: Number of simulations per scenario
            
        Returns:
            Stress test results
        """
        if scenarios is None:
            # Default scenarios
            scenarios = {
                'market_crash_20': {asset: -0.20 for asset in self.asset_names},
                'volatility_spike_3x': {asset: 3.0 for asset in self.asset_names},
                'correlation_breakdown': {asset: 0.0 for asset in self.asset_names},  # Uncorrelated
                'bitcoin_halving': {'BTCUSDT': -0.30, 'ETHUSDT': -0.15},
                'defi_crash': {'BTCUSDT': -0.10, 'ETHUSDT': -0.25, 'SOLUSDT': -0.35}
            }
        
        results = {}
        
        for scenario_name, shock in scenarios.items():
            # Apply scenario shocks
            shocked_returns = self.returns.copy()
            
            for asset, multiplier in shock.items():
                if asset in shocked_returns.columns:
                    shocked_returns[asset] = shocked_returns[asset] * multiplier
            
            # Calculate CVaR for shocked scenario
            stressed_engine = MultiAssetCVaR(shocked_returns)
            stressed_cvar = stressed_engine.portfolio_cvar(0.01, n_simulations, self.weights)
            
            results[scenario_name] = {
                'cvar_1pct': stressed_cvar,
                'expected_loss': stressed_engine.simulate_portfolio(n_simulations, self.weights).mean()
            }
        
        return results
    
    # =========================================================================
    # 9️⃣ Efficient Frontier
    # =========================================================================
    def efficient_frontier(
        self, 
        n_points: int = 50,
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            n_points: Number of points on frontier
            n_simulations: Monte Carlo simulations per point
            
        Returns:
            DataFrame with frontier data
        """
        mean_returns = self.returns.mean()
        
        # Target returns from min to max
        min_return = mean_returns.min()
        max_return = mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = []
        
        for target in target_returns:
            try:
                # Optimize with return constraint
                result = minimize(
                    lambda w: self.portfolio_cvar(0.01, n_simulations, w),
                    self.weights,
                    method='SLSQP',
                    bounds=[(0, 1) for _ in range(self.n_assets)],
                    constraints=[
                        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                        {'type': 'eq', 'fun': lambda w: w @ mean_returns - target}
                    ]
                )
                
                if result.success:
                    vol = np.sqrt(result.x @ self.returns.corr().values @ result.x)
                    frontier.append({
                        'target_return': target,
                        'volatility': vol,
                        'sharpe': (target - 0.02) / vol if vol > 0 else 0,
                        'cvar': result.fun
                    })
            except:
                continue
        
        return pd.DataFrame(frontier)
