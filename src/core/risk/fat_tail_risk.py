"""
Fat-Tail Risk Modeling Module
Student-t distribution, Cornish-Fisher expansion, and EVT for risk estimation
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FatTailRisk:
    """
    Fat-tail risk modeling using Student-t distribution and EVT.
    
    Provides:
    - Parametric VaR with Student-t
    - Expected Shortfall (CVaR)
    - Cornish-Fisher VaR
    - Peak-Over-Threshold (POT) for extreme events
    """
    
    def __init__(self, returns: pd.Series, volatility: Optional[pd.Series] = None):
        """
        Initialize fat-tail risk model.
        
        Args:
            returns: Series of returns
            volatility: Optional conditional volatility from GARCH
        """
        self.returns = returns.dropna()
        self.volatility = volatility
        self.nu = None  # degrees of freedom
        self.loc = None  # location parameter
        self.scale = None  # scale parameter
        self.standardized_returns = None
        
        if volatility is not None:
            self.standardized_returns = self.returns / self.volatility
        
    # =========================================================================
    # 1️⃣ Fit Student-t Distribution
    # =========================================================================
    def fit_student_t(self) -> Tuple[float, float, float]:
        """
        Fit Student-t distribution to standardized returns.
        
        Returns:
            Tuple of (degrees of freedom, location, scale)
        """
        if self.standardized_returns is None:
            data = self.returns
        else:
            data = self.standardized_returns
            
        self.nu, self.loc, self.scale = stats.t.fit(data)
        
        logger.info(f"Student-t fit: df={self.nu:.2f}, loc={self.loc:.4f}, scale={self.scale:.4f}")
        
        return self.nu, self.loc, self.scale
    
    # =========================================================================
    # 2️⃣ Parametric VaR (Student-t)
    # =========================================================================
    def parametric_var(self, alpha: float = 0.01) -> float:
        """
        Compute VaR using Student-t distribution.
        
        Args:
            alpha: Significance level (e.g., 0.01 for 99% VaR)
            
        Returns:
            VaR as positive number (loss)
        """
        if self.nu is None:
            self.fit_student_t()
        
        # Quantile from Student-t
        q = stats.t.ppf(alpha, df=self.nu, loc=self.loc, scale=self.scale)
        
        # If using conditional volatility
        if self.volatility is not None:
            var = -q * self.volatility.iloc[-1]
        else:
            var = -q
            
        logger.debug(f"Parametric VaR ({1-alpha:.0%}): {var:.4f}")
        return var
    
    # =========================================================================
    # 3️⃣ Parametric CVaR (Expected Shortfall)
    # =========================================================================
    def parametric_cvar(self, alpha: float = 0.01) -> float:
        """
        Compute Expected Shortfall (CVaR) using Student-t.
        
        CVaR = E[X | X <= VaR]
        
        Args:
            alpha: Significance level
            
        Returns:
            CVaR as positive number (loss)
        """
        if self.nu is None:
            self.fit_student_t()
        
        # VaR threshold
        var_threshold = stats.t.ppf(alpha, df=self.nu, loc=self.loc, scale=self.scale)
        
        # CVaR formula for Student-t
        # E[X | X <= x_alpha] = -scale * (nu + x_alpha^2) / (nu - 1) * T_nu(x_alpha) / alpha
        # where T_nu is the PDF of Student-t
        
        if self.volatility is not None:
            vol = self.volatility.iloc[-1]
        else:
            vol = 1.0
        
        # Simplified: use numerical integration
        x = np.linspace(-10 * self.scale, var_threshold, 1000)
        pdf = stats.t.pdf(x, df=self.nu, loc=self.loc, scale=self.scale)
        cvar = -np.trapz(x * pdf, x) / alpha * vol
        
        # Alternative: analytical formula
        x_alpha = (var_threshold - self.loc) / self.scale
        if self.nu > 1:
            analytical_cvar = -self.scale * (self.nu + x_alpha**2) / (self.nu - 1) * \
                            stats.t.pdf(x_alpha, df=self.nu) / alpha * vol
        else:
            analytical_cvar = cvar
            
        logger.debug(f"Parametric CVaR ({1-alpha:.0%}): {analytical_cvar:.4f}")
        
        return analytical_cvar
    
    # =========================================================================
    # 4️⃣ Cornish-Fisher Expansion
    # =========================================================================
    def cornish_fisher_var(self, alpha: float = 0.01) -> float:
        """
        Cornish-Fisher VaR - adjusts Gaussian VaR for skewness and kurtosis.
        
        VaR_cf = VaR_normal * (1 + S*z/6 + (K-3)*z^2/24 - ...)
        
        Args:
            alpha: Significance level
            
        Returns:
            Cornish-Fisher VaR as positive number
        """
        # Standard returns
        if self.standardized_returns is not None:
            data = self.standardized_returns
        else:
            data = (self.returns - self.returns.mean()) / self.returns.std()
        
        # Calculate moments
        mu = data.mean()
        sigma = data.std()
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)  # Excess kurtosis
        
        # Standard normal quantile
        z = stats.norm.ppf(alpha)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                skew * (z**2 - 1) / 6 + 
                kurt * (z**3 - 3*z) / 24 - 
                skew**2 * (2*z**3 - 5*z) / 36)
        
        # VaR
        if self.volatility is not None:
            vol = self.volatility.iloc[-1]
        else:
            vol = sigma
            
        var = -(mu + z_cf * vol)
        
        logger.debug(f" Cornish-Fisher VaR: {var:.4f} (skew={skew:.3f}, kurt={kurt:.3f})")
        
        return var
    
    # =========================================================================
    # 5️⃣ Peak-Over-Threshold (POT) - EVT
    # =========================================================================
    class POT:
        """Peak-Over-Threshold model for extreme events."""
        
        def __init__(self, returns: pd.Series, threshold: float = None):
            self.returns = returns.dropna()
            
            # Auto-select threshold (95th percentile)
            if threshold is None:
                self.threshold = np.percentile(self.returns, 95)
            else:
                self.threshold = threshold
                
            # Extract exceedances
            self.exceedances = self.returns[self.returns > self.threshold]
            
            # Fit GPD
            self.fit_gpd()
            
        def fit_gpd(self):
            """Fit Generalized Pareto Distribution to exceedances."""
            data = self.exceedances - self.threshold
            
            # MLE for GPD
            # xi (shape), beta (scale)
            def neg_loglik(params):
                xi, beta = params
                if beta <= 0 or xi <= -0.5:
                    return 1e10
                    
                n = len(data)
                if xi == 0:
                    return n * np.log(beta) + np.sum(data) / beta
                else:
                    if np.any(1 + xi * data / beta <= 0):
                        return 1e10
                    return n * np.log(beta) + (1/xi + 1) * np.sum(np.log(1 + xi * data / beta))
            
            # Initial guess
            result = optimize.minimize(neg_loglik, [0.1, self.exceedances.std()])
            self.xi, self.beta = result.x
            
            logger.info(f"POT fit: xi={self.xi:.3f}, beta={self.beta:.4f}, threshold={self.threshold:.4f}")
            
        def var_pot(self, alpha: float = 0.01) -> float:
            """
            VaR using POT method.
            
            VaR = threshold + (beta/xi) * ((n/N_u * alpha)^(-xi) - 1)
            """
            n = len(self.returns)
            N_u = len(self.exceedances)
            
            # Probability of exceeding threshold
            p = N_u / n
            
            # VaR
            if abs(self.xi) < 1e-6:
                var = self.threshold + self.beta * np.log(p / alpha)
            else:
                var = self.threshold + (self.beta / self.xi) * ((p / alpha) ** (-self.xi) - 1)
                
            return var
        
        def cvar_pot(self, alpha: float = 0.01) -> float:
            """CVaR using POT method."""
            var = self.var_pot(alpha)
            
            if abs(self.xi) < 1e-6:
                cvar = self.threshold + self.beta * (1 + np.log(p / alpha))
            else:
                cvar = self.threshold + (self.beta / self.xi) * (1 / (1 - self.xi)) * \
                       ((p / alpha) ** (1 - self.xi) - 1) - self.beta / (1 - self.xi)
                
            return cvar
    
    def pot_model(self, threshold: float = None) -> 'POT':
        """
        Create POT model.
        
        Args:
            threshold: Threshold for exceedances
            
        Returns:
            POT object
        """
        return self.POT(self.returns, threshold)
    
    # =========================================================================
    # 6️⃣ Monte Carlo VaR with Fat-Tails
    # =========================================================================
    def monte_carlo_var(
        self, 
        n_simulations: int = 100000, 
        alpha: float = 0.01,
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation using Student-t distribution.
        
        Args:
            n_simulations: Number of simulations
            alpha: Significance level
            horizon: Time horizon (days)
            
        Returns:
            Dict with VaR, CVaR, and simulation results
        """
        if self.nu is None:
            self.fit_student_t()
        
        # Current volatility
        if self.volatility is not None:
            vol = self.volatility.iloc[-1]
        else:
            vol = self.returns.std()
        
        # Simulate returns
        simulated = stats.t.rvs(
            df=self.nu, 
            loc=self.loc * vol, 
            scale=self.scale * vol,
            size=(n_simulations, horizon)
        )
        
        # Cumulative returns
        if horizon > 1:
            cumulative_returns = (1 + simulated).prod(axis=1) - 1
        else:
            cumulative_returns = simulated.flatten()
        
        # VaR and CVaR
        var = -np.percentile(cumulative_returns, alpha * 100)
        
        # CVaR: mean of losses beyond VaR
        worst_losses = cumulative_returns[cumulative_returns <= -var]
        cvar = -worst_losses.mean() if len(worst_losses) > 0 else var
        
        return {
            'var': var,
            'cvar': cvar,
            'simulations': cumulative_returns,
            'mean': cumulative_returns.mean(),
            'std': cumulative_returns.std(),
            'p5': np.percentile(cumulative_returns, 5),
            'p1': np.percentile(cumulative_returns, 1),
            'worst': cumulative_returns.min()
        }
    
    # =========================================================================
    # 7️⃣ Full Risk Report
    # =========================================================================
    def full_risk_report(self, alpha: float = 0.01) -> Dict:
        """
        Generate comprehensive fat-tail risk report.
        
        Args:
            alpha: Significance level
            
        Returns:
            Complete risk metrics dictionary
        """
        report = {
            'distribution': 'Student-t',
            'degrees_of_freedom': self.nu,
            'location': self.loc,
            'scale': self.scale,
        }
        
        # Basic moments
        if self.standardized_returns is not None:
            data = self.standardized_returns
        else:
            data = self.returns
            
        report['mean'] = data.mean()
        report['std'] = data.std()
        report['skewness'] = stats.skew(data)
        report['kurtosis'] = stats.kurtosis(data)
        
        # VaR methods
        report['var_parametric'] = self.parametric_var(alpha)
        report['var_cornish_fisher'] = self.cornish_fisher_var(alpha)
        
        # CVaR
        report['cvar_parametric'] = self.parametric_cvar(alpha)
        
        # POT
        try:
            pot = self.pot_model()
            report['var_pot'] = pot.var_pot(alpha)
            report['cvar_pot'] = pot.cvar_pot(alpha)
        except:
            report['var_pot'] = None
            report['cvar_pot'] = None
        
        # Monte Carlo
        mc = self.monte_carlo_var(alpha=alpha)
        report['monte_carlo_var'] = mc['var']
        report['monte_carlo_cvar'] = mc['cvar']
        
        return report


# =============================================================================
# HILL ESTIMATOR (Simple EVT)
# =============================================================================

class HillEstimator:
    """
    Hill estimator for tail index - simple extreme value method.
    """
    
    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
        
    def fit(self, k: int = None) -> Dict:
        """
        Fit Hill estimator.
        
        Args:
            k: Number of upper order statistics to use
            
        Returns:
            Tail index and VaR estimate
        """
        sorted_returns = np.sort(self.returns)[::-1]  # Descending
        
        if k is None:
            k = int(len(sorted_returns) * 0.1)  # 10% of data
            
        # Hill estimator for tail index
        log_ratios = np.log(sorted_returns[:k]) - np.log(sorted_returns[k])
        alpha = 1 / np.mean(log_ratios)
        
        # VaR at alpha level
        x_k = sorted_returns[k]
        var = x_k * (k / (len(self.returns) * 0.01)) ** alpha
        
        return {
            'tail_index': alpha,
            'k': k,
            'var_1pct': var,
            'hill_plot_data': sorted_returns[:k]
        }
