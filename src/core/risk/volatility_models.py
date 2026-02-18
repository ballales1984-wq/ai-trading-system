"""
Advanced Volatility Modeling Module
GARCH, EGARCH, GJR-GARCH, and Stochastic Volatility models
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GARCHModel:
    """
    GARCH(1,1) volatility model with various extensions.
    
    Models:
    - GARCH(1,1): Standard symmetric model
    - EGARCH(1,1): Asymmetric (leverage effects)
    - GJR-GARCH(1,1): Threshold GARCH
    """
    
    def __init__(self, p: int = 1, q: int = 1, model_type: str = "GARCH"):
        """
        Initialize GARCH model.
        
        Args:
            p: GARCH order
            q: ARCH order
            model_type: 'GARCH', 'EGARCH', or 'GJR'
        """
        self.p = p
        self.q = q
        self.model_type = model_type.upper()
        self.params = None
        self.residuals = None
        self.conditional_vol = None
        
    # =========================================================================
    # GARCH(1,1) - Standard Model
    # =========================================================================
    def fit_garch(self, returns: pd.Series) -> Dict:
        """
        Fit GARCH(1,1) model.
        
        r_t = mu + epsilon_t
        epsilon_t = sigma_t * z_t
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with parameters and diagnostics
        """
        returns = returns.dropna()
        n = len(returns)
        
        # Initial estimates
        mu = returns.mean()
        var = returns.var()
        
        def garch_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            sigma2 = np.zeros(n)
            sigma2[0] = var
            
            for t in range(1, n):
                sigma2[t] = omega + alpha * (returns.iloc[t-1] - mu)**2 + beta * sigma2[t-1]
            
            loglik = -0.5 * np.sum(
                np.log(2 * np.pi) + np.log(sigma2) + (returns - mu)**2 / sigma2
            )
            
            return -loglik
        
        # Optimize
        result = optimize.minimize(
            garch_loglik,
            x0=[var * 0.1, 0.05, 0.9],
            method='SLSQP',
            bounds=[(1e-6, var), (0.001, 0.3), (0.5, 0.98)]
        )
        
        omega, alpha, beta = result.x
        self.params = {'omega': omega, 'alpha': alpha, 'beta': beta, 'mu': mu}
        
        # Compute conditional volatility
        self.conditional_vol = self._compute_conditional_vol(returns)
        
        # Diagnostics
        standardized = (returns - mu) / self.conditional_vol
        
        return {
            'params': self.params,
            'aic': 2 * result.fun + 2 * 3 / n,
            'bic': 2 * result.fun + 3 * np.log(n) / n,
            'log_likelihood': -result.fun,
            'standardized_residuals': standardized,
            'conditional_vol': self.conditional_vol,
            'persistence': alpha + beta,
            'half_life': np.log(0.5) / np.log(alpha + beta) if alpha + beta < 1 else np.inf
        }
    
    # =========================================================================
    # EGARCH - Exponential GARCH (asymmetric)
    # =========================================================================
    def fit_egarch(self, returns: pd.Series) -> Dict:
        """
        Fit EGARCH(1,1) model.
        
        log(sigma_t^2) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma_{t-1}^2)
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with parameters and diagnostics
        """
        returns = returns.dropna()
        n = len(returns)
        
        mu = returns.mean()
        var = returns.var()
        
        def egarch_loglik(params):
            omega, alpha, gamma, beta = params
            if beta >= 1 or beta <= 0 or alpha <= 0:
                return 1e10
            
            log_sigma2 = np.zeros(n)
            log_sigma2[0] = np.log(var)
            
            for t in range(1, n):
                z = (returns.iloc[t-1] - mu) / np.sqrt(var)
                log_sigma2[t] = omega + alpha * abs(z) + gamma * z + beta * log_sigma2[t-1]
            
            sigma2 = np.exp(log_sigma2)
            
            loglik = -0.5 * np.sum(
                np.log(2 * np.pi) + log_sigma2 + (returns - mu)**2 / sigma2
            )
            
            return -loglik
        
        result = optimize.minimize(
            egarch_loglik,
            x0=[-0.1, 0.1, -0.05, 0.95],
            method='SLSQP'
        )
        
        omega, alpha, gamma, beta = result.x
        self.params = {'omega': omega, 'alpha': alpha, 'gamma': gamma, 'beta': beta, 'mu': mu}
        
        # Compute conditional volatility
        self.conditional_vol = self._compute_egarch_vol(returns)
        
        standardized = (returns - mu) / self.conditional_vol
        
        return {
            'params': self.params,
            'aic': 2 * result.fun + 2 * 4 / n,
            'bic': 2 * result.fun + 4 * np.log(n) / n,
            'log_likelihood': -result.fun,
            'standardized_residuals': standardized,
            'conditional_vol': self.conditional_vol,
            'leverage_effect': gamma < 0  # Negative gamma = negative shock increases vol
        }
    
    # =========================================================================
    # GJR-GARCH - Threshold GARCH
    # =========================================================================
    def fit_gjr_garch(self, returns: pd.Series, threshold: float = 0) -> Dict:
        """
        Fit GJR-GARCH(1,1) model with leverage effect.
        
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + gamma * I_{t-1} * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        
        Args:
            returns: Series of returns
            threshold: Threshold for indicator function (default 0)
            
        Returns:
            Dictionary with parameters and diagnostics
        """
        returns = returns.dropna()
        n = len(returns)
        
        mu = returns.mean()
        var = returns.var()
        
        def gjr_loglik(params):
            omega, alpha, gamma, beta = params
            if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0 or alpha + gamma/2 + beta >= 1:
                return 1e10
            
            sigma2 = np.zeros(n)
            sigma2[0] = var
            
            for t in range(1, n):
                eps = returns.iloc[t-1] - mu
                indicator = 1 if eps < threshold else 0
                sigma2[t] = omega + alpha * eps**2 + gamma * indicator * eps**2 + beta * sigma2[t-1]
            
            loglik = -0.5 * np.sum(
                np.log(2 * np.pi) + np.log(sigma2) + (returns - mu)**2 / sigma2
            )
            
            return -loglik
        
        result = optimize.minimize(
            gjr_loglik,
            x0=[var * 0.1, 0.05, 0.05, 0.9],
            method='SLSQP'
        )
        
        omega, alpha, gamma, beta = result.x
        self.params = {'omega': omega, 'alpha': alpha, 'gamma': gamma, 'beta': beta, 'mu': mu}
        
        self.conditional_vol = self._compute_gjr_vol(returns)
        
        standardized = (returns - mu) / self.conditional_vol
        
        return {
            'params': self.params,
            'aic': 2 * result.fun + 2 * 4 / n,
            'bic': 2 * result.fun + 4 * np.log(n) / n,
            'log_likelihood': -result.fun,
            'standardized_residuals': standardized,
            'conditional_vol': self.conditional_vol,
            'leverage_effect': gamma > 0  # Positive gamma = negative shock increases vol
        }
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    def _compute_conditional_vol(self, returns: pd.Series) -> pd.Series:
        """Compute conditional volatility for GARCH."""
        n = len(returns)
        omega = self.params['omega']
        alpha = self.params['alpha']
        beta = self.params['beta']
        mu = self.params['mu']
        
        var = returns.var()
        sigma2 = np.zeros(n)
        sigma2[0] = var
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * (returns.iloc[t-1] - mu)**2 + beta * sigma2[t-1]
        
        return pd.Series(np.sqrt(sigma2), index=returns.index)
    
    def _compute_egarch_vol(self, returns: pd.Series) -> pd.Series:
        """Compute conditional volatility for EGARCH."""
        n = len(returns)
        omega = self.params['omega']
        alpha = self.params['alpha']
        gamma = self.params['gamma']
        beta = self.params['beta']
        mu = self.params['mu']
        
        var = returns.var()
        log_sigma2 = np.zeros(n)
        log_sigma2[0] = np.log(var)
        
        for t in range(1, n):
            z = (returns.iloc[t-1] - mu) / np.sqrt(var)
            log_sigma2[t] = omega + alpha * abs(z) + gamma * z + beta * log_sigma2[t-1]
        
        return pd.Series(np.exp(log_sigma2 / 2), index=returns.index)
    
    def _compute_gjr_vol(self, returns: pd.Series) -> pd.Series:
        """Compute conditional volatility for GJR-GARCH."""
        n = len(returns)
        omega = self.params['omega']
        alpha = self.params['alpha']
        gamma = self.params['gamma']
        beta = self.params['beta']
        mu = self.params['mu']
        
        var = returns.var()
        sigma2 = np.zeros(n)
        sigma2[0] = var
        
        for t in range(1, n):
            eps = returns.iloc[t-1] - mu
            indicator = 1 if eps < 0 else 0
            sigma2[t] = omega + alpha * eps**2 + gamma * indicator * eps**2 + beta * sigma2[t-1]
        
        return pd.Series(np.sqrt(sigma2), index=returns.index)
    
    # =========================================================================
    # Forecasting
    # =========================================================================
    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast future volatility.
        
        Args:
            horizon: Forecast horizon (days)
            
        Returns:
            Forecasted volatility (annualized)
        """
        if self.params is None:
            raise ValueError("Model not fitted")
        
        alpha = self.params['alpha']
        beta = self.params['beta']
        omega = self.params['omega']
        
        # Long-run variance
        long_run_var = omega / (1 - alpha - beta)
        
        # Convergence to long-run variance
        forecast_var = long_run_var + (alpha + beta)**horizon * (
            self.conditional_vol.iloc[-1]**2 - long_run_var
        )
        
        return np.sqrt(forecast_var * 252)  # Annualized
    
    # =========================================================================
    # Fit All Models and Compare
    # =========================================================================
    def fit_all(self, returns: pd.Series) -> Dict:
        """
        Fit all GARCH variants and compare.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with results from all models
        """
        results = {}
        
        try:
            results['GARCH'] = self.fit_garch(returns)
        except Exception as e:
            logger.error(f"GARCH fit failed: {e}")
            results['GARCH'] = None
        
        try:
            results['EGARCH'] = self.fit_egarch(returns)
        except Exception as e:
            logger.error(f"EGARCH fit failed: {e}")
            results['EGARCH'] = None
        
        try:
            results['GJR-GARCH'] = self.fit_gjr_garch(returns)
        except Exception as e:
            logger.error(f"GJR-GARCH fit failed: {e}")
            results['GJR-GARCH'] = None
        
        return results


# =============================================================================
# REGIME-SWITCHING VOLATILITY
# =============================================================================

class RegimeSwitchingVolatility:
    """
    Regime-switching volatility model for capturing market regimes.
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize regime-switching model.
        
        Args:
            n_regimes: Number of volatility regimes
        """
        self.n_regimes = n_regimes
        self.params = None
        self.regime_probs = None
        
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit regime-switching model using EM algorithm.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with regime parameters
        """
        returns = returns.dropna()
        
        # Initialize parameters
        mu = returns.mean()
        std = returns.std()
        
        # Two regimes: low vol and high vol
        mu_low = mu
        sigma_low = std * 0.5
        
        mu_high = mu
        sigma_high = std * 1.5
        
        # Transition probabilities (simplified)
        p_ll = 0.95  # Low to Low
        p_lh = 0.05  # Low to High
        p_hh = 0.90  # High to High
        p_hl = 0.10  # High to Low
        
        # Compute regime probabilities (forward-backward simplified)
        low_vol_probs = self._compute_regime_prob(
            returns, mu_low, sigma_low, mu_high, sigma_high
        )
        
        self.params = {
            'mu_low': mu_low,
            'sigma_low': sigma_low,
            'mu_high': mu_high,
            'sigma_high': sigma_high,
            'p_ll': p_ll,
            'p_lh': p_lh,
            'p_hh': p_hh,
            'p_hl': p_hl
        }
        
        self.regime_probs = low_vol_probs
        
        return {
            'params': self.params,
            'regime_probabilities': low_vol_probs,
            'current_regime': 'low_vol' if low_vol_probs.iloc[-1] > 0.5 else 'high_vol',
            'volatility_regime': sigma_low if low_vol_probs.iloc[-1] > 0.5 else sigma_high
        }
    
    def _compute_regime_prob(
        self, 
        returns: pd.Series, 
        mu1: float, 
        sigma1: float,
        mu2: float, 
        sigma2: float
    ) -> pd.Series:
        """Compute probability of being in low-volatility regime."""
        ll1 = stats.norm.pdf(returns, mu1, sigma1)
        ll2 = stats.norm.pdf(returns, mu2, sigma2)
        
        # Normalize
        prob_low = ll1 / (ll1 + ll2)
        
        # Smooth with rolling average
        prob_low = pd.Series(prob_low, index=returns.index).rolling(5).mean()
        
        return prob_low.fillna(0.5)


# =============================================================================
# REALIZED VOLATILITY
# =============================================================================

class RealizedVolatility:
    """
    Realized volatility estimators.
    """
    
    @staticmethod
    def parkinson( highs: pd.Series, lows: pd.Series, window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator.
        
        sqrt((1 / (4 * log(2))) * ((log(H/L))^2))
        
        Args:
            highs: High prices
            lows: Low prices
            window: Rolling window size
            
        Returns:
            Realized volatility
        """
        hl_ratio = np.log(highs / lows)
        parkinson = np.sqrt((1 / (4 * np.log(2))) * hl_ratio**2)
        
        return parkinson.rolling(window).mean() * np.sqrt(252)
    
    @staticmethod
    def garman_klass(
        highs: pd.Series, 
        lows: pd.Series, 
        opens: pd.Series, 
        closes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Garman-Klass volatility estimator.
        
        More efficient than close-to-close.
        
        Args:
            highs: High prices
            lows: Low prices
            opens: Open prices
            closes: Close prices
            window: Rolling window size
            
        Returns:
            Realized volatility
        """
        log_hl = np.log(highs / lows)
        log_co = np.log(closes / opens)
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        
        return np.sqrt(gk.rolling(window).mean() * 252)
    
    @staticmethod
    def rogers_satchell(
        highs: pd.Series, 
        lows: pd.Series, 
        opens: pd.Series, 
        closes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Rogers-Satchell volatility estimator.
        
        Unbiased for drift.
        
        Args:
            highs: High prices
            lows: Low prices
            opens: Open prices
            closes: Close prices
            window: Rolling window size
            
        Returns:
            Realized volatility
        """
        log_hc = np.log(highs / closes)
        log_ho = np.log(highs / opens)
        log_lc = np.log(lows / closes)
        log_lo = np.log(lows / opens)
        
        rs = log_hc * log_ho + log_lc * log_lo
        
        return np.sqrt(rs.rolling(window).mean() * 252)
    
    @staticmethod
    def yang_zhang(
        highs: pd.Series, 
        lows: pd.Series, 
        opens: pd.Series, 
        closes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Yang-Zhang volatility estimator.
        
        Combines overnight, open-close, and Rogers-Satchell.
        
        Args:
            highs: High prices
            lows: Low prices
            opens: Open prices
            closes: Close prices
            window: Rolling window size
            
        Returns:
            Realized volatility
        """
        # Overnight volatility
        log_oc = np.log(opens / closes.shift(1))
        overnight_var = log_oc.rolling(window).var()
        
        # Open-close volatility
        log_co = np.log(closes / opens)
        oc_var = log_co.rolling(window).var()
        
        # Rogers-Satchell
        log_hc = np.log(highs / closes)
        log_ho = np.log(highs / opens)
        log_lc = np.log(lows / closes)
        log_lo = np.log(lows / opens)
        rs = log_hc * log_ho + log_lc * log_lo
        rs_var = rs.rolling(window).var()
        
        # Yang-Zhang
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
        
        return np.sqrt(yz_var * 252)
