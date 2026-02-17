"""
Portfolio Optimizer
=================
Advanced portfolio optimization using Markowitz, Risk Parity, and Black-Litterman.

Features:
- Markowitz Mean-Variance Optimization
- Risk Parity
- Black-Litterman Model
- Ensemble Optimizer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimizer with multiple methods.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
    
    # ======================
    # MARKOWITZ OPTIMIZER
    # ======================
    
    def markowitz(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
    ) -> np.ndarray:
        """
        Markowitz Mean-Variance Optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (higher = less risk)
            
        Returns:
            Optimal weights
        """
        n = len(expected_returns)
        
        try:
            # Inverse covariance matrix
            cov_inv = np.linalg.inv(cov_matrix + 1e-8 * np.eye(n))
            
            # Calculate optimal weights
            numerator = cov_inv @ expected_returns
            denominator = np.ones(n) @ cov_inv @ expected_returns
            
            weights = numerator / denominator
            
            # Apply risk aversion
            weights = weights / (1 + risk_aversion)
            
            # Ensure positive weights and normalize
            weights = np.maximum(weights, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n) / n
            
            return weights
            
        except np.linalg.LinAlgError:
            logger.warning("Markowitz: using equal weights due to singular matrix")
            return np.ones(n) / n
    
    def max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Maximize Sharpe Ratio.
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            
        Returns:
            Weights that maximize Sharpe ratio
        """
        return self.markowitz(expected_returns, cov_matrix, risk_aversion=0.0)
    
    # ======================
    # RISK PARITY
    # ======================
    
    def risk_parity(
        self,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Risk Parity optimization.
        Each asset contributes equally to portfolio risk.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Risk parity weights
        """
        # Calculate volatility (standard deviation)
        vol = np.sqrt(np.diag(cov_matrix))
        
        # Inverse volatility weighting
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        
        return weights
    
    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 4,
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity.
        
        Args:
            returns: Returns DataFrame
            n_clusters: Number of clusters
            
        Returns:
            HRP weights
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Calculate correlation matrix
            corr = returns.corr()
            
            # Convert to distance matrix
            dist = np.sqrt(2 * (1 - corr))
            dist_matrix = squareform(dist)
            
            # Hierarchical clustering
            linkage_matrix = linkage(dist_matrix, method='ward')
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Initialize weights
            n = len(returns.columns)
            weights = np.zeros(n)
            
            # Assign weights by cluster
            for i in range(1, n_clusters + 1):
                mask = clusters == i
                cluster_assets = np.where(mask)[0]
                
                if len(cluster_assets) > 0:
                    # Use inverse vol within cluster
                    cluster_vol = np.sqrt(np.diag(cov_matrix.values))
                    cluster_vol = cluster_vol[mask]
                    inv_vol = 1 / cluster_vol
                    cluster_weights = inv_vol / inv_vol.sum()
                    
                    weights[cluster_assets] = cluster_weights / n_clusters
            
            return weights
            
        except Exception as e:
            logger.warning(f"HRP failed: {e}, using equal weights")
            return np.ones(len(returns.columns)) / len(returns.columns)
    
    # ======================
    # BLACK-LITTERMAN
    # ======================
    
    def black_litterman(
        self,
        cov_matrix: np.ndarray,
        market_cap_weights: np.ndarray,
        views: Optional[Dict[int, float]] = None,
        tau: float = 0.05,
    ) -> np.ndarray:
        """
        Black-Litterman Model.
        
        Args:
            cov_matrix: Covariance matrix
            market_cap_weights: Market cap weights (equilibrium)
            views: Dict of {asset_index: expected_return}
            tau: Uncertainty in prior (scalar)
            
        Returns:
            Black-Litterman weights
        """
        n = len(market_cap_weights)
        
        # Calculate implied returns (equilibrium)
        equilibrium_returns = market_cap_weights * (
            self.risk_free_rate + 
            2 * np.diag(cov_matrix).mean()
        )
        
        if views is None or len(views) == 0:
            # No views - use equilibrium
            return market_cap_weights
        
        # Build views matrix
        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = view_return
        
        # Black-Litterman formula
        try:
            tau_cov = tau * cov_matrix
            M = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ P)
            post_returns = M @ (np.linalg.inv(tau_cov) @ equilibrium_returns + P.T @ Q)
            
            # Optimize with posterior returns
            weights = self.markowitz(post_returns, cov_matrix)
            return weights
            
        except np.linalg.LinAlgError:
            logger.warning("Black-Litterman failed, using market weights")
            return market_cap_weights
    
    # ======================
    # MINIMUM VARIANCE
    # ======================
    
    def minimum_variance(
        self,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Minimum Variance Portfolio.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Minimum variance weights
        """
        try:
            cov_inv = np.linalg.inv(cov_matrix + 1e-8 * np.eye(len(cov_matrix)))
            ones = np.ones(len(cov_matrix))
            weights = cov_inv @ ones
            weights = weights / weights.sum()
            return np.maximum(weights, 0)
        except:
            return np.ones(len(cov_matrix)) / len(cov_matrix)
    
    # ======================
    # MAXIMUM DIVERSIFICATION
    # ======================
    
    def max_diversification(
        self,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Maximum Diversification Portfolio.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Max diversification weights
        """
        vol = np.sqrt(np.diag(cov_matrix))
        cov_vol = cov_matrix / np.outer(vol, vol)
        
        try:
            cov_vol_inv = np.linalg.inv(cov_vol)
            ones = np.ones(len(vol))
            weights = cov_vol_inv @ ones
            weights = weights * vol
            weights = weights / weights.sum()
            return np.maximum(weights, 0)
        except:
            return self.risk_parity(cov_matrix)
    
    # ======================
    # KELLY CRITERION
    # ======================
    
    def kelly_weight(
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
            avg_loss: Average loss amount (positive)
            fraction: Kelly fraction to use
            
        Returns:
            Kelly weight
        """
        if avg_loss <= 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fraction
        kelly = kelly * fraction
        
        return max(kelly, 0)
    
    # ======================
    # ENSEMBLE OPTIMIZER
    # ======================
    
    def ensemble(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights_dict: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Ensemble optimizer combining multiple methods.
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            weights_dict: Dict of {method: weight}
            
        Returns:
            Ensemble weights
        """
        if weights_dict is None:
            weights_dict = {
                'markowitz': 0.4,
                'risk_parity': 0.3,
                'min_variance': 0.3,
            }
        
        # Calculate weights from each method
        w_mkt = self.markowitz(expected_returns, cov_matrix)
        w_rp = self.risk_parity(cov_matrix)
        w_mv = self.minimum_variance(cov_matrix)
        
        # Combine
        final_weights = (
            weights_dict.get('markowitz', 0.4) * w_mkt +
            weights_dict.get('risk_parity', 0.3) * w_rp +
            weights_dict.get('min_variance', 0.3) * w_mv
        )
        
        # Normalize
        final_weights = np.maximum(final_weights, 0)
        final_weights = final_weights / final_weights.sum()
        
        return final_weights
    
    # ======================
    # UTILITY FUNCTIONS
    # ======================
    
    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            
        Returns:
            Dictionary of metrics
        """
        # Expected return
        port_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return {
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
        }


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    # Create optimizer
    po = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Example data
    assets = ['BTC', 'ETH', 'SOL', 'BNB']
    n = len(assets)
    
    # Expected returns (annual)
    expected_returns = np.array([0.15, 0.12, 0.20, 0.10])
    
    # Covariance matrix
    cov_matrix = np.array([
        [0.04, 0.02, 0.015, 0.01],
        [0.02, 0.05, 0.018, 0.012],
        [0.015, 0.018, 0.06, 0.015],
        [0.01, 0.012, 0.015, 0.03],
    ])
    
    print("📊 Portfolio Optimization Results:")
    print("=" * 50)
    
    # Markowitz
    w_mkt = po.markowitz(expected_returns, cov_matrix)
    print("\n📈 Markowitz:")
    for a, w in zip(assets, w_mkt):
        print(f"  {a}: {w*100:.1f}%")
    
    # Risk Parity
    w_rp = po.risk_parity(cov_matrix)
    print("\n📊 Risk Parity:")
    for a, w in zip(assets, w_rp):
        print(f"  {a}: {w*100:.1f}%")
    
    # Minimum Variance
    w_mv = po.minimum_variance(cov_matrix)
    print("\n📉 Minimum Variance:")
    for a, w in zip(assets, w_mv):
        print(f"  {a}: {w*100:.1f}%")
    
    # Ensemble
    w_ens = po.ensemble(expected_returns, cov_matrix)
    print("\n🎯 Ensemble:")
    for a, w in zip(assets, w_ens):
        print(f"  {a}: {w*100:.1f}%")
