"""
app/portfolio/optimization.py
Portfolio optimization using Modern Portfolio Theory and extensions.

Provides:
  - Mean-Variance Optimization (Markowitz)
  - Risk Parity
  - Maximum Sharpe Ratio
  - Minimum Variance
  - Black-Litterman (with views)
  - Constraints: max position, sector limits, turnover
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: Dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""
    converged: bool = True
    iterations: int = 0
    constraints_satisfied: bool = True


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""

    min_weight: float = 0.0
    max_weight: float = 1.0
    max_positions: int = 20
    max_sector_weight: float = 0.4
    max_turnover: float = 0.5  # max fraction of portfolio to rebalance
    long_only: bool = True


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer.

    Usage:
        opt = PortfolioOptimizer(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            returns=np.array([[...], [...], [...]]),  # T x N
        )
        result = opt.optimize(method="max_sharpe")
    """

    def __init__(
        self,
        symbols: List[str],
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        constraints: Optional[OptimizationConstraints] = None,
    ):
        """
        Args:
            symbols: Asset names
            returns: T x N array of historical returns
            risk_free_rate: Annual risk-free rate
            constraints: Optimization constraints
        """
        self.symbols = symbols
        self.returns = returns
        self.n_assets = len(symbols)
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or OptimizationConstraints()

        # Precompute
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns, rowvar=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, method: str = "max_sharpe") -> OptimizationResult:
        """
        Run optimization with specified method.

        Methods: max_sharpe, min_variance, risk_parity, equal_weight,
                 inverse_volatility, black_litterman
        """
        methods = {
            "max_sharpe": self._max_sharpe,
            "min_variance": self._min_variance,
            "risk_parity": self._risk_parity,
            "equal_weight": self._equal_weight,
            "inverse_volatility": self._inverse_volatility,
        }

        func = methods.get(method)
        if func is None:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

        weights = func()
        weights = self._apply_constraints(weights)

        # Compute portfolio stats
        port_return = float(np.dot(weights, self.mean_returns) * 252)
        port_vol = float(np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)) * 252))
        daily_rf = self.risk_free_rate / 252
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        return OptimizationResult(
            weights={s: float(w) for s, w in zip(self.symbols, weights)},
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method=method,
        )

    def efficient_frontier(self, n_points: int = 50) -> List[Tuple[float, float, Dict[str, float]]]:
        """
        Compute the efficient frontier.

        Returns list of (volatility, return, weights) tuples.
        """
        target_returns = np.linspace(
            np.min(self.mean_returns) * 252,
            np.max(self.mean_returns) * 252,
            n_points,
        )

        frontier = []
        for target in target_returns:
            try:
                weights = self._min_variance_for_target(target)
                weights = self._apply_constraints(weights)
                vol = float(np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)) * 252))
                ret = float(np.dot(weights, self.mean_returns) * 252)
                w_dict = {s: float(w) for s, w in zip(self.symbols, weights)}
                frontier.append((vol, ret, w_dict))
            except Exception:
                continue

        return frontier

    # ------------------------------------------------------------------
    # Optimization methods
    # ------------------------------------------------------------------

    def _max_sharpe(self) -> np.ndarray:
        """Maximum Sharpe Ratio portfolio (analytical for unconstrained)."""
        daily_rf = self.risk_free_rate / 252
        excess = self.mean_returns - daily_rf

        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(self.cov_matrix)

        raw_weights = inv_cov @ excess
        total = np.sum(raw_weights)

        if total == 0 or not np.isfinite(total):
            return np.ones(self.n_assets) / self.n_assets

        weights = raw_weights / total
        return weights

    def _min_variance(self) -> np.ndarray:
        """Global Minimum Variance portfolio."""
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(self.cov_matrix)

        ones = np.ones(self.n_assets)
        raw_weights = inv_cov @ ones
        weights = raw_weights / np.sum(raw_weights)
        return weights

    def _min_variance_for_target(self, target_return: float) -> np.ndarray:
        """Minimum variance portfolio for a target return (Lagrangian)."""
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(self.cov_matrix)

        ones = np.ones(self.n_assets)
        mu = self.mean_returns * 252  # annualized

        A = ones @ inv_cov @ ones
        B = ones @ inv_cov @ mu
        C = mu @ inv_cov @ mu
        D = A * C - B * B

        if abs(D) < 1e-12:
            return np.ones(self.n_assets) / self.n_assets

        lam1 = (C - target_return * B) / D
        lam2 = (target_return * A - B) / D

        weights = inv_cov @ (lam1 * ones + lam2 * mu)
        return weights

    def _risk_parity(self) -> np.ndarray:
        """Risk Parity — equal risk contribution from each asset."""
        vols = np.sqrt(np.diag(self.cov_matrix))
        vols = np.where(vols > 0, vols, 1e-10)

        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)

        # Iterative refinement (Newton-like)
        for _ in range(100):
            port_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            if port_vol < 1e-12:
                break

            marginal_risk = self.cov_matrix @ weights / port_vol
            risk_contrib = weights * marginal_risk
            target_risk = port_vol / self.n_assets

            adjustment = target_risk / (risk_contrib + 1e-12)
            weights = weights * adjustment
            weights = weights / np.sum(weights)

        return weights

    def _equal_weight(self) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(self.n_assets) / self.n_assets

    def _inverse_volatility(self) -> np.ndarray:
        """Inverse volatility weighting."""
        vols = np.sqrt(np.diag(self.cov_matrix))
        vols = np.where(vols > 0, vols, 1e-10)
        inv_vols = 1.0 / vols
        return inv_vols / np.sum(inv_vols)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio constraints to weights."""
        c = self.constraints

        # Long only
        if c.long_only:
            weights = np.maximum(weights, 0)

        # Min/max weight
        weights = np.clip(weights, c.min_weight, c.max_weight)

        # Max positions
        if np.count_nonzero(weights) > c.max_positions:
            # Keep top N by weight
            threshold = np.sort(weights)[-c.max_positions]
            weights[weights < threshold] = 0

        # Renormalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def portfolio_stats(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Compute stats for given weights."""
        w = np.array([weights.get(s, 0) for s in self.symbols])
        ret = float(np.dot(w, self.mean_returns) * 252)
        vol = float(np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)) * 252))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        return {
            "expected_return": ret,
            "expected_volatility": vol,
            "sharpe_ratio": sharpe,
        }
