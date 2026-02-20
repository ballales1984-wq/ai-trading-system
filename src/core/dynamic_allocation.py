"""
Dynamic Capital Allocation Engine
==================================
Institutional-grade position sizing and capital allocation.

This module provides:
- Kelly Criterion with dynamic adjustment
- Volatility targeting
- Risk parity allocation
- Adaptive leverage control
- Regime-based allocation (HMM)
- Multi-strategy capital allocation

Author: AI Trading System
Version: 1.0.0
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================

class AllocationMethod(str, Enum):
    """Capital allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    REGIME_BASED = "regime_based"
    ADAPTIVE = "adaptive"


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class AllocationConfig:
    """Configuration for capital allocation."""
    # Target volatility (annualized)
    target_volatility: float = 0.15  # 15% annualized
    
    # Kelly parameters
    kelly_fraction: float = 0.5  # Half-Kelly by default
    max_kelly_leverage: float = 2.0
    min_kelly_fraction: float = 0.1
    
    # Risk parity parameters
    risk_parity_risk_budget: Optional[Dict[str, float]] = None
    
    # Leverage limits
    max_leverage: float = 3.0
    min_leverage: float = 0.1
    
    # Regime multipliers
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 0.8,
        MarketRegime.RANGING: 1.0,
        MarketRegime.HIGH_VOLATILITY: 0.5,
        MarketRegime.CRISIS: 0.2,
        MarketRegime.RECOVERY: 0.7
    })
    
    # Lookback periods
    volatility_lookback: int = 21  # Trading days
    return_lookback: int = 63  # Trading days
    correlation_lookback: int = 63
    
    # Smoothing
    allocation_smoothing: float = 0.3  # EMA smoothing factor
    
    # Constraints
    min_position_weight: float = 0.01  # 1% minimum
    max_position_weight: float = 0.20  # 20% maximum


@dataclass
class AssetMetrics:
    """Metrics for a single asset."""
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    kelly_fraction: float
    var_95: float
    cvar_95: float
    beta: float = 1.0
    sector: str = "unknown"


@dataclass
class AllocationResult:
    """Result of allocation calculation."""
    weights: Dict[str, float]
    method: AllocationMethod
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    leverage: float
    regime: MarketRegime
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# KELLY CRITERION CALCULATOR
# =============================================================================

class KellyCalculator:
    """
    Kelly Criterion calculator with adjustments.
    
    Features:
    - Full Kelly calculation
    - Fractional Kelly
    - Multiple asset Kelly
    - Drawdown-adjusted Kelly
    """
    
    @staticmethod
    def calculate_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction from win rate and average win/loss.
        
        Kelly = W - (1-W) / (avg_win/avg_loss)
        
        Where W = win rate
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        return max(0, kelly)
    
    @staticmethod
    def calculate_kelly_from_returns(returns: np.ndarray) -> float:
        """
        Calculate Kelly fraction from return series.
        
        Kelly = mean(r) / var(r) for log returns
        """
        if len(returns) < 10:
            return 0.0
        
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance == 0:
            return 0.0
        
        kelly = mean_return / variance
        
        # Annualize if daily returns
        kelly = kelly * 252
        
        return max(0, min(kelly, 1.0))
    
    @staticmethod
    def calculate_multiple_asset_kelly(
        returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Kelly weights for multiple assets.
        
        Uses: w = Σ^(-1) * μ
        """
        try:
            # Add small regularization
            cov_reg = cov_matrix + np.eye(len(cov_matrix)) * 1e-6
            cov_inv = np.linalg.inv(cov_reg)
            
            mean_returns = np.mean(returns, axis=0)
            kelly_weights = cov_inv @ mean_returns
            
            # Normalize to sum to 1
            kelly_weights = kelly_weights / np.sum(np.abs(kelly_weights))
            
            return kelly_weights
        except Exception as e:
            logger.error(f"Error calculating multi-asset Kelly: {e}")
            return np.ones(len(returns[0])) / len(returns[0])
    
    @staticmethod
    def adjust_for_drawdown(
        kelly: float,
        max_drawdown: float,
        target_drawdown: float = 0.10
    ) -> float:
        """
        Adjust Kelly fraction based on historical drawdown.
        
        Reduce Kelly if drawdown exceeds target.
        """
        if max_drawdown <= target_drawdown:
            return kelly
        
        # Reduce proportionally
        adjustment = target_drawdown / max_drawdown
        return kelly * adjustment


# =============================================================================
# VOLATILITY TARGETING
# =============================================================================

class VolatilityTargeting:
    """
    Volatility targeting position sizing.
    
    Adjusts position sizes to achieve target portfolio volatility.
    """
    
    def __init__(self, target_vol: float = 0.15):
        self.target_vol = target_vol
    
    def calculate_leverage(
        self,
        current_vol: float,
        min_leverage: float = 0.1,
        max_leverage: float = 3.0
    ) -> float:
        """
        Calculate leverage to achieve target volatility.
        
        leverage = target_vol / current_vol
        """
        if current_vol <= 0:
            return min_leverage
        
        leverage = self.target_vol / current_vol
        
        return max(min_leverage, min(max_leverage, leverage))
    
    def adjust_weights(
        self,
        weights: Dict[str, float],
        asset_vols: Dict[str, float],
        target_vol: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Adjust weights to achieve target volatility contribution.
        """
        target = target_vol or self.target_vol
        
        # Calculate current portfolio volatility
        current_vol = self._calculate_portfolio_vol(weights, asset_vols)
        
        if current_vol <= 0:
            return weights
        
        # Scale weights
        scale = target / current_vol
        
        adjusted = {
            symbol: min(weight * scale, 0.20)  # Cap at 20%
            for symbol, weight in weights.items()
        }
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted
    
    def _calculate_portfolio_vol(
        self,
        weights: Dict[str, float],
        asset_vols: Dict[str, float]
    ) -> float:
        """Calculate portfolio volatility from weights and asset vols."""
        variance = 0.0
        for symbol, weight in weights.items():
            vol = asset_vols.get(symbol, 0.20)
            variance += (weight ** 2) * (vol ** 2)
        
        # Add correlation term (assume 0.5 average correlation)
        symbols = list(weights.keys())
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                correlation = 0.5  # Default correlation
                variance += 2 * weights[s1] * weights[s2] * asset_vols.get(s1, 0.20) * asset_vols.get(s2, 0.20) * correlation
        
        return np.sqrt(variance)


# =============================================================================
# RISK PARITY
# =============================================================================

class RiskParityAllocator:
    """
    Risk parity allocation engine.
    
    Allocates capital so each asset contributes equally to portfolio risk.
    """
    
    def __init__(self, config: AllocationConfig):
        self.config = config
    
    def calculate_weights(
        self,
        returns: np.ndarray,
        symbols: List[str],
        risk_budget: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights.
        
        Each asset contributes equally to total portfolio risk.
        """
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns.T)
        
        # Default to equal risk budget
        if risk_budget is None:
            risk_budget = {s: 1.0/n_assets for s in symbols}
        
        target_risk = np.array([risk_budget.get(s, 1.0/n_assets) for s in symbols])
        target_risk = target_risk / np.sum(target_risk)
        
        # Optimize for risk parity
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights
            risk_contribution = weights * marginal_risk / portfolio_vol
            
            # Minimize difference from target risk contribution
            return np.sum((risk_contribution / portfolio_vol - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        bounds = [(self.config.min_position_weight, self.config.max_position_weight) 
                  for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            weights = result.x
            
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")
            weights = np.ones(n_assets) / n_assets
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return {symbol: weight for symbol, weight in zip(symbols, weights)}


# =============================================================================
# DYNAMIC ALLOCATION ENGINE
# =============================================================================

class DynamicAllocationEngine:
    """
    Institutional-grade dynamic capital allocation.
    
    Features:
    - Kelly Criterion with dynamic adjustment
    - Volatility targeting
    - Risk parity allocation
    - Regime-based allocation
    - Multi-strategy allocation
    - Adaptive leverage control
    """
    
    def __init__(
        self,
        config: Optional[AllocationConfig] = None,
        initial_capital: float = 100000.0
    ):
        self.config = config or AllocationConfig()
        self.initial_capital = initial_capital
        
        # Sub-components
        self.kelly = KellyCalculator()
        self.vol_targeting = VolatilityTargeting(self.config.target_volatility)
        self.risk_parity = RiskParityAllocator(self.config)
        
        # State tracking
        self._current_regime = MarketRegime.RANGING
        self._current_leverage = 1.0
        self._allocation_history: deque = deque(maxlen=100)
        self._return_history: Dict[str, deque] = {}
        
        # Threading
        self._lock = threading.RLock()
        
        # Callbacks
        self._allocation_callbacks: List[Callable] = []
        
        logger.info(
            f"Dynamic Allocation Engine initialized with "
            f"target_vol={self.config.target_volatility*100}%, "
            f"kelly_fraction={self.config.kelly_fraction}"
        )
    
    # =========================================================================
    # MAIN ALLOCATION METHODS
    # =========================================================================
    
    def calculate_allocation(
        self,
        assets: Dict[str, AssetMetrics],
        returns: Optional[np.ndarray] = None,
        symbols: Optional[List[str]] = None,
        method: AllocationMethod = AllocationMethod.ADAPTIVE,
        regime: Optional[MarketRegime] = None
    ) -> AllocationResult:
        """
        Calculate optimal capital allocation.
        
        Args:
            assets: Dictionary of asset metrics
            returns: Historical returns matrix (optional)
            symbols: List of symbols (optional)
            method: Allocation method
            regime: Current market regime (optional)
        
        Returns:
            AllocationResult with weights and metrics
        """
        with self._lock:
            # Update regime
            if regime:
                self._current_regime = regime
            
            # Get symbols
            if symbols is None:
                symbols = list(assets.keys())
            
            if not symbols:
                return self._empty_result()
            
            # Calculate weights based on method
            if method == AllocationMethod.ADAPTIVE:
                weights = self._adaptive_allocation(assets, returns, symbols)
            elif method == AllocationMethod.KELLY:
                weights = self._kelly_allocation(assets, symbols)
            elif method == AllocationMethod.HALF_KELLY:
                weights = self._kelly_allocation(assets, symbols, fraction=0.5)
            elif method == AllocationMethod.VOLATILITY_TARGETING:
                weights = self._volatility_targeted_allocation(assets, symbols)
            elif method == AllocationMethod.RISK_PARITY:
                weights = self._risk_parity_allocation(returns, symbols)
            elif method == AllocationMethod.EQUAL_WEIGHT:
                weights = {s: 1.0/len(symbols) for s in symbols}
            else:
                weights = {s: 1.0/len(symbols) for s in symbols}
            
            # Apply regime adjustment
            weights = self._apply_regime_adjustment(weights)
            
            # Apply constraints
            weights = self._apply_constraints(weights)
            
            # Calculate expected metrics
            expected_return = self._calculate_expected_return(weights, assets)
            expected_vol = self._calculate_expected_volatility(weights, assets)
            expected_sharpe = expected_return / expected_vol if expected_vol > 0 else 0
            
            # Calculate leverage
            leverage = self._calculate_leverage(expected_vol)
            
            result = AllocationResult(
                weights=weights,
                method=method,
                expected_return=expected_return,
                expected_volatility=expected_vol,
                expected_sharpe=expected_sharpe,
                leverage=leverage,
                regime=self._current_regime
            )
            
            # Store history
            self._allocation_history.append(result)
            
            return result
    
    def _adaptive_allocation(
        self,
        assets: Dict[str, AssetMetrics],
        returns: Optional[np.ndarray],
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Adaptive allocation combining multiple methods.
        
        Blends Kelly, Risk Parity, and Volatility Targeting
        based on market conditions.
        """
        # Calculate all methods
        kelly_weights = self._kelly_allocation(assets, symbols, fraction=0.5)
        
        if returns is not None and len(returns) > 20:
            rp_weights = self._risk_parity_allocation(returns, symbols)
        else:
            rp_weights = {s: 1.0/len(symbols) for s in symbols}
        
        vol_weights = self._volatility_targeted_allocation(assets, symbols)
        
        # Blend based on regime
        regime = self._current_regime
        
        if regime == MarketRegime.TRENDING_UP:
            # Favor Kelly in trending markets
            blend = {'kelly': 0.5, 'rp': 0.3, 'vol': 0.2}
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Favor risk parity in volatile markets
            blend = {'kelly': 0.2, 'rp': 0.5, 'vol': 0.3}
        elif regime == MarketRegime.CRISIS:
            # Favor volatility targeting in crisis
            blend = {'kelly': 0.1, 'rp': 0.3, 'vol': 0.6}
        else:
            # Balanced blend
            blend = {'kelly': 0.35, 'rp': 0.35, 'vol': 0.3}
        
        # Combine weights
        combined = {}
        for symbol in symbols:
            combined[symbol] = (
                blend['kelly'] * kelly_weights.get(symbol, 0) +
                blend['rp'] * rp_weights.get(symbol, 0) +
                blend['vol'] * vol_weights.get(symbol, 0)
            )
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def _kelly_allocation(
        self,
        assets: Dict[str, AssetMetrics],
        symbols: List[str],
        fraction: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate Kelly-based allocation."""
        fraction = fraction or self.config.kelly_fraction
        
        weights = {}
        for symbol in symbols:
            if symbol in assets:
                asset = assets[symbol]
                
                # Calculate Kelly
                kelly = self.kelly.calculate_kelly(
                    asset.win_rate,
                    asset.avg_win,
                    asset.avg_loss
                )
                
                # Adjust for drawdown
                kelly = self.kelly.adjust_for_drawdown(
                    kelly,
                    asset.max_drawdown
                )
                
                # Apply fraction
                weight = kelly * fraction
                
                # Apply constraints
                weight = max(self.config.min_position_weight,
                           min(weight, self.config.max_position_weight))
                
                weights[symbol] = weight
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _volatility_targeted_allocation(
        self,
        assets: Dict[str, AssetMetrics],
        symbols: List[str]
    ) -> Dict[str, float]:
        """Calculate volatility-targeted allocation."""
        # Inverse volatility weighting
        weights = {}
        for symbol in symbols:
            if symbol in assets:
                vol = assets[symbol].volatility
                if vol > 0:
                    weights[symbol] = 1.0 / vol
                else:
                    weights[symbol] = 1.0
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Adjust for target volatility
        asset_vols = {s: assets[s].volatility for s in symbols if s in assets}
        weights = self.vol_targeting.adjust_weights(weights, asset_vols)
        
        return weights
    
    def _risk_parity_allocation(
        self,
        returns: Optional[np.ndarray],
        symbols: List[str]
    ) -> Dict[str, float]:
        """Calculate risk parity allocation."""
        if returns is None or len(returns) < 10:
            return {s: 1.0/len(symbols) for s in symbols}
        
        return self.risk_parity.calculate_weights(returns, symbols)
    
    # =========================================================================
    # REGIME-BASED ADJUSTMENT
    # =========================================================================
    
    def _apply_regime_adjustment(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply regime-based adjustment to weights."""
        multiplier = self.config.regime_multipliers.get(
            self._current_regime, 1.0
        )
        
        # Scale weights
        adjusted = {
            symbol: weight * multiplier
            for symbol, weight in weights.items()
        }
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted
    
    def update_regime(self, regime: MarketRegime) -> None:
        """Update current market regime."""
        with self._lock:
            self._current_regime = regime
            logger.info(f"Market regime updated to: {regime.value}")
    
    def detect_regime_from_returns(
        self,
        returns: np.ndarray,
        volatility: float
    ) -> MarketRegime:
        """
        Detect market regime from returns.
        
        Simple heuristic-based detection.
        """
        if len(returns) < 20:
            return MarketRegime.RANGING
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = self._calculate_skewness(returns)
        
        # Crisis detection
        if volatility > 0.40 or (mean_return < -0.02 and std_return > 0.03):
            return MarketRegime.CRISIS
        
        # High volatility
        if volatility > 0.25:
            return MarketRegime.HIGH_VOLATILITY
        
        # Trending
        if abs(mean_return) > 0.005:
            if mean_return > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Recovery (positive skew after drawdown)
        if skewness > 0.5 and mean_return > 0:
            return MarketRegime.RECOVERY
        
        return MarketRegime.RANGING
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        return np.mean(((returns - mean) / std) ** 3)
    
    # =========================================================================
    # LEVERAGE CALCULATION
    # =========================================================================
    
    def _calculate_leverage(self, expected_vol: float) -> float:
        """Calculate optimal leverage for target volatility."""
        leverage = self.vol_targeting.calculate_leverage(
            expected_vol,
            self.config.min_leverage,
            self.config.max_leverage
        )
        
        # Apply regime adjustment
        regime_mult = self.config.regime_multipliers.get(
            self._current_regime, 1.0
        )
        leverage *= regime_mult
        
        # Apply constraints
        leverage = max(self.config.min_leverage,
                      min(leverage, self.config.max_leverage))
        
        self._current_leverage = leverage
        
        return leverage
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _apply_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply position constraints."""
        constrained = {}
        
        for symbol, weight in weights.items():
            # Apply min/max
            weight = max(self.config.min_position_weight,
                        min(weight, self.config.max_position_weight))
            constrained[symbol] = weight
        
        # Normalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v/total for k, v in constrained.items()}
        
        return constrained
    
    def _calculate_expected_return(
        self,
        weights: Dict[str, float],
        assets: Dict[str, AssetMetrics]
    ) -> float:
        """Calculate expected portfolio return."""
        expected = 0.0
        for symbol, weight in weights.items():
            if symbol in assets:
                expected += weight * assets[symbol].expected_return
        return expected
    
    def _calculate_expected_volatility(
        self,
        weights: Dict[str, float],
        assets: Dict[str, AssetMetrics]
    ) -> float:
        """Calculate expected portfolio volatility."""
        variance = 0.0
        
        # Individual variances
        for symbol, weight in weights.items():
            if symbol in assets:
                vol = assets[symbol].volatility
                variance += (weight ** 2) * (vol ** 2)
        
        # Correlation term (assume 0.5 average)
        symbols = list(weights.keys())
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                if s1 in assets and s2 in assets:
                    correlation = 0.5
                    variance += 2 * weights[s1] * weights[s2] * \
                               assets[s1].volatility * assets[s2].volatility * correlation
        
        return np.sqrt(variance)
    
    def _empty_result(self) -> AllocationResult:
        """Return empty allocation result."""
        return AllocationResult(
            weights={},
            method=AllocationMethod.EQUAL_WEIGHT,
            expected_return=0.0,
            expected_volatility=0.0,
            expected_sharpe=0.0,
            leverage=1.0,
            regime=self._current_regime
        )
    
    # =========================================================================
    # MULTI-STRATEGY ALLOCATION
    # =========================================================================
    
    def allocate_across_strategies(
        self,
        strategy_returns: Dict[str, np.ndarray],
        strategy_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Allocate capital across multiple strategies.
        
        Uses risk parity approach across strategies.
        """
        if not strategy_returns:
            return {}
        
        strategies = list(strategy_returns.keys())
        n_strategies = len(strategies)
        
        # Calculate strategy volatilities
        strategy_vols = {}
        for strategy, returns in strategy_returns.items():
            strategy_vols[strategy] = np.std(returns) * np.sqrt(252)
        
        # Inverse volatility weighting
        weights = {}
        for strategy in strategies:
            vol = strategy_vols.get(strategy, 0.15)
            if vol > 0:
                weights[strategy] = 1.0 / vol
            else:
                weights[strategy] = 1.0
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Adjust based on Sharpe ratio
        if strategy_metrics:
            sharpe_adjusted = {}
            for strategy in strategies:
                base_weight = weights.get(strategy, 1.0/n_strategies)
                sharpe = strategy_metrics.get(strategy, {}).get('sharpe', 0)
                
                # Increase weight for higher Sharpe
                adjustment = max(0.5, min(2.0, 1.0 + sharpe / 2))
                sharpe_adjusted[strategy] = base_weight * adjustment
            
            # Normalize
            total = sum(sharpe_adjusted.values())
            if total > 0:
                weights = {k: v/total for k, v in sharpe_adjusted.items()}
        
        return weights
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def calculate_position_size(
        self,
        symbol: str,
        asset: AssetMetrics,
        portfolio_value: float,
        method: AllocationMethod = AllocationMethod.HALF_KELLY
    ) -> float:
        """
        Calculate position size for a single asset.
        
        Returns position size in currency units.
        """
        # Calculate Kelly fraction
        kelly = self.kelly.calculate_kelly(
            asset.win_rate,
            asset.avg_win,
            asset.avg_loss
        )
        
        # Apply fraction (half-Kelly by default)
        fraction = self.config.kelly_fraction
        if method == AllocationMethod.HALF_KELLY:
            fraction = 0.5
        elif method == AllocationMethod.KELLY:
            fraction = 1.0
        
        weight = kelly * fraction
        
        # Apply constraints
        weight = max(self.config.min_position_weight,
                    min(weight, self.config.max_position_weight))
        
        # Apply regime adjustment
        regime_mult = self.config.regime_multipliers.get(
            self._current_regime, 1.0
        )
        weight *= regime_mult
        
        # Calculate position size
        position_size = portfolio_value * weight * self._current_leverage
        
        return position_size
    
    # =========================================================================
    # STATUS AND MONITORING
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current allocation engine status."""
        with self._lock:
            return {
                "current_regime": self._current_regime.value,
                "current_leverage": self._current_leverage,
                "target_volatility": self.config.target_volatility,
                "kelly_fraction": self.config.kelly_fraction,
                "allocation_history_count": len(self._allocation_history),
                "config": {
                    "max_leverage": self.config.max_leverage,
                    "max_position_weight": self.config.max_position_weight,
                    "min_position_weight": self.config.min_position_weight,
                    "regime_multipliers": {
                        k.value: v for k, v in self.config.regime_multipliers.items()
                    }
                }
            }
    
    def register_callback(self, callback: Callable) -> None:
        """Register callback for allocation updates."""
        self._allocation_callbacks.append(callback)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_allocation_engine(
    target_volatility: float = 0.15,
    max_leverage: float = 3.0
) -> DynamicAllocationEngine:
    """Create allocation engine with default settings."""
    config = AllocationConfig(
        target_volatility=target_volatility,
        max_leverage=max_leverage
    )
    return DynamicAllocationEngine(config=config)
