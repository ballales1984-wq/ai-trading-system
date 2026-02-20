"""
Dynamic Capital Allocation Engine
=================================
Institutional-grade capital allocation with:
- Kelly Criterion (dynamic)
- Volatility Targeting
- Risk Parity
- Adaptive Leverage Control
- Regime-Based Allocation (HMM)

Author: AI Trading System
Version: 1.0.0
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================

class AllocationMethod(str, Enum):
    """Capital allocation methods."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"
    REGIME_BASED = "regime_based"
    ADAPTIVE = "adaptive"


class MarketRegime(str, Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


@dataclass
class AllocationConfig:
    """Configuration for capital allocation."""
    # Kelly Criterion
    kelly_fraction: float = 0.25          # Use 25% of Kelly (conservative)
    min_kelly: float = 0.0                # Minimum Kelly fraction
    max_kelly: float = 0.50               # Maximum Kelly fraction
    
    # Volatility Targeting
    target_volatility: float = 0.15       # 15% annualized target
    vol_lookback: int = 20                # Days for vol calculation
    max_vol_adjustment: float = 2.0       # Max vol adjustment factor
    
    # Risk Parity
    risk_parity_iterations: int = 100     # Max iterations for optimization
    risk_parity_tolerance: float = 1e-6   # Convergence tolerance
    
    # Leverage Control
    max_leverage: float = 3.0             # Maximum portfolio leverage
    leverage_decay: float = 0.95          # Decay factor for leverage reduction
    
    # Regime-Based
    regime_lookback: int = 60             # Days for regime detection
    regime_confidence_threshold: float = 0.7
    
    # General
    min_position_size: float = 0.01       # Minimum 1% position
    max_position_size: float = 0.20       # Maximum 20% position
    rebalance_threshold: float = 0.05     # 5% drift before rebalance


@dataclass
class AssetMetrics:
    """Metrics for a single asset."""
    symbol: str
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    current_weight: float = 0.0
    target_weight: float = 0.0
    kelly_fraction: float = 0.0
    risk_contribution: float = 0.0
    regime: MarketRegime = MarketRegime.SIDEWAYS


@dataclass
class AllocationResult:
    """Result of capital allocation."""
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

class KellyCriterion:
    """
    Kelly Criterion calculator for position sizing.
    
    Kelly = W - (1-W)/R
    Where:
    - W = Win probability
    - R = Win/Loss ratio
    """
    
    def __init__(self, config: AllocationConfig):
        self.config = config
    
    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            win_rate: Probability of winning
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)
            
        Returns:
            Kelly fraction (0 to max_kelly)
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        
        # Win/Loss ratio
        r = avg_win / avg_loss
        
        # Kelly formula
        kelly = win_rate - (1 - win_rate) / r
        
        # Apply constraints
        kelly = max(self.config.min_kelly, min(kelly, self.config.max_kelly))
        
        # Apply fractional Kelly
        kelly *= self.config.kelly_fraction
        
        return kelly
    
    def calculate_from_returns(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Kelly from return series.
        
        Uses the formula: Kelly = μ / σ²
        """
        if len(returns) < 10:
            return 0.0
        
        mu = np.mean(returns)
        sigma_sq = np.var(returns)
        
        if sigma_sq <= 0:
            return 0.0
        
        # Kelly = μ / σ²
        kelly = mu / sigma_sq
        
        # Annualize if daily returns
        kelly *= 252  # Annualization factor
        
        # Apply constraints
        kelly = max(self.config.min_kelly, min(kelly, self.config.max_kelly))
        kelly *= self.config.kelly_fraction
        
        return kelly
    
    def calculate_optimal_leverage(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate optimal leverage using Kelly.
        
        Leverage = (μ - r) / σ²
        """
        if volatility <= 0:
            return 1.0
        
        excess_return = expected_return - risk_free_rate
        leverage = excess_return / (volatility ** 2)
        
        # Apply constraints
        leverage = max(0.0, min(leverage, self.config.max_leverage))
        
        return leverage


# =============================================================================
# VOLATILITY TARGETING
# =============================================================================

class VolatilityTargeting:
    """
    Volatility targeting for position sizing.
    
    Adjusts positions to achieve target portfolio volatility.
    """
    
    def __init__(self, config: AllocationConfig):
        self.config = config
        self._vol_history: Dict[str, deque] = {}
    
    def calculate_volatility(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Calculate realized volatility."""
        if len(returns) < 5:
            return 0.0
        
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize daily volatility
        
        return vol
    
    def calculate_vol_adjustment(
        self,
        current_vol: float,
        target_vol: Optional[float] = None
    ) -> float:
        """
        Calculate volatility adjustment factor.
        
        Factor = Target Vol / Current Vol
        """
        target = target_vol or self.config.target_volatility
        
        if current_vol <= 0:
            return 1.0
        
        factor = target / current_vol
        
        # Apply constraints
        factor = max(
            1.0 / self.config.max_vol_adjustment,
            min(factor, self.config.max_vol_adjustment)
        )
        
        return factor
    
    def adjust_weights_for_vol(
        self,
        weights: Dict[str, float],
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust weights based on individual volatilities."""
        adjusted = {}
        
        for symbol, weight in weights.items():
            vol = volatilities.get(symbol, self.config.target_volatility)
            factor = self.calculate_vol_adjustment(vol)
            adjusted[symbol] = weight * factor
        
        # Normalize to sum to 1
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted


# =============================================================================
# RISK PARITY
# =============================================================================

class RiskParity:
    """
    Risk Parity allocation.
    
    Allocates capital so each asset contributes equally to portfolio risk.
    """
    
    def __init__(self, config: AllocationConfig):
        self.config = config
    
    def calculate_risk_parity_weights(
        self,
        cov_matrix: np.ndarray,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights using iterative optimization.
        
        Each asset contributes equally to total portfolio risk.
        """
        n = len(symbols)
        if n == 0 or cov_matrix.shape[0] != n:
            return {s: 1.0 / n for s in symbols}
        
        # Initialize with equal weights
        weights = np.ones(n) / n
        
        # Iterative optimization
        for _ in range(self.config.risk_parity_iterations):
            # Calculate marginal risk contribution
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            if portfolio_vol < 1e-10:
                break
            
            marginal_risk = cov_matrix @ weights / portfolio_vol
            
            # Risk contribution of each asset
            risk_contrib = weights * marginal_risk
            
            # Target: equal risk contribution
            target_risk = portfolio_vol / n
            
            # Adjust weights
            adjustment = target_risk / (risk_contrib + 1e-10)
            weights = weights * np.sqrt(adjustment)
            
            # Normalize
            weights = weights / weights.sum()
            
            # Check convergence
            current_risk_contrib = weights * (cov_matrix @ weights) / portfolio_vol
            if np.std(current_risk_contrib) < self.config.risk_parity_tolerance:
                break
        
        # Convert to dictionary
        return {symbols[i]: weights[i] for i in range(n)}
    
    def calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if portfolio_vol < 1e-10:
            return np.zeros_like(weights)
        
        marginal_risk = cov_matrix @ weights / portfolio_vol
        risk_contrib = weights * marginal_risk
        
        # Normalize to percentages
        return risk_contrib / portfolio_vol


# =============================================================================
# REGIME-BASED ALLOCATION
# =============================================================================

class RegimeBasedAllocation:
    """
    Regime-based capital allocation.
    
    Adjusts allocation based on detected market regime.
    """
    
    # Regime-specific allocation adjustments
    REGIME_MULTIPLIERS = {
        MarketRegime.BULL: {
            "risk_on": 1.2,
            "leverage": 1.1,
            "equity_bias": 1.2
        },
        MarketRegime.BEAR: {
            "risk_on": 0.6,
            "leverage": 0.7,
            "equity_bias": 0.5
        },
        MarketRegime.SIDEWAYS: {
            "risk_on": 0.8,
            "leverage": 0.9,
            "equity_bias": 1.0
        },
        MarketRegime.HIGH_VOLATILITY: {
            "risk_on": 0.5,
            "leverage": 0.5,
            "equity_bias": 0.7
        },
        MarketRegime.LOW_VOLATILITY: {
            "risk_on": 1.1,
            "leverage": 1.2,
            "equity_bias": 1.1
        },
        MarketRegime.CRISIS: {
            "risk_on": 0.2,
            "leverage": 0.3,
            "equity_bias": 0.3
        }
    }
    
    def __init__(self, config: AllocationConfig):
        self.config = config
    
    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: float,
        trend: float
    ) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime.
        
        Returns:
            Tuple of (regime, confidence)
        """
        if len(returns) < self.config.regime_lookback:
            return MarketRegime.SIDEWAYS, 0.5
        
        # Calculate metrics
        avg_return = np.mean(returns)
        vol = volatility if volatility > 0 else np.std(returns) * np.sqrt(252)
        
        # Crisis detection
        recent_drawdown = self._calculate_drawdown(returns)
        if recent_drawdown > 0.20 or vol > 0.50:
            return MarketRegime.CRISIS, 0.8
        
        # High volatility
        if vol > 0.30:
            return MarketRegime.HIGH_VOLATILITY, 0.7
        
        # Low volatility
        if vol < 0.10:
            return MarketRegime.LOW_VOLATILITY, 0.6
        
        # Trend-based
        if trend > 0.15:
            return MarketRegime.BULL, 0.7
        elif trend < -0.15:
            return MarketRegime.BEAR, 0.7
        
        return MarketRegime.SIDEWAYS, 0.5
    
    def _calculate_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)
    
    def adjust_for_regime(
        self,
        weights: Dict[str, float],
        regime: MarketRegime,
        leverage: float
    ) -> Tuple[Dict[str, float], float]:
        """
        Adjust weights and leverage for regime.
        
        Returns:
            Tuple of (adjusted_weights, adjusted_leverage)
        """
        multipliers = self.REGIME_MULTIPLIERS.get(regime, {})
        
        risk_mult = multipliers.get("risk_on", 1.0)
        leverage_mult = multipliers.get("leverage", 1.0)
        
        # Adjust weights
        adjusted = {k: v * risk_mult for k, v in weights.items()}
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        # Adjust leverage
        adjusted_leverage = leverage * leverage_mult
        adjusted_leverage = min(adjusted_leverage, self.config.max_leverage)
        
        return adjusted, adjusted_leverage


# =============================================================================
# DYNAMIC CAPITAL ALLOCATION ENGINE
# =============================================================================

class DynamicCapitalAllocationEngine:
    """
    Main engine for dynamic capital allocation.
    
    Combines multiple allocation methods:
    - Kelly Criterion for position sizing
    - Volatility Targeting for risk adjustment
    - Risk Parity for diversification
    - Regime-Based for market adaptation
    """
    
    def __init__(
        self,
        config: Optional[AllocationConfig] = None,
        initial_capital: float = 100000.0
    ):
        self.config = config or AllocationConfig()
        self.initial_capital = initial_capital
        
        # Initialize sub-components
        self.kelly = KellyCriterion(self.config)
        self.vol_targeting = VolatilityTargeting(self.config)
        self.risk_parity = RiskParity(self.config)
        self.regime_allocator = RegimeBasedAllocation(self.config)
        
        # State
        self.current_weights: Dict[str, float] = {}
        self.current_leverage: float = 1.0
        self.current_regime: MarketRegime = MarketRegime.SIDEWAYS
        self.asset_metrics: Dict[str, AssetMetrics] = {}
        
        # History
        self._allocation_history: List[AllocationResult] = []
    
    def calculate_allocation(
        self,
        symbols: List[str],
        returns_data: Dict[str, np.ndarray],
        method: AllocationMethod = AllocationMethod.ADAPTIVE,
        current_positions: Optional[Dict[str, float]] = None
    ) -> AllocationResult:
        """
        Calculate optimal capital allocation.
        
        Args:
            symbols: List of symbols to allocate
            returns_data: Historical returns for each symbol
            method: Allocation method to use
            current_positions: Current position weights
            
        Returns:
            AllocationResult with optimal weights
        """
        if not symbols:
            return AllocationResult(
                weights={},
                method=method,
                expected_return=0.0,
                expected_volatility=0.0,
                expected_sharpe=0.0,
                leverage=1.0,
                regime=self.current_regime
            )
        
        # Calculate asset metrics
        self._update_asset_metrics(symbols, returns_data)
        
        # Detect regime
        self._detect_market_regime(returns_data)
        
        # Calculate weights based on method
        if method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation(symbols)
        elif method == AllocationMethod.KELLY_CRITERION:
            weights = self._kelly_allocation(symbols)
        elif method == AllocationMethod.VOLATILITY_TARGETING:
            weights = self._volatility_targeted_allocation(symbols)
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation(symbols, returns_data)
        elif method == AllocationMethod.REGIME_BASED:
            weights = self._regime_based_allocation(symbols)
        else:  # ADAPTIVE
            weights = self._adaptive_allocation(symbols, returns_data)
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        # Calculate portfolio metrics
        expected_return, expected_vol, sharpe = self._calculate_portfolio_metrics(
            weights, returns_data
        )
        
        # Calculate optimal leverage
        leverage = self.kelly.calculate_optimal_leverage(
            expected_return, expected_vol
        )
        
        # Adjust for regime
        weights, leverage = self.regime_allocator.adjust_for_regime(
            weights, self.current_regime, leverage
        )
        
        # Create result
        result = AllocationResult(
            weights=weights,
            method=method,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            expected_sharpe=sharpe,
            leverage=leverage,
            regime=self.current_regime,
            metadata={
                "num_assets": len(symbols),
                "regime_confidence": getattr(self, '_regime_confidence', 0.5)
            }
        )
        
        # Store
        self.current_weights = weights
        self.current_leverage = leverage
        self._allocation_history.append(result)
        
        return result
    
    def _update_asset_metrics(
        self,
        symbols: List[str],
        returns_data: Dict[str, np.ndarray]
    ) -> None:
        """Update metrics for all assets."""
        for symbol in symbols:
            returns = returns_data.get(symbol, np.array([]))
            
            if len(returns) < 5:
                continue
            
            metrics = AssetMetrics(symbol=symbol)
            
            # Basic stats
            metrics.expected_return = np.mean(returns) * 252  # Annualized
            metrics.volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming 0% risk-free)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.expected_return / metrics.volatility
            
            # Win rate
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            metrics.win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.5
            metrics.avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            metrics.avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
            
            # Kelly fraction
            metrics.kelly_fraction = self.kelly.calculate_kelly(
                metrics.win_rate,
                metrics.avg_win,
                metrics.avg_loss
            )
            
            self.asset_metrics[symbol] = metrics
    
    def _detect_market_regime(self, returns_data: Dict[str, np.ndarray]) -> None:
        """Detect overall market regime."""
        # Use first symbol as market proxy (or average)
        all_returns = []
        for returns in returns_data.values():
            if len(returns) > 0:
                all_returns.append(returns)
        
        if not all_returns:
            self.current_regime = MarketRegime.SIDEWAYS
            return
        
        # Average returns across assets
        min_len = min(len(r) for r in all_returns)
        avg_returns = np.mean([r[-min_len:] for r in all_returns], axis=0)
        
        # Calculate trend
        cumulative = np.cumprod(1 + avg_returns)
        trend = (cumulative[-1] / cumulative[0]) - 1 if len(cumulative) > 0 else 0
        
        # Calculate volatility
        vol = np.std(avg_returns) * np.sqrt(252)
        
        # Detect regime
        self.current_regime, self._regime_confidence = self.regime_allocator.detect_regime(
            avg_returns, vol, trend
        )
    
    def _equal_weight_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Simple equal weight allocation."""
        weight = 1.0 / len(symbols)
        return {s: weight for s in symbols}
    
    def _kelly_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Kelly Criterion based allocation."""
        weights = {}
        
        for symbol in symbols:
            metrics = self.asset_metrics.get(symbol)
            if metrics:
                weights[symbol] = metrics.kelly_fraction
            else:
                weights[symbol] = self.config.min_position_size
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _volatility_targeted_allocation(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """Volatility targeted allocation."""
        weights = {}
        volatilities = {}
        
        # Start with equal weights
        for symbol in symbols:
            metrics = self.asset_metrics.get(symbol)
            volatilities[symbol] = metrics.volatility if metrics else self.config.target_volatility
            weights[symbol] = 1.0 / len(symbols)
        
        # Adjust for volatility
        return self.vol_targeting.adjust_weights_for_vol(weights, volatilities)
    
    def _risk_parity_allocation(
        self,
        symbols: List[str],
        returns_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Risk parity allocation."""
        # Build covariance matrix
        min_len = min(len(r) for r in returns_data.values() if len(r) > 0)
        
        if min_len < 5:
            return self._equal_weight_allocation(symbols)
        
        returns_matrix = np.column_stack([
            returns_data[s][-min_len:] for s in symbols if s in returns_data
        ])
        
        cov_matrix = np.cov(returns_matrix.T)
        
        return self.risk_parity.calculate_risk_parity_weights(cov_matrix, symbols)
    
    def _regime_based_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Regime-based allocation."""
        # Start with equal weights
        weights = self._equal_weight_allocation(symbols)
        
        # Adjust for regime
        weights, _ = self.regime_allocator.adjust_for_regime(
            weights, self.current_regime, 1.0
        )
        
        return weights
    
    def _adaptive_allocation(
        self,
        symbols: List[str],
        returns_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Adaptive allocation combining multiple methods.
        
        Weights methods based on recent performance.
        """
        # Get allocations from each method
        kelly_weights = self._kelly_allocation(symbols)
        vol_weights = self._volatility_targeted_allocation(symbols)
        rp_weights = self._risk_parity_allocation(symbols, returns_data)
        
        # Combine with regime-based weights
        regime_mult = RegimeBasedAllocation.REGIME_MULTIPLIERS.get(
            self.current_regime, {}
        ).get("risk_on", 1.0)
        
        # Adaptive weighting based on regime
        if self.current_regime == MarketRegime.CRISIS:
            # Favor risk parity in crisis
            weights = {s: rp_weights.get(s, 0) * 0.6 + vol_weights.get(s, 0) * 0.4
                      for s in symbols}
        elif self.current_regime == MarketRegime.BULL:
            # Favor Kelly in bull market
            weights = {s: kelly_weights.get(s, 0) * 0.5 + rp_weights.get(s, 0) * 0.3 + 
                      vol_weights.get(s, 0) * 0.2 for s in symbols}
        else:
            # Balanced approach
            weights = {s: kelly_weights.get(s, 0) * 0.3 + 
                      rp_weights.get(s, 0) * 0.4 + 
                      vol_weights.get(s, 0) * 0.3 for s in symbols}
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply position size constraints."""
        constrained = {}
        
        for symbol, weight in weights.items():
            # Apply min/max
            weight = max(self.config.min_position_size, 
                        min(weight, self.config.max_position_size))
            constrained[symbol] = weight
        
        # Normalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {k: v / total for k, v in constrained.items()}
        
        return constrained
    
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns_data: Dict[str, np.ndarray]
    ) -> Tuple[float, float, float]:
        """Calculate expected portfolio metrics."""
        if not weights or not returns_data:
            return 0.0, 0.0, 0.0
        
        # Weighted return
        expected_return = 0.0
        for symbol, weight in weights.items():
            metrics = self.asset_metrics.get(symbol)
            if metrics:
                expected_return += weight * metrics.expected_return
        
        # Portfolio volatility (simplified - assumes some correlation)
        variance = 0.0
        for symbol, weight in weights.items():
            metrics = self.asset_metrics.get(symbol)
            if metrics:
                variance += (weight ** 2) * (metrics.volatility ** 2)
        
        # Add correlation effect (assume 0.5 average correlation)
        n = len(weights)
        if n > 1:
            avg_correlation = 0.5
            for s1, w1 in weights.items():
                for s2, w2 in weights.items():
                    if s1 != s2:
                        m1 = self.asset_metrics.get(s1)
                        m2 = self.asset_metrics.get(s2)
                        if m1 and m2:
                            variance += w1 * w2 * m1.volatility * m2.volatility * avg_correlation / (n - 1)
        
        volatility = np.sqrt(variance)
        
        # Sharpe ratio
        sharpe = expected_return / volatility if volatility > 0 else 0.0
        
        return expected_return, volatility, sharpe
    
    def get_position_sizes(
        self,
        capital: float,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Convert weights to position sizes.
        
        Args:
            capital: Available capital
            prices: Current prices for each symbol
            
        Returns:
            Dictionary of position sizes (in units)
        """
        positions = {}
        
        for symbol, weight in self.current_weights.items():
            price = prices.get(symbol, 0)
            if price > 0:
                position_value = capital * weight * self.current_leverage
                positions[symbol] = position_value / price
        
        return positions
    
    def should_rebalance(
        self,
        current_weights: Dict[str, float]
    ) -> bool:
        """Check if rebalancing is needed."""
        if not self.current_weights or not current_weights:
            return False
        
        # Check drift
        for symbol in self.current_weights:
            target = self.current_weights.get(symbol, 0)
            current = current_weights.get(symbol, 0)
            
            drift = abs(target - current)
            if drift > self.config.rebalance_threshold:
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current allocation status."""
        return {
            "current_weights": self.current_weights,
            "current_leverage": self.current_leverage,
            "current_regime": self.current_regime.value,
            "num_assets": len(self.current_weights),
            "allocation_history_count": len(self._allocation_history)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_allocation_engine(
    target_volatility: float = 0.15,
    max_leverage: float = 3.0,
    kelly_fraction: float = 0.25
) -> DynamicCapitalAllocationEngine:
    """Create allocation engine with common settings."""
    config = AllocationConfig(
        target_volatility=target_volatility,
        max_leverage=max_leverage,
        kelly_fraction=kelly_fraction
    )
    return DynamicCapitalAllocationEngine(config=config)
