"""
Multi-Skill Composition Module
==============================
Provides high-level composed strategies that combine multiple quant skills.

This module implements Level 4 orchestration - combining multiple skills
into sophisticated trading strategies that leverage HMM regime detection,
GARCH volatility forecasting, Monte Carlo simulation, and portfolio optimization.

Usage:
    from composed_strategies import (
        regime_aware_mc_portfolio,
        full_risk_analysis,
        smart_portfolio_optimization
    )
    
    # Regime-aware Monte Carlo portfolio
    result = regime_aware_mc_portfolio(
        symbol="BTC",
        initial_price=50000,
        assets=["BTC", "ETH", "SOL"]
    )
    
    # Full risk analysis pipeline
    risk_report = full_risk_analysis(
        symbols=["BTC", "ETH"],
        portfolio_value=100000
    )
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import the intent router to execute individual skills
# Note: These imports will work once the intent_router is properly configured
# For now, we use a lazy import pattern to avoid circular dependencies


# =============================================================================
# Strategy Parameters and Results
# =============================================================================

@dataclass
class RegimeAwareParams:
    """Parameters for regime-aware Monte Carlo portfolio strategy."""
    symbol: str
    initial_price: float
    assets: List[str]
    n_paths: int = 5000
    days_ahead: int = 30
    confidence_level: float = 0.95


@dataclass
class RiskAnalysisParams:
    """Parameters for full risk analysis pipeline."""
    symbols: List[str]
    portfolio_value: float
    positions: Optional[Dict[str, float]] = None  # symbol -> value
    lookback_days: int = 90
    mc_simulations: int = 10000


@dataclass
class PortfolioOptimizationParams:
    """Parameters for smart portfolio optimization."""
    assets: List[str]
    objective: str = "max_sharpe"  # max_sharpe, min_volatility, max_return, risk_parity
    risk_free_rate: float = 0.02
    include_regime_adjustment: bool = True
    rebalance_threshold: float = 0.05


# =============================================================================
# Lazy Import Functions (to avoid circular dependencies)
# =============================================================================

def _import_intent_router():
    """Lazy import of intent_router to avoid circular imports."""
    try:
        from intent_router import route_intent
        return route_intent
    except ImportError:
        # Fallback: try importing from the package
        from .intent_router import route_intent
        return route_intent


# =============================================================================
# Regime-Aware Monte Carlo Portfolio Strategy
# =============================================================================

def regime_aware_mc_portfolio(
    symbol: str,
    initial_price: float,
    assets: List[str],
    n_paths: int = 5000,
    days_ahead: int = 30,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Execute a regime-aware Monte Carlo portfolio strategy.
    
    This Level 4 strategy combines:
    1. HMM regime detection to identify market state (bull/bear/sideways)
    2. Monte Carlo simulation with regime-adjusted parameters
    3. Portfolio optimization based on the risk profile
    
    Args:
        symbol: Primary trading symbol to analyze
        initial_price: Current price of the primary asset
        assets: List of assets to include in the portfolio
        n_paths: Number of Monte Carlo simulation paths
        days_ahead: Forecast horizon in days
        confidence_level: Confidence level for VaR calculations
        
    Returns:
        Dictionary containing:
        - regime: HMM regime detection results
        - monte_carlo: Monte Carlo simulation results
        - optimizer: Portfolio optimization results
        - metadata: Strategy execution metadata
    """
    route_intent = _import_intent_router()
    
    execution_start = datetime.now()
    
    # Step 1: Detect market regime
    hmm_result = route_intent("regime_analysis", {"symbol": symbol})
    regime = hmm_result.get("current_state", "sideways")
    regime_confidence = hmm_result.get("confidence", 0.0)
    
    # Step 2: Adjust parameters based on regime
    if regime == "bear":
        volatility = 0.04
        expected_return = -0.0005
        risk_level = "high"
    elif regime == "bull":
        volatility = 0.025
        expected_return = 0.0015
        risk_level = "low"
    else:  # sideways
        volatility = 0.03
        expected_return = 0.0005
        risk_level = "medium"
    
    # Step 3: Run Monte Carlo simulation with regime-adjusted parameters
    mc_result = route_intent(
        "simulate_paths",
        {
            "initial_price": initial_price,
            "expected_return": expected_return,
            "volatility": volatility,
            "n_paths": n_paths,
            "days_ahead": days_ahead,
        }
    )
    
    # Step 4: Optimize portfolio
    opt_result = route_intent(
        "portfolio_optimization",
        {
            "assets": assets,
            "objective": "max_sharpe",
        }
    )
    
    execution_time = (datetime.now() - execution_start).total_seconds()
    
    return {
        "regime": {
            "current_state": regime,
            "confidence": regime_confidence,
            "volatility": hmm_result.get("volatility", volatility),
            "transition_probabilities": hmm_result.get("transition_matrix", {}),
        },
        "monte_carlo": {
            "percentiles": mc_result.get("percentiles", {}),
            "var": mc_result.get("var", {}),
            "cvar": mc_result.get("cvar", {}),
            "probability_profit": mc_result.get("probability_profit", 0.0),
            "expected_value": mc_result.get("expected_value", initial_price),
        },
        "optimizer": {
            "weights": opt_result.get("weights", {}),
            "expected_return": opt_result.get("expected_return", 0.0),
            "volatility": opt_result.get("volatility", 0.0),
            "sharpe_ratio": opt_result.get("sharpe_ratio", 0.0),
        },
        "metadata": {
            "risk_level": risk_level,
            "regime_adjusted_volatility": volatility,
            "regime_adjusted_return": expected_return,
            "execution_time_seconds": execution_time,
            "n_paths": n_paths,
            "days_ahead": days_ahead,
            "confidence_level": confidence_level,
        }
    }


# =============================================================================
# Full Risk Analysis Pipeline
# =============================================================================

def full_risk_analysis(
    symbols: List[str],
    portfolio_value: float,
    positions: Optional[Dict[str, float]] = None,
    lookback_days: int = 90,
    mc_simulations: int = 10000
) -> Dict[str, Any]:
    """
    Execute a comprehensive risk analysis combining multiple skills.
    
    This Level 4 pipeline combines:
    1. HMM regime detection for each symbol
    2. GARCH volatility forecasting
    3. Monte Carlo simulation for VaR/CVaR
    4. Portfolio-level risk metrics
    
    Args:
        symbols: List of symbols to analyze
        portfolio_value: Total portfolio value in USD
        positions: Optional dict of symbol -> position value
        lookback_days: Historical data lookback period
        mc_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dictionary containing comprehensive risk analysis
    """
    route_intent = _import_intent_router()
    
    execution_start = datetime.now()
    
    # Initialize results
    regime_results = {}
    volatility_results = {}
    mc_results = {}
    
    # Process each symbol
    for symbol in symbols:
        # Get regime detection
        try:
            regime_results[symbol] = route_intent(
                "regime_analysis",
                {"symbol": symbol, "lookback_days": lookback_days}
            )
        except Exception as e:
            regime_results[symbol] = {"error": str(e)}
        
        # Get GARCH volatility
        try:
            volatility_results[symbol] = route_intent(
                "volatility_analysis",
                {"symbol": symbol, "forecast_horizon": 5}
            )
        except Exception as e:
            volatility_results[symbol] = {"error": str(e)}
    
    # Get overall portfolio Monte Carlo if we have positions
    if positions:
        total_value = sum(positions.values())
        avg_volatility = sum(
            v.get("volatility", 0.03) 
            for v in volatility_results.values() 
            if "volatility" in v
        ) / len(positions) if positions else 0.03
        
        try:
            mc_results["portfolio"] = route_intent(
                "simulate_paths",
                {
                    "initial_price": total_value,
                    "expected_return": 0.0005,
                    "volatility": avg_volatility,
                    "n_paths": mc_simulations,
                    "days_ahead": 1,
                }
            )
        except Exception as e:
            mc_results["portfolio"] = {"error": str(e)}
    
    execution_time = (datetime.now() - execution_start).total_seconds()
    
    # Calculate aggregate risk metrics
    portfolio_var = mc_results.get("portfolio", {}).get("var", {}).get("95%", 0)
    portfolio_cvar = mc_results.get("portfolio", {}).get("cvar", {}).get("95%", 0)
    
    return {
        "regimes": regime_results,
        "volatilities": volatility_results,
        "monte_carlo": mc_results,
        "risk_metrics": {
            "portfolio_value": portfolio_value,
            "value_at_risk_95": portfolio_var,
            "conditional_var_95": portfolio_cvar,
            "var_percentage": (portfolio_var / portfolio_value * 100) if portfolio_value > 0 else 0,
            "cvar_percentage": (portfolio_cvar / portfolio_value * 100) if portfolio_value > 0 else 0,
        },
        "metadata": {
            "symbols_analyzed": symbols,
            "lookback_days": lookback_days,
            "mc_simulations": mc_simulations,
            "execution_time_seconds": execution_time,
            "timestamp": execution_start.isoformat(),
        }
    }


# =============================================================================
# Smart Portfolio Optimization
# =============================================================================

def smart_portfolio_optimization(
    assets: List[str],
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    include_regime_adjustment: bool = True,
    rebalance_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Execute smart portfolio optimization with regime awareness.
    
    This Level 4 strategy:
    1. Optionally detects regime for each asset
    2. Adjusts optimization parameters based on regime
    3. Runs portfolio optimization
    4. Provides rebalancing recommendations
    
    Args:
        assets: List of asset symbols
        objective: Optimization objective
        risk_free_rate: Risk-free rate for Sharpe calculation
        include_regime_adjustment: Whether to incorporate regime analysis
        rebalance_threshold: Threshold for triggering rebalance
        
    Returns:
        Dictionary containing optimization results and recommendations
    """
    route_intent = _import_intent_router()
    
    execution_start = datetime.now()
    
    regime_adjustments = {}
    
    # Step 1: Optional regime detection
    if include_regime_adjustment:
        for asset in assets:
            try:
                regime_result = route_intent(
                    "regime_analysis",
                    {"symbol": asset}
                )
                regime = regime_result.get("current_state", "sideways")
                regime_adjustments[asset] = {
                    "regime": regime,
                    "confidence": regime_result.get("confidence", 0.0),
                }
            except Exception:
                regime_adjustments[asset] = {"regime": "unknown", "confidence": 0.0}
    
    # Step 2: Run portfolio optimization
    opt_result = route_intent(
        "portfolio_optimization",
        {
            "assets": assets,
            "objective": objective,
            "risk_free_rate": risk_free_rate,
        }
    )
    
    # Step 3: Generate recommendations
    recommendations = _generate_recommendations(
        opt_result,
        regime_adjustments,
        rebalance_threshold
    )
    
    execution_time = (datetime.now() - execution_start).total_seconds()
    
    return {
        "optimization": {
            "weights": opt_result.get("weights", {}),
            "expected_return": opt_result.get("expected_return", 0.0),
            "volatility": opt_result.get("volatility", 0.0),
            "sharpe_ratio": opt_result.get("sharpe_ratio", 0.0),
        },
        "regimes": regime_adjustments if include_regime_adjustment else {},
        "recommendations": recommendations,
        "metadata": {
            "objective": objective,
            "risk_free_rate": risk_free_rate,
            "include_regime_adjustment": include_regime_adjustment,
            "execution_time_seconds": execution_time,
        }
    }


def _generate_recommendations(
    optimization_result: Dict[str, Any],
    regime_adjustments: Dict[str, Any],
    threshold: float
) -> List[Dict[str, Any]]:
    """Generate portfolio rebalancing recommendations based on regime and optimization."""
    recommendations = []
    
    weights = optimization_result.get("weights", {})
    
    for asset, weight in weights.items():
        recommendation = {
            "asset": asset,
            "current_weight": weight,
            "action": "hold",
            "reason": "",
        }
        
        # Check regime-based adjustments
        if asset in regime_adjustments:
            regime = regime_adjustments[asset].get("regime", "unknown")
            
            if regime == "bear":
                recommendation["reason"] = f"Regime is bearish - consider reducing exposure"
                if weight > 0.3:
                    recommendation["action"] = "reduce"
            elif regime == "bull" and weight < 0.2:
                recommendation["reason"] = "Regime is bullish - consider increasing exposure"
                recommendation["action"] = "increase"
        
        # Check threshold
        if weight > (1.0 / len(weights)) * (1 + threshold):
            recommendation["action"] = "reduce"
            recommendation["reason"] = f"Weight {weight:.1%} exceeds target by threshold"
        elif weight < (1.0 / len(weights)) * (1 - threshold):
            recommendation["action"] = "increase"
            recommendation["reason"] = f"Weight {weight:.1%} below target by threshold"
        
        recommendations.append(recommendation)
    
    return recommendations


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_compositions() -> List[str]:
    """
    Get list of available composed strategies.
    
    Returns:
        List of available composition function names
    """
    return [
        "regime_aware_mc_portfolio",
        "full_risk_analysis",
        "smart_portfolio_optimization",
    ]


def describe_composition(name: str) -> Dict[str, Any]:
    """
    Get description and parameters for a composed strategy.
    
    Args:
        name: Name of the composition function
        
    Returns:
        Dictionary with description and parameter specs
    """
    descriptions = {
        "regime_aware_mc_portfolio": {
            "description": "Combines HMM regime detection, Monte Carlo simulation, "
                          "and portfolio optimization with regime-adjusted parameters",
            "parameters": {
                "symbol": "Primary trading symbol",
                "initial_price": "Current price of primary asset",
                "assets": "List of assets for portfolio",
                "n_paths": "Number of Monte Carlo paths (default: 5000)",
                "days_ahead": "Forecast horizon in days (default: 30)",
            }
        },
        "full_risk_analysis": {
            "description": "Comprehensive risk analysis combining regime detection, "
                          "GARCH volatility, and Monte Carlo VaR",
            "parameters": {
                "symbols": "List of symbols to analyze",
                "portfolio_value": "Total portfolio value",
                "positions": "Optional dict of symbol -> position value",
                "lookback_days": "Historical lookback (default: 90)",
                "mc_simulations": "Monte Carlo simulations (default: 10000)",
            }
        },
        "smart_portfolio_optimization": {
            "description": "Portfolio optimization with optional regime-aware adjustments",
            "parameters": {
                "assets": "List of asset symbols",
                "objective": "Optimization objective (default: max_sharpe)",
                "risk_free_rate": "Risk-free rate (default: 0.02)",
                "include_regime_adjustment": "Whether to use regime analysis",
                "rebalance_threshold": "Rebalance threshold (default: 0.05)",
            }
        },
    }
    
    return descriptions.get(name, {"error": "Composition not found"})
