"""
OpenClaw Skill: Portfolio Optimization
=====================================
Direct Python execution for portfolio optimization using Modern Portfolio Theory.

This skill runs locally, using PyPortfolioOpt for optimization.

Usage in OpenClaw:
 Skill: portfolio_optimizer
  Input: assets, target_return, risk_aversion
  Output: optimal weights, efficient frontier, risk/return
"""

import json
from typing import Dict, Any, List
from datetime import datetime

# Try to import quant libraries
try:
    import numpy as np
    import pandas as pd
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False
    import random


def get_price_data(assets: List[str], days: int = 252) -> Dict[str, List[float]]:
    """Get simulated price data for assets."""
    np.random.seed(42)
    
    # Base prices and expected returns
    base_data = {
        "BTC": {"price": 50000, "mu": 0.001, "sigma": 0.03},
        "ETH": {"price": 3000, "mu": 0.0008, "sigma": 0.035},
        "SOL": {"price": 100, "mu": 0.0012, "sigma": 0.045},
        "SPY": {"price": 450, "mu": 0.0004, "sigma": 0.01},
        "GLD": {"price": 180, "mu": 0.0002, "sigma": 0.008},
    }
    
    prices = {}
    for asset in assets:
        if asset in base_data:
            data = base_data[asset]
        else:
            # Default for unknown assets
            data = {"price": 100, "mu": 0.0005, "sigma": 0.02}
        
        # Generate price paths
        returns = np.random.normal(data["mu"], data["sigma"], days)
        price_path = [data["price"]]
        for r in returns:
            price_path.append(price_path[-1] * (1 + r))
        prices[asset] = price_path
    
    return prices


def optimize_portfolio(
    assets: List[str],
    objective: str = "max_sharpe",
    target_return: float = None,
    risk_aversion: float = 1.0
) -> Dict[str, Any]:
    """
    Optimize portfolio using Modern Portfolio Theory.
    
    Args:
        assets: List of asset symbols
        objective: Optimization objective (max_sharpe, min_volatility, max_return)
        target_return: Target expected return (for efficient frontier)
        risk_aversion: Risk aversion parameter (higher = more conservative)
        
    Returns:
        Dict with optimal weights and portfolio metrics
    """
    if not HAS_PYPFOPT:
        # Fallback random weights
        n = len(assets)
        weights = [1.0/n + random.uniform(-0.05, 0.05) for _ in range(n)]
        weights = [w / sum(weights) for w in weights]
        
        return {
            "assets": assets,
            "objective": objective,
            "weights": {a: float(w) for a, w in zip(assets, weights)},
            "error": "PyPortfolioOpt not available",
            "expected_return": 0.08,
            "volatility": 0.15,
            "sharpe_ratio": 0.53
        }
    
    try:
        # Get price data
        prices_dict = get_price_data(assets)
        
        # Create DataFrame
        df = pd.DataFrame(prices_dict)
        
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        
        # Optimize
        ef = EfficientFrontier(mu, S)
        
        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=0.02)
        elif objective == "min_volatility":
            ef.min_volatility()
        elif objective == "max_return":
            if target_return:
                ef.efficient_return(target_return=target_return)
            else:
                ef.max_return()
        elif objective == "risk_parity":
            # Simple risk parity approximation
            vol = np.sqrt(np.diag(S))
            weights = 1 / vol
            weights = weights / weights.sum()
            ef._weights = weights
        else:
            ef.max_sharpe(risk_free_rate=0.02)
        
        # Get clean weights
        weights = dict(ef.clean_weights())
        
        # Portfolio metrics
        portfolio_return = sum(weights[i] * mu[i] for i in assets)
        portfolio_vol = np.sqrt(sum(
            weights[i] * weights[j] * S.loc[i, j] 
            for i in assets for j in assets
        ))
        
        # Sharpe ratio (annualized)
        sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate efficient frontier points
        n_points = 10
        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        frontier_returns = []
        frontier_vols = []
        
        for target in target_returns:
            try:
                ef_temp = EfficientFrontier(mu, S)
                ef_temp.efficient_return(target)
                w = dict(ef_temp.clean_weights())
                ret = sum(w[i] * mu[i] for i in assets)
                vol = np.sqrt(sum(
                    w[i] * w[j] * S.loc[i, j] for i in assets for j in assets
                ))
                frontier_returns.append(float(ret))
                frontier_vols.append(float(vol))
            except:
                pass
        
        return {
            "metadata": {
                "assets": assets,
                "objective": objective,
                "optimized_at": datetime.utcnow().isoformat()
            },
            "weights": weights,
            "portfolio_metrics": {
                "expected_return_annual": float(portfolio_return * 252),
                "volatility_annual": float(portfolio_vol * np.sqrt(252)),
                "sharpe_ratio": float(sharpe),
                "risk_free_rate": 0.02
            },
            "asset_allocation": {
                asset: {
                    "weight": weights.get(asset, 0),
                    "expected_return": float(mu.get(asset, 0) * 252),
                    "volatility": float(np.sqrt(S.loc[asset, asset]) * np.sqrt(252)) if asset in S.index else 0
                }
                for asset in assets
            },
            "efficient_frontier": {
                "returns": frontier_returns,
                "volatilities": frontier_vols
            }
        }
        
    except Exception as e:
        return {
            "assets": assets,
            "error": str(e),
            "recommendation": "Check asset data and try different objective"
        }


def format_output(result: Dict[str, Any]) -> str:
    """Format portfolio optimization results for chat."""
    if "error" in result and "weights" not in result:
        return f"❌ Error: {result['error']}"
    
    meta = result["metadata"]
    metrics = result["portfolio_metrics"]
    weights = result.get("weights", {})
    allocation = result.get("asset_allocation", {})
    
    output = f"""
💼 **Portfolio Optimization: {meta['objective'].replace('_', ' ').title()}**

**Assets:** {', '.join(meta['assets'])}

**Optimal Weights:**
"""
    
    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for asset, weight in sorted_weights:
        if weight > 0.001:
            bar = "█" * int(weight * 40)
            output += f"- {asset}: {weight*100:5.1f}% {bar}\n"
    
    output += f"""
**Portfolio Metrics:**
- Expected Return (Annual): {metrics['expected_return_annual']*100:.2f}%
- Volatility (Annual): {metrics['volatility_annual']*100:.2f}%
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Risk-Free Rate: {metrics['risk_free_rate']*100:.1f}%

**Asset Details:**
"""
    
    for asset, alloc in allocation.items():
        if alloc["weight"] > 0.001:
            output += f"- {asset}: Return {alloc['expected_return']*100:.1f}%, Vol {alloc['volatility']*100:.1f}%\n"
    
    return output


# CLI for testing
if __name__ == "__main__":
    assets = ["BTC", "ETH", "SOL", "SPY", "GLD"]
    result = optimize_portfolio(assets, objective="max_sharpe")
    print(format_output(result))
