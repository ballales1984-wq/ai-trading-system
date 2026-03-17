"""
OpenClaw Skill: Monte Carlo Simulation
======================================
Direct Python execution for Monte Carlo path generation.

This skill runs locally on the OpenClaw agent's Python environment,
generating thousands of potential price paths for risk analysis.

Usage in OpenClaw:
 Skill: monte_carlo_paths
  Input: symbol, n_paths, days_ahead, initial_price
  Output: path distribution, percentiles, VaR estimates
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Try to import quant libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    import random


def generate_price_paths(
    initial_price: float,
    expected_return: float,
    volatility: float,
    n_paths: int,
    days_ahead: int,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate Monte Carlo price paths using Geometric Brownian Motion.
    
    Args:
        initial_price: Starting price of the asset
        expected_return: Expected daily return (drift)
        volatility: Daily volatility (standard deviation)
        n_paths: Number of simulation paths
        days_ahead: Number of days to simulate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with path statistics and percentiles
    """
    if not HAS_NUMPY:
        # Fallback to random for testing
        random.seed(seed)
        paths = []
        for _ in range(n_paths):
            price = initial_price
            path = [price]
            for _ in range(days_ahead):
                change = random.gauss(expected_return, volatility)
                price = price * (1 + change)
                path.append(price)
            paths.append(path)
        
        final_prices = [p[-1] for p in paths]
        return {
            "initial_price": initial_price,
            "n_paths": n_paths,
            "days_ahead": days_ahead,
            "final_prices": final_prices,
            "mean": sum(final_prices) / len(final_prices),
            "error": "Using random fallback - install numpy for real simulation"
        }
    
    np.random.seed(seed)
    
    # Generate random returns
    dt = 1  # Daily
    drift = expected_return * dt
    diffusion = volatility * np.sqrt(dt)
    
    # Generate paths
    random_returns = np.random.normal(
        drift, 
        diffusion, 
        (days_ahead, n_paths)
    )
    
    # Calculate cumulative returns
    log_returns = np.cumsum(random_returns, axis=0)
    
    # Calculate price paths
    price_paths = initial_price * np.exp(log_returns)
    price_paths = np.vstack([np.full(n_paths, initial_price), price_paths])
    
    # Get final prices
    final_prices = price_paths[-1, :]
    
    # Calculate statistics
    mean_final = np.mean(final_prices)
    std_final = np.std(final_prices)
    min_final = np.min(final_prices)
    max_final = np.max(final_prices)
    
    # Calculate percentiles
    percentiles = {
        "p1": float(np.percentile(final_prices, 1)),
        "p5": float(np.percentile(final_prices, 5)),
        "p10": float(np.percentile(final_prices, 10)),
        "p25": float(np.percentile(final_prices, 25)),
        "p50": float(np.percentile(final_prices, 50)),
        "p75": float(np.percentile(final_prices, 75)),
        "p90": float(np.percentile(final_prices, 90)),
        "p95": float(np.percentile(final_prices, 95)),
        "p99": float(np.percentile(final_prices, 99)),
    }
    
    # Calculate VaR
    var_95 = initial_price - percentiles["p5"]
    var_99 = initial_price - percentiles["p1"]
    
    # Expected Shortfall (CVaR)
    cvar_95 = initial_price - np.mean(final_prices[final_prices <= percentiles["p5"]])
    cvar_99 = initial_price - np.mean(final_prices[final_prices <= percentiles["p1"]])
    
    # Probability of profit
    prob_profit = np.mean(final_prices > initial_price)
    
    # Calculate returns distribution
    returns_dist = (final_prices - initial_price) / initial_price * 100
    
    return {
        "metadata": {
            "initial_price": initial_price,
            "expected_return": expected_return,
            "volatility": volatility,
            "n_paths": n_paths,
            "days_ahead": days_ahead,
            "generated_at": datetime.utcnow().isoformat()
        },
        "statistics": {
            "mean": float(mean_final),
            "std": float(std_final),
            "min": float(min_final),
            "max": float(max_final),
            "expected_return_pct": float((mean_final - initial_price) / initial_price * 100)
        },
        "percentiles": percentiles,
        "risk_metrics": {
            "var_95": float(var_95),
            "var_99": float(var_99),
            "cvar_95": float(cvar_95),
            "cvar_99": float(cvar_99),
            "prob_profit": float(prob_profit)
        },
        "returns_distribution": {
            "mean": float(np.mean(returns_dist)),
            "std": float(np.std(returns_dist)),
            "skewness": float(np.mean(((returns_dist - np.mean(returns_dist)) / np.std(returns_dist)) ** 3)),
            "kurtosis": float(np.mean(((returns_dist - np.mean(returns_dist)) / np.std(returns_dist)) ** 4))
        }
    }


def format_output(result: Dict[str, Any]) -> str:
    """Format Monte Carlo results for chat display."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    meta = result["metadata"]
    stats = result["statistics"]
    percentiles = result["percentiles"]
    risk = result["risk_metrics"]
    
    initial = meta["initial_price"]
    days = meta["days_ahead"]
    
    output = f"""
🎲 **Monte Carlo Simulation: {meta['n_paths']:,} Paths**

**Parameters:**
- Initial Price: ${initial:,.2f}
- Expected Daily Return: {meta['expected_return']*100:.2f}%
- Volatility: {meta['volatility']*100:.2f}%
- Horizon: {days} days

**Price Distribution:**
- 📊 Mean: ${stats['mean']:,.2f} ({stats['expected_return_pct']:+.2f}%)
- 📉 Min: ${stats['min']:,.2f}
- 📈 Max: ${stats['max']:,.2f}
- σ Std Dev: ${stats['std']:,.2f}

**Key Percentiles:**
| Percentile | Price | Return |
|------------|-------|--------|
| P5 (Worst) | ${percentiles['p5']:,.2f} | {((percentiles['p5']/initial)-1)*100:+.1f}% |
| P25 | ${percentiles['p25']:,.2f} | {((percentiles['p25']/initial)-1)*100:+.1f}% |
| P50 (Median) | ${percentiles['p50']:,.2f} | {((percentiles['p50']/initial)-1)*100:+.1f}% |
| P75 | ${percentiles['p75']:,.2f} | {((percentiles['p75']/initial)-1)*100:+.1f}% |
| P95 (Best) | ${percentiles['p95']:,.2f} | {((percentiles['p95']/initial)-1)*100:+.1f}% |

**Risk Metrics:**
- ⚠️ VaR (95%): ${risk['var_95']:,.2f} ({risk['var_95']/initial*100:.1f}%)
- ⛔ VaR (99%): ${risk['var_99']:,.2f} ({risk['var_99']/initial*100:.1f}%)
- 📉 CVaR (95%): ${risk['cvar_95']:,.2f}
- 📉 CVaR (99%): ${risk['cvar_99']:,.2f}
- 💰 Prob. Profit: {risk['prob_profit']*100:.1f}%
"""
    return output


# CLI for testing
if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    initial = float(sys.argv[2]) if len(sys.argv) > 2 else 50000
    n_paths = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    
    # Typical crypto params
    result = generate_price_paths(
        initial_price=initial,
        expected_return=0.001,  # 0.1% daily
        volatility=0.03,  # 3% daily
        n_paths=n_paths,
        days_ahead=days
    )
    
    print(format_output(result))
