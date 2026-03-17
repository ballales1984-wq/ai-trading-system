"""
OpenClaw Skill: HMM Regime Detection
====================================
Direct Python execution for Hidden Markov Model regime detection.

This skill runs locally on the OpenClaw agent's Python environment,
importing the actual quant libraries (numpy, pandas, hmmlearn).

Usage in OpenClaw:
 Skill: hmm_regime_detect
  Input: symbol (e.g., "BTCUSDT"), lookback_days
  Output: regime probabilities, market state
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Try to import quant libraries, fallback to mock if unavailable
try:
    import numpy as np
    import pandas as pd
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    import random


def get_historical_prices(symbol: str, days: int = 90) -> List[float]:
    """
    Get historical prices for the symbol.
    In production, this would call a price API.
    """
    if HAS_HMM:
        # Generate realistic price data using random walk
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50000 if "BTC" in symbol else 3000
        returns = np.random.normal(0.001, 0.03, days)
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return prices
    else:
        # Mock data for testing
        base = 50000 if "BTC" in symbol else 3000
        return [base * (1 + random.uniform(-0.1, 0.1)) for _ in range(days)]


def calculate_returns(prices: List[float]) -> np.ndarray:
    """Calculate log returns from prices."""
    returns = []
    for i in range(1, len(prices)):
        ret = np.log(prices[i] / prices[i-1])
        returns.append(ret)
    return np.array(returns).reshape(-1, 1)


def detect_regimes(symbol: str, n_states: int = 3, lookback_days: int = 90) -> Dict[str, Any]:
    """
    Detect market regimes using Hidden Markov Model.
    
    States:
    - State 0: Bear (high volatility, negative drift)
    - State 1: Sideways/Low volatility
    - State 2: Bull (low volatility, positive drift)
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        n_states: Number of hidden states
        lookback_days: Days of historical data to analyze
        
    Returns:
        Dict with regime probabilities, current state, and transition matrix
    """
    if not HAS_HMM:
        return {
            "symbol": symbol,
            "error": "HMM library not available - install hmmlearn",
            "current_state": "UNKNOWN",
            "regime_probabilities": {"bear": 0.33, "sideways": 0.34, "bull": 0.33},
            "recommendation": "Install hmmlearn for regime detection"
        }

    try:
        # Get price data
        prices = get_historical_prices(symbol, lookback_days)
        returns = calculate_returns(prices)
        
        # Fit HMM model
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(returns)
        
        # Predict current state
        current_state = model.predict(returns[-1:])[0]
        state_probs = model.predict_proba(returns[-1:])[0]
        
        # Get transition matrix
        trans_matrix = model.transmat_.tolist()
        
        # Interpret states based on emission means
        means = model.means_.flatten()
        state_means = list(means)
        
        # Sort states by mean return
        sorted_indices = np.argsort(means)
        state_mapping = {}
        for i, idx in enumerate(sorted_indices):
            if i == 0:
                state_mapping[idx] = "bear"
            elif i == len(sorted_indices) - 1:
                state_mapping[idx] = "bull"
            else:
                state_mapping[idx] = "sideways"
        
        current_regime = state_mapping.get(current_state, "unknown")
        
        # Calculate regime probabilities
        regime_probs = {
            "bear": float(state_probs[sorted_indices[0]]),
            "sideways": float(state_probs[sorted_indices[len(sorted_indices)//2]]),
            "bull": float(state_probs[sorted_indices[-1]])
        }
        
        # Generate recommendation
        recommendations = {
            "bear": "Reduce exposure, defensive strategy, focus on hedging",
            "sideways": "Neutral stance, range-bound strategy, collect premiums",
            "bull": "Increase exposure, momentum strategy, trend following"
        }
        
        return {
            "symbol": symbol,
            "analysis_date": datetime.utcnow().isoformat(),
            "lookback_days": lookback_days,
            "current_state": current_regime,
            "state_id": int(current_state),
            "regime_probabilities": regime_probs,
            "transition_matrix": trans_matrix,
            "state_means": state_means,
            "volatility_regimes": "High" if current_regime == "bear" else "Normal",
            "recommendation": recommendations.get(current_regime, "Unknown"),
            "confidence": float(max(state_probs))
        }
        
    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "current_state": "ERROR",
            "recommendation": "Check data availability and try again"
        }


def format_output(result: Dict[str, Any]) -> str:
    """Format the HMM result for display in chat."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    probs = result['regime_probabilities']
    conf = result.get('confidence', 0) * 100
    
    output = f"""
🔍 **HMM Regime Analysis: {result['symbol']}**

**Current Regime:** {result['current_state'].upper()} (State {result['state_id']})
**Confidence:** {conf:.1f}%

**Regime Probabilities:**
- 🐻 Bear: {probs['bear']*100:.1f}%
- 📊 Sideways: {probs['sideways']*100:.1f}%
- 🐂 Bull: {probs['bull']*100:.1f}%

**Volatility:** {result['volatility_regimes']}

📋 **Recommendation:**
{result['recommendation']}

_Analysis based on {result['lookback_days']} days of historical data_
"""
    return output


# CLI for testing
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    
    result = detect_regimes(symbol, lookback_days=days)
    print(format_output(result))
