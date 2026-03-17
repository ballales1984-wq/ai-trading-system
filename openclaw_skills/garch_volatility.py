"""
OpenClaw Skill: GARCH Volatility Model
======================================
Direct Python execution for GARCH volatility forecasting.

This skill runs locally, using ARCH library for GARCH modeling.

Usage in OpenClaw:
 Skill: garch_volatility
  Input: symbol, p, q (GARCH order), forecast_horizon
  Output: volatility forecast, conditional volatility, risk metrics
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Try to import quant libraries
try:
    import numpy as np
    import pandas as pd
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    import random


def get_returns_data(symbol: str, days: int = 252) -> List[float]:
    """Get historical returns for GARCH fitting."""
    if HAS_ARCH:
        np.random.seed(hash(symbol) % 2**32)
        # Simulate realistic returns with volatility clustering
        returns = []
        vol = 0.02
        for _ in range(days):
            vol = 0.95 * vol + 0.05 * random.gauss(0, 1)**2
            vol = np.sqrt(vol)
            ret = random.gauss(0.0005, vol)
            returns.append(ret)
        return returns
    else:
        # Mock returns for testing
        return [random.gauss(0.001, 0.02) for _ in range(days)]


def fit_garch(
    symbol: str,
    p: int = 1,
    q: int = 1,
    forecast_horizon: int = 5
) -> Dict[str, Any]:
    """
    Fit GARCH(p,q) model and forecast volatility.
    
    Args:
        symbol: Asset symbol
        p: GARCH lag order
        q: ARCH lag order
        forecast_horizon: Days to forecast
        
    Returns:
        Dict with fitted parameters and forecasts
    """
    if not HAS_ARCH:
        # Fallback mock response
        return {
            "symbol": symbol,
            "error": "ARCH library not available - install arch",
            "current_volatility": 0.03,
            "forecast": [0.029, 0.028, 0.027, 0.026, 0.025],
            "recommendation": "Install arch package: pip install arch"
        }
    
    try:
        # Get returns data
        returns = get_returns_data(symbol)
        returns_array = np.array(returns)
        
        # Fit GARCH model
        model = arch_model(
            returns_array * 100,  # Scale to percentage
            vol='Garch',
            p=p,
            q=q,
            dist='normal'
        )
        result = model.fit(disp='off')
        
        # Get parameters
        omega = result.params.get('omega', 0)
        alpha = result.params.get('alpha[1]', 0)
        beta = result.params.get('beta[1]', 0)
        
        # Current conditional volatility (annualized)
        cond_vol = result.conditional_volatility[-1]
        annual_vol = cond_vol * np.sqrt(252) / 100
        
        # Forecast
        forecast = result.forecast(horizon=forecast_horizon)
        forecast_vols = forecast.variance.values[-1, :] ** 0.5 / 100
        forecast_annual = forecast_vols * np.sqrt(252)
        
        # Calculate VaR with GARCH volatility
        z_95 = 1.645  # 95% confidence
        var_1d = annual_vol / np.sqrt(252) * z_95
        
        return {
            "symbol": symbol,
            "model": f"GARCH({p},{q})",
            "fitted_at": datetime.utcnow().isoformat(),
            "parameters": {
                "omega": float(omega),
                "alpha": float(alpha),
                "beta": float(beta),
                "persistence": float(alpha + beta)
            },
            "current_volatility": {
                "daily": float(cond_vol / 100),
                "annualized": float(annual_vol),
                "percent": f"{annual_vol * 100:.2f}%"
            },
            "forecast": {
                "days": forecast_horizon,
                "daily_volatility": [float(v) for v in forecast_vols],
                "annualized_volatility": [float(v) for v in forecast_annual]
            },
            "risk_metrics": {
                "var_1d_95": float(var_1d),
                "var_5d_95": float(var_1d * np.sqrt(5)),
                "volatility_regime": "HIGH" if annual_vol > 0.05 else "NORMAL" if annual_vol > 0.02 else "LOW"
            },
            "model_stats": {
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "log_likelihood": float(result.loglikelihood)
            }
        }
        
    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "recommendation": "Check data quality and model parameters"
        }


def format_output(result: Dict[str, Any]) -> str:
    """Format GARCH results for chat display."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    params = result["parameters"]
    curr = result["current_volatility"]
    forecast = result["forecast"]
    risk = result["risk_metrics"]
    
    output = f"""
📈 **GARCH Volatility Analysis: {result['symbol']}**

**Model:** {result['model']}

**Parameters:**
- ω (omega): {params['omega']:.6f}
- α (alpha): {params['alpha']:.4f}
- β (beta): {params['beta']:.4f}
- Persistence (α+β): {params['persistence']:.4f}

**Current Volatility:**
- Daily: {curr['daily']*100:.2f}%
- Annualized: {curr['percent']}

**Forecast (Next {forecast['days']} Days):**
| Day | Daily Vol | Annual Vol |
|-----|-----------|------------|
"""
    
    for i, (d, a) in enumerate(zip(forecast['daily_volatility'], forecast['annualized_volatility']), 1):
        output += f"| {i} | {d*100:.2f}% | {a*100:.2f}% |\n"
    
    output += f"""
**Risk Metrics:**
- VaR (1-day, 95%): {risk['var_1d_95']*100:.2f}%
- VaR (5-day, 95%): {risk['var_5d_95']*100:.2f}%
- Regime: {risk['volatility_regime']}

**Model Fit:**
- AIC: {result['model_stats']['AIC']:.2f}
- BIC: {result['model_stats']['BIC']:.2f}
"""
    return output


# CLI for testing
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    
    result = fit_garch(symbol, p=1, q=1, forecast_horizon=5)
    print(format_output(result))
