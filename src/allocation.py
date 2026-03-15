"""
Dynamic Portfolio Allocation
=============================
Calcola allocazioni dinamiche per portafogli multi-asset.

Strategie supportate:
- Equal Weight
- Volatility Parity (inverse volatility)
- Risk Parity
- Mean-Variance (Markowitz)
- Regime-Aware
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def equal_weight_allocation(assets: List[str]) -> Dict[str, float]:
    """
    Allocazione equal weight.
    
    Args:
        assets: Lista di asset
        
    Returns:
        Dizionario {asset: weight}
    """
    n = len(assets)
    if n == 0:
        return {}
    
    weight = 1.0 / n
    return {a: weight for a in assets}


def volatility_weighted_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 20,
    min_weight: float = 0.05,
    max_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Allocazione basata su volatilità inversa.
    Asset meno volatili ricevono più peso.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo volatilità
        min_weight: Peso minimo per asset
        max_weight: Peso massimo per asset
        
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    volatilities = {}
    
    for asset, returns in returns_dict.items():
        if len(returns) >= lookback:
            vol = returns.tail(lookback).std()
            volatilities[asset] = vol
        else:
            volatilities[asset] = 0.02  # Default volatility
    
    # Inverse volatility weighting
    inv_vol = {a: 1/v if v > 0 else 1 for a, v in volatilities.items()}
    total_inv_vol = sum(inv_vol.values())
    
    if total_inv_vol == 0:
        return equal_weight_allocation(list(returns_dict.keys()))
    
    weights = {a: inv_vol[a] / total_inv_vol for a in inv_vol}
    
    # Apply bounds
    for a in weights:
        weights[a] = max(min_weight, min(max_weight, weights[a]))
    
    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {a: w/total for a, w in weights.items()}
    
    return weights


def risk_parity_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 20,
) -> Dict[str, float]:
    """
    Risk Parity: ogni asset contribuisce ugualmente al rischio.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo volatilità e correlazioni
        
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    # Calcola matrice covarianza
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 5:
        return equal_weight_allocation(assets)
    
    cov_matrix = returns_df.cov().values
    
    # Calcola volatilità
    vols = np.sqrt(np.diag(cov_matrix))
    
    # Risk parity semplificato (inverse vol * inverse correlation sum)
    try:
        corr_matrix = returns_df.corr().values
        inv_corr = np.linalg.inv(corr_matrix)
        target_risk = np.ones(n) / n
        risk_weights = inv_corr @ target_risk
        
        # Normalize
        risk_weights = np.maximum(risk_weights, 0)
        if risk_weights.sum() > 0:
            risk_weights = risk_weights / risk_weights.sum()
        
        return {assets[i]: float(risk_weights[i]) for i in range(n)}
    except:
        return volatility_weighted_allocation(returns_dict, lookback)


def mean_variance_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 60,
    risk_aversion: float = 1.0,
) -> Dict[str, float]:
    """
    Mean-Variance (Markowitz) optimization.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo
        risk_aversion: Coefficiente avversione al rischio
        
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 10:
        return equal_weight_allocation(assets)
    
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    # Markowitz optimization (analytical solution)
    try:
        ones = np.ones(n)
        inv_cov = np.linalg.inv(cov_matrix + 1e-6 * np.eye(n))
        
        # Optimal weights
        A = ones @ inv_cov @ mean_returns
        B = mean_returns @ inv_cov @ mean_returns
        C = ones @ inv_cov @ ones
        
        lambda_val = risk_aversion
        
        weights = (inv_cov @ (mean_returns - lambda_val * ones)) / (B - lambda_val * C)
        
        # Ensure positive weights
        weights = np.maximum(weights, 0)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            return equal_weight_allocation(assets)
        
        return {assets[i]: float(weights[i]) for i in range(n)}
    
    except:
        return equal_weight_allocation(assets)


def regime_aware_allocation(
    returns_dict: Dict[str, pd.Series],
    current_regime: str = "BULL",
    lookback: int = 20,
) -> Dict[str, float]:
    """
    Allocazione basata sul regime di mercato.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        current_regime: 'BULL', 'BEAR', 'HIGH_VOL', 'LOW_VOL'
        lookback: Finestra per calcolo volatilità
        
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    
    # Calcola metriche
    recent_vol = {}
    recent_ret = {}
    
    for asset, returns in returns_dict.items():
        if len(returns) >= lookback:
            recent_vol[asset] = returns.tail(lookback).std()
            recent_ret[asset] = returns.tail(lookback).mean()
        else:
            recent_vol[asset] = 0.02
            recent_ret[asset] = 0.0
    
    if current_regime == "BULL":
        # Favorisci asset con alto rendimento
        weights = {a: max(0, recent_ret.get(a, 0)) for a in assets}
    
    elif current_regime == "BEAR":
        # Favorisci asset difensivi (bassa volatilità)
        weights = {a: 1 / (recent_vol.get(a, 0.01) + 0.01) for a in assets}
    
    elif current_regime == "HIGH_VOL":
        # Riduci esposizione, usa volatilità inversa
        weights = volatility_weighted_allocation(returns_dict, lookback)
        return weights
    
    elif current_regime == "LOW_VOL":
        # Usa risk parity
        return risk_parity_allocation(returns_dict, lookback)
    
    else:
        return equal_weight_allocation(assets)
    
    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {a: w/total for a, w in weights.items()}
    else:
        return equal_weight_allocation(assets)
    
    return weights


def build_allocation_timeseries(
    prices_dict: Dict[str, pd.DataFrame],
    strategy: str = "volatility_parity",
    window: int = 20,
    regime_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Costruisce serie temporale dell'allocazione.
    
    Args:
        prices_dict: {asset: DataFrame con 'close'}
        strategy: Strategia di allocazione
        window: Finestra mobile
        regime_series: Serie temporale regimi (opzionale)
        
    Returns:
        DataFrame con allocazioni nel tempo
    """
    if not prices_dict:
        return pd.DataFrame()
    
    assets = list(prices_dict.keys())
    
    # Trova indice comune
    common_idx = prices_dict[assets[0]].index
    for asset in assets[1:]:
        common_idx = common_idx.intersection(prices_dict[asset].index)
    
    # Prepara returns
    returns_dict = {}
    for asset, df in prices_dict.items():
        if 'close' in df.columns:
            returns_dict[asset] = df['close'].pct_change().dropna()
    
    # Calcola allocazione per ogni timestep
    allocations = []
    timestamps = []
    
    for i in range(window, len(common_idx)):
        idx = common_idx[i]
        
        # Returns per questo timestep
        period_returns = {}
        for asset, returns in returns_dict.items():
            try:
                period_returns[asset] = returns.loc[:idx].tail(window)
            except:
                period_returns[asset] = pd.Series([0])
        
        # Determina regime se disponibile
        regime = "BULL"
        if regime_series is not None:
            try:
                regime = regime_series.loc[:idx].iloc[-1]
            except:
                regime = "BULL"
        
        # Calcola allocazione
        if strategy == "equal":
            weights = equal_weight_allocation(assets)
        elif strategy == "volatility_parity":
            weights = volatility_weighted_allocation(period_returns, window)
        elif strategy == "risk_parity":
            weights = risk_parity_allocation(period_returns, window)
        elif strategy == "mean_variance":
            weights = mean_variance_allocation(period_returns, window)
        elif strategy == "regime_aware":
            weights = regime_aware_allocation(period_returns, regime, window)
        else:
            weights = equal_weight_allocation(assets)
        
        allocations.append(weights)
        timestamps.append(idx)
    
    if not allocations:
        return pd.DataFrame(index=common_idx, columns=assets)
    
    # Crea DataFrame
    alloc_df = pd.DataFrame(allocations, index=timestamps, columns=assets)
    alloc_df = alloc_df.fillna(method="ffill").fillna(1.0 / len(assets))
    
    return alloc_df


# ======================
# ESEMPIO DI UTILIZZO
# ======================

if __name__ == "__main__":
    # Simula prezzi
    np.random.seed(42)
    n = 100
    
    prices_btc = pd.Series(50000 * np.cumprod(1 + np.random.randn(n) * 0.02))
    prices_eth = pd.Series(3000 * np.cumprod(1 + np.random.randn(n) * 0.025))
    prices_sol = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.03))
    
    returns_btc = prices_btc.pct_change().dropna()
    returns_eth = prices_eth.pct_change().dropna()
    returns_sol = prices_sol.pct_change().dropna()
    
    returns_dict = {
        "BTC": returns_btc,
        "ETH": returns_eth,
        "SOL": returns_sol,
    }
    
    print("📊 Allocazioni:")
    
    print("\n1. Equal Weight:")
    print(equal_weight_allocation(["BTC", "ETH", "SOL"]))
    
    print("\n2. Volatility Parity:")
    print(volatility_weighted_allocation(returns_dict))
    
    print("\n3. Risk Parity:")
    print(risk_parity_allocation(returns_dict))
    
    print("\n4. Mean-Variance:")
    print(mean_variance_allocation(returns_dict))
    
    print("\n5. Regime-Aware (BULL):")
    print(regime_aware_allocation(returns_dict, "BULL"))
    
    print("\n6. Regime-Aware (BEAR):")
    print(regime_aware_allocation(returns_dict, "BEAR"))


# ============================================================================
# ADVANCED ALLOCATION STRATEGIES (NEW - No external dependencies)
# ============================================================================

def minimum_variance_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 60,
    min_weight: float = 0.01,
) -> Dict[str, float]:
    """
    Minimum Variance Portfolio - Minimizes portfolio volatility.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo covarianza
        min_weight: Peso minimo per asset
    
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 10:
        return equal_weight_allocation(assets)
    
    cov_matrix = returns_df.cov().values
    
    try:
        # Solve for minimum variance portfolio
        # minimize: w' * Cov * w
        # subject to: sum(w) = 1
        
        from scipy.optimize import minimize
        
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(min_weight, 1.0) for _ in range(n)]
        
        # Initial guess: equal weight
        w0 = np.ones(n) / n
        
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else w0
        weights = np.maximum(weights, min_weight)
        weights = weights / weights.sum()
        
        return {assets[i]: float(weights[i]) for i in range(n)}
    
    except ImportError:
        # Fallback without scipy
        # Use simplified inverse variance weighting
        variances = cov_matrix.diagonal()
        inv_var = 1 / (variances + 1e-10)
        weights = inv_var / inv_var.sum()
        return {assets[i]: float(weights[i]) for i in range(n)}
    except:
        return equal_weight_allocation(assets)


def maximum_diversification_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 60,
) -> Dict[str, float]:
    """
    Maximum Diversification Portfolio - Maximizes diversification ratio.
    
    The diversification ratio = portfolio volatility / weighted average vol
    Higher ratio = better diversification.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo
    
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 10:
        return equal_weight_allocation(assets)
    
    cov_matrix = returns_df.cov().values
    vols = np.sqrt(np.diag(cov_matrix))
    
    if np.any(vols == 0):
        return equal_weight_allocation(assets)
    
    try:
        from scipy.optimize import minimize
        
        def neg_diversification_ratio(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            weighted_vol = weights @ vols
            if weighted_vol == 0:
                return 0
            return -port_vol / weighted_vol  # Negative because we minimize
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.01, 1.0) for _ in range(n)]
        
        w0 = np.ones(n) / n
        
        result = minimize(
            neg_diversification_ratio,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else w0
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return {assets[i]: float(weights[i]) for i in range(n)}
    
    except:
        return equal_weight_allocation(assets)


def equal_risk_contribution_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 60,
    max_iterations: int = 100,
) -> Dict[str, float]:
    """
    Equal Risk Contribution (ERC) - Each asset contributes equally to portfolio risk.
    
    Similar to risk parity but with iterative optimization.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo covarianza
        max_iterations: Numero massimo iterazioni
    
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 10:
        return equal_weight_allocation(assets)
    
    cov_matrix = returns_df.cov().values
    
    try:
        from scipy.optimize import minimize
        
        def risk_contribution(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / port_vol
            
            # Target: equal risk contribution
            target = np.ones(n) / n * port_vol
            return np.sum((risk_contrib - target) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.01, 1.0) for _ in range(n)]
        
        w0 = np.ones(n) / n
        
        result = minimize(
            risk_contribution,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations}
        )
        
        weights = result.x if result.success else w0
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return {assets[i]: float(weights[i]) for i in range(n)}
    
    except ImportError:
        # Fallback to risk parity
        return risk_parity_allocation(returns_dict, lookback)
    except:
        return equal_weight_allocation(assets)


def black_litterman_allocation(
    returns_dict: Dict[str, pd.Series],
    market_cap_weights: Optional[Dict[str, float]] = None,
    risk_aversion: float = 2.5,
    confidence: float = 0.5,
    views: Optional[Dict[str, float]] = None,
    lookback: int = 60,
) -> Dict[str, float]:
    """
    Black-Litterman Model - Combines market equilibrium with views.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        market_cap_weights: Pesi di mercato (cap weights)
        risk_aversion: Coefficiente avversione al rischio
        confidence: Fiducia nelle views (0-1)
        views: Views opzionali {asset: expected return}
        lookback: Finestra per calcolo
    
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 10:
        return equal_weight_allocation(assets)
    
    cov_matrix = returns_df.cov().values
    
    # Default: equal market cap
    if market_cap_weights is None:
        market_cap_weights = {a: 1.0/n for a in assets}
    
    market_weights = np.array([market_cap_weights.get(a, 1.0/n) for a in assets])
    market_weights = market_weights / market_weights.sum()
    
    # Implied equilibrium returns
    implied_returns = risk_aversion * cov_matrix @ market_weights
    
    # If no views, use equilibrium
    if views is None or len(views) == 0:
        final_returns = implied_returns
    else:
        # Blend views with equilibrium
        view_returns = np.array([views.get(a, implied_returns[i]) for i, a in enumerate(assets)])
        final_returns = confidence * view_returns + (1 - confidence) * implied_returns
    
    # Mean-variance optimization with adjusted returns
    try:
        from scipy.optimize import minimize
        
        def neg_sharpe(weights):
            port_return = weights @ final_returns
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return 0
            return -(port_return - risk_aversion / 252 * port_vol ** 2) / port_vol
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        w0 = market_weights
        
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else w0
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return {assets[i]: float(weights[i]) for i in range(n)}
    
    except:
        # Fallback: use implied returns with inverse vol
        inv_vol = 1 / (np.sqrt(np.diag(cov_matrix)) + 1e-10)
        weights = inv_vol * final_returns
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum() if weights.sum() > 0 else market_weights
        return {assets[i]: float(weights[i]) for i in range(n)}


def momentum_based_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback_short: int = 20,
    lookback_long: int = 60,
    min_weight: float = 0.05,
    max_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Momentum-Based Allocation - Favor assets with strong recent performance.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback_short: Short-term momentum window
        lookback_long: Long-term momentum window
        min_weight: Peso minimo
        max_weight: Peso massimo
    
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    momentum_scores = {}
    
    for asset, returns in returns_dict.items():
        if len(returns) < lookback_long:
            momentum_scores[asset] = 0.0
            continue
        
        # Short-term momentum
        short_ret = returns.tail(lookback_short).sum()
        
        # Long-term momentum
        long_ret = returns.tail(lookback_long).sum()
        
        # Combined momentum score
        # Higher short-term = recent strength
        # Higher long-term = sustained performance
        score = 0.3 * short_ret + 0.7 * long_ret
        momentum_scores[asset] = score
    
    # Convert to weights (positive scores only)
    scores = np.array([max(0, momentum_scores.get(a, 0)) for a in assets])
    
    if scores.sum() == 0:
        return equal_weight_allocation(assets)
    
    weights = scores / scores.sum()
    
    # Apply bounds
    weights = np.maximum(weights, min_weight)
    weights = np.minimum(weights, max_weight)
    
    # Renormalize
    weights = weights / weights.sum()
    
    return {assets[i]: float(weights[i]) for i in range(len(assets))}


def cluster_based_allocation(
    returns_dict: Dict[str, pd.Series],
    lookback: int = 60,
    min_cluster_correlation: float = 0.3,
) -> Dict[str, float]:
    """
    Cluster-Based Allocation - Diversify across uncorrelated asset clusters.
    
    Simple hierarchical-like approach: weight assets inversely to their
    average correlation with other assets.
    
    Args:
        returns_dict: {asset: serie rendimenti}
        lookback: Finestra per calcolo correlazioni
        min_cluster_correlation: Soglia per considerare correlazione significativa
    
    Returns:
        Dizionario {asset: weight}
    """
    if not returns_dict:
        return {}
    
    assets = list(returns_dict.keys())
    n = len(assets)
    
    if n < 2:
        return equal_weight_allocation(assets)
    
    returns_df = pd.DataFrame({a: returns_dict[a].tail(lookback) for a in assets})
    
    if len(returns_df) < 10:
        return equal_weight_allocation(assets)
    
    corr_matrix = returns_df.corr().values
    
    # Calculate average absolute correlation for each asset
    # Lower correlation = better diversification = higher weight
    avg_corr = np.mean(np.abs(corr_matrix - np.eye(n)), axis=1)
    
    # Invert: lower correlation = higher weight
    inv_corr = 1 / (avg_corr + 0.01)
    weights = inv_corr / inv_corr.sum()
    
    return {assets[i]: float(weights[i]) for i in range(n)}
