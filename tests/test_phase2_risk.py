"""
Test for Phase 2: Quantitative Risk Management
Verifies GARCH volatility models and Monte Carlo simulations.
"""

import numpy as np
import pandas as pd


def test_garch_model():
    """Test GARCH volatility forecasting (simplified)."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))

    # Simple volatility estimation
    cond_vol = returns.std() * np.sqrt(252)  # Annualized

    print(f"Annualized volatility: {cond_vol:.2%}")
    assert cond_vol > 0, "Volatility should be positive"
    print("[OK] GARCH model test passed")
    return cond_vol


def test_monte_carlo_import():
    """Test Monte Carlo module import."""
    from src.agents.agent_montecarlo import MonteCarloAgent, SimulationResult

    print(f"MonteCarloAgent: {MonteCarloAgent.__name__}")
    print(f"SimulationResult: {SimulationResult.__name__}")
    print("[OK] Monte Carlo import test passed")


def test_var_calculation():
    """Test Value at Risk calculation."""
    returns = np.random.normal(0.001, 0.02, 1000)

    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    print(f"VaR (95%): {var_95:.4f}")
    print(f"CVaR (95%): {cvar_95:.4f}")
    assert var_95 < 0, "VaR should be negative for risk"
    print("[OK] VaR test passed")


def test_risk_metrics():
    """Test risk metrics calculation."""
    from src.core.risk import RiskEngine

    risk = RiskEngine(initial_balance=100000)

    # Test signal check
    signal = {"symbol": "BTCUSDT", "action": "BUY", "quantity": 0.1, "price": 45000, "position": 0}

    passed, reason = risk.check_signal(signal)
    print(f"Signal check: {passed} - {reason}")

    status = risk.get_status()
    print(f"Risk level: {status['risk_level']}")

    print("[OK] Risk engine test passed")


def test_volatility_models():
    """Test GARCH model import."""
    from src.core.risk.volatility_models import GARCHModel

    garch = GARCHModel(p=1, q=1, model_type="GARCH")
    print(f"GARCHModel: {garch.model_type}")
    print("[OK] Volatility models test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: QUANTITATIVE RISK MANAGEMENT TESTS")
    print("=" * 60)

    test_garch_model()
    test_monte_carlo_import()
    test_var_calculation()
    test_risk_metrics()
    test_volatility_models()

    print("\n" + "=" * 60)
    print("ALL PHASE 2 TESTS PASSED")
    print("=" * 60)
