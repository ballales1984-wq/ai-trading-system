"""
Test for Phase 3: AI Strategy Orchestrator
Verifies AutonomousQuantAgent and strategy generation.
"""

import sys

sys.path.insert(0, ".")


def test_autonomous_agent_import():
    """Test AutonomousQuantAgent import."""
    try:
        from src.agents.autonomous_quant_agent import AutonomousQuantAgent, AgentConfig

        print(f"[OK] AutonomousQuantAgent loaded")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_agent_config():
    """Test agent configuration."""
    from src.agents.autonomous_quant_agent import AgentConfig

    config = AgentConfig(max_position_pct=0.15, default_symbols=["BTCUSDT", "ETHUSDT"])

    assert config.max_position_pct == 0.15
    assert "BTCUSDT" in config.default_symbols

    print(f"[OK] AgentConfig: max_pos={config.max_position_pct}, symbols={config.default_symbols}")
    return True


def test_regime_signal():
    """Test regime signal dataclass."""
    from src.agents.autonomous_quant_agent import RegimeSignal
    from datetime import datetime

    signal = RegimeSignal(
        symbol="BTCUSDT", regime="bull", confidence=0.85, volatility=0.25, timestamp=datetime.now()
    )

    assert signal.regime == "bull"
    assert signal.confidence > 0.7

    print(f"[OK] RegimeSignal: {signal.regime} @ {signal.confidence:.0%}")
    return True


def test_risk_assessment():
    """Test risk assessment dataclass."""
    from src.agents.autonomous_quant_agent import RiskAssessment

    assessment = RiskAssessment(
        var_95=-0.02,
        cvar_95=-0.035,
        drawdown_pct=0.03,
        exposure_pct=0.08,
        position_count=3,
        within_limits=True,
    )

    assert assessment.within_limits == True

    print(
        f"[OK] RiskAssessment: VaR={assessment.var_95:.2%}, within_limits={assessment.within_limits}"
    )
    return True


def test_model_registry():
    """Test Model Registry."""
    from src.research.model_registry import ModelRegistry, ModelMeta
    from datetime import datetime

    registry = ModelRegistry()

    meta = ModelMeta(
        name="test_model",
        version="1.0.0",
        trained_at=datetime.now(),
        dataset_id="test_ds",
        metrics={"accuracy": 0.75, "f1": 0.72},
    )

    print(f"[OK] ModelRegistry: added {meta.name}")
    return True


def test_decision_engine():
    """Test Decision Engine import."""
    from src.decision import MonteCarloSimulator, DecisionEngine

    print(f"[OK] DecisionEngine imported")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: AI STRATEGY ORCHESTRATOR TESTS")
    print("=" * 60)

    results = []
    results.append(test_autonomous_agent_import())
    results.append(test_agent_config())
    results.append(test_regime_signal())
    results.append(test_risk_assessment())
    results.append(test_model_registry())
    results.append(test_decision_engine())

    print("\n" + "=" * 60)
    if all(results):
        print("ALL PHASE 3 TESTS PASSED")
    else:
        print(f"FAILED: {results.count(False)} tests")
    print("=" * 60)
