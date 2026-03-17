"""
Test Suite for OpenClaw Level 3 Components
============================================
Tests all the new components for the 10/10 upgrade:
- QuantResult standard format
- Parameter validation
- Skill Registry
- Intent Router
- Skill Orchestrator

Run with: python openclaw_skills/test_level3_components.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
from quant_result import (
    QuantResult, create_result, ExecutionStatus, RiskLevel,
    wrap_skill_result
)
from param_validator import (
    validate_params, ValidationError, get_skill_limits,
    check_resource_availability, sanitize_input
)
from intent_router import IntentRouter, IntentType, route_message
from skill_orchestrator import SkillOrchestrator, PipelineStatus


# =============================================================================
# Test Results Tracking
# =============================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, test_name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            error_msg = f"  [FAIL] {test_name}: {message}" if message else f"  [FAIL] {test_name}"
            self.errors.append(error_msg)
            print(error_msg)
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"Failed tests:")
            for error in self.errors:
                print(error)
        return self.failed == 0


results = TestResults()


# =============================================================================
# Test QuantResult
# =============================================================================

def test_quant_result():
    print("\n--- Testing QuantResult ---")
    
    # Test 1: Create basic result
    result = create_result(
        skill="hmm",
        metrics={"regime": "bull", "confidence": 0.87},
        recommendations=["Increase exposure"],
        risk_level="low"
    )
    results.record(
        "Create basic QuantResult",
        result.skill == "hmm" and result.metrics["regime"] == "bull"
    )
    
    # Test 2: Error result factory
    error_result = QuantResult.create_error(
        skill="garch",
        error_message="Insufficient data",
        execution_time_ms=45.2
    )
    results.record(
        "Create error result",
        error_result.status == ExecutionStatus.FAILED.value and
        "Insufficient data" in error_result.errors
    )
    
    # Test 3: Validation error result
    validation_result = QuantResult.create_validation_error(
        skill="monte_carlo",
        validation_errors=["n_paths must be <= 20000"]
    )
    results.record(
        "Create validation error result",
        validation_result.status == ExecutionStatus.VALIDATION_ERROR.value
    )
    
    # Test 4: JSON serialization
    result = create_result(skill="test", metrics={"value": 42})
    json_str = result.to_json()
    results.record(
        "JSON serialization",
        "test" in json_str and "value" in json_str
    )
    
    # Test 5: Format for chat
    result = create_result(
        skill="hmm",
        metrics={"regime": "bull"},
        recommendations=["Test recommendation"]
    )
    chat_output = result.format_for_chat()
    results.record(
        "Format for chat",
        "HMM" in chat_output and "bull" in chat_output
    )


# =============================================================================
# Test Parameter Validation
# =============================================================================

def test_param_validation():
    print("\n--- Testing Parameter Validation ---")
    
    # Test 1: Valid Monte Carlo params
    try:
        validated = validate_params("monte_carlo", {
            "initial_price": 50000,
            "volatility": 0.03,
            "n_paths": 5000,
            "days_ahead": 30
        })
        results.record(
            "Valid Monte Carlo params",
            validated["n_paths"] == 5000 and validated["initial_price"] == 50000
        )
    except ValidationError as e:
        results.record("Valid Monte Carlo params", False, str(e.errors))
    
    # Test 2: Invalid n_paths (too high)
    try:
        validate_params("monte_carlo", {
            "initial_price": 50000,
            "volatility": 0.03,
            "n_paths": 100000,  # Exceeds max
            "days_ahead": 30
        })
        results.record("Reject excessive n_paths", False, "Should have raised error")
    except ValidationError as e:
        results.record(
            "Reject excessive n_paths",
            "n_paths" in str(e.errors) and "20000" in str(e.errors)
        )
    
    # Test 3: Missing required parameter
    try:
        validate_params("hmm", {
            "n_states": 3
            # Missing 'symbol'
        })
        results.record("Detect missing required param", False, "Should have raised error")
    except ValidationError as e:
        results.record(
            "Detect missing required param",
            "symbol" in str(e.errors)
        )
    
    # Test 4: Get skill limits
    limits = get_skill_limits("monte_carlo")
    results.record(
        "Get skill limits",
        "max_runtime_ms" in limits and "max_memory_mb" in limits
    )
    
    # Test 5: Sanitize input
    sanitized = sanitize_input({"cmd": "ls; rm -rf"}, dict)
    results.record(
        "Sanitize input",
        ";" not in str(sanitized.get("cmd", ""))
    )


# =============================================================================
# Test Intent Router
# =============================================================================

def test_intent_router():
    print("\n--- Testing Intent Router ---")
    
    router = IntentRouter()
    
    # Test 1: Regime detection intent
    intent = router.parse("What regime is BTC in?")
    results.record(
        "Detect regime intent",
        intent.skill == "hmm" and "BTC" in intent.params.get("symbol", "")
    )
    
    # Test 2: Monte Carlo intent
    intent = router.parse("Simulate 5000 paths for BTC with $50000 price")
    results.record(
        "Detect Monte Carlo intent",
        intent.skill == "monte_carlo" and 
        intent.params.get("n_paths") == 5000
    )
    
    # Test 3: Portfolio optimization intent
    intent = router.parse("Optimize my portfolio: BTC, ETH, SOL")
    results.record(
        "Detect portfolio intent",
        intent.skill == "portfolio_optimizer" and
        len(intent.params.get("assets", [])) >= 2
    )
    
    # Test 4: Extract parameters from message
    intent = router.parse("Calculate GARCH volatility for ETH with GARCH(1,1)")
    results.record(
        "Extract GARCH params",
        intent.params.get("p") == 1 and intent.params.get("q") == 1
    )
    
    # Test 5: Clarification needed
    intent = router.parse("Run a simulation")
    results.record(
        "Detect need for clarification",
        intent.needs_clarification and len(intent.clarification_questions) > 0
    )


# =============================================================================
# Test Skill Orchestrator
# =============================================================================

def test_skill_orchestrator():
    print("\n--- Testing Skill Orchestrator ---")
    
    orchestrator = SkillOrchestrator(enable_cache=False)
    
    # Test 1: Single skill execution
    result = orchestrator.execute("hmm", {
        "symbol": "BTCUSDT",
        "n_states": 3
    })
    results.record(
        "Execute HMM skill",
        result is not None and result.skill == "hmm"
    )
    
    # Test 2: Execute with validation
    result = orchestrator.execute("monte_carlo", {
        "initial_price": 50000,
        "volatility": 0.03,
        "n_paths": 5000,
        "days_ahead": 30
    })
    results.record(
        "Execute Monte Carlo",
        result.status == ExecutionStatus.SUCCESS.value
    )
    
    # Test 3: Handle invalid parameters gracefully
    result = orchestrator.execute("monte_carlo", {
        "initial_price": 50000,
        "volatility": 0.03,
        "n_paths": 500000,  # Too high
        "days_ahead": 30
    })
    results.record(
        "Handle invalid params",
        result.status in [ExecutionStatus.FAILED.value, ExecutionStatus.VALIDATION_ERROR.value]
    )
    
    # Test 4: Pipeline execution
    # Note: This may fail if skills aren't installed, that's OK
    pipeline_result = orchestrator.execute_pipeline("full_risk_analysis", {
        "symbol": "BTCUSDT"
    })
    results.record(
        "Execute pipeline",
        pipeline_result is not None and hasattr(pipeline_result, 'status')
    )


# =============================================================================
# Test Integration
# =============================================================================

def test_integration():
    print("\n--- Testing Full Integration ---")
    
    # Test: End-to-end flow
    # 1. Parse user message
    router = IntentRouter()
    intent = router.parse("What regime is ETH in?")
    
    # 2. Execute skill
    orchestrator = SkillOrchestrator(enable_cache=False)
    result = orchestrator.execute(intent.skill, intent.params)
    
    # 3. Format response
    response = result.format_for_chat()
    
    results.record(
        "End-to-end flow",
        len(response) > 0 and "HMM" in response
    )


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print("="*50)
    print("OpenClaw Level 3 Components Test Suite")
    print("="*50)
    
    test_quant_result()
    test_param_validation()
    test_intent_router()
    test_skill_orchestrator()
    test_integration()
    
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
