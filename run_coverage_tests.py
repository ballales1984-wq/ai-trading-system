"""
Comprehensive Coverage Test Runner
==============================
This script runs all the coverage tests together.
"""

import subprocess
import sys
import os

# List of all coverage test files
COVERAGE_TESTS = [
    "tests/test_portfolio_coverage.py",
    "tests/test_market_data_coverage.py",
    "tests/test_execution_coverage.py",
    "tests/test_strategies_coverage.py",
    "tests/test_database_coverage.py",
    "tests/test_performance_coverage.py",
    "tests/test_ml_coverage.py",
    "tests/test_external_coverage.py",
    "tests/test_live_coverage.py",
    "tests/test_decision_coverage.py",
    "tests/test_production_coverage.py",
    "tests/test_research_coverage.py",
    "tests/test_additional_modules_coverage.py",
    "tests/test_app_coverage_extended.py",
    "tests/test_core_coverage.py",
    "tests/test_uncovered_modules.py",
    "tests/test_trading_completo_coverage.py",
]

def run_tests():
    """Run all coverage tests."""
    print("=" * 60)
    print("Running Coverage Tests")
    print("=" * 60)
    
    # Run pytest with all coverage tests
    test_files = " ".join(COVERAGE_TESTS)
    
    cmd = f"python -m pytest {test_files} -v --tb=short"
    
    print(f"\nRunning command: {cmd}\n")
    
    result = os.system(cmd)
    
    return result

if __name__ == "__main__":
    sys.exit(run_tests())

