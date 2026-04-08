"""Test Phase 4: Monitoring Enterprise - Minimal"""

import os
import sys

print("=" * 60)
print("PHASE 4: MONITORING ENTERPRISE TESTS")
print("=" * 60)

results = []

# Test 1: Check prometheus_metrics.py exists
prom_file = "src/core/performance/prometheus_metrics.py"
if os.path.exists(prom_file):
    print(f"[OK] prometheus_metrics.py exists")
    results.append(True)
else:
    print(f"[FAIL] prometheus_metrics.py not found")
    results.append(False)

# Test 2: Check resource_monitor.py exists
res_file = "src/core/resource_monitor.py"
if os.path.exists(res_file):
    print(f"[OK] resource_monitor.py exists")
    results.append(True)
else:
    print(f"[FAIL] resource_monitor.py not found")
    results.append(False)

# Test 3: Check health endpoint
health_file = "app/api/routes/health.py"
if os.path.exists(health_file):
    print(f"[OK] health.py exists")
    results.append(True)
else:
    print(f"[FAIL] health.py not found")
    results.append(False)

# Test 4: Check validation_dashboard
dash_file = "tests/validation_dashboard.py"
if os.path.exists(dash_file):
    print(f"[OK] validation_dashboard.py exists")
    results.append(True)
else:
    print(f"[FAIL] validation_dashboard.py not found")
    results.append(False)

print("\n" + "=" * 60)
if all(results):
    print("ALL PHASE 4 TESTS PASSED")
else:
    print(f"FAILED: {results.count(False)} tests")
print("=" * 60)
