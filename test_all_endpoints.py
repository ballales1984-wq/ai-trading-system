#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive API Test Suite
============================
Tests all endpoints of the Hedge Fund Trading System API.
"""

import sys
import io
import requests
import json
from typing import Dict, Any
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_test(name: str, status: str, details: str = ""):
    """Print test result."""
    color = Colors.GREEN if status == "OK" else Colors.RED
    print(f"{color}{status}{Colors.RESET} {name}")
    if details:
        print(f"   {details}")

def test_endpoint(method: str, endpoint: str, expected_status: int = 200, 
                  data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> tuple:
    """Test a single endpoint."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=5)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=5)
        elif method.upper() == "PATCH":
            response = requests.patch(url, json=data, timeout=5)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=5)
        else:
            return False, f"Unknown method: {method}"
        
        success = response.status_code == expected_status
        details = f"Status: {response.status_code}"
        
        if success:
            try:
                json_data = response.json()
                if isinstance(json_data, dict) and len(json_data) > 0:
                    details += f" | Response keys: {list(json_data.keys())[:3]}..."
                elif isinstance(json_data, list):
                    details += f" | Items: {len(json_data)}"
            except:
                pass
        
        return success, details
        
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - Is server running?"
    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Run all API tests."""
    print_header("HEDGE FUND TRADING SYSTEM - API TEST SUITE")
    print(f"Testing API at: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    # ========================================================================
    # HEALTH ENDPOINTS
    # ========================================================================
    print_header("HEALTH ENDPOINTS")
    
    tests = [
        ("GET", "/health", 200),
        (f"GET", f"{API_PREFIX}/health", 200),
        (f"GET", f"{API_PREFIX}/ready", 200),
        (f"GET", f"{API_PREFIX}/live", 200),
    ]
    
    for method, endpoint, status in tests:
        total_tests += 1
        success, details = test_endpoint(method, endpoint, status)
        if success:
            passed_tests += 1
            print_test(f"{method} {endpoint}", "OK", details)
        else:
            failed_tests.append((method, endpoint, details))
            print_test(f"{method} {endpoint}", "FAIL", details)
    
    # ========================================================================
    # MARKET DATA ENDPOINTS
    # ========================================================================
    print_header("MARKET DATA ENDPOINTS")
    
    tests = [
        (f"GET", f"{API_PREFIX}/market/price/BTCUSDT", 200),
        (f"GET", f"{API_PREFIX}/market/prices", 200),
        (f"GET", f"{API_PREFIX}/market/candles/BTCUSDT", 200, None, {"interval": "1h", "limit": 10}),
        (f"GET", f"{API_PREFIX}/market/orderbook/BTCUSDT", 200),
        (f"GET", f"{API_PREFIX}/market/ticker/24h/BTCUSDT", 200),
        (f"GET", f"{API_PREFIX}/market/trades/BTCUSDT", 200, None, {"limit": 10}),
        (f"GET", f"{API_PREFIX}/market/futures/funding/BTCUSDT", 200),
        (f"GET", f"{API_PREFIX}/market/index/BTCUSDT", 200),
    ]
    
    for method, endpoint, status, *rest in tests:
        total_tests += 1
        data = rest[0] if len(rest) > 0 else None
        params = rest[1] if len(rest) > 1 else None
        success, details = test_endpoint(method, endpoint, status, data, params)
        if success:
            passed_tests += 1
            print_test(f"{method} {endpoint}", "OK", details)
        else:
            failed_tests.append((method, endpoint, details))
            print_test(f"{method} {endpoint}", "FAIL", details)
    
    # ========================================================================
    # PORTFOLIO ENDPOINTS
    # ========================================================================
    print_header("PORTFOLIO ENDPOINTS")
    
    tests = [
        (f"GET", f"{API_PREFIX}/portfolio/summary", 200),
        (f"GET", f"{API_PREFIX}/portfolio/positions", 200),
        (f"GET", f"{API_PREFIX}/portfolio/performance", 200),
        (f"GET", f"{API_PREFIX}/portfolio/allocation", 200),
        (f"GET", f"{API_PREFIX}/portfolio/history", 200),
        (f"GET", f"{API_PREFIX}/portfolio/history", 200, None, {"days": 7}),
    ]
    
    for method, endpoint, status, *rest in tests:
        total_tests += 1
        data = rest[0] if len(rest) > 0 else None
        params = rest[1] if len(rest) > 1 else None
        success, details = test_endpoint(method, endpoint, status, data, params)
        if success:
            passed_tests += 1
            print_test(f"{method} {endpoint}", "OK", details)
        else:
            failed_tests.append((method, endpoint, details))
            print_test(f"{method} {endpoint}", "FAIL", details)
    
    # Get a position ID for testing
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/portfolio/positions", timeout=5)
        if response.status_code == 200:
            positions = response.json()
            if positions and len(positions) > 0:
                position_id = positions[0].get("position_id")
                if position_id:
                    total_tests += 1
                    success, details = test_endpoint("GET", f"{API_PREFIX}/portfolio/positions/{position_id}", 200)
                    if success:
                        passed_tests += 1
                        print_test(f"GET {API_PREFIX}/portfolio/positions/{position_id[:8]}...", "✓", details)
                    else:
                        failed_tests.append(("GET", f"{API_PREFIX}/portfolio/positions/{position_id}", details))
                        print_test(f"GET {API_PREFIX}/portfolio/positions/{position_id[:8]}...", "✗", details)
    except:
        pass
    
    # ========================================================================
    # RISK ENDPOINTS
    # ========================================================================
    print_header("RISK MANAGEMENT ENDPOINTS")
    
    tests = [
        (f"GET", f"{API_PREFIX}/risk/metrics", 200),
        (f"GET", f"{API_PREFIX}/risk/limits", 200),
        (f"GET", f"{API_PREFIX}/risk/positions", 200),
        (f"GET", f"{API_PREFIX}/risk/var/monte_carlo", 200, None, {"simulations": 1000, "confidence": 0.95}),
        (f"GET", f"{API_PREFIX}/risk/stress_test", 200),
        (f"GET", f"{API_PREFIX}/risk/correlation", 200),
    ]
    
    for method, endpoint, status, *rest in tests:
        total_tests += 1
        data = rest[0] if len(rest) > 0 else None
        params = rest[1] if len(rest) > 1 else None
        success, details = test_endpoint(method, endpoint, status, data, params)
        if success:
            passed_tests += 1
            print_test(f"{method} {endpoint}", "OK", details)
        else:
            failed_tests.append((method, endpoint, details))
            print_test(f"{method} {endpoint}", "FAIL", details)
    
    # Test POST endpoint for risk check
    total_tests += 1
    risk_check_data = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 1.5,
        "price": 43500.0
    }
    success, details = test_endpoint("POST", f"{API_PREFIX}/risk/check_order", 200, risk_check_data)
    if success:
        passed_tests += 1
        print_test(f"POST {API_PREFIX}/risk/check_order", "✓", details)
    else:
        failed_tests.append(("POST", f"{API_PREFIX}/risk/check_order", details))
        print_test(f"POST {API_PREFIX}/risk/check_order", "✗", details)
    
    # ========================================================================
    # STRATEGY ENDPOINTS
    # ========================================================================
    print_header("STRATEGY ENDPOINTS")
    
    tests = [
        (f"GET", f"{API_PREFIX}/strategy/", 200),
        (f"GET", f"{API_PREFIX}/strategy/", 200, None, {"enabled_only": "true"}),
    ]
    
    for method, endpoint, status, *rest in tests:
        total_tests += 1
        data = rest[0] if len(rest) > 0 else None
        params = rest[1] if len(rest) > 1 else None
        success, details = test_endpoint(method, endpoint, status, data, params)
        if success:
            passed_tests += 1
            print_test(f"{method} {endpoint}", "OK", details)
        else:
            failed_tests.append((method, endpoint, details))
            print_test(f"{method} {endpoint}", "FAIL", details)
    
    # Get a strategy ID for testing
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/strategy/", timeout=5)
        if response.status_code == 200:
            strategies = response.json()
            if strategies and len(strategies) > 0:
                strategy_id = strategies[0].get("strategy_id")
                if strategy_id:
                    strategy_tests = [
                        ("GET", f"{API_PREFIX}/strategy/{strategy_id}", 200),
                        ("GET", f"{API_PREFIX}/strategy/{strategy_id}/signals", 200, None, {"limit": 10}),
                        ("GET", f"{API_PREFIX}/strategy/{strategy_id}/performance", 200),
                        ("POST", f"{API_PREFIX}/strategy/{strategy_id}/run", 200),
                    ]
                    
                    for method, endpoint, status, *rest in strategy_tests:
                        total_tests += 1
                        data = rest[0] if len(rest) > 0 else None
                        params = rest[1] if len(rest) > 1 else None
                        success, details = test_endpoint(method, endpoint, status, data, params)
                        if success:
                            passed_tests += 1
                            print_test(f"{method} {endpoint[:60]}...", "OK", details)
                        else:
                            failed_tests.append((method, endpoint, details))
                            print_test(f"{method} {endpoint[:60]}...", "FAIL", details)
    except:
        pass
    
    # Test creating a new strategy
    total_tests += 1
    new_strategy = {
        "name": "Test Strategy",
        "description": "Test strategy for API testing",
        "strategy_type": "momentum",
        "asset_classes": ["crypto"],
        "parameters": {"lookback": 20},
        "enabled": True
    }
    success, details = test_endpoint("POST", f"{API_PREFIX}/strategy/", 201, new_strategy)
    if success:
        passed_tests += 1
        print_test(f"POST {API_PREFIX}/strategy/", "✓", details)
    else:
        failed_tests.append(("POST", f"{API_PREFIX}/strategy/", details))
        print_test(f"POST {API_PREFIX}/strategy/", "✗", details)
    
    # ========================================================================
    # ORDER ENDPOINTS
    # ========================================================================
    print_header("ORDER ENDPOINTS")
    
    # Test creating an order
    total_tests += 1
    new_order = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": 0.1,
        "broker": "binance"
    }
    success, details = test_endpoint("POST", f"{API_PREFIX}/orders/", 201, new_order)
    order_id = None
    if success:
        passed_tests += 1
        print_test(f"POST {API_PREFIX}/orders/", "✓", details)
        # Try to extract order ID from response
        try:
            response = requests.post(f"{BASE_URL}{API_PREFIX}/orders/", json=new_order, timeout=5)
            if response.status_code == 201:
                order_data = response.json()
                order_id = order_data.get("order_id")
        except:
            pass
    else:
        failed_tests.append(("POST", f"{API_PREFIX}/orders/", details))
        print_test(f"POST {API_PREFIX}/orders/", "✗", details)
    
    # Test listing orders
    tests = [
        (f"GET", f"{API_PREFIX}/orders/", 200),
        (f"GET", f"{API_PREFIX}/orders/", 200, None, {"symbol": "BTCUSDT"}),
        (f"GET", f"{API_PREFIX}/orders/", 200, None, {"status": "PENDING"}),
    ]
    
    for method, endpoint, status, *rest in tests:
        total_tests += 1
        data = rest[0] if len(rest) > 0 else None
        params = rest[1] if len(rest) > 1 else None
        success, details = test_endpoint(method, endpoint, status, data, params)
        if success:
            passed_tests += 1
            print_test(f"{method} {endpoint}", "OK", details)
        else:
            failed_tests.append((method, endpoint, details))
            print_test(f"{method} {endpoint}", "FAIL", details)
    
    # Test order operations if we have an order ID
    if order_id:
        order_tests = [
            ("GET", f"{API_PREFIX}/orders/{order_id}", 200),
            ("PATCH", f"{API_PREFIX}/orders/{order_id}", 200, {"quantity": 0.15}),
            ("POST", f"{API_PREFIX}/orders/{order_id}/execute", 200),
        ]
        
        for method, endpoint, status, *rest in order_tests:
            total_tests += 1
            data = rest[0] if len(rest) > 0 else None
            success, details = test_endpoint(method, endpoint, status, data)
            if success:
                passed_tests += 1
                print_test(f"{method} {endpoint[:60]}...", "✓", details)
            else:
                failed_tests.append((method, endpoint, details))
                print_test(f"{method} {endpoint[:60]}...", "✗", details)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("TEST SUMMARY")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {len(failed_tests)}{Colors.RESET}")
    print(f"Success Rate: {Colors.BOLD}{success_rate:.1f}%{Colors.RESET}\n")
    
    if failed_tests:
        print(f"{Colors.YELLOW}Failed Tests:{Colors.RESET}")
        for method, endpoint, details in failed_tests:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {method} {endpoint}")
            print(f"    {details}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_header("END OF TEST SUITE")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
        exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.RESET}")
        exit(1)
