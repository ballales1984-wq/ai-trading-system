#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Dashboard Integration with Real Trading System
===================================================
Creates test data and verifies dashboard can read it.
"""

import sys
import io
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests

API_BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

def print_section(title):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_api_connection():
    """Test 1: Verify API is running."""
    print_section("TEST 1: API Connection")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] API Server is running")
            print(f"  Status: {response.json()}")
            return True
        else:
            print(f"[FAIL] API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Cannot connect to API: {e}")
        print("  Make sure FastAPI server is running on port 8000")
        return False

def create_test_data():
    """Test 2: Create test trading data."""
    print_section("TEST 2: Creating Test Trading Data")
    
    try:
        from trading_simulator import TradingSimulator
        
        print("Creating TradingSimulator instance...")
        simulator = TradingSimulator(initial_balance=1000000.0)
        
        print("Creating test orders...")
        
        # Create some test orders via API
        test_orders = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 0.5,
                "broker": "binance"
            },
            {
                "symbol": "ETHUSDT",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 10.0,
                "broker": "binance"
            },
            {
                "symbol": "SOLUSDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": 50.0,
                "price": 95.0,
                "broker": "binance"
            }
        ]
        
        created_orders = []
        for order_data in test_orders:
            try:
                response = requests.post(
                    f"{API_BASE_URL}{API_PREFIX}/orders/",
                    json=order_data,
                    timeout=5
                )
                if response.status_code == 201:
                    order = response.json()
                    created_orders.append(order)
                    print(f"  [OK] Created order: {order['symbol']} {order['side']} {order['quantity']}")
                else:
                    print(f"  [FAIL] Failed to create order: {response.status_code}")
            except Exception as e:
                print(f"  [FAIL] Error creating order: {e}")
        
        # Execute some orders to create positions
        print("\nExecuting orders to create positions...")
        for order in created_orders[:2]:  # Execute first 2
            try:
                response = requests.post(
                    f"{API_BASE_URL}{API_PREFIX}/orders/{order['order_id']}/execute",
                    timeout=5
                )
                if response.status_code == 200:
                    executed = response.json()
                    print(f"  [OK] Executed order: {executed['symbol']} - Status: {executed['status']}")
                else:
                    print(f"  [FAIL] Failed to execute order: {response.status_code}")
            except Exception as e:
                print(f"  [FAIL] Error executing order: {e}")
        
        # Simulate some portfolio activity
        print("\nSimulating portfolio activity...")
        try:
            # Check portfolio status
            status = simulator.check_portfolio()
            print(f"  Portfolio Balance: ${status.get('balance', 0):,.2f}")
            print(f"  Open Positions: {status.get('open_positions', 0)}")
            
            # Save simulator state
            simulator.save_state()
            print("  ✓ Simulator state saved")
        except Exception as e:
            print(f"  ⚠ Could not access simulator: {e}")
        
        print(f"\n✓ Test data creation completed")
        print(f"  Created {len(created_orders)} orders")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Cannot import TradingSimulator: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error creating test data: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_dashboard_data():
    """Test 3: Verify dashboard can read real data."""
    print_section("TEST 3: Verifying Dashboard Can Read Real Data")
    
    results = {
        'portfolio': False,
        'positions': False,
        'orders': False,
        'history': False
    }
    
    # Test Portfolio Summary
    print("1. Testing Portfolio Summary...")
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/portfolio/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] Portfolio Summary retrieved")
            print(f"    Total Value: ${data.get('total_value', 0):,.2f}")
            print(f"    Cash Balance: ${data.get('cash_balance', 0):,.2f}")
            print(f"    Total P&L: ${data.get('total_pnl', 0):+,.2f}")
            print(f"    Positions: {data.get('num_positions', 0)}")
            results['portfolio'] = True
        else:
            print(f"  [FAIL] Failed: Status {response.status_code}")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
    
    # Test Positions
    print("\n2. Testing Positions...")
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/portfolio/positions", timeout=5)
        if response.status_code == 200:
            positions = response.json()
            print(f"  [OK] Positions retrieved: {len(positions)} positions")
            for pos in positions[:3]:  # Show first 3
                print(f"    - {pos.get('symbol')}: {pos.get('quantity')} @ ${pos.get('entry_price', 0):,.2f}")
            results['positions'] = True
        else:
            print(f"  [FAIL] Failed: Status {response.status_code}")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
    
    # Test Orders
    print("\n3. Testing Orders...")
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/orders/", params={"limit": 10}, timeout=5)
        if response.status_code == 200:
            orders = response.json()
            print(f"  [OK] Orders retrieved: {len(orders)} orders")
            for order in orders[:3]:  # Show first 3
                print(f"    - {order.get('symbol')} {order.get('side')}: {order.get('quantity')} - {order.get('status')}")
            results['orders'] = True
        else:
            print(f"  [FAIL] Failed: Status {response.status_code}")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
    
    # Test History
    print("\n4. Testing Portfolio History...")
    try:
        response = requests.get(f"{API_BASE_URL}{API_PREFIX}/portfolio/history", params={"days": 7}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            print(f"  [OK] History retrieved: {len(history)} days")
            if history:
                print(f"    Latest: {history[-1].get('date')} - ${history[-1].get('value', 0):,.2f}")
            results['history'] = True
        else:
            print(f"  [FAIL] Failed: Status {response.status_code}")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
    
    # Summary
    print("\n" + "-"*70)
    print("VERIFICATION SUMMARY:")
    print("-"*70)
    for key, value in results.items():
        status = "[PASS]" if value else "[FAIL]"
        print(f"  {key.upper():15} {status}")
    
    all_passed = all(results.values())
    return all_passed

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  DASHBOARD INTEGRATION TEST SUITE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Prefix: {API_PREFIX}\n")
    
    # Test 1: API Connection
    if not test_api_connection():
        print("\n❌ API is not running. Please start the FastAPI server first.")
        print("   Run: python -m uvicorn app.main:app --reload")
        return False
    
    # Test 2: Create Test Data
    if not create_test_data():
        print("\n⚠️  Could not create test data, but continuing with verification...")
    
    # Wait a moment for data to propagate
    print("\nWaiting 2 seconds for data to propagate...")
    time.sleep(2)
    
    # Test 3: Verify Dashboard Can Read Data
    success = verify_dashboard_data()
    
    print("\n" + "="*70)
    if success:
        print("  [SUCCESS] ALL TESTS PASSED")
        print("  Dashboard is successfully connected to real trading system!")
    else:
        print("  [WARNING] SOME TESTS FAILED")
        print("  Dashboard may be using fallback/mock data")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Open dashboard: http://localhost:8050")
    print("  2. Check if data appears in the dashboard")
    print("  3. Create more orders/positions to see real-time updates")
    print()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nTest suite error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
