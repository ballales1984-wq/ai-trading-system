#!/usr/bin/env python3
"""Test completo di tutte le API e funzionalità dell'app"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_endpoint(name, method, endpoint, data=None):
    """Testa un endpoint e mostra il risultato"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=5)
        elif method == "POST":
            r = requests.post(url, json=data, timeout=5)
        else:
            r = requests.request(method, url, json=data, timeout=5)
        
        status = "✅" if r.status_code in [200, 201] else "❌"
        print(f"{status} {name}: {r.status_code}")
        
        if r.status_code in [200, 201]:
            try:
                return r.json()
            except:
                return r.text[:100]
        else:
            print(f"   Error: {r.text[:100]}")
            return None
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)[:50]}")
        return None

def main():
    print("=" * 60)
    print("🚀 TEST COMPLETO AI TRADING SYSTEM")
    print("=" * 60)
    print()
    
    # 1. Health Check
    test_endpoint("1. Health Check", "GET", "/health")
    print()
    
    # 2. Portfolio
    test_endpoint("2. Portfolio Summary", "GET", "/portfolio/summary")
    test_endpoint("3. Portfolio Positions", "GET", "/portfolio/positions")
    test_endpoint("4. Portfolio History", "GET", "/portfolio/history?days=7")
    print()
    
    # 3. Market
    test_endpoint("5. Market Prices", "GET", "/market/prices")
    test_endpoint("6. Market BTC/USD", "GET", "/market/BTCUSDT/price")
    print()
    
    # 4. Orders
    orders = test_endpoint("7. List Orders", "GET", "/orders")
    
    # Create order
    order_data = {
        "symbol": "BTCUSDT",
        "side": "BUY", 
        "order_type": "MARKET",
        "quantity": 0.1
    }
    new_order = test_endpoint("8. Create Order", "POST", "/orders", order_data)
    
    if new_order and "order_id" in new_order:
        order_id = new_order["order_id"]
        test_endpoint(f"9. Get Order {order_id[:8]}...", "GET", f"/orders/{order_id}")
    print()
    
    # 5. Risk
    test_endpoint("10. Risk Metrics", "GET", "/risk/metrics")
    print()
    
    # 6. Strategy
    test_endpoint("11. Strategy Signals", "GET", "/strategy/signals")
    print()
    
    # 7. Emergency Stop
    test_endpoint("12. Emergency Status", "GET", "/orders/status/emergency")
    
    # Activate emergency stop
    stop_data = {"reason": "Test emergency stop", "cancel_all_orders": True}
    test_endpoint("13. Emergency Stop", "POST", "/orders/emergency-stop", stop_data)
    
    # Check status again
    test_endpoint("14. Emergency Status (after stop)", "GET", "/orders/status/emergency")
    
    # Resume trading
    test_endpoint("15. Emergency Resume", "POST", "/orders/emergency-resume")
    
    # Final status
    test_endpoint("16. Emergency Status (final)", "GET", "/orders/status/emergency")
    print()
    
    print("=" * 60)
    print("✅ TEST COMPLETATO!")
    print("=" * 60)

if __name__ == "__main__":
    main()
