#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Create and Manage Orders
=================================
Demonstrates how to use the Orders API endpoints.
"""

import sys
import io
import requests
import json
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

def print_json(data):
    """Pretty print JSON."""
    print(json.dumps(data, indent=2, default=str))

def create_order():
    """Example: Create a new order."""
    print("="*70)
    print("CREATING A NEW ORDER")
    print("="*70)
    
    order_data = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": 0.1,
        "broker": "binance"
    }
    
    print("\nRequest:")
    print_json(order_data)
    
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/orders/",
        json=order_data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 201:
        print("\n[OK] Order created successfully!")
        print("\nResponse:")
        print_json(response.json())
        return response.json().get("order_id")
    else:
        print("\n[ERROR] Error creating order:")
        print_json(response.json())
        return None

def create_limit_order():
    """Example: Create a limit order."""
    print("\n" + "="*70)
    print("CREATING A LIMIT ORDER")
    print("="*70)
    
    order_data = {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": 1.5,
        "price": 2300.0,
        "time_in_force": "GTC",
        "broker": "binance"
    }
    
    print("\nRequest:")
    print_json(order_data)
    
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/orders/",
        json=order_data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 201:
        print("\n[OK] Limit order created successfully!")
        print("\nResponse:")
        print_json(response.json())
        return response.json().get("order_id")
    else:
        print("\n[ERROR] Error creating order:")
        print_json(response.json())
        return None

def list_orders(symbol=None, status=None, limit=10):
    """Example: List orders with filters."""
    print("\n" + "="*70)
    print("LISTING ORDERS")
    print("="*70)
    
    params = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    if status:
        params["status"] = status
    
    print(f"\nFilters: {params}")
    
    response = requests.get(
        f"{BASE_URL}{API_PREFIX}/orders/",
        params=params
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        orders = response.json()
        print(f"\n[OK] Found {len(orders)} orders")
        print("\nOrders:")
        for order in orders[:5]:  # Show first 5
            print(f"  - {order['order_id'][:8]}... | {order['symbol']} | {order['side']} | {order['status']}")
        return orders
    else:
        print("\n[ERROR] Error listing orders:")
        print_json(response.json())
        return []

def get_order(order_id):
    """Example: Get order by ID."""
    print("\n" + "="*70)
    print(f"GETTING ORDER: {order_id[:8]}...")
    print("="*70)
    
    response = requests.get(
        f"{BASE_URL}{API_PREFIX}/orders/{order_id}"
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n[OK] Order retrieved:")
        print_json(response.json())
        return response.json()
    else:
        print("\n[ERROR] Error getting order:")
        print_json(response.json())
        return None

def update_order(order_id, quantity=None, price=None):
    """Example: Update an order."""
    print("\n" + "="*70)
    print(f"UPDATING ORDER: {order_id[:8]}...")
    print("="*70)
    
    update_data = {}
    if quantity is not None:
        update_data["quantity"] = quantity
    if price is not None:
        update_data["price"] = price
    
    print("\nUpdate data:")
    print_json(update_data)
    
    response = requests.patch(
        f"{BASE_URL}{API_PREFIX}/orders/{order_id}",
        json=update_data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n[OK] Order updated:")
        print_json(response.json())
        return response.json()
    else:
        print("\n[ERROR] Error updating order:")
        print_json(response.json())
        return None

def execute_order(order_id):
    """Example: Execute an order."""
    print("\n" + "="*70)
    print(f"EXECUTING ORDER: {order_id[:8]}...")
    print("="*70)
    
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/orders/{order_id}/execute"
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n[OK] Order executed:")
        print_json(response.json())
        return response.json()
    else:
        print("\n[ERROR] Error executing order:")
        print_json(response.json())
        return None

def cancel_order(order_id):
    """Example: Cancel an order."""
    print("\n" + "="*70)
    print(f"CANCELLING ORDER: {order_id[:8]}...")
    print("="*70)
    
    response = requests.delete(
        f"{BASE_URL}{API_PREFIX}/orders/{order_id}"
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 204:
        print("\n[OK] Order cancelled successfully")
        return True
    else:
        print("\n[ERROR] Error cancelling order:")
        print_json(response.json())
        return False

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ORDERS API - EXAMPLES")
    print("="*70)
    print(f"\nAPI Base URL: {BASE_URL}{API_PREFIX}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Create a market order
    order_id_1 = create_order()
    
    # 2. Create a limit order
    order_id_2 = create_limit_order()
    
    # 3. List all orders
    orders = list_orders(limit=10)
    
    # 4. Get specific order
    if order_id_1:
        get_order(order_id_1)
    
    # 5. Update order (if we have a pending order)
    if order_id_2:
        update_order(order_id_2, quantity=2.0, price=2350.0)
    
    # 6. Execute order
    if order_id_1:
        execute_order(order_id_1)
    
    # 7. List orders filtered by status
    list_orders(status="FILLED", limit=5)
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
