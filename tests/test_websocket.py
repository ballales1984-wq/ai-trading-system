#!/usr/bin/env python3
"""Quick WebSocket test for Binance"""

import websocket
import json
import time

def on_message(ws, message):
    print(f"Received: {message[:200]}...")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket opened")
    
    # For single stream, just subscribe via URL - no message needed
    # The stream is already in the URL path
    pass

# Test different endpoints
endpoints = [
    "wss://stream.binance.com:9443/ws/btcusdt@kline_1h",
    "wss://stream.binance.com:9443/ws/ethusdt@kline_1h",
    "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1h",
]

print("Testing Binance WebSocket connections...")
print("=" * 60)

for url in endpoints:
    print(f"\nTesting: {url}")
    try:
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run for 5 seconds
        ws.run_forever(ping_interval=10, ping_timeout=5)
        
    except Exception as e:
        print(f"Exception: {e}")
    
    time.sleep(1)

print("\n" + "=" * 60)
print("Test completed")
