#!/usr/bin/env python
"""Quick test script to verify Binance API connection"""
import os
import sys

# Import from config
import config

print("="*50)
print("BINANCE API CONNECTION TEST")
print("="*50)
print(f"API Key: {config.BINANCE_API_KEY[:10]}...")
print(f"Testnet: {config.USE_BINANCE_TESTNET}")
print()

try:
    import ccxt
    
    # Create exchange instance using config values
    exchange = ccxt.binance({
        'apiKey': config.BINANCE_API_KEY,
        'secret': config.BINANCE_SECRET_KEY,
        'testnet': config.USE_BINANCE_TESTNET,
    })
    
    # Fetch ticker
    print("Fetching BTC/USDT price...")
    ticker = exchange.fetch_ticker('BTC/USDT')
    
    print(f"SUCCESS!")
    print(f"   BTC/USDT: ${ticker['last']:,.2f}")
    print(f"   24h High: ${ticker['high']:,.2f}")
    print(f"   24h Low:  ${ticker['low']:,.2f}")
    print(f"   24h Vol:  {ticker['quoteVolume']:,.0f} USDT")
    
except ccxt.AuthenticationError as e:
    print(f"Authentication Error: {e}")
    print("   Check your API keys in .env file")
except ccxt.NetworkError as e:
    print(f"Network Error: {e}")
    print("   Check your internet connection")
except Exception as e:
    print(f"Error: {e}")

print()
print("="*50)

