#!/usr/bin/env python3
"""Test CoinMarketCap API connection"""
import os
from src.external.coinmarketcap_client import CoinMarketCapClient

# Get API key from environment
api_key = os.getenv('COINMARKETCAP_API_KEY', '')
if not api_key:
    print("❌ COINMARKETCAP_API_KEY not set in environment")
    exit(1)

# Test connection
client = CoinMarketCapClient()
print(f"API Key loaded: {bool(client.api_key)}")

# Test connection
result = client.test_connection()
print(f"Connection test: {result}")

# Test get listings
if result:
    listings = client.get_listings(limit=5)
    print(f"\nTop 5 cryptocurrencies:")
    for coin in listings:
        print(f"  {coin['symbol']}: ${coin['price']:.2f}")
    
    # Test get quote for BTC
    btc = client.get_quote("BTC")
    if btc:
        print(f"\nBTC: ${btc['price']:.2f}")
    
    # Test global metrics
    global_data = client.get_global_metrics()
    if global_data:
        print(f"\nGlobal Market Cap: ${global_data.get('total_market_cap', 0):,.0f}")
        print(f"BTC Dominance: {global_data.get('btc_dominance', 0):.1f}%")

