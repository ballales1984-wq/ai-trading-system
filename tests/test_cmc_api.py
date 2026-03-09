#!/usr/bin/env python3
"""Test CoinMarketCap API key"""
import os

# Test environment variable is set
cmc_key = os.getenv('COINMARKETCAP_API_KEY', '')
print(f"COINMARKETCAP_API_KEY from env: {cmc_key}")

# Now test with app config
from app.core.config import get_settings
settings = get_settings()
print(f"CMC API Key from settings: {settings.coinmarketcap_api_key}")

if settings.coinmarketcap_api_key:
    print("✅ CoinMarketCap API key loaded successfully!")
else:
    print("❌ CoinMarketCap API key NOT loaded")

