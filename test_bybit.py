"""Test Bybit Client"""
import asyncio
import sys
sys.path.insert(0, '.')

from src.external.bybit_client import BybitClient

async def test():
    client = BybitClient()
    await client.connect()
    
    try:
        # Get BTC ticker
        ticker = await client.get_ticker("BTCUSDT")
        print(f"BTC Price: ${ticker.last_price:,.2f}")
        print(f"24h Change: {ticker.price_change_pct_24h:.2f}%")
        
        # Get klines
        klines = await client.get_klines("BTCUSDT", interval="15", limit=3)
        print(f"Klines: {len(klines)}")
        
        print("\nBybit API working!")
    finally:
        await client.disconnect()

asyncio.run(test())
