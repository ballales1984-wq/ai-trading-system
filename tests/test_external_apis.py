"""
Test External API Clients
=========================
Integration tests for all external API clients.
"""

import asyncio
import os
import sys
import traceback
sys.path.insert(0, '.')


async def test_binance():
    """Test Binance client."""
    print("\n=== TESTING: Binance ===")
    try:
        from src.external.market_data_apis import BinanceMarketClient
        client = BinanceMarketClient(testnet=True)
        health = await client.health_check()
        print(f"Health: {'OK' if health else 'FAIL'}")
        if not health:
            return False
        records = await client.fetch(symbol="BTCUSDT", interval="1h", limit=5)
        print(f"Records: {len(records)}")
        return len(records) > 0
    except Exception as e:
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return False


async def test_coingecko():
    """Test CoinGecko client."""
    print("\n=== TESTING: CoinGecko ===")
    try:
        from src.external.market_data_apis import CoinGeckoClient
        client = CoinGeckoClient()
        health = await client.health_check()
        print(f"Health: {'OK' if health else 'FAIL'}")
        if not health:
            return False
        records = await client.fetch(symbol="BTCUSDT", limit=5)
        print(f"Records: {len(records)}")
        return len(records) > 0
    except Exception as e:
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return False


async def test_bybit():
    """Test Bybit client."""
    print("\n=== TESTING: Bybit ===")
    try:
        from src.external.bybit_client import BybitClient
        client = BybitClient()
        await client.connect()
        try:
            ticker = await client.get_ticker("BTCUSDT")
            print(f"BTC Price: ${ticker.last_price:,.2f}")
            return True
        finally:
            await client.disconnect()
    except Exception as e:
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return False


async def test_okx():
    """Test OKX client."""
    print("\n=== TESTING: OKX ===")
    try:
        from src.external.okx_client import OKXClient
        client = OKXClient()
        await client.connect()
        try:
            ticker = await client.get_ticker("BTC-USDT")
            print(f"BTC Price: ${ticker.last_price:,.2f}")
            return True
        finally:
            await client.disconnect()
    except Exception as e:
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return False


async def test_registry():
    """Test API Registry."""
    print("\n=== TESTING: API Registry ===")
    try:
        from src.external.api_registry import APIRegistry
        from src.external.market_data_apis import create_market_data_clients
        
        registry = APIRegistry()
        clients = create_market_data_clients(binance_testnet=True)
        for c in clients:
            registry.register(c)
        
        print(f"Registered: {len(registry._clients)} clients")
        health = await registry.health_check_all()
        for name, ok in health.items():
            print(f"  {name}: {'OK' if ok else 'FAIL'}")
        return True
    except Exception as e:
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return False


async def run_all():
    print("="*60)
    print("STARTING API TESTS")
    print("="*60)
    
    results = {}
    
    tests = [
        ("Binance", test_binance),
        ("CoinGecko", test_coingecko),
        ("Bybit", test_bybit),
        ("OKX", test_okx),
        ("Registry", test_registry),
    ]
    
    for name, test_fn in tests:
        print(f"\n>>> Running {name}...")
        try:
            results[name] = await test_fn()
        except Exception as e:
            print(f"FATAL ERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {name:15s}: {status}")
        if ok:
            passed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f">>> Total: {passed}/{len(results)} PASSED <<<")
    print("="*60)
    
    if failed > 0:
        print(f"\n⚠️  FAILED ({failed}):")
        for name, ok in results.items():
            if not ok:
                print(f"  - {name}")
    else:
        print("\n🎉 ALL TESTS PASSED!")
    
    return passed == len(results)


if __name__ == "__main__":
    asyncio.run(run_all())

