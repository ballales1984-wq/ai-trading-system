#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test API Reali - Verifica cosa funziona e cosa no
==================================================
Questo script testa tutte le API per vedere quali funzionano
con configurazione reale (senza API key a pagamento).
"""

import os
import sys
import requests
import json
from datetime import datetime

# Fix encoding per Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("TEST API REALI - AI TRADING SYSTEM")
print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

results = []

def test_api(name, url, parser=None, timeout=10):
    """Testa una API e ritorna il risultato."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if parser:
                result = parser(data)
            else:
                result = "OK"
            results.append((name, "OK", result))
            print(f"\n[OK] {name}")
            print(f"     {result}")
            return True
        else:
            results.append((name, "FAIL", f"HTTP {r.status_code}"))
            print(f"\n[FAIL] {name} - HTTP {r.status_code}")
            return False
    except Exception as e:
        results.append((name, "ERROR", str(e)[:50]))
        print(f"\n[ERROR] {name} - {e}")
        return False

# ============================================
# API GRATIS (SENZA KEY)
# ============================================

print("\n" + "=" * 60)
print("1. API GRATIS (SENZA KEY NECESSARIA)")
print("=" * 60)

# CoinGecko
test_api(
    "CoinGecko - Crypto Prices",
    "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=5",
    lambda d: f"BTC: ${d[0]['current_price']:,.0f}, ETH: ${d[1]['current_price']:,.0f}"
)

# Binance Public
test_api(
    "Binance Public - BTC/USDT",
    "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
    lambda d: f"BTC/USDT: ${float(d['price']):,.0f}"
)

# Binance 24h Stats
test_api(
    "Binance - 24h Stats",
    "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
    lambda d: f"High: ${float(d['highPrice']):,.0f}, Low: ${float(d['lowPrice']):,.0f}, Volume: {float(d['volume']):,.0f}"
)

# CoinCap
test_api(
    "CoinCap - Crypto Data",
    "https://api.coincap.io/v2/assets?limit=3",
    lambda d: f"BTC: ${float(d['data'][0]['priceUsd']):,.0f}, ETH: ${float(d['data'][1]['priceUsd']):,.0f}"
)

# Open-Meteo (Weather)
test_api(
    "Open-Meteo - Weather Milan",
    "https://api.open-meteo.com/v1/forecast?latitude=45.46&longitude=9.19&current_weather=true",
    lambda d: f"Milano: {d['current_weather']['temperature']}C, Wind: {d['current_weather']['windspeed']} km/h"
)

# CoinGecko News (Status updates)
test_api(
    "CoinGecko - Status Updates",
    "https://api.coingecko.com/api/v3/status_updates?per_page=3",
    lambda d: f"Found {len(d.get('status_updates', []))} updates"
)

# Binance Exchange Info
test_api(
    "Binance - Exchange Info",
    "https://api.binance.com/api/v3/exchangeInfo",
    lambda d: f"Server time: {datetime.fromtimestamp(d['serverTime']/1000)}"
)

# ============================================
# API CHE RICHIEDONO KEY (TEST SENZA KEY)
# ============================================

print("\n" + "=" * 60)
print("2. API CHE RICHIEDONO KEY (TEST SENZA KEY)")
print("=" * 60)

# NewsAPI
test_api(
    "NewsAPI - Richiede Key",
    "https://newsapi.org/v2/top-headlines?country=us&apiKey=DEMO_KEY",
    lambda d: f"Status: {d.get('status', 'unknown')}"
)

# Alpha Vantage
test_api(
    "Alpha Vantage - Richiede Key",
    "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=demo",
    lambda d: f"IBM Price: {d.get('Global Quote', {}).get('05. price', 'N/A')}"
)

# Twitter/X (non ha endpoint pubblico gratuito)
print("\n[SKIP] Twitter/X API - Richiede Bearer Token a pagamento")

# ============================================
# TEST BINANCE TESTNET
# ============================================

print("\n" + "=" * 60)
print("3. BINANCE TESTNET (PAPER TRADING)")
print("=" * 60)

# Testnet endpoints
test_api(
    "Binance Testnet - API Status",
    "https://testnet.binance.vision/api/v3/ping",
    lambda d: "Testnet attivo"
)

test_api(
    "Binance Testnet - Server Time",
    "https://testnet.binance.vision/api/v3/time",
    lambda d: f"Server time: {datetime.fromtimestamp(d['serverTime']/1000)}"
)

test_api(
    "Binance Testnet - BTC Price",
    "https://testnet.binance.vision/api/v3/ticker/price?symbol=BTCUSDT",
    lambda d: f"BTC/USDT: ${float(d['price']):,.0f}"
)

# ============================================
# TEST FUNZIONALITA INTERNE
# ============================================

print("\n" + "=" * 60)
print("4. TEST MODULI INTERNI")
print("=" * 60)

# Test import moduli
modules_to_test = [
    ("config", "Configurazione"),
    ("data_collector", "Data Collector"),
    ("decision_engine", "Decision Engine"),
    ("technical_analysis", "Technical Analysis"),
    ("sentiment_news", "Sentiment News"),
    ("ml_predictor", "ML Predictor"),
]

for module_name, display_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"\n[OK] {display_name} ({module_name})")
        results.append((f"Module: {module_name}", "OK", "Importato"))
    except Exception as e:
        print(f"\n[FAIL] {display_name} ({module_name}) - {e}")
        results.append((f"Module: {module_name}", "FAIL", str(e)[:50]))

# ============================================
# RIEPILOGO
# ============================================

print("\n" + "=" * 60)
print("RIEPILOGO RISULTATI")
print("=" * 60)

ok_count = sum(1 for r in results if r[1] == "OK")
fail_count = sum(1 for r in results if r[1] == "FAIL")
error_count = sum(1 for r in results if r[1] == "ERROR")

print(f"\nTotali: {len(results)} test")
print(f"  OK: {ok_count}")
print(f"  FAIL: {fail_count}")
print(f"  ERROR: {error_count}")

print("\nDettaglio:")
for name, status, detail in results:
    status_icon = "[OK]" if status == "OK" else "[FAIL]" if status == "FAIL" else "[ERROR]"
    print(f"  {status_icon} {name}: {detail[:40]}")

# ============================================
# CONCLUSIONI
# ============================================

print("\n" + "=" * 60)
print("CONCLUSIONI")
print("=" * 60)

print("""
API GRATUIRE FUNZIONANTI:
- CoinGecko (crypto prices, news)
- Binance Public (market data)
- CoinCap (crypto data)
- Open-Meteo (weather)
- Binance Testnet (paper trading)

API CHE RICHIEDONO KEY:
- NewsAPI (news feed completo)
- Alpha Vantage (stocks, forex)
- Twitter/X (sentiment)

CONFIGURAZIONE CONSIGLIATA:
- SIMULATION_MODE=false (dati reali)
- USE_BINANCE_TESTNET=true (soldi finti)
- API keys opzionali per funzionalita extra
""")

print("=" * 60)
