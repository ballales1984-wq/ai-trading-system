#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Trading Launcher - DATI REALI + SOLDI FINTI
=================================================
Questo script avvia il sistema in modalita:
- DATI REALI: Prezzi live da Binance, CoinGecko, ecc.
- SOLDI FINTI: Trading su Binance Testnet (nessun rischio)

Usage:
    python start_paper_trading.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix encoding per Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Carica configurazione paper trading
env_file = Path(__file__).parent / ".env.paper_trading"
if env_file.exists():
    load_dotenv(env_file)
    print("[OK] Caricato config da .env.paper_trading")
else:
    # Imposta variabili direttamente
    os.environ['SIMULATION_MODE'] = 'false'  # USA DATI REALI
    os.environ['USE_BINANCE_TESTNET'] = 'true'  # USA SOLDI FINTI
    print("[OK] Configurazione paper trading attiva")

# Verifica configurazione
print("\n" + "="*50)
print("PAPER TRADING MODE - DATI REALI")
print("="*50)
print(f"  SIMULATION_MODE: {os.getenv('SIMULATION_MODE', 'false')}")
print(f"  USE_BINANCE_TESTNET: {os.getenv('USE_BINANCE_TESTNET', 'true')}")
print("="*50 + "\n")

# Test connessione API reali
print("Test connessione API reali...")
import requests

try:
    # Test CoinGecko
    r = requests.get('https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=3', timeout=10)
    data = r.json()
    print("\n[OK] COINGECKO - Prezzi reali:")
    for coin in data[:3]:
        print(f"   {coin['symbol'].upper()}: ${coin['current_price']:,.2f}")
except Exception as e:
    print(f"[WARN] CoinGecko: {e}")

try:
    # Test Binance
    r = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=10)
    data = r.json()
    print(f"\n[OK] BINANCE - BTC/USDT: ${float(data['price']):,.2f}")
except Exception as e:
    print(f"[WARN] Binance: {e}")

try:
    # Test Open-Meteo (weather)
    r = requests.get('https://api.open-meteo.com/v1/forecast?latitude=45.46&longitude=9.19&current_weather=true', timeout=10)
    data = r.json()
    temp = data['current_weather']['temperature']
    print(f"\n[OK] OPEN-METEO - Milano: {temp}C")
except Exception as e:
    print(f"[WARN] Open-Meteo: {e}")

print("\n" + "="*50)
print("Avvio Dashboard...")
print("="*50 + "\n")

# Avvia dashboard
import subprocess
subprocess.run([sys.executable, "dashboard.py"])
