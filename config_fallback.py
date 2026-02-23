"""
Configurazione fallback per avvio senza API keys
Rende le chiavi opzionali per permettere deploy su Render/Vercel
"""
import os
import warnings

# Fallback per chiavi API mancanti - permette avvio in demo mode
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "demo_key")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "demo_secret")
USE_BINANCE_TESTNET = os.getenv("USE_BINANCE_TESTNET", "true").lower() == "true"

# Avviso se in demo mode
if BINANCE_API_KEY == "demo_key":
    warnings.warn(
        "⚠️  AVVIO IN DEMO MODE: Chiavi API Binance non configurate. "
        "Aggiungi BINANCE_API_KEY e BINANCE_API_SECRET in Environment Variables su Render."
    )
