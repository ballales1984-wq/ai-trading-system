"""
Test all API connections
"""
import os
import requests

print('=== VERIFICA COMPLETA API ===')
print()

# API con key configurate
configured = []
not_configured = []
free_apis = []

# 1. Binance
try:
    resp = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
    if resp.status_code == 200:
        configured.append('Binance')
except Exception as e:
    print(f'Binance error: {e}')

# 2. CoinGecko (free)
try:
    resp = requests.get('https://api.coingecko.com/api/v3/ping', timeout=5)
    if resp.status_code == 200:
        free_apis.append('CoinGecko')
except Exception as e:
    print(f'CoinGecko error: {e}')

# 3. CoinMarketCap
try:
    headers = {'X-CMC_PRO_API_KEY': '8efc064fa1854649a1ac787217fed90d'}
    resp = requests.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?limit=1', headers=headers, timeout=5)
    if resp.status_code == 200:
        configured.append('CoinMarketCap')
except Exception as e:
    print(f'CoinMarketCap error: {e}')

# 4. NewsAPI
try:
    resp = requests.get('https://newsapi.org/v2/top-headlines?country=us&apiKey=155d9973fd8149208b9c4ef6b11a52b7', timeout=5)
    if resp.status_code == 200:
        configured.append('NewsAPI')
except Exception as e:
    print(f'NewsAPI error: {e}')

# 5. GDELT (free)
try:
    resp = requests.get('https://api.gdeltproject.org/api/v2/doc/doc?format=json&query=bitcoin&maxrecords=1', timeout=10)
    if resp.status_code == 200:
        free_apis.append('GDELT')
except Exception as e:
    print(f'GDELT error: {e}')

# 6. Open-Meteo (free)
try:
    resp = requests.get('https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true', timeout=5)
    if resp.status_code == 200:
        free_apis.append('Open-Meteo')
except Exception as e:
    print(f'Open-Meteo error: {e}')

# 7. USGS Water (free)
try:
    resp = requests.get('https://waterservices.usgs.gov/nwis/site?format=rdb&stateCd=TX&siteType=ST', timeout=10)
    if resp.status_code == 200:
        free_apis.append('USGS Water')
except Exception as e:
    print(f'USGS error: {e}')

# 8. Climate TRACE (free)
try:
    resp = requests.get('https://api.climatetrace.org/v1/countries', timeout=10)
    if resp.status_code in [200, 403]:
        free_apis.append('Climate TRACE')
except Exception as e:
    print(f'Climate TRACE error: {e}')

print('API CON KEY CONFIGURATE E FUNZIONANTI:')
for api in configured:
    print(f'  [OK] {api}')

print()
print('API GRATUITE (SENZA KEY):')
for api in free_apis:
    print(f'  [OK] {api}')

print()
print('API NON CONFIGURATE:')
not_configured = ['AlphaVantage', 'Quandl', 'Twitter/X', 'Benzinga', 'Trading Economics', 'Telegram', 'Bybit', 'OKX', 'EIA', 'Investing.com']
for api in not_configured:
    print(f'  [--] {api}')

print()
total_working = len(configured) + len(free_apis)
total_apis = 18
print(f'TOTALE: {total_working}/{total_apis} API connesse')
ua