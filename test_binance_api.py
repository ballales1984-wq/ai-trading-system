#!/usr/bin/env python3
"""Quick test to check Binance API connection"""

import ccxt
import config
import sys

def main():
    print('=== Testing Binance API Connection ===')
    print(f'API Key present: {bool(config.BINANCE_API_KEY)}')
    print(f'Secret Key present: {bool(config.BINANCE_SECRET_KEY)}')
    print(f'Testnet: {config.USE_BINANCE_TESTNET}')
    print()

    try:
        exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'testnet': config.USE_BINANCE_TESTNET,
        })
        
        # Test connection
        print('Testing connection to Binance...')
        time = exchange.fetch_time()
        print(f'Binance connection OK! Server time: {time}')
        
        # Try to get BTC price
        print('Fetching BTC/USDT price...')
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f'BTC/USDT price: ${ticker["last"]:,.2f}')
        
        print('\n✅ Binance API is working!')
        return 0
        
    except ccxt.AuthenticationError as e:
        print(f'❌ Authentication Error: {e}')
        print('   The API keys may be invalid or have expired.')
        return 1
    except ccxt.NetworkError as e:
        print(f'❌ Network Error: {e}')
        print('   Check your internet connection.')
        return 1
    except Exception as e:
        print(f'❌ Error: {type(e).__name__}: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main())

