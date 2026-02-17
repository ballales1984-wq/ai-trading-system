"""
Binance Research Module
Fetches market analysis and research data from Binance
"""

import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Binance API endpoints
BINANCE_SPOT_API = "https://api.binance.com"
BINANCE_RESEARCH_API = "https://research.binance.com"


class BinanceResearch:
    """
    Binance Research data fetcher
    Provides market analysis, reports, and insights
    """
    
    def __init__(self):
        """Initialize Binance Research module"""
        self.base_url = BINANCE_SPOT_API
        self.research_url = BINANCE_RESEARCH_API
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_market_summary(self, symbol: str) -> Dict:
        """
        Get market summary for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Dict with market data
        """
        try:
            # Remove / from symbol if present
            symbol = symbol.replace('/', '').upper()
            
            # Get 24hr ticker
            ticker = self.session.get(
                f"{self.base_url}/api/v3/ticker/24hr",
                params={'symbol': symbol},
                timeout=10
            )
            
            if ticker.status_code == 200:
                data = ticker.json()
                return {
                    'status': 'OK',
                    'symbol': symbol,
                    'price': float(data.get('lastPrice', 0)),
                    'price_change': float(data.get('priceChange', 0)),
                    'price_change_percent': float(data.get('priceChangePercent', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0)),
                    'volume_24h': float(data.get('volume', 0)),
                    'quote_volume_24h': float(data.get('quoteVolume', 0)),
                    'trades_24h': int(data.get('count', 0)),
                }
            else:
                return {'status': 'ERROR', 'message': f'API error: {ticker.status_code}'}
                
        except Exception as e:
            logger.warning(f"Error fetching market summary for {symbol}: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get order book depth for a symbol
        
        Args:
            symbol: Trading pair
            limit: Number of bids/asks (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dict with order book data
        """
        try:
            symbol = symbol.replace('/', '').upper()
            
            response = self.session.get(
                f"{self.base_url}/api/v3/depth",
                params={'symbol': symbol, 'limit': limit},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'OK',
                    'symbol': symbol,
                    'bids': [[float(p), float(q)] for p, q in data.get('bids', [])],
                    'asks': [[float(p), float(q)] for p, q in data.get('asks', [])],
                }
            else:
                return {'status': 'ERROR', 'message': f'API error: {response.status_code}'}
                
        except Exception as e:
            logger.warning(f"Error fetching order book for {symbol}: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get recent trades for a symbol
        
        Args:
            symbol: Trading pair
            limit: Number of trades to retrieve
            
        Returns:
            List of recent trades
        """
        try:
            symbol = symbol.replace('/', '').upper()
            
            response = self.session.get(
                f"{self.base_url}/api/v3/trades",
                params={'symbol': symbol, 'limit': limit},
                timeout=10
            )
            
            if response.status_code == 200:
                trades = response.json()
                return [
                    {
                        'id': t.get('id'),
                        'price': float(t.get('price', 0)),
                        'qty': float(t.get('qty', 0)),
                        'time': datetime.fromtimestamp(t.get('time', 0) / 1000),
                        'isBuyerMaker': t.get('isBuyerMaker'),
                    }
                    for t in trades
                ]
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Error fetching recent trades for {symbol}: {e}")
            return []
    
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price or None
        """
        try:
            symbol = symbol.replace('/', '').upper()
            
            response = self.session.get(
                f"{self.base_url}/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('price', 0))
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching price for {symbol}: {e}")
            return None
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Get all available trading pair prices
        
        Returns:
            Dict mapping symbols to prices
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/v3/ticker/price",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {t.get('symbol'): float(t.get('price', 0)) for t in data}
            return {}
            
        except Exception as e:
            logger.warning(f"Error fetching all prices: {e}")
            return {}
    
    def get_24hr_tickers(self) -> List[Dict]:
        """
        Get 24hr ticker data for all symbols
        
        Returns:
            List of ticker data
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/v3/ticker/24hr",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Filter for USDT pairs only
                usdt_pairs = [t for t in data if t.get('symbol', '').endswith('USDT')]
                return [
                    {
                        'symbol': t.get('symbol'),
                        'price': float(t.get('lastPrice', 0)),
                        'price_change': float(t.get('priceChange', 0)),
                        'price_change_percent': float(t.get('priceChangePercent', 0)),
                        'high': float(t.get('highPrice', 0)),
                        'low': float(t.get('lowPrice', 0)),
                        'volume': float(t.get('volume', 0)),
                        'quote_volume': float(t.get('quoteVolume', 0)),
                    }
                    for t in usdt_pairs
                ]
            return []
            
        except Exception as e:
            logger.warning(f"Error fetching 24hr tickers: {e}")
            return []
    
    def get_market_cap(self) -> Dict:
        """
        Get cryptocurrency market cap data
        
        Returns:
            Dict with market cap information
        """
        try:
            # Get BTC dominance and total market data
            tickers = self.get_24hr_tickers()
            
            if not tickers:
                return {'status': 'ERROR', 'message': 'No data available'}
            
            # Calculate total volume
            total_volume = sum(t.get('quote_volume', 0) for t in tickers)
            
            # Get BTC and ETH volumes
            btc_data = next((t for t in tickers if t.get('symbol') == 'BTCUSDT'), {})
            eth_data = next((t for t in tickers if t.get('symbol') == 'ETHUSDT'), {})
            
            btc_volume = btc_data.get('quote_volume', 0)
            eth_volume = eth_data.get('quote_volume', 0)
            
            btc_dominance = (btc_volume / total_volume * 100) if total_volume > 0 else 0
            eth_dominance = (eth_volume / total_volume * 100) if total_volume > 0 else 0
            
            return {
                'status': 'OK',
                'total_volume_24h': total_volume,
                'btc_volume_24h': btc_volume,
                'eth_volume_24h': eth_volume,
                'btc_dominance': btc_dominance,
                'eth_dominance': eth_dominance,
                'num_pairs': len(tickers),
                'timestamp': datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.warning(f"Error fetching market cap: {e}")
            return {'status': 'ERROR', 'message': str(e)}


# Singleton instance
_research_instance = None

def get_binance_research() -> BinanceResearch:
    """Get singleton instance of BinanceResearch"""
    global _research_instance
    if _research_instance is None:
        _research_instance = BinanceResearch()
    return _research_instance


if __name__ == "__main__":
    # Test the module
    research = BinanceResearch()
    
    print("=" * 50)
    print("BINANCE RESEARCH DATA")
    print("=" * 50)
    
    # Get BTC summary
    btc = research.get_market_summary('BTCUSDT')
    print(f"\nBTC/USDT:")
    print(f"   Price: ${btc.get('price', 0):,.2f}")
    print(f"   24h Change: {btc.get('price_change_percent', 0):+.2f}%")
    print(f"   Volume: ${btc.get('quote_volume_24h', 0):,.0f}")
    
    # Get market cap
    market = research.get_market_cap()
    if market.get('status') == 'OK':
        print(f"\nMarket Cap:")
        print(f"   BTC Dominance: {market.get('btc_dominance', 0):.1f}%")
        print(f"   ETH Dominance: {market.get('eth_dominance', 0):.1f}%")
        print(f"   Total Pairs: {market.get('num_pairs', 0)}")
