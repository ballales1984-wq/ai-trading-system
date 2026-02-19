"""
Data Collector Module
Collects crypto and commodity token data from exchanges
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os

import pandas as pd
import numpy as np
import requests

# Try to import ccxt, fallback to manual implementation
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not available. Using simulated data mode.")

import config

# Configure logging
logger = logging.getLogger(__name__)


# ==================== DATA STRUCTURES ====================

@dataclass
class PriceData:
    """Single price data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "exchange"


@dataclass
class MarketData:
    """Complete market data for an asset"""
    symbol: str
    name: str
    current_price: float
    price_change_24h: float
    price_change_percent_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    market_cap: float
    timestamps: List[datetime] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    candles: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class CorrelationData:
    """Correlation data between assets"""
    asset1: str
    asset2: str
    correlation: float
    timeframe: str
    sample_size: int
    timestamp: datetime


# ==================== DATA COLLECTOR CLASS ====================

class DataCollector:
    """
    Collects market data from crypto exchanges and commodity token markets.
    Supports both real API calls and simulation mode.
    """
    
    def __init__(self, exchange: str = 'binance', simulation: bool = True):
        """
        Initialize the data collector.
        
        Args:
            exchange: Exchange name to use (binance, kucoin, etc.)
            simulation: Use simulated data instead of real API
        """
        self.exchange_name = exchange
        self.simulation = simulation
        self.exchange = None
        self.price_cache = {}
        self.last_update = {}
        
        # Initialize exchange if ccxt available and not in simulation mode
        if CCXT_AVAILABLE and not self.simulation:
            self._init_exchange()
        
        # Initialize CoinMarketCap client for enriched market data
        self._cmc_client = None
        try:
            from src.external.coinmarketcap_client import CoinMarketCapClient
            self._cmc_client = CoinMarketCapClient()
            if self._cmc_client.test_connection():
                logger.info("CoinMarketCap API available for data enrichment")
            else:
                self._cmc_client = None
        except Exception:
            pass
        
        logger.info(f"DataCollector initialized (simulation={simulation}, exchange={exchange})")
    
    def _init_exchange(self):
        """Initialize the exchange API"""
        try:
            if self.exchange_name == 'binance':
                # Check if we have API keys for authenticated requests
                if config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY:
                    self.exchange = ccxt.binance({
                        'apiKey': config.BINANCE_API_KEY,
                        'secret': config.BINANCE_SECRET_KEY,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'},
                        # Use testnet if enabled
                        'testnet': config.USE_BINANCE_TESTNET,
                    })
                    logger.info("Binance API initialized with authentication")
                else:
                    # Public API (rate limited)
                    self.exchange = ccxt.binance({
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'}
                    })
                    logger.info("Binance API initialized (public, no auth)")
            elif self.exchange_name == 'kucoin':
                self.exchange = ccxt.kucoin({'enableRateLimit': True})
            elif self.exchange_name == 'bybit':
                self.exchange = ccxt.bybit({'enableRateLimit': True})
            elif self.exchange_name == 'coinbase':
                self.exchange = ccxt.coinbase({'enableRateLimit': True})
            elif self.exchange_name == 'kraken':
                self.exchange = ccxt.kraken({'enableRateLimit': True})
            elif self.exchange_name == 'okx':
                self.exchange = ccxt.okx({'enableRateLimit': True})
            elif self.exchange_name == 'gateio':
                self.exchange = ccxt.gateio({'enableRateLimit': True})
            elif self.exchange_name == 'huobi':
                self.exchange = ccxt.huobi({'enableRateLimit': True})
            else:
                self.exchange = ccxt.binance({'enableRateLimit': True})
            
            # Test connection
            self.exchange.fetch_time()
            logger.info(f"Connected to {self.exchange_name}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to {self.exchange_name}: {e}. Using simulation mode.")
            self.simulation = True
    
    # ==================== PRICE FETCHING ====================
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """
        Fetch current price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price or None if failed
        """
        # Check cache first
        cache_key = f"{symbol}_{self.exchange_name}"
        if cache_key in self.price_cache:
            cached_time, cached_price = self.price_cache[cache_key]
            if time.time() - cached_time < 60:  # Use cache for 1 minute
                return cached_price
        
        if self.simulation:
            price = self._simulate_price(symbol)
        else:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                self.price_cache[cache_key] = (time.time(), price)
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                return None
        
        return price
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                    limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.simulation:
            return self._simulate_ohlcv(symbol, timeframe, limit)
        
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return self._simulate_ohlcv(symbol, timeframe, limit)
    
    def fetch_market_data(self, symbol: str, name: str = "") -> MarketData:
        """
        Fetch complete market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            name: Display name for the asset
            
        Returns:
            MarketData object
        """
        current_price = self.fetch_current_price(symbol)
        
        if current_price is None:
            current_price = 0.0
        
        # Generate simulated 24h change for demo
        if self.simulation:
            price_change = current_price * random.uniform(-0.05, 0.05)
            price_change_percent = random.uniform(-5, 5)
            volume = random.uniform(1_000_000, 100_000_000)
            high_24h = current_price * 1.03
            low_24h = current_price * 0.97
        else:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price_change = ticker['change']
                price_change_percent = ticker['percentage']
                volume = ticker['quoteVolume']
                high_24h = ticker['high']
                low_24h = ticker['low']
            except:
                price_change = 0
                price_change_percent = 0
                volume = 0
                high_24h = current_price
                low_24h = current_price
        
        market_cap = 0
        
        # Enrich with CoinMarketCap data if available
        if self._cmc_client:
            try:
                coin_sym = symbol.replace('USDT', '').replace('/', '').replace('USD', '')
                cmc_data = self._cmc_client.get_quote(coin_sym)
                if cmc_data:
                    market_cap = cmc_data.get('market_cap', 0) or 0
                    # Use CMC price if exchange price unavailable
                    if current_price == 0 and cmc_data.get('price'):
                        current_price = cmc_data['price']
            except Exception as e:
                logger.debug(f"CMC enrichment failed for {symbol}: {e}")
        
        return MarketData(
            symbol=symbol,
            name=name or symbol,
            current_price=current_price,
            price_change_24h=price_change,
            price_change_percent_24h=price_change_percent,
            volume_24h=volume,
            high_24h=high_24h,
            low_24h=low_24h,
            market_cap=market_cap
        )
    
    def fetch_multiple_markets(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch market data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Dictionary of symbol -> MarketData
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_market_data(symbol)
                results[symbol] = data
                
                # Rate limiting
                if not self.simulation:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return results
    
    # ==================== HISTORICAL DATA ====================
    
    def fetch_historical_data(self, symbol: str, days: int = 30, 
                              timeframe: str = '1h') -> pd.DataFrame:
        """
        Fetch historical data for analysis.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days to fetch
            timeframe: Timeframe for candles
            
        Returns:
            DataFrame with historical data
        """
        limit = days * 24 if timeframe == '1h' else days * 24 * 4
        
        return self.fetch_ohlcv(symbol, timeframe, min(limit, 500))
    
    def get_price_history(self, symbol: str, hours: int = 24) -> List[float]:
        """
        Get price history for correlation analysis.
        
        Args:
            symbol: Trading pair symbol
            hours: Number of hours of history
            
        Returns:
            List of historical prices
        """
        df = self.fetch_ohlcv(symbol, '1h', hours)
        
        if df is not None and not df.empty:
            return df['close'].tolist()
        
        return []
    
    # ==================== CORRELATION ANALYSIS ====================
    
    def calculate_correlation(self, symbol1: str, symbol2: str, 
                             hours: int = 24) -> CorrelationData:
        """
        Calculate correlation between two assets.
        
        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol
            hours: Number of hours for calculation
            
        Returns:
            CorrelationData object
        """
        prices1 = self.get_price_history(symbol1, hours)
        prices2 = self.get_price_history(symbol2, hours)
        
        # Ensure same length
        min_len = min(len(prices1), len(prices2))
        
        if min_len < 2:
            return CorrelationData(
                asset1=symbol1,
                asset2=symbol2,
                correlation=0.0,
                timeframe=f"{hours}h",
                sample_size=min_len,
                timestamp=datetime.now()
            )
        
        # Calculate correlation
        prices1 = prices1[:min_len]
        prices2 = prices2[:min_len]
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        
        return CorrelationData(
            asset1=symbol1,
            asset2=symbol2,
            correlation=correlation if not np.isnan(correlation) else 0.0,
            timeframe=f"{hours}h",
            sample_size=min_len,
            timestamp=datetime.now()
        )
    
    def calculate_correlation_matrix(self, symbols: List[str], 
                                     hours: int = 24) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.
        
        Args:
            symbols: List of asset symbols
            hours: Number of hours for calculation
            
        Returns:
            DataFrame with correlation matrix
        """
        # Get price histories
        price_data = {}
        
        for symbol in symbols:
            prices = self.get_price_history(symbol, hours)
            if prices:
                # Use percentage returns instead of raw prices
                returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else []
                price_data[symbol] = returns
        
        # Create returns DataFrame
        min_len = min(len(v) for v in price_data.values()) if price_data else 0
        
        if min_len < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame({
            symbol: data[:min_len] 
            for symbol, data in price_data.items()
        })
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    # ==================== SIMULATION HELPERS ====================
    
    def _simulate_price(self, symbol: str) -> float:
        """Generate simulated price for a symbol"""
        # Use symbol hash for consistent base price
        base_prices = {
            'BTC/USDT': 67500.0,
            'ETH/USDT': 3450.0,
            'XRP/USDT': 0.62,
            'SOL/USDT': 145.0,
            'ADA/USDT': 0.58,
            'DOT/USDT': 7.50,
            'AVAX/USDT': 38.0,
            'MATIC/USDT': 0.85,
            'PAXG/USDT': 2030.0,
            'XAUT/USDT': 2025.0,
            'STETH/USDT': 3450.0,
            'PEUR/USDT': 1.08,
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add some randomness
        volatility = config.SIMULATED_VOLATILITY
        change = random.gauss(0, volatility)
        
        return base_price * (1 + change)
    
    def _simulate_ohlcv(self, symbol: str, timeframe: str, 
                        limit: int) -> pd.DataFrame:
        """Generate simulated OHLCV data"""
        base_price = self._simulate_price(symbol)
        
        # Generate timestamps
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, 
            '4h': 240, '1d': 1440, '1w': 10080
        }
        
        interval = timeframe_minutes.get(timeframe, 60)
        
        # Generate candle data
        data = []
        current_price = base_price
        
        for i in range(limit):
            timestamp = datetime.now() - timedelta(minutes=interval * (limit - i))
            
            # Random walk for price
            change = random.gauss(0, 0.01)  # 1% volatility
            current_price *= (1 + change)
            
            # Generate OHLC from close price
            high = current_price * random.uniform(1.001, 1.02)
            low = current_price * random.uniform(0.98, 0.999)
            open_price = random.uniform(low, high)
            
            volume = random.uniform(1000, 100000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    # ==================== UTILITY METHODS ====================
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return list(config.CRYPTO_SYMBOLS.values()) + \
               [v['api_symbol'] for v in config.COMMODITY_TOKENS.values()]
    
    def get_commodity_symbols(self) -> List[str]:
        """Get list of commodity-backed token symbols"""
        return [v['api_symbol'] for v in config.COMMODITY_TOKENS.values()]
    
    def save_data_to_file(self, data: pd.DataFrame, filename: str):
        """Save data to CSV file"""
        filepath = config.DATA_DIR / filename
        data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
    
    def load_data_from_file(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        filepath = config.DATA_DIR / filename
        
        if filepath.exists():
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        return None


# ==================== STANDALONE FUNCTIONS ====================

def get_collector(exchange: str = 'binance', simulation: bool = True) -> DataCollector:
    """
    Factory function to create a DataCollector instance.
    
    Args:
        exchange: Exchange name
        simulation: Use simulation mode
        
    Returns:
        DataCollector instance
    """
    return DataCollector(exchange=exchange, simulation=simulation)


if __name__ == "__main__":
    # Test the data collector
    logging.basicConfig(level=logging.INFO)
    
    collector = DataCollector(simulation=True)
    
    print("\n" + "="*60)
    print("DATA COLLECTOR TEST")
    print("="*60)
    
    # Test fetching prices
    print("\n📊 Fetching crypto prices...")
    for symbol in ['BTC/USDT', 'ETH/USDT', 'PAXG/USDT']:
        price = collector.fetch_current_price(symbol)
        print(f"  {symbol}: ${price:,.2f}" if price else f"  {symbol}: Failed")
    
    # Test OHLCV
    print("\n📈 Fetching OHLCV data...")
    df = collector.fetch_ohlcv('BTC/USDT', '1h', 24)
    if df is not None:
        print(f"  Got {len(df)} candles")
        print(f"  Latest: {df['close'].iloc[-1]:,.2f}")
    
    # Test correlation
    print("\n🔗 Calculating correlation...")
    corr = collector.calculate_correlation('BTC/USDT', 'ETH/USDT', 24)
    print(f"  BTC/ETH correlation: {corr.correlation:.4f}")
    
    print("\n✅ Test complete!")

