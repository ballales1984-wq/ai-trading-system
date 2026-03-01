"""
Configuration Settings for Crypto Commodity Trading System
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== PROJECT PATHS ====================
PROJECT_ROOT: Path = Path(__file__).parent
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CACHE_DIR: Path = PROJECT_ROOT / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ==================== CRYPTO CONFIGURATION ====================
CRYPTO_SYMBOLS: Dict[str, str] = {
    # Major cryptocurrencies
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'XRP': 'XRP/USDT',
    'SOL': 'SOL/USDT',
    'ADA': 'ADA/USDT',
    'DOT': 'DOT/USDT',
    'AVAX': 'AVAX/USDT',
    'MATIC': 'MATIC/USDT',
    # Additional cryptocurrencies
    'BNB': 'BNB/USDT',
    'DOGE': 'DOGE/USDT',
    'LINK': 'LINK/USDT',
    'ATOM': 'ATOM/USDT',
    'UNI': 'UNI/USDT',
    'LTC': 'LTC/USDT',
    'NEAR': 'NEAR/USDT',
    'APT': 'APT/USDT',
    'ARB': 'ARB/USDT',
    'OP': 'OP/USDT',
    'INJ': 'INJ/USDT',
    'SUI': 'SUI/USDT',
    'SEI': 'SEI/USDT',
    'TIA': 'TIA/USDT',
    # New additions
    'ETC': 'ETC/USDT',
    'XLM': 'XLM/USDT',
    'FIL': 'FIL/USDT',
    'HBAR': 'HBAR/USDT',
    'VET': 'VET/USDT',
    'ALGO': 'ALGO/USDT',
    'FTM': 'FTM/USDT',
    'PEPE': 'PEPE/USDT',
    'SHIB': 'SHIB/USDT',
    'TRX': 'TRX/USDT',
}

# Commodity-linked tokens (stablecoins backed by commodities)
COMMODITY_TOKENS: Dict[str, Dict[str, str]] = {
    # Precious Metals
    'PAXG': {
        'symbol': 'PAXG/USDT',
        'name': 'Paxos Gold',
        'underlying': 'Gold',
        'api_symbol': 'PAXG/USDT',
        'category': 'precious_metal'
    },
    'XAUT': {
        'symbol': 'XAUT/USDT',
        'name': 'Tether Gold',
        'underlying': 'Gold',
        'api_symbol': 'XAUT/USDT',
        'category': 'precious_metal'
    },
    # Energy Tokens (when available)
    'WTI': {
        'symbol': 'WTI/USDT',
        'name': 'Wrapped Tokenized Oil',
        'underlying': 'Crude Oil WTI',
        'api_symbol': 'WTI/USDT',
        'category': 'energy'
    },
    'NG': {
        'symbol': 'NG/USDT',
        'name': 'Natural Gas Token',
        'underlying': 'Natural Gas',
        'api_symbol': 'NG/USDT',
        'category': 'energy'
    },
    # Fiat Tokens
    'PEUR': {
        'symbol': 'PEUR/USDT',
        'name': 'Perp Euro',
        'underlying': 'EUR',
        'api_symbol': 'PEUR/USDT',
        'category': 'fiat'
    },
    'PGBP': {
        'symbol': 'PGBP/USDT',
        'name': 'Perp British Pound',
        'underlying': 'GBP',
        'api_symbol': 'PGBP/USDT',
        'category': 'fiat'
    },
    'PJPY': {
        'symbol': 'PJPY/USDT',
        'name': 'Perp Japanese Yen',
        'underlying': 'JPY',
        'api_symbol': 'PJPY/USDT',
        'category': 'fiat'
    },
    # Staking Tokens
    'STETH': {
        'symbol': 'STETH/USDT',
        'name': 'Lido Staked Ether',
        'underlying': 'ETH',
        'api_symbol': 'STETH/USDT',
        'category': 'staking'
    },
    'BNB': {
        'symbol': 'BNB/USDT',
        'name': 'Binance Staked BNB',
        'underlying': 'BNB',
        'api_symbol': 'BNB/USDT',
        'category': 'staking'
    },
}

# Commodities we track but may not have direct trading pairs
# These are for simulation and tracking purposes
TRACKED_COMMODITIES: Dict[str, Dict[str, Any]] = {
    # Precious Metals
    'GOLD': {
        'name': 'Gold (XAU/USD)',
        'symbol': 'XAU',
        'price_source': 'PAXG/USDT',
        'category': 'precious_metal',
        'typical_range': (1900, 2100)
    },
    'SILVER': {
        'name': 'Silver (XAG/USD)',
        'symbol': 'XAG',
        'price_source': 'XAG/USDT',
        'category': 'precious_metal',
        'typical_range': (22, 28)
    },
    'PLATINUM': {
        'name': 'Platinum (XPT/USD)',
        'symbol': 'XPT',
        'price_source': None,
        'category': 'precious_metal',
        'typical_range': (900, 1100)
    },
    'PALLADIUM': {
        'name': 'Palladium (XPD/USD)',
        'symbol': 'XPD',
        'price_source': None,
        'category': 'precious_metal',
        'typical_range': (1000, 1400)
    },
    # Energy
    'WTI_OIL': {
        'name': 'Crude Oil WTI',
        'symbol': 'CL',
        'price_source': 'WTI/USDT',
        'category': 'energy',
        'typical_range': (70, 90)
    },
    'BRENT_OIL': {
        'name': 'Brent Crude',
        'symbol': 'BZ',
        'price_source': None,
        'category': 'energy',
        'typical_range': (75, 95)
    },
    'NATURAL_GAS': {
        'name': 'Natural Gas',
        'symbol': 'NG',
        'price_source': 'NG/USDT',
        'category': 'energy',
        'typical_range': (2.5, 4.0)
    },
    # Agricultural
    'WHEAT': {
        'name': 'Wheat',
        'symbol': 'ZW',
        'price_source': None,
        'category': 'agricultural',
        'typical_range': (5.5, 7.5)
    },
    'CORN': {
        'name': 'Corn',
        'symbol': 'ZC',
        'price_source': None,
        'category': 'agricultural',
        'typical_range': (4.5, 6.5)
    },
    'SOYBEANS': {
        'name': 'Soybeans',
        'symbol': 'ZS',
        'price_source': None,
        'category': 'agricultural',
        'typical_range': (11, 15)
    },
    # Indices (via crypto proxies)
    'SP500': {
        'name': 'S&P 500',
        'symbol': 'SPX',
        'price_source': None,
        'category': 'index',
        'typical_range': (4500, 5500)
    },
    'NASDAQ': {
        'name': 'NASDAQ 100',
        'symbol': 'NDX',
        'price_source': None,
        'category': 'index',
        'typical_range': (15000, 19000)
    },
    # Crypto Indices
    'BTC_ETH': {
        'name': 'Bitcoin + Ethereum Index',
        'symbol': 'BTCETH',
        'price_source': None,
        'category': 'crypto_index',
        'typical_range': (40000, 75000)
    },
}

# ==================== EXCHANGE API CONFIGURATION ====================
# Primary exchange for crypto data
DEFAULT_EXCHANGE: str = 'binance'

# Fallback exchanges
FALLBACK_EXCHANGES: List[str] = ['kucoin', 'bybit', 'okx']

# API Rate limits
RATE_LIMIT_REQUESTS: int = 10  # requests per minute
RATE_LIMIT_SECONDS: int = 6    # seconds between requests

# ==================== DATA COLLECTION ====================
# Timeframes for analysis
TIMEFRAMES: Dict[str, str] = {
    '1m': '1 minute',
    '5m': '5 minutes', 
    '15m': '15 minutes',
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day',
    '1w': '1 week',
}

# Default timeframe for analysis
DEFAULT_TIMEFRAME: str = '1h'

# Historical data limit (max candles to fetch)
MAX_CANDLES: int = 500

# Data refresh interval (seconds)
REFRESH_INTERVAL: int = 60

# ==================== TECHNICAL INDICATORS ====================
INDICATOR_SETTINGS: Dict[str, Any] = {
    # RSI
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    
    # EMA
    'ema_short': 12,
    'ema_medium': 26,
    'ema_long': 50,
    
    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2,
    
    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    # ATR (Average True Range)
    'atr_period': 14,
    
    # Stochastic
    'stoch_period': 14,
    'stoch_smooth': 3,
}

# ==================== DECISION ENGINE ====================
DECISION_SETTINGS: Dict[str, Any] = {
    # Signal generation thresholds
    'min_signal_confidence': 0.55,
    'strong_signal_threshold': 0.70,
    
    # Risk management
    'max_position_size': 0.1,
    'stop_loss_percent': 0.02,
    'take_profit_percent': 0.05,
    
    # Correlation thresholds
    'correlation_lookback': 24,
    'high_correlation': 0.7,
    'negative_correlation': -0.3,
    
    # Position scoring weights
    'weights': {
        'technical': 0.30,
        'momentum': 0.25,
        'correlation': 0.20,
        'sentiment': 0.15,
        'volatility': 0.10,
    },
    
    # ML Blackbox Agent settings
    'ml_enabled': True,
    'ml_weight': 0.15,
    'ml_confidence_weight': 0.30,
    'ml_model_path': 'models/ml_predictor.pkl',
}

# ==================== NEWS & SENTIMENT ====================
NEWS_SETTINGS: Dict[str, Any] = {
    # News sources (API keys needed)
    'news_api_key': os.getenv('NEWS_API_KEY', ''),
    'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN', ''),
    'benzinga_api_key': os.getenv('BENZINGA_API_KEY', ''),
    
    # Keywords for different asset classes
    'crypto_keywords': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi'],
    'gold_keywords': ['gold', 'gold price', 'precious metals', 'xau'],
    'oil_keywords': ['oil', 'crude', 'petroleum', 'opec'],
    'forex_keywords': ['usd', 'euro', 'forex', 'currency', 'fed', 'ecb'],
    'geopolitical_keywords': ['war', 'conflict', 'sanctions', 'election', 'fed rates'],
    
    # Sentiment scoring
    'sentiment_period': 24,
    
    # Sources priority
    'preferred_sources': [
        'Reuters', 'Bloomberg', 'CoinDesk', 'CoinTelegraph',
        'Financial Times', 'Wall Street Journal'
    ],
}

# ==================== DASHBOARD ====================
DASHBOARD_CONFIG: Dict[str, Any] = {
    'host': '0.0.0.0',
    'port': 8050,
    'debug': False,
    
    # Theme
    'theme': 'plotly_dark',
    
    # Update interval (seconds)
    'refresh_interval': 30,
    
    # Charts
    'max_charts': 10,
    'candlestick_range': 100,
    
    # Correlation heatmap
    'correlation_assets': list(CRYPTO_SYMBOLS.keys()) + list(COMMODITY_TOKENS.keys()),
}

# ==================== LOGGING ====================
LOGGING_CONFIG: Dict[str, Any] = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(LOGS_DIR / 'trading_system.log'),
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# ==================== SIMULATION MODE ====================
# If True, use simulated prices instead of real API calls
SIMULATION_MODE: bool = os.getenv('SIMULATION_MODE', 'true').lower() != 'false'

# ==================== BINANCE API CONFIGURATION ====================
# Get your API keys from: https://www.binance.com/en/my/settings/api-management
BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY: str = os.getenv('BINANCE_SECRET_KEY', '')

# Testnet (use for testing without real money)
USE_BINANCE_TESTNET: bool = os.getenv('USE_BINANCE_TESTNET', 'false').lower() == 'true'

# ==================== COINMARKETCAP API CONFIGURATION ====================
# Get your API key from: https://coinmarketcap.com/api/
COINMARKETCAP_API_KEY: str = os.getenv('COINMARKETCAP_API_KEY', '')

# ==================== EIA API CONFIGURATION ====================
# Get your API key from: https://www.eia.gov/opendata/

EIA_API_KEY: str = os.getenv('EIA_API_KEY', 'dx4gm4LcTfp9bYfmMmTd5ADkfjfT1W1rIkK6l6jH')

# ==================== ALPHA VANTAGE API CONFIGURATION ====================
# Get your API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', 'H2PEP00G1RBV2CKJ')

# Simulated price ranges for testing
SIMULATED_PRICES: Dict[str, Tuple[float, float]] = {
    # Major cryptos
    'BTC/USDT': (40000, 70000),
    'ETH/USDT': (2000, 3500),
    'XRP/USDT': (0.5, 0.8),
    # Precious Metals (Commodity-backed tokens)
    'PAXG/USDT': (1900, 2100),
    'XAUT/USDT': (1900, 2100),
    # Energy tokens
    'WTI/USDT': (70, 90),
    'NG/USDT': (2.5, 4.0),
    # Fiat tokens
    'PEUR/USDT': (1.0, 1.1),
    'PGBP/USDT': (1.2, 1.35),
    'PJPY/USDT': (0.006, 0.007),
    # Additional cryptos
    'SOL/USDT': (100, 200),
    'ADA/USDT': (0.3, 0.6),
    'DOT/USDT': (5, 10),
    'AVAX/USDT': (25, 45),
    'MATIC/USDT': (0.6, 1.2),
    'BNB/USDT': (300, 500),
    'DOGE/USDT': (0.05, 0.15),
    'LINK/USDT': (10, 20),
    'ATOM/USDT': (7, 15),
    'UNI/USDT': (5, 12),
    'LTC/USDT': (60, 120),
    'NEAR/USDT': (3, 8),
    'APT/USDT': (5, 15),
    'ARB/USDT': (0.8, 2.0),
    'OP/USDT': (1.5, 4.0),
    'INJ/USDT': (15, 40),
    'SUI/USDT': (1, 4),
    'SEI/USDT': (0.2, 0.8),
    'TIA/USDT': (10, 30),
    # Staking tokens
    'STETH/USDT': (2000, 3500),
    # Other tokens
    'FXS/USDT': (3, 10),
}

# Simulated commodities (not directly tradeable on Binance)
SIMULATED_COMMODITIES: Dict[str, Dict[str, float]] = {
    'GOLD': {'price': 2000, 'volatility': 0.01},
    'SILVER': {'price': 25, 'volatility': 0.015},
    'PLATINUM': {'price': 1000, 'volatility': 0.012},
    'PALLADIUM': {'price': 1200, 'volatility': 0.015},
    'WTI_OIL': {'price': 80, 'volatility': 0.02},
    'BRENT_OIL': {'price': 85, 'volatility': 0.02},
    'NATURAL_GAS': {'price': 3.0, 'volatility': 0.03},
    'WHEAT': {'price': 6.5, 'volatility': 0.025},
    'CORN': {'price': 5.5, 'volatility': 0.02},
    'SOYBEANS': {'price': 13, 'volatility': 0.02},
    'SP500': {'price': 5000, 'volatility': 0.008},
    'NASDAQ': {'price': 17000, 'volatility': 0.01},
}

# Price volatility for simulation
SIMULATED_VOLATILITY: float = 0.02

# ==================== TELEGRAM NOTIFICATIONS ====================
TELEGRAM_ENABLED: bool = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
TELEGRAM_RATE_LIMIT: int = int(os.getenv('TELEGRAM_RATE_LIMIT', '5'))

# Alert types
TELEGRAM_ALERT_TRADES: bool = os.getenv('TELEGRAM_ALERT_TRADES', 'true').lower() == 'true'
TELEGRAM_ALERT_RISK: bool = os.getenv('TELEGRAM_ALERT_RISK', 'true').lower() == 'true'
TELEGRAM_ALERT_REGIME: bool = os.getenv('TELEGRAM_ALERT_REGIME', 'true').lower() == 'true'
TELEGRAM_ALERT_PORTFOLIO: bool = os.getenv('TELEGRAM_ALERT_PORTFOLIO', 'false').lower() == 'true'
TELEGRAM_HEARTBEAT_INTERVAL: int = int(os.getenv('TELEGRAM_HEARTBEAT_INTERVAL', '3600'))  # seconds
