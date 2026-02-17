"""
Configuration Settings for Crypto Commodity Trading System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ==================== CRYPTO CONFIGURATION ====================
CRYPTO_SYMBOLS = {
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
}

# Commodity-linked tokens (stablecoins backed by commodities)
COMMODITY_TOKENS = {
    'PAX Gold': {
        'symbol': 'PAXG/USDT',
        'name': 'Paxos Gold',
        'underlying': 'Gold',
        'api_symbol': 'PAXG/USDT'
    },
    'Tether Gold': {
        'symbol': 'XAUT/USDT',
        'name': 'Tether Gold',
        'underlying': 'Gold',
        'api_symbol': 'XAUT/USDT'
    },
    'Perp Euro': {
        'symbol': 'PEUR/USDT',
        'name': 'Perp Euro',
        'underlying': 'EUR',
        'api_symbol': 'PEUR/USDT'
    },
    'Staked Ether': {
        'symbol': 'STETH/USDT',
        'name': 'Lido Staked Ether',
        'underlying': 'ETH',
        'api_symbol': 'STETH/USDT'
    },
    'Staked BNB': {
        'symbol': 'BNBUSDT',  # BNB staking on BNB Smart Chain
        'name': 'Binance Staked BNB',
        'underlying': 'BNB',
        'api_symbol': 'BNB/USDT'
    },
    'Frax Share': {
        'symbol': 'FXS/USDT',
        'name': 'Frax Share',
        'underlying': 'Crypto',
        'api_symbol': 'FXS/USDT'
    },
}

# ==================== EXCHANGE API CONFIGURATION ====================
# Primary exchange for crypto data
DEFAULT_EXCHANGE = 'binance'

# Fallback exchanges
FALLBACK_EXCHANGES = ['kucoin', 'bybit', 'okx']

# API Rate limits
RATE_LIMIT_REQUESTS = 10  # requests per minute
RATE_LIMIT_SECONDS = 6    # seconds between requests

# ==================== DATA COLLECTION ====================
# Timeframes for analysis
TIMEFRAMES = {
    '1m': '1 minute',
    '5m': '5 minutes', 
    '15m': '15 minutes',
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day',
    '1w': '1 week',
}

# Default timeframe for analysis
DEFAULT_TIMEFRAME = '1h'

# Historical data limit (max candles to fetch)
MAX_CANDLES = 500

# Data refresh interval (seconds)
REFRESH_INTERVAL = 60  # 1 minute

# ==================== TECHNICAL INDICATORS ====================
INDICATOR_SETTINGS = {
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
DECISION_SETTINGS = {
    # Signal generation thresholds
    'min_signal_confidence': 0.55,  # Minimum confidence to generate signal
    'strong_signal_threshold': 0.70,  # Threshold for strong signal
    
    # Risk management
    'max_position_size': 0.1,  # Max 10% of portfolio per position
    'stop_loss_percent': 0.02,  # 2% stop loss
    'take_profit_percent': 0.05,  # 5% take profit
    
    # Correlation thresholds
    'correlation_lookback': 24,  # Hours to calculate correlation
    'high_correlation': 0.7,
    'negative_correlation': -0.3,
    
    # Position scoring weights
    'weights': {
        'technical': 0.30,
        'momentum': 0.25,
        'correlation': 0.20,
        'sentiment': 0.15,
        'volatility': 0.10,
    }
}

# ==================== NEWS & SENTIMENT ====================
NEWS_SETTINGS = {
    # News sources (API keys needed)
    'news_api_key': os.getenv('NEWS_API_KEY', ''),
    
    # Keywords for different asset classes
    'crypto_keywords': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi'],
    'gold_keywords': ['gold', 'gold price', 'precious metals', 'xau'],
    'oil_keywords': ['oil', 'crude', 'petroleum', 'opec'],
    'forex_keywords': ['usd', 'euro', 'forex', 'currency', 'fed', 'ecb'],
    'geopolitical_keywords': ['war', 'conflict', 'sanctions', 'election', 'fed rates'],
    
    # Sentiment scoring
    'sentiment_period': 24,  # Hours to aggregate news
    
    # Sources priority
    'preferred_sources': [
        'Reuters', 'Bloomberg', 'CoinDesk', 'CoinTelegraph',
        'Financial Times', 'Wall Street Journal'
    ],
}

# ==================== DASHBOARD ====================
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 8050,
    'debug': False,
    
    # Theme
    'theme': 'plotly_dark',
    
    # Update interval (seconds)
    'refresh_interval': 30,
    
    # Charts
    'max_charts': 10,
    'candlestick_range': 100,  # Number of candles to display
    
    # Correlation heatmap
    'correlation_assets': list(CRYPTO_SYMBOLS.keys()) + list(COMMODITY_TOKENS.keys()),
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(LOGS_DIR / 'trading_system.log'),
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# ==================== SIMULATION MODE ====================
# If True, use simulated prices instead of real API calls
SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'true').lower() != 'false'  # Default to simulation

# ==================== BINANCE API CONFIGURATION ====================
# Get your API keys from: https://www.binance.com/en/my/settings/api-management
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')

# Testnet (use for testing without real money)
USE_BINANCE_TESTNET = os.getenv('USE_BINANCE_TESTNET', 'false').lower() == 'true'

# Simulated price ranges for testing
SIMULATED_PRICES = {
    # Major cryptos
    'BTC/USDT': (40000, 70000),
    'ETH/USDT': (2000, 3500),
    'XRP/USDT': (0.5, 0.8),
    # Commodities
    'PAXG/USDT': (1900, 2100),
    'XAUT/USDT': (1900, 2100),
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
    # Other tokens
    'STETH/USDT': (2000, 3500),
    'PEUR/USDT': (1.0, 1.1),
    'FXS/USDT': (3, 10),
}

# Price volatility for simulation
SIMULATED_VOLATILITY = 0.02  # 2% random price movement

