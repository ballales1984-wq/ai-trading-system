ma"""
Unified Configuration for AI Trading System
===========================================
Centralized configuration management combining:
- FastAPI app settings
- Trading system settings  
- Dashboard settings
- SaaS settings (multi-tenant ready)
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pydantic.config import ConfigDict


# ==================== PROJECT PATHS ====================
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, CACHE_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)


class CryptoSymbols:
    """Cryptocurrency symbols configuration."""
    
    MAJOR: Dict[str, str] = {
        'BTC': 'BTC/USDT',
        'ETH': 'ETH/USDT',
        'XRP': 'XRP/USDT',
        'SOL': 'SOL/USDT',
        'ADA': 'ADA/USDT',
        'DOT': 'DOT/USDT',
        'AVAX': 'AVAX/USDT',
        'MATIC': 'MATIC/USDT',
    }
    
    EXTENDED: Dict[str, str] = {
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
    
    MEME: Dict[str, str] = {
        'PEPE': 'PEPE/USDT',
        'SHIB': 'SHIB/USDT',
    }
    
    @classmethod
    def all(cls) -> Dict[str, str]:
        """Get all symbols."""
        return {**cls.MAJOR, **cls.EXTENDED, **cls.MEME}
    
    @classmethod
    def get_top(cls, n: int = 10) -> List[str]:
        """Get top N symbols by market cap (approximate)."""
        all_symbols = list(cls.MAJOR.values()) + list(cls.EXTENDED.values())[:n-len(cls.MAJOR)]
        return all_symbols[:n]


class Settings(BaseSettings):
    """
    Unified application settings with environment variable support.
    Combines FastAPI, trading, and SaaS configurations.
    """
    
    # Allow extra fields from .env file
    model_config = ConfigDict(
        extra='ignore',
        env_file='.env',
        env_file_encoding='utf-8'
    )
    
    # ==================== APP INFO ====================
    app_name: str = "AI Trading System"
    app_version: str = "2.1.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # ==================== SERVER ====================
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"
    workers: int = Field(default=4, env="WORKERS")
    worker_timeout: int = 300
    
    # ==================== CORS ====================
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # ==================== LOGGING ====================
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = "logs/trading_system.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    log_format: str = "json"  # json or text
    
    # ==================== DATABASE ====================
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/hedge_fund",
        env="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # ==================== TRADING CONFIGURATION ====================
    max_leverage: float = 10.0
    max_position_size: float = 100000.0  # USD
    default_risk_per_trade: float = 0.02  # 2%
    max_daily_loss: float = 0.05  # 5%
    
    # ==================== RISK LIMITS ====================
    max_var_percent: float = 0.02  # 2% VaR limit
    max_cvar_percent: float = 0.05  # 5% CVaR limit
    max_correlation: float = 0.7
    max_sector_exposure: float = 0.3  # 30% per sector
    
    # ==================== ORDER EXECUTION ====================
    max_slippage_pct: float = 0.001  # 0.1%
    max_slippage_abs: float = 0.01  # $0.01 for crypto
    order_timeout: int = 30  # seconds
    max_retries: int = 3
    
    # ==================== BROKER CONFIGURATION ====================
    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    
    bybit_api_key: str = Field(default="", env="BYBIT_API_KEY")
    bybit_secret_key: str = Field(default="", env="BYBIT_SECRET_KEY")
    bybit_testnet: bool = Field(default=True, env="BYBIT_TESTNET")
    
    # ==================== CLOUDFLARE RADAR ====================
    cloudflare_radar_api_key: str = Field(default="", env="CLOUDFLARE_RADAR_API_KEY")
    
    # ==================== PAPER TRADING
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    paper_initial_balance: float = 1000000.0
    
    # ==================== SECURITY ====================
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    api_key_header: str = "X-API-Key"
    
    # ==================== MARKET DATA ====================
    data_feed_interval: int = 1  # seconds
    price_cache_ttl: int = 60  # seconds
    history_lookback_days: int = 365
    
    # ==================== STRATEGY CONFIGURATION ====================
    momentum_lookback: int = 20
    mean_reversion_lookback: int = 50
    ml_model_path: str = "models/"
    
    # ==================== SAAS CONFIGURATION ====================
    # Stripe
    stripe_api_key: str = Field(default="", env="STRIPE_API_KEY")
    stripe_webhook_secret: str = Field(default="", env="STRIPE_WEBHOOK_SECRET")
    stripe_default_price_id: str = Field(default="", env="STRIPE_DEFAULT_PRICE_ID")
    stripe_success_url: str = Field(default="", env="STRIPE_SUCCESS_URL")
    stripe_cancel_url: str = Field(default="", env="STRIPE_CANCEL_URL")
    
    # Pricing - One-time purchase
    lifetime_price: float = 49.99  # EUR one-time
    trial_days: int = 7
    
    # Stripe Price IDs (from Stripe Dashboard)
    stripe_lifetime_price_id: str = Field(default="", env="STRIPE_LIFETIME_PRICE_ID")
    # Stripe Trial Price ID (for 7-day trial)
    stripe_trial_price_id: str = Field(default="", env="STRIPE_TRIAL_PRICE_ID")
    
    # Multi-tenancy
    max_users_per_instance: int = 100
    user_data_isolation: bool = True
    
    # Email (for notifications and marketing)
    smtp_host: str = Field(default="", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: str = Field(default="", env="SMTP_USER")
    smtp_password: str = Field(default="", env="SMTP_PASSWORD")
    email_from: str = Field(default="noreply@aitrading.systems", env="EMAIL_FROM")
    
    # Feature flags
    enable_monte_carlo: bool = True
    enable_multi_asset: bool = True
    enable_ml_signals: bool = True
    enable_telegram_alerts: bool = True
    
    # ==================== DASHBOARD ====================
    dashboard_title: str = "🤖 AI Trading System - Hedge Fund Edition"
    dashboard_update_interval: int = 5000  # milliseconds
    dashboard_max_candles: int = 500
    
    # ==================== PATHS ====================
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT
    
    @property
    def data_dir(self) -> Path:
        return DATA_DIR
    
    @property
    def logs_dir(self) -> Path:
        return LOGS_DIR
    
    @property
    def cache_dir(self) -> Path:
        return CACHE_DIR
    
    @property
    def models_dir(self) -> Path:
        return MODELS_DIR
    
    # ==================== VALIDATORS ====================
    @validator('environment')
    def validate_environment(cls, v: str) -> str:
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper
    
    # ==================== HELPERS ====================
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def get_crypto_symbols(self, top_n: Optional[int] = None) -> Dict[str, str]:
        """Get cryptocurrency symbols."""
        if top_n:
            return {k: v for k, v in list(CryptoSymbols.all().items())[:top_n]}
        return CryptoSymbols.all()
    
    def get_database_url_async(self) -> str:
        """Get async database URL."""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        return self.database_url


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()

# Export for backward compatibility
CRYPTO_SYMBOLS = CryptoSymbols.all()
