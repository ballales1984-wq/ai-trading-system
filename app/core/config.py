"""
Application Configuration
=========================
Centralized configuration management using Pydantic settings.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic.config import ConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Allow extra fields from .env file
    model_config = ConfigDict(extra='ignore')
    
    # App Info
    app_name: str = "Hedge Fund Trading System"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    api_prefix: str = "/api/v1"
    
    # CORS - Environment-based origin configuration
    # In development: allow all origins
    # In production: restrict to specific domains
    cors_origins: List[str] = ["*"] if debug else [
        "https://ai-trading-system.vercel.app",  # Production frontend
        "https://*.vercel.app",  # Preview deployments (note: wildcard not supported by browsers)
        "http://localhost:3000",  # Local React dev server
        "http://localhost:5173",  # Local Vite dev server
    ]

    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/trading_system.log"
    
    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/hedge_fund",
        env="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Trading Configuration
    max_leverage: float = 10.0
    max_position_size: float = 100000.0  # USD
    default_risk_per_trade: float = 0.02  # 2%
    max_daily_loss: float = 0.05  # 5%
    
    # Risk Limits
    max_var_percent: float = 0.02  # 2% VaR limit
    max_cvar_percent: float = 0.05  # 5% CVaR limit
    max_correlation: float = 0.7
    max_sector_exposure: float = 0.3  # 30% per sector
    
    # Order Execution
    max_slippage_pct: float = 0.001  # 0.1%
    max_slippage_abs: float = 0.01  # $0.01 for crypto
    order_timeout: int = 30  # seconds
    max_retries: int = 3
    
    # Broker Configuration
    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", env="BINANCE_SECRET_KEY")
    binance_testnet: bool = True
    
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    
    bybit_api_key: str = Field(default="", env="BYBIT_API_KEY")
    bybit_secret_key: str = Field(default="", env="BYBIT_SECRET_KEY")
    bybit_testnet: bool = True
    
    # Paper Trading
    paper_trading: bool = True
    paper_initial_balance: float = 1000000.0
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    
    # Market Data
    data_feed_interval: int = 1  # seconds
    price_cache_ttl: int = 60  # seconds
    history_lookback_days: int = 365
    
    # Strategy Configuration
    momentum_lookback: int = 20
    mean_reversion_lookback: int = 50
    ml_model_path: str = "models/"
    
    # Performance
    workers: int = 4
    worker_timeout: int = 300


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings
settings = get_settings()
