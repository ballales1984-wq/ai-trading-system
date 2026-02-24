"""
Tests for Unified Configuration Module
=======================================
Tests for the centralized configuration management.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.unified_config import CryptoSymbols, Settings


class TestCryptoSymbols:
    """Test CryptoSymbols class."""
    
    def test_major_symbols(self):
        """Test major cryptocurrency symbols."""
        assert 'BTC' in CryptoSymbols.MAJOR
        assert 'ETH' in CryptoSymbols.MAJOR
        assert 'XRP' in CryptoSymbols.MAJOR
        assert 'SOL' in CryptoSymbols.MAJOR
        
        assert CryptoSymbols.MAJOR['BTC'] == 'BTC/USDT'
        assert CryptoSymbols.MAJOR['ETH'] == 'ETH/USDT'
    
    def test_extended_symbols(self):
        """Test extended cryptocurrency symbols."""
        assert 'BNB' in CryptoSymbols.EXTENDED
        assert 'DOGE' in CryptoSymbols.EXTENDED
        assert 'LINK' in CryptoSymbols.EXTENDED
    
    def test_meme_symbols(self):
        """Test meme cryptocurrency symbols."""
        assert 'PEPE' in CryptoSymbols.MEME
        assert 'SHIB' in CryptoSymbols.MEME
    
    def test_all_symbols(self):
        """Test getting all symbols."""
        all_symbols = CryptoSymbols.all()
        
        # Should contain all major, extended, and meme symbols
        assert 'BTC' in all_symbols
        assert 'BNB' in all_symbols
        assert 'PEPE' in all_symbols
        
        # Should be a combination of all
        expected_count = len(CryptoSymbols.MAJOR) + len(CryptoSymbols.EXTENDED) + len(CryptoSymbols.MEME)
        assert len(all_symbols) == expected_count
    
    def test_get_top_symbols(self):
        """Test getting top N symbols."""
        top_5 = CryptoSymbols.get_top(5)
        
        assert len(top_5) == 5
        assert all(isinstance(s, str) for s in top_5)
    
    def test_get_top_default(self):
        """Test getting top symbols with default count."""
        top_10 = CryptoSymbols.get_top()
        
        assert len(top_10) == 10
    
    def test_get_top_more_than_major(self):
        """Test getting more symbols than major count."""
        top_15 = CryptoSymbols.get_top(15)
        
        assert len(top_15) == 15


class TestSettings:
    """Test Settings class."""
    
    def test_default_initialization(self):
        """Test default Settings initialization."""
        settings = Settings()
        
        assert settings.app_name == "AI Trading System"
        assert settings.app_version == "2.1.0"
        assert settings.host == "0.0.0.0"
        assert settings.port > 0  # Port may be overridden by .env file
        assert settings.api_prefix == "/api/v1"
    
    def test_default_trading_config(self):
        """Test default trading configuration."""
        settings = Settings()
        
        assert settings.max_leverage == 10.0
        assert settings.max_position_size == 100000.0
        assert settings.default_risk_per_trade == 0.02
        assert settings.max_daily_loss == 0.05
    
    def test_default_risk_limits(self):
        """Test default risk limits."""
        settings = Settings()
        
        assert settings.max_var_percent == 0.02
        assert settings.max_cvar_percent == 0.05
        assert settings.max_correlation == 0.7
        assert settings.max_sector_exposure == 0.3
    
    def test_default_order_execution(self):
        """Test default order execution settings."""
        settings = Settings()
        
        assert settings.max_slippage_pct == 0.001
        assert settings.max_slippage_abs == 0.01
        assert settings.order_timeout == 30
        assert settings.max_retries == 3
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG', 'DEBUG': 'false'}):
            settings = Settings()
            
            assert settings.log_level == 'DEBUG'
    
    def test_database_url_default(self):
        """Test default database URL."""
        settings = Settings()
        
        assert 'postgresql' in settings.database_url
    
    def test_redis_url_default(self):
        """Test default Redis URL."""
        settings = Settings()
        
        assert 'redis' in settings.redis_url
    
    def test_cors_origins_default(self):
        """Test default CORS origins."""
        settings = Settings()
        
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0
    
    def test_log_format_default(self):
        """Test default log format."""
        settings = Settings()
        
        assert settings.log_format in ['json', 'text']
    
    def test_workers_default(self):
        """Test default workers count."""
        settings = Settings()
        
        assert settings.workers > 0
    
    def test_database_pool_settings(self):
        """Test database pool settings."""
        settings = Settings()
        
        assert settings.database_pool_size > 0
        assert settings.database_max_overflow > 0
