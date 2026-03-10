"""
Test Suite for Unified Config Module
====================================
Comprehensive tests for unified configuration system.
"""

import pytest
from app.core.unified_config import (
    CryptoSymbols,
    Settings,
    PROJECT_ROOT,
    DATA_DIR,
    LOGS_DIR,
    CACHE_DIR,
    MODELS_DIR,
)


class TestCryptoSymbols:
    """Tests for CryptoSymbols class."""
    
    def test_major_symbols(self):
        """Test major cryptocurrency symbols."""
        assert 'BTC' in CryptoSymbols.MAJOR
        assert CryptoSymbols.MAJOR['BTC'] == 'BTC/USDT'
        assert CryptoSymbols.MAJOR['ETH'] == 'ETH/USDT'
    
    def test_extended_symbols(self):
        """Test extended cryptocurrency symbols."""
        assert 'BNB' in CryptoSymbols.EXTENDED
        assert CryptoSymbols.EXTENDED['BNB'] == 'BNB/USDT'
    
    def test_meme_symbols(self):
        """Test meme cryptocurrency symbols."""
        assert 'PEPE' in CryptoSymbols.MEME
        assert CryptoSymbols.MEME['PEPE'] == 'PEPE/USDT'
    
    def test_all_symbols(self):
        """Test getting all symbols."""
        all_symbols = CryptoSymbols.all()
        assert 'BTC' in all_symbols
        assert 'BNB' in all_symbols
        assert 'PEPE' in all_symbols
        assert len(all_symbols) > len(CryptoSymbols.MAJOR)
    
    def test_get_top_symbols(self):
        """Test getting top N symbols."""
        top = CryptoSymbols.get_top(5)
        assert len(top) <= 5
        assert 'BTC/USDT' in top


class TestSettings:
    """Tests for Settings class."""
    
    def test_settings_creation(self):
        """Test settings creation with defaults."""
        settings = Settings()
        assert settings.app_name == "AI Trading System"
        assert settings.app_version == "2.1.0"
    
    def test_settings_environment(self):
        """Test settings environment."""
        settings = Settings()
        assert settings.environment in ["development", "production", "test"]
    
    def test_settings_debug(self):
        """Test settings debug flag."""
        settings = Settings()
        assert isinstance(settings.debug, bool)
    
    def test_settings_database_url(self):
        """Test settings database URL."""
        settings = Settings()
        assert settings.database_url is not None


class TestProjectPaths:
    """Tests for project paths."""
    
    def test_project_root_exists(self):
        """Test project root exists."""
        assert PROJECT_ROOT.exists()
    
    def test_data_dir_exists(self):
        """Test data directory exists."""
        assert DATA_DIR.exists()
    
    def test_logs_dir_exists(self):
        """Test logs directory exists."""
        assert LOGS_DIR.exists()
    
    def test_cache_dir_exists(self):
        """Test cache directory exists."""
        assert CACHE_DIR.exists()
    
    def test_models_dir_exists(self):
        """Test models directory exists."""
        assert MODELS_DIR.exists()
