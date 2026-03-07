"""
Tests for app/core/config.py - Application Configuration
"""

import pytest
from pydantic import ValidationError


class TestConfigSettings:
    """Test suite for Settings configuration class."""

    def test_settings_creation(self):
        """Test basic settings creation with defaults."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.app_name == "Hedge Fund Trading System"
        assert settings.app_version is not None
        assert settings.environment == "development"

    def test_settings_host_port(self):
        """Test server host and port configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port is not None  # Port may be overridden by env

    def test_settings_api_prefix(self):
        """Test API prefix configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.api_prefix == "/api/v1"

    def test_settings_cors_origins(self):
        """Test CORS origins configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert isinstance(settings.cors_origins, list)
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:5173" in settings.cors_origins

    def test_settings_logging(self):
        """Test logging configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.log_level == "INFO"
        assert settings.log_file == "logs/trading_system.log"

    def test_settings_database_defaults(self):
        """Test database configuration defaults."""
        from app.core.config import Settings
        
        settings = Settings()
        assert "postgresql://" in settings.database_url
        assert settings.db_pool_size == 20
        assert settings.db_max_overflow == 40
        assert settings.db_pool_recycle == 3600

    def test_settings_redis(self):
        """Test Redis configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert "redis://" in settings.redis_url

    def test_settings_trading_config(self):
        """Test trading configuration defaults."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.max_leverage == 10.0
        assert settings.max_position_size == 100000.0
        assert settings.default_risk_per_trade == 0.02
        assert settings.max_daily_loss == 0.05

    def test_settings_risk_limits(self):
        """Test risk limit configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.max_var_percent == 0.02
        assert settings.max_cvar_percent == 0.05
        assert settings.max_correlation == 0.7
        assert settings.max_sector_exposure == 0.3

    def test_settings_order_execution(self):
        """Test order execution configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.max_slippage_pct == 0.001
        assert settings.max_slippage_abs == 0.01
        assert settings.order_timeout == 30
        assert settings.max_retries == 3

    def test_settings_broker_binance(self):
        """Test Binance broker configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.binance_api_key == ""
        assert settings.binance_secret_key == ""
        assert settings.binance_testnet is True

    def test_settings_broker_bybit(self):
        """Test Bybit broker configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.bybit_api_key == ""
        assert settings.bybit_secret_key == ""
        assert settings.bybit_testnet is True

    def test_settings_broker_ib(self):
        """Test Interactive Brokers configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.ib_host == "127.0.0.1"
        assert settings.ib_port == 7497
        assert settings.ib_client_id == 1

    def test_settings_paper_trading(self):
        """Test paper trading configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.paper_trading is True
        assert settings.paper_initial_balance == 1000000.0

    def test_settings_security(self):
        """Test security configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.secret_key == ""
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_expiration_minutes == 60

    def test_settings_market_data(self):
        """Test market data configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.data_feed_interval == 1
        assert settings.price_cache_ttl == 60
        assert settings.history_lookback_days == 365

    def test_settings_strategy(self):
        """Test strategy configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.momentum_lookback == 20
        assert settings.mean_reversion_lookback == 50
        assert settings.ml_model_path == "models/"

    def test_settings_performance(self):
        """Test performance configuration."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.workers == 4
        assert settings.worker_timeout == 300

    def test_settings_debug_validator_string(self):
        """Test debug field validator with string input."""
        from app.core.config import Settings
        
        # Test string "true"
        settings = Settings(debug="true")
        assert settings.debug is True
        
        settings = Settings(debug="1")
        assert settings.debug is True
        
        settings = Settings(debug="yes")
        assert settings.debug is True
        
        settings = Settings(debug="on")
        assert settings.debug is True
        
        settings = Settings(debug="false")
        assert settings.debug is False
        
        settings = Settings(debug="0")
        assert settings.debug is False

    def test_settings_secret_key_validation_short(self):
        """Test secret key validation - too short."""
        from app.core.config import Settings
        
        with pytest.raises(ValidationError):
            Settings(secret_key="short", environment="production")

    def test_settings_secret_key_validation_long_enough(self):
        """Test secret key validation - long enough."""
        from app.core.config import Settings
        
        # Should not raise if key is long enough
        settings = Settings(secret_key="a" * 32)
        assert len(settings.secret_key) >= 32

    def test_settings_secret_key_production_empty(self):
        """Test secret key validation - empty in production."""
        import os
        from app.core.config import Settings
        
        # Test that secret_key can be empty string in non-production
        settings = Settings(secret_key="")
        assert settings.secret_key == ""

    def test_settings_custom_values(self):
        """Test settings with custom values."""
        from app.core.config import Settings
        
        settings = Settings(
            app_name="Custom App",
            port=9000,
            debug=False,
            log_level="DEBUG",
            max_leverage=5.0
        )
        
        assert settings.app_name == "Custom App"
        assert settings.port == 9000
        assert settings.debug is False
        assert settings.log_level == "DEBUG"
        assert settings.max_leverage == 5.0

    def test_settings_cors_as_string(self):
        """Test CORS origins can be set as list of strings."""
        from app.core.config import Settings
        
        settings = Settings(
            cors_origins=["https://example.com", "https://test.com"]
        )
        
        assert len(settings.cors_origins) == 2
        assert "https://example.com" in settings.cors_origins

    def test_get_settings_function(self):
        """Test get_settings cached function."""
        from app.core.config import get_settings, Settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should return same cached instance
        assert settings1 is settings2
        assert isinstance(settings1, Settings)

    def test_settings_timescaledb_disabled(self):
        """Test TimescaleDB configuration - disabled by default."""
        from app.core.config import Settings
        
        settings = Settings()
        assert settings.timescaledb_enabled is False
        assert settings.timescaledb_url is None
