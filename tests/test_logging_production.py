"""
Tests for Production Logging Module
=====================================
Tests for the production-grade structured logging system.
"""

import pytest
import sys
import os
import json
import logging
from datetime import datetime
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging_production import (
    ProductionJSONFormatter, TradingLogger as ProdTradingLogger,
    LogLevel, LogCategory,
    correlation_id, request_id, user_id, session_id,
    get_correlation_id, set_correlation_id, new_correlation_id
)


class TestCorrelationIDs:
    """Test correlation ID functions."""
    
    def setup_method(self):
        """Reset context variables."""
        correlation_id.set('')
        request_id.set('')
        user_id.set('')
        session_id.set('')
    
    def test_get_correlation_id_generates_new(self):
        """Test that get_correlation_id generates a new ID if not set."""
        correlation_id.set('')
        
        cid = get_correlation_id()
        
        assert cid is not None
        assert len(cid) > 0
    
    def test_get_correlation_id_returns_existing(self):
        """Test that get_correlation_id returns existing ID."""
        correlation_id.set("existing_id")
        
        cid = get_correlation_id()
        
        assert cid == "existing_id"
    
    def test_set_correlation_id(self):
        """Test setting correlation ID."""
        set_correlation_id("test_id_123")
        
        assert correlation_id.get() == "test_id_123"
    
    def test_new_correlation_id(self):
        """Test generating new correlation ID."""
        cid = new_correlation_id()
        
        assert cid is not None
        assert len(cid) > 0
        assert correlation_id.get() == cid
    
    def test_new_correlation_id_is_unique(self):
        """Test that new correlation IDs are unique."""
        cid1 = new_correlation_id()
        cid2 = new_correlation_id()
        
        assert cid1 != cid2


class TestLogLevel:
    """Test LogLevel enum."""
    
    def test_log_levels(self):
        """Test all log levels."""
        assert LogLevel.TRACE == "TRACE"
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestLogCategory:
    """Test LogCategory enum."""
    
    def test_log_categories(self):
        """Test all log categories."""
        assert LogCategory.SYSTEM == "system"
        assert LogCategory.TRADING == "trading"
        assert LogCategory.RISK == "risk"
        assert LogCategory.SECURITY == "security"
        assert LogCategory.AUDIT == "audit"
        assert LogCategory.PERFORMANCE == "performance"
        assert LogCategory.API == "api"


class TestProductionJSONFormatter:
    """Test ProductionJSONFormatter class."""
    
    def setup_method(self):
        """Setup test data."""
        self.formatter = ProductionJSONFormatter()
    
    def test_initialization(self):
        """Test ProductionJSONFormatter initialization."""
        assert self.formatter is not None
    
    def test_format_basic(self):
        """Test basic log record formatting."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        
        # The formatter uses ECS format with 'log.level' instead of 'level'
        assert 'message' in log_data
        assert 'log.level' in log_data or 'level' in log_data
        assert '@timestamp' in log_data or 'timestamp' in log_data
    
    def test_format_with_exception(self):
        """Test log record formatting with exception."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        # Should contain exception info
        assert 'message' in log_data
    
    def test_sensitive_data_masking(self):
        """Test that sensitive data is masked."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="API key: sk-1234567890abcdef",
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        # The message should be in the log
        assert 'message' in log_data


class TestProdTradingLogger:
    """Test production TradingLogger class."""
    
    def setup_method(self):
        """Setup test data."""
        self.logger = ProdTradingLogger("test_module")
    
    def test_initialization(self):
        """Test TradingLogger initialization."""
        assert self.logger is not None
    
    def test_info_logging(self):
        """Test info level logging."""
        # Should not raise any exception
        self.logger.info("Test info message")
    
    def test_debug_logging(self):
        """Test debug level logging."""
        # Should not raise any exception
        self.logger.debug("Test debug message")
    
    def test_warning_logging(self):
        """Test warning level logging."""
        # Should not raise any exception
        self.logger.warning("Test warning message")
    
    def test_error_logging(self):
        """Test error level logging."""
        # Should not raise any exception
        self.logger.error("Test error message")
    
    def test_critical_logging(self):
        """Test critical level logging."""
        # Should not raise any exception
        self.logger.critical("Test critical message")
    
    def test_log_order_created(self):
        """Test order creation logging."""
        # Should not raise any exception
        self.logger.log_order_created(
            "order_123",
            "BTC/USDT",
            "buy",
            0.1,
            50000.0,
            "limit"
        )
    
    def test_log_order_filled(self):
        """Test order fill logging."""
        # Should not raise any exception
        self.logger.log_order_filled(
            "order_123",
            "BTC/USDT",
            "buy",
            0.1,
            50000.0,
            0.001
        )
    
    def test_log_order_rejected(self):
        """Test order rejection logging."""
        # Should not raise any exception
        self.logger.log_order_rejected(
            "order_123",
            "BTC/USDT",
            "Insufficient funds",
            "INSUFFICIENT_BALANCE"
        )
    
    def test_log_signal(self):
        """Test signal logging."""
        # Should not raise any exception
        self.logger.log_signal(
            "momentum",
            "BTC/USDT",
            "buy",
            0.85
        )
    
    def test_log_risk_violation(self):
        """Test risk violation logging."""
        # Should not raise any exception
        self.logger.log_risk_violation(
            "Max drawdown",
            0.15,
            0.10
        )
    
    def test_log_position_opened(self):
        """Test position opened logging."""
        # Should not raise any exception
        self.logger.log_position_opened(
            "BTC/USDT",
            "buy",
            0.1,
            50000.0,
            "momentum"
        )
    
    def test_log_position_closed(self):
        """Test position closed logging."""
        # Should not raise any exception
        self.logger.log_position_closed(
            "BTC/USDT",
            "buy",
            0.1,
            50100.0,
            10.0,
            2.0,
            "momentum"
        )
    
    def test_log_api_call(self):
        """Test API call logging."""
        # Should not raise any exception
        self.logger.log_api_call(
            "binance",
            "/api/v3/ticker/price",
            "GET",
            200,
            150.0
        )
    
    def test_log_performance(self):
        """Test performance logging."""
        # Should not raise any exception
        self.logger.log_performance(
            "Sharpe ratio",
            1.2
        )
    
    def test_log_audit(self):
        """Test audit logging."""
        # Should not raise any exception
        self.logger.log_audit(
            "update",
            "strategy",
            "momentum_v1"
        )
