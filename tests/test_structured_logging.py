"""
Tests for Structured Logging Module
=====================================
Tests for the structured logging system.
"""

import pytest
import sys
import os
import json
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.structured_logging import (
    StructuredFormatter, TradingLogger,
    request_id_ctx, user_id_ctx, session_id_ctx
)


class TestStructuredFormatter:
    """Test StructuredFormatter class."""
    
    def setup_method(self):
        """Setup test data."""
        self.formatter = StructuredFormatter()
    
    def test_initialization(self):
        """Test StructuredFormatter initialization."""
        assert self.formatter is not None
        assert self.formatter.include_extra is True
    
    def test_initialization_without_extra(self):
        """Test StructuredFormatter initialization without extra."""
        formatter = StructuredFormatter(include_extra=False)
        
        assert formatter.include_extra is False
    
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
        
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
        assert 'timestamp' in log_data
        assert 'logger' in log_data
    
    def test_format_with_context(self):
        """Test log record formatting with context variables."""
        # Set context variables
        request_id_ctx.set("req_123")
        user_id_ctx.set("user_456")
        session_id_ctx.set("sess_789")
        
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
        log_data = json.loads(formatted)
        
        assert log_data.get('request_id') == 'req_123'
        assert log_data.get('user_id') == 'user_456'
        assert log_data.get('session_id') == 'sess_789'
        
        # Reset context variables
        request_id_ctx.set('')
        user_id_ctx.set('')
        session_id_ctx.set('')
    
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
        
        assert 'exception' in log_data
        assert log_data['exception']['type'] == 'ValueError'
        assert log_data['exception']['message'] == 'Test exception'


class TestTradingLogger:
    """Test TradingLogger class."""
    
    def setup_method(self):
        """Setup test data."""
        self.logger = TradingLogger("test_module")
    
    def test_initialization(self):
        """Test TradingLogger initialization."""
        assert self.logger is not None
        assert self.logger.name == "test_module"
        assert self.logger.context == {}
    
    def test_initialization_with_context(self):
        """Test TradingLogger initialization with context."""
        context = {'service': 'trading', 'version': '2.0'}
        logger = TradingLogger("test_module", context=context)
        
        assert logger.context == context
    
    def test_info_logging(self):
        """Test info level logging."""
        # Should not raise any exception
        self.logger.info("Test info message", key="value")
    
    def test_debug_logging(self):
        """Test debug level logging."""
        # Should not raise any exception
        self.logger.debug("Test debug message", key="value")
    
    def test_warning_logging(self):
        """Test warning level logging."""
        # Should not raise any exception
        self.logger.warning("Test warning message", key="value")
    
    def test_error_logging(self):
        """Test error level logging."""
        # Should not raise any exception
        self.logger.error("Test error message", key="value")
    
    def test_critical_logging(self):
        """Test critical level logging."""
        # Should not raise any exception
        self.logger.critical("Test critical message", key="value")
    
    def test_trade_logging(self):
        """Test trade-specific logging."""
        # Should not raise any exception
        self.logger.trade(
            action="BUY",
            symbol="BTC/USDT",
            price=50000,
            quantity=0.1
        )
    
    def test_order_logging(self):
        """Test order-specific logging."""
        # Should not raise any exception
        self.logger.order(
            order_id="order_123",
            status="FILLED",
            symbol="BTC/USDT"
        )
    
    def test_risk_logging(self):
        """Test risk-specific logging."""
        # Should not raise any exception
        self.logger.risk(
            event="DRAWDOWN_ALERT",
            level="WARNING",
            drawdown=0.05
        )
    
    def test_performance_logging(self):
        """Test performance-specific logging."""
        # Should not raise any exception
        self.logger.performance(
            metric="latency_ms",
            value=5.2,
            endpoint="/api/v1/orders"
        )
    
    def test_api_logging(self):
        """Test API-specific logging."""
        # Should not raise any exception
        self.logger.api(
            endpoint="/api/v1/orders",
            method="POST",
            status_code=200,
            duration_ms=15.5
        )


class TestContextVariables:
    """Test context variables for request tracking."""
    
    def test_request_id_default(self):
        """Test default request ID."""
        # Reset to default
        request_id_ctx.set('')
        
        assert request_id_ctx.get() == ''
    
    def test_request_id_set(self):
        """Test setting request ID."""
        request_id_ctx.set("req_123")
        
        assert request_id_ctx.get() == "req_123"
        
        # Reset
        request_id_ctx.set('')
    
    def test_user_id_default(self):
        """Test default user ID."""
        user_id_ctx.set('')
        
        assert user_id_ctx.get() == ''
    
    def test_user_id_set(self):
        """Test setting user ID."""
        user_id_ctx.set("user_456")
        
        assert user_id_ctx.get() == "user_456"
        
        # Reset
        user_id_ctx.set('')
    
    def test_session_id_default(self):
        """Test default session ID."""
        session_id_ctx.set('')
        
        assert session_id_ctx.get() == ''
    
    def test_session_id_set(self):
        """Test setting session ID."""
        session_id_ctx.set("sess_789")
        
        assert session_id_ctx.get() == "sess_789"
        
        # Reset
        session_id_ctx.set('')
