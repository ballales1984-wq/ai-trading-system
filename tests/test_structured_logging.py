"""
Test Suite for Structured Logging Module
=========================================
Comprehensive tests for structured logging system.
"""

import pytest
import logging
import json
from datetime import datetime
from app.core.structured_logging import (
    StructuredFormatter,
    TradingLogger,
    request_id_ctx,
    user_id_ctx,
    session_id_ctx,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter class."""
    
    def test_formatter_creation(self):
        """Test formatter creation."""
        formatter = StructuredFormatter()
        assert formatter.include_extra is True
    
    def test_formatter_creation_no_extra(self):
        """Test formatter creation without extra."""
        formatter = StructuredFormatter(include_extra=False)
        assert formatter.include_extra is False
    
    def test_format_basic(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["module"] == "test"
    
    def test_format_with_extra(self):
        """Test log formatting with extra fields."""
        formatter = StructuredFormatter(include_extra=True)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra = {"custom_field": "value"}
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["extra"]["custom_field"] == "value"


class TestTradingLogger:
    """Tests for TradingLogger class."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = TradingLogger("test")
        assert logger.name == "test"
        assert logger.context == {}
    
    def test_logger_creation_with_context(self):
        """Test logger creation with context."""
        context = {"symbol": "BTC", "exchange": "binance"}
        logger = TradingLogger("test", context=context)
        assert logger.context == context
    
    def test_logger_info(self):
        """Test logger info method."""
        logger = TradingLogger("test")
        # This should not raise an exception
        logger.info("Test message", symbol="BTC")
    
    def test_logger_debug(self):
        """Test logger debug method."""
        logger = TradingLogger("test")
        logger.debug("Debug message", value=123)
    
    def test_logger_warning(self):
        """Test logger warning method."""
        logger = TradingLogger("test")
        logger.warning("Warning message", code=404)
    
    def test_logger_error(self):
        """Test logger error method."""
        logger = TradingLogger("test")
        logger.error("Error message", error_code=500)
    
    def test_logger_critical(self):
        """Test logger critical method."""
        logger = TradingLogger("test")
        logger.critical("Critical message", severity="high")


class TestContextVariables:
    """Tests for context variables."""
    
    def test_request_id_context(self):
        """Test request_id context variable."""
        token = request_id_ctx.set("test-request-id")
        assert request_id_ctx.get() == "test-request-id"
        request_id_ctx.reset(token)
    
    def test_user_id_context(self):
        """Test user_id context variable."""
        token = user_id_ctx.set("test-user-id")
        assert user_id_ctx.get() == "test-user-id"
        user_id_ctx.reset(token)
    
    def test_session_id_context(self):
        """Test session_id context variable."""
        token = session_id_ctx.set("test-session-id")
        assert session_id_ctx.get() == "test-session-id"
        session_id_ctx.reset(token)
    
    def test_context_reset(self):
        """Test context variable reset."""
        token = request_id_ctx.set("test-request-id")
        request_id_ctx.reset(token)
        assert request_id_ctx.get() == ""
