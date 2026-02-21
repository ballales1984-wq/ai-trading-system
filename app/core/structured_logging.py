"""
Structured Logging System for AI Trading System
================================================
Provides JSON structured logging with:
- Request ID tracking
- User context
- Performance metrics
- Error tracking
- Log rotation
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import time

from loguru import logger

# Context variables for request tracking
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')
user_id_ctx: ContextVar[str] = ContextVar('user_id', default='')
session_id_ctx: ContextVar[str] = ContextVar('session_id', default='')


class StructuredFormatter:
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_extra: bool = True):
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context variables
        if request_id := request_id_ctx.get():
            log_data["request_id"] = request_id
        if user_id := user_id_ctx.get():
            log_data["user_id"] = user_id
        if session_id := session_id_ctx.get():
            log_data["session_id"] = session_id
        
        # Add extra fields
        if self.include_extra and hasattr(record, 'extra'):
            log_data["extra"] = record.extra
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": logging.Formatter().formatException(record.exc_info)
            }
        
        return json.dumps(log_data, default=str)


class TradingLogger:
    """
    Enhanced logger for trading system with structured logging.
    
    Usage:
        logger = TradingLogger("execution")
        logger.info("Order executed", symbol="BTC/USDT", price=95000, quantity=0.1)
    """
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.name = name
        self.context = context or {}
        self._logger = logger.bind(module=name)
    
    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method with context."""
        extra = {**self.context, **kwargs}
        getattr(self._logger, level)(message, **extra)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self._log("critical", message, **kwargs)
    
    def trade(self, action: str, symbol: str, **kwargs: Any) -> None:
        """Log trade-related events."""
        self.info(
            f"TRADE: {action}",
            trade_action=action,
            symbol=symbol,
            category="trade",
            **kwargs
        )
    
    def order(self, order_id: str, status: str, **kwargs: Any) -> None:
        """Log order events."""
        self.info(
            f"ORDER: {status}",
            order_id=order_id,
            order_status=status,
            category="order",
            **kwargs
        )
    
    def risk(self, event: str, level: str, **kwargs: Any) -> None:
        """Log risk management events."""
        self.warning(
            f"RISK: {event}",
            risk_event=event,
            risk_level=level,
            category="risk",
            **kwargs
        )
    
    def performance(self, metric: str, value: float, **kwargs: Any) -> None:
        """Log performance metrics."""
        self.debug(
            f"PERF: {metric}",
            performance_metric=metric,
            performance_value=value,
            category="performance",
            **kwargs
        )
    
    def api(self, endpoint: str, method: str, status_code: int, duration_ms: float, **kwargs: Any) -> None:
        """Log API calls."""
        self.info(
            f"API: {method} {endpoint}",
            api_endpoint=endpoint,
            api_method=method,
            api_status_code=status_code,
            api_duration_ms=duration_ms,
            category="api",
            **kwargs
        )
    
    def with_context(self, **kwargs: Any) -> "TradingLogger":
        """Create a new logger with additional context."""
        return TradingLogger(self.name, {**self.context, **kwargs})


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_rotation: str = "10 MB",
    log_retention: str = "30 days",
    log_format: str = "json",
    serialize: bool = True
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_rotation: Log rotation size/time
        log_retention: Log retention period
        log_format: Output format (json or text)
        serialize: Whether to serialize logs as JSON
    """
    # Remove default handler
    logger.remove()
    
    # Custom format
    if log_format == "json":
        format_str = "{{message}}"
    else:
        format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=format_str,
        serialize=(log_format == "json"),
        colorize=(log_format == "text"),
        enqueue=True
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format=format_str,
            rotation=log_rotation,
            retention=log_retention,
            serialize=serialize,
            enqueue=True,
            compression="gz"
        )
    
    # Error file handler (separate file for errors)
    if log_file:
        error_log = log_file.replace(".log", "_errors.log")
        logger.add(
            error_log,
            level="ERROR",
            format=format_str,
            rotation=log_rotation,
            retention=log_retention,
            serialize=serialize,
            enqueue=True,
            compression="gz"
        )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str, **context: Any) -> TradingLogger:
    """
    Get a trading logger instance.
    
    Args:
        name: Logger name (usually module name)
        **context: Additional context to include in all logs
    
    Returns:
        TradingLogger instance
    """
    return TradingLogger(name, context)


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID for context tracking."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_ctx.set(rid)
    return rid


def set_user_id(user_id: str) -> None:
    """Set user ID for context tracking."""
    user_id_ctx.set(user_id)


def set_session_id(session_id: str) -> None:
    """Set session ID for context tracking."""
    session_id_ctx.set(session_id)


def get_request_id() -> str:
    """Get current request ID."""
    return request_id_ctx.get()


def timed(logger: Optional[TradingLogger] = None):
    """
    Decorator to time function execution.
    
    Usage:
        @timed(logger)
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                if logger:
                    logger.performance(
                        f"{func.__name__}_duration_ms",
                        duration_ms
                    )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                if logger:
                    logger.error(
                        f"{func.__name__} failed",
                        error=str(e),
                        duration_ms=duration_ms
                    )
                raise
        return wrapper
    return decorator


def async_timed(logger: Optional[TradingLogger] = None):
    """
    Decorator to time async function execution.
    
    Usage:
        @async_timed(logger)
        async def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                if logger:
                    logger.performance(
                        f"{func.__name__}_duration_ms",
                        duration_ms
                    )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                if logger:
                    logger.error(
                        f"{func.__name__} failed",
                        error=str(e),
                        duration_ms=duration_ms
                    )
                raise
        return wrapper
    return decorator


# Convenience function for backward compatibility
def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Configure logging (backward compatible)."""
    setup_logging(log_level=level, log_file=log_file, **kwargs)