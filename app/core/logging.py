"""
Structured Logging Configuration
================================
Institutional-grade logging with JSON formatting and context.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
    """
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    running_on_vercel = bool(os.getenv("VERCEL"))
    if running_on_vercel:
        # Vercel runtime filesystem is read-only for app code paths.
        file_path = None
    
    # Create logs directory if needed
    if file_path:
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fall back to console-only logging when filesystem is not writable.
            file_path = None
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with JSON format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if file_path:
        try:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(file_handler)
        except OSError:
            logging.warning("File logging disabled: unable to open log file")
    
    # Set third-party loggers to WARNING
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured: level={level}, file={file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """
    Specialized logger for trading operations.
    Adds trading-specific context to logs.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs: Any) -> None:
        """Set contextual information for subsequent logs."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear contextual information."""
        self.context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs: Any) -> None:
        """Log message with context."""
        extra = {**self.context, **kwargs}
        log_func = getattr(self.logger, level.lower())
        log_func(message, extra={"extra": extra})
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log_with_context("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log_with_context("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log_with_context("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self._log_with_context("CRITICAL", message, **kwargs)
    
    # Trading-specific methods
    def log_order(self, order_id: str, symbol: str, side: str, 
                  quantity: float, price: float) -> None:
        """Log order details."""
        self.info(
            f"Order {order_id}: {side} {quantity} {symbol} @ {price}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            event="order"
        )
    
    def log_trade(self, trade_id: str, symbol: str, side: str,
                  quantity: float, price: float, pnl: float) -> None:
        """Log trade execution."""
        self.info(
            f"Trade {trade_id}: {side} {quantity} {symbol} @ {price}, PnL: {pnl}",
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            event="trade"
        )
    
    def log_signal(self, strategy: str, symbol: str, signal: str,
                   confidence: float) -> None:
        """Log trading signal."""
        self.info(
            f"Signal: {strategy} on {symbol}: {signal} (confidence: {confidence})",
            strategy=strategy,
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            event="signal"
        )
    
    def log_risk_violation(self, limit_type: str, current: float,
                           limit: float) -> None:
        """Log risk limit violation."""
        self.warning(
            f"Risk violation: {limit_type} = {current:.4f} exceeds limit {limit:.4f}",
            limit_type=limit_type,
            current=current,
            limit=limit,
            event="risk_violation"
        )
