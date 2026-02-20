"""
Production-Grade Structured Logging
====================================
Institutional logging with JSON formatting, correlation IDs,
log aggregation support, and compliance-ready audit trails.
"""

import json
import logging
import sys
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from contextvars import ContextVar
import uuid
import socket
import threading
from functools import wraps
import time


# ============================================================================
# CONTEXT VARIABLES FOR CORRELATION
# ============================================================================

# Request/correlation ID tracking
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')
request_id: ContextVar[str] = ContextVar('request_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')
session_id: ContextVar[str] = ContextVar('session_id', default='')


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(cid)


def new_correlation_id() -> str:
    """Generate and set new correlation ID."""
    cid = str(uuid.uuid4())[:8]
    correlation_id.set(cid)
    return cid


# ============================================================================
# LOG LEVELS AND CATEGORIES
# ============================================================================

class LogLevel(str, Enum):
    """Enhanced log levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    ALERT = "ALERT"
    EMERGENCY = "EMERGENCY"


class LogCategory(str, Enum):
    """Log categories for filtering and routing."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    API = "api"
    DATA = "data"
    ML = "ml"
    COMPLIANCE = "compliance"


# ============================================================================
# JSON FORMATTER
# ============================================================================

class ProductionJSONFormatter(logging.Formatter):
    """
    Production-grade JSON formatter for structured logging.
    
    Features:
    - ISO 8601 timestamps with timezone
    - Correlation IDs for distributed tracing
    - Host and process information
    - Structured extra fields
    - Exception stack traces
    - Sensitive data masking
    """
    
    # Fields to mask for security
    SENSITIVE_FIELDS = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'api_key',
        'api_secret', 'private_key', 'access_token', 'refresh_token',
        'authorization', 'credit_card', 'ssn', 'social_security'
    }
    
    def __init__(
        self,
        service_name: str = "ai-trading-system",
        environment: str = "production",
        include_hostname: bool = True,
        include_process_info: bool = True,
        mask_sensitive: bool = True
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.include_hostname = include_hostname
        self.include_process_info = include_process_info
        self.mask_sensitive = mask_sensitive
        self._hostname = socket.gethostname() if include_hostname else None
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as production-ready JSON."""
        # Base log entry
        log_entry: Dict[str, Any] = {
            # Timestamp in ISO 8601 format with timezone
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "@version": "1",
            
            # Message
            "message": record.getMessage(),
            
            # Log level
            "log.level": record.levelname.lower(),
            "log.logger": record.name,
            
            # Source information
            "log.origin": {
                "file": {
                    "name": record.filename,
                    "line": record.lineno,
                    "function": record.funcName
                },
                "module": record.module
            },
            
            # Service information
            "service": {
                "name": self.service_name,
                "environment": self.environment,
                "type": "trading"
            },
            
            # Event category
            "event": {
                "category": getattr(record, 'category', LogCategory.SYSTEM.value),
                "type": getattr(record, 'event_type', 'info'),
                "severity": record.levelname.lower()
            }
        }
        
        # Add hostname
        if self.include_hostname and self._hostname:
            log_entry["host"] = {"name": self._hostname}
        
        # Add process information
        if self.include_process_info:
            log_entry["process"] = {
                "pid": record.process,
                "name": record.processName,
                "thread": {
                    "id": record.thread,
                    "name": record.threadName
                }
            }
        
        # Add correlation IDs
        cid = get_correlation_id()
        if cid:
            log_entry["correlation_id"] = cid
        
        rid = request_id.get()
        if rid:
            log_entry["request_id"] = rid
        
        uid = user_id.get()
        if uid:
            log_entry["user"] = {"id": uid}
        
        sid = session_id.get()
        if sid:
            log_entry["session_id"] = sid
        
        # Add extra fields from record
        extra_fields = {}
        if hasattr(record, 'extra') and isinstance(record.extra, dict):
            extra_fields = self._mask_sensitive_data(record.extra)
        
        # Add standard extra attributes
        for attr in ['symbol', 'order_id', 'trade_id', 'strategy', 'portfolio_id',
                     'duration_ms', 'status_code', 'error_code', 'stack_trace']:
            if hasattr(record, attr):
                extra_fields[attr] = getattr(record, attr)
        
        if extra_fields:
            log_entry["data"] = extra_fields
        
        # Add exception info if present
        if record.exc_info:
            log_entry["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "stack_trace": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive fields in data."""
        if not self.mask_sensitive:
            return data
        
        result = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                result[key] = "***MASKED***"
            elif isinstance(value, dict):
                result[key] = self._mask_sensitive_data(value)
            elif isinstance(value, list):
                result[key] = [
                    self._mask_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# ============================================================================
# LOG HANDLERS
# ============================================================================

class RotatingJSONFileHandler(logging.Handler):
    """
    Rotating file handler with JSON formatting.
    Supports size-based and time-based rotation.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        filename_prefix: str = "trading",
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        compress_backups: bool = True
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.filename_prefix = filename_prefix
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress_backups = compress_backups
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self._get_current_file()
        self._file_handle = None
        self._lock = threading.Lock()
    
    def _get_current_file(self) -> Path:
        """Get current log file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{self.filename_prefix}-{date_str}.json"
    
    def _open_file(self):
        """Open log file for writing."""
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = open(self.current_file, 'a', encoding='utf-8')
    
    def emit(self, record: logging.LogRecord) -> None:
        """Write log record to file."""
        with self._lock:
            try:
                # Check if we need to rotate
                if self.current_file != self._get_current_file():
                    self._rotate()
                
                self._open_file()
                
                msg = self.format(record)
                self._file_handle.write(msg + '\n')
                self._file_handle.flush()
                
                # Check size rotation
                if self._file_handle.tell() > self.max_bytes:
                    self._rotate_size()
                    
            except Exception:
                self.handleError(record)
    
    def _rotate(self):
        """Rotate to new date-based file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self.current_file = self._get_current_file()
    
    def _rotate_size(self):
        """Rotate based on file size."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        
        # Rename current file with timestamp
        timestamp = datetime.now().strftime("%H%M%S")
        rotated_name = f"{self.current_file.stem}-{timestamp}.json"
        rotated_path = self.log_dir / rotated_name
        self.current_file.rename(rotated_path)
        
        # Compress if enabled
        if self.compress_backups:
            import gzip
            with open(rotated_path, 'rb') as f_in:
                with gzip.open(f"{rotated_path}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            rotated_path.unlink()
        
        # Clean old backups
        self._clean_old_backups()
        
        self.current_file = self._get_current_file()
    
    def _clean_old_backups(self):
        """Remove old backup files."""
        pattern = f"{self.filename_prefix}-*.json.gz"
        backups = sorted(self.log_dir.glob(pattern), reverse=True)
        
        for backup in backups[self.backup_count:]:
            backup.unlink()
    
    def close(self):
        """Close the handler."""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
        super().close()


class ElasticsearchHandler(logging.Handler):
    """
    Handler for sending logs to Elasticsearch.
    Compatible with Elastic Common Schema (ECS).
    """
    
    def __init__(
        self,
        hosts: List[str],
        index_prefix: str = "trading-logs",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0
    ):
        super().__init__()
        self.hosts = hosts
        self.index_prefix = index_prefix
        self.username = username
        self.password = password
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._buffer: List[str] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Buffer log record for batch sending."""
        with self._lock:
            self._buffer.append(self.format(record))
            
            if len(self._buffer) >= self.batch_size or \
               time.time() - self._last_flush >= self.flush_interval:
                self._flush()
    
    def _flush(self):
        """Send buffered logs to Elasticsearch."""
        if not self._buffer:
            return
        
        try:
            import requests
            
            # Build bulk request body
            bulk_body = ""
            today = datetime.now().strftime("%Y.%m.%d")
            index_name = f"{self.index_prefix}-{today}"
            
            for doc in self._buffer:
                bulk_body += f'{{"index":{{"_index":"{index_name}"}}}}\n'
                bulk_body += doc + '\n'
            
            headers = {"Content-Type": "application/json"}
            auth = None
            
            if self.username and self.password:
                auth = (self.username, self.password)
            elif self.api_key:
                headers["Authorization"] = f"ApiKey {self.api_key}"
            
            for host in self.hosts:
                try:
                    response = requests.post(
                        f"{host}/_bulk",
                        data=bulk_body,
                        headers=headers,
                        auth=auth,
                        timeout=10
                    )
                    if response.status_code == 200:
                        break
                except Exception:
                    continue
            
            self._buffer.clear()
            self._last_flush = time.time()
            
        except Exception as e:
            # Fallback: write to local file
            fallback_path = Path("logs/elasticsearch-fallback.json")
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fallback_path, 'a') as f:
                for doc in self._buffer:
                    f.write(doc + '\n')
            self._buffer.clear()
    
    def close(self):
        """Flush remaining logs and close handler."""
        with self._lock:
            self._flush()
        super().close()


# ============================================================================
# TRADING-SPECIFIC LOGGER
# ============================================================================

class TradingLogger:
    """
    Specialized logger for trading operations.
    Provides domain-specific logging methods with structured context.
    """
    
    def __init__(
        self,
        name: str,
        category: LogCategory = LogCategory.TRADING
    ):
        self.logger = logging.getLogger(name)
        self.category = category
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs: Any) -> None:
        """Set persistent context for all subsequent log calls."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()
    
    def _log(
        self,
        level: int,
        message: str,
        log_category: Optional[LogCategory] = None,
        event_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Internal logging method with context."""
        extra_data = {**self._context, **kwargs}
        extra = {
            'extra': extra_data,
            'log_category': (log_category or self.category).value,
            'event_type': event_type
        }
        self.logger.log(level, message, extra=extra)
    
    # Standard log methods
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)
    
    # Trading-specific methods
    def log_order_created(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float],
        order_type: str
    ) -> None:
        """Log order creation."""
        self.info(
            f"Order created: {order_id}",
            event_type="order_created",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type
        )
    
    def log_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float,
        pnl: Optional[float] = None
    ) -> None:
        """Log order fill."""
        self.info(
            f"Order filled: {order_id} @ {price}",
            event_type="order_filled",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            pnl=pnl
        )
    
    def log_order_rejected(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        error_code: Optional[str] = None
    ) -> None:
        """Log order rejection."""
        self.warning(
            f"Order rejected: {order_id} - {reason}",
            event_type="order_rejected",
            order_id=order_id,
            symbol=symbol,
            reason=reason,
            error_code=error_code
        )
    
    def log_signal(
        self,
        strategy: str,
        symbol: str,
        action: str,
        confidence: float,
        factors: Optional[Dict] = None
    ) -> None:
        """Log trading signal."""
        self.info(
            f"Signal: {strategy} -> {action} {symbol} ({confidence:.2%})",
            event_type="signal",
            log_category=LogCategory.TRADING,
            strategy=strategy,
            symbol=symbol,
            action=action,
            confidence=confidence,
            factors=factors
        )
    
    def log_risk_violation(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        severity: str = "warning"
    ) -> None:
        """Log risk limit violation."""
        level = logging.WARNING if severity == "warning" else logging.ERROR
        self._log(
            level,
            f"Risk violation: {limit_type} = {current_value:.4f} (limit: {limit_value:.4f})",
            log_category=LogCategory.RISK,
            event_type="risk_violation",
            limit_type=limit_type,
            current_value=current_value,
            limit_value=limit_value,
            severity=severity
        )
    
    def log_position_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        strategy: str
    ) -> None:
        """Log position opening."""
        self.info(
            f"Position opened: {side} {quantity} {symbol} @ {entry_price}",
            event_type="position_opened",
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            strategy=strategy
        )
    
    def log_position_closed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        strategy: str
    ) -> None:
        """Log position closing."""
        level = logging.INFO if pnl >= 0 else logging.WARNING
        self._log(
            level,
            f"Position closed: {symbol} @ {exit_price}, PnL: {pnl:.2f} ({pnl_pct:.2%})",
            event_type="position_closed",
            symbol=symbol,
            side=side,
            quantity=quantity,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=strategy
        )
    
    def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        error: Optional[str] = None
    ) -> None:
        """Log API call."""
        level = logging.INFO if status_code < 400 else logging.WARNING
        self._log(
            level,
            f"API call: {method} {endpoint} -> {status_code} ({duration_ms:.0f}ms)",
            log_category=LogCategory.API,
            event_type="api_call",
            api_name=api_name,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            error=error
        )
    
    def log_performance(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict] = None
    ) -> None:
        """Log performance metric."""
        self.info(
            f"Performance: {metric_name} = {value}{unit}",
            log_category=LogCategory.PERFORMANCE,
            event_type="metric",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            tags=tags
        )
    
    def log_audit(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        changes: Optional[Dict] = None
    ) -> None:
        """Log audit event for compliance."""
        self.info(
            f"Audit: {action} on {resource_type}/{resource_id}",
            log_category=LogCategory.AUDIT,
            event_type="audit",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            changes=changes
        )


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_production_logging(
    service_name: str = "ai-trading-system",
    environment: str = "production",
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_elasticsearch: bool = False,
    elasticsearch_hosts: Optional[List[str]] = None,
    elasticsearch_api_key: Optional[str] = None
) -> None:
    """
    Configure production-grade logging.
    
    Args:
        service_name: Name of the service
        environment: Environment (development, staging, production)
        log_level: Minimum log level
        log_dir: Directory for log files
        enable_file_logging: Enable rotating file handler
        enable_elasticsearch: Enable Elasticsearch handler
        elasticsearch_hosts: List of Elasticsearch hosts
        elasticsearch_api_key: API key for Elasticsearch
    """
    # Create formatter
    formatter = ProductionJSONFormatter(
        service_name=service_name,
        environment=environment
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file_logging:
        file_handler = RotatingJSONFileHandler(
            log_dir=log_dir,
            filename_prefix=service_name,
            max_bytes=100 * 1024 * 1024,  # 100MB
            backup_count=10
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Elasticsearch handler
    if enable_elasticsearch and elasticsearch_hosts:
        es_handler = ElasticsearchHandler(
            hosts=elasticsearch_hosts,
            api_key=elasticsearch_api_key
        )
        es_handler.setLevel(getattr(logging, log_level.upper()))
        es_handler.setFormatter(formatter)
        root_logger.addHandler(es_handler)
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    # Log startup
    root_logger.info(
        "Logging configured",
        extra={
            "extra": {
                "service_name": service_name,
                "environment": environment,
                "log_level": log_level
            }
        }
    )


def get_trading_logger(name: str, category: LogCategory = LogCategory.TRADING) -> TradingLogger:
    """
    Get a trading-specific logger instance.
    
    Args:
        name: Logger name (typically __name__)
        category: Log category for filtering
    
    Returns:
        TradingLogger instance
    """
    return TradingLogger(name, category)


# ============================================================================
# DECORATORS
# ============================================================================

def log_execution(
    logger: Optional[TradingLogger] = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False
):
    """
    Decorator to log function execution.
    
    Usage:
        @log_execution(logger, include_args=True)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_trading_logger(func.__module__)
            
            start_time = time.time()
            func_name = func.__qualname__
            
            extra = {}
            if include_args:
                extra['args'] = str(args)[:200]
                extra['kwargs'] = str(kwargs)[:200]
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                if include_result:
                    extra['result'] = str(result)[:200]
                
                func_logger._log(
                    level,
                    f"Function {func_name} completed in {duration_ms:.2f}ms",
                    event_type="function_execution",
                    duration_ms=duration_ms,
                    **extra
                )
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                func_logger.error(
                    f"Function {func_name} failed after {duration_ms:.2f}ms: {e}",
                    event_type="function_error",
                    duration_ms=duration_ms,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    **extra
                )
                raise
        
        return wrapper
    return decorator


def with_correlation_id(func):
    """
    Decorator to create new correlation ID for function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_correlation_id()
        return func(*args, **kwargs)
    return wrapper
