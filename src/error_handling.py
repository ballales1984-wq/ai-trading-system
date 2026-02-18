"""
Centralized Error Handling Module
Provides consistent error handling across the application
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TradingSystemError(Exception):
    """Base exception for trading system"""
    def __init__(self, message: str, details: Dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class DataError(TradingSystemError):
    """Data-related errors"""
    pass


class MLModelError(TradingSystemError):
    """ML model errors"""
    pass


class APIError(TradingSystemError):
    """API-related errors"""
    pass


class SignalGenerationError(TradingSystemError):
    """Signal generation errors"""
    pass


class RiskManagementError(TradingSystemError):
    """Risk management errors"""
    pass


class ErrorHandler:
    """
    Centralized error handler with logging and recovery
    """
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 100
    
    def handle_error(self, error: Exception, context: Dict = None) -> Dict:
        """
        Handle error with logging and tracking
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            Error information dict
        """
        error_type = type(error).__name__
        timestamp = datetime.now().isoformat()
        
        # Count errors
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create error info
        error_info = {
            'timestamp': timestamp,
            'type': error_type,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log error
        if isinstance(error, TradingSystemError):
            logger.error(f"{error_type}: {error.message}", extra=error.details)
        else:
            logger.error(f"{error_type}: {str(error)}")
        
        return error_info
    
    def get_error_stats(self) -> Dict:
        """Get error statistics"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'by_type': self.error_counts.copy(),
            'recent_errors': len(self.error_history)
        }
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors"""
        return self.error_history[-limit:]
    
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_counts.clear()


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler"""
    return _error_handler


def with_error_handling(error_type: str = "Error", default_return: Any = None):
    """
    Decorator for consistent error handling
    
    Args:
        error_type: Type of error to log
        default_return: Value to return on error
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = _error_handler.handle_error(
                    e, 
                    {
                        'function': func.__name__,
                        'args': str(args)[:100],
                        'kwargs': str(kwargs)[:100]
                    }
                )
                
                if default_return is not None:
                    return default_return
                raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value to return on error
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        _error_handler.handle_error(
            e,
            {'function': func.__name__}
        )
        return default


# Error recovery strategies
class ErrorRecovery:
    """Error recovery strategies"""
    
    @staticmethod
    def retry_with: Callable, max_backoff(func_retries: int = 3, 
                          initial_delay: float = 1.0) -> Any:
        """
        Retry function with exponential backoff
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            
        Returns:
            Function result
        """
        import time
        
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        raise last_error
    
    @staticmethod
    def fallback_to_default(func: Callable, default_value: Any) -> Any:
        """
        Fallback to default value on error
        
        Args:
            func: Function to execute
            default_value: Default value to return on error
            
        Returns:
            Function result or default value
        """
        try:
            return func()
        except Exception as e:
            _error_handler.handle_error(e, {'function': func.__name__})
            return default_value


if __name__ == "__main__":
    # Test error handling
    handler = ErrorHandler()
    
    # Test custom exception
    try:
        raise MLModelError("Model not trained", {'model_id': 'test'})
    except TradingSystemError as e:
        info = handler.handle_error(e, {'context': 'testing'})
        print(f"Error handled: {info['type']}")
    
    # Test stats
    print(f"Stats: {handler.get_error_stats()}")
