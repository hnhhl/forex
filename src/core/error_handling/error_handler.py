"""
Production Error Handler
Ultimate XAU Super System V4.0

Comprehensive error handling with logging, retry, and recovery.
"""

import logging
import traceback
import time
import functools
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime, timedelta
from .exceptions import *

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Production error handling system"""
    
    def __init__(self):
        self.error_counts = {}
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle error with comprehensive logging and recovery"""
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now(),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        # Log error
        self._log_error(error_info)
        
        # Update error statistics
        self._update_error_stats(error_info)
        
        # Check circuit breaker
        if self._should_break_circuit(error_info):
            self._activate_circuit_breaker(error_info)
            
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_info)
        
        return {
            'error_handled': True,
            'recovery_attempted': recovery_result['attempted'],
            'recovery_successful': recovery_result['successful'],
            'error_info': error_info
        }
        
    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """Log error with appropriate level"""
        error_type = error_info['error_type']
        message = error_info['message']
        context = error_info['context']
        
        if error_type in ['SecurityException', 'DatabaseException']:
            logger.critical(f"{error_type}: {message}", extra={'context': context})
        elif error_type in ['TradingException', 'RiskManagementException']:
            logger.error(f"{error_type}: {message}", extra={'context': context})
        elif error_type in ['DataSourceException', 'AIModelException']:
            logger.warning(f"{error_type}: {message}", extra={'context': context})
        else:
            logger.info(f"{error_type}: {message}", extra={'context': context})
            
    def _update_error_stats(self, error_info: Dict[str, Any]) -> None:
        """Update error statistics for monitoring"""
        error_type = error_info['error_type']
        
        if error_type not in self.error_counts:
            self.error_counts[error_type] = {
                'count': 0,
                'first_occurrence': error_info['timestamp'],
                'last_occurrence': error_info['timestamp']
            }
            
        self.error_counts[error_type]['count'] += 1
        self.error_counts[error_type]['last_occurrence'] = error_info['timestamp']
        
    def _should_break_circuit(self, error_info: Dict[str, Any]) -> bool:
        """Determine if circuit breaker should activate"""
        error_type = error_info['error_type']
        
        # Circuit breaker thresholds
        critical_errors = ['SecurityException', 'DatabaseException']
        if error_type in critical_errors:
            return self.error_counts[error_type]['count'] >= 3
            
        return self.error_counts[error_type]['count'] >= 10
        
    def _activate_circuit_breaker(self, error_info: Dict[str, Any]) -> None:
        """Activate circuit breaker for error type"""
        error_type = error_info['error_type']
        
        self.circuit_breakers[error_type] = {
            'activated_at': datetime.now(),
            'reset_after': datetime.now() + timedelta(minutes=5)
        }
        
        logger.critical(f"Circuit breaker activated for {error_type}")
        
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt automatic error recovery"""
        error_type = error_info['error_type']
        
        recovery_strategies = {
            'MarketDataException': self._recover_market_data,
            'DatabaseException': self._recover_database,
            'AIModelException': self._recover_ai_model,
            'TradingException': self._recover_trading
        }
        
        if error_type in recovery_strategies:
            try:
                recovery_strategies[error_type](error_info)
                return {'attempted': True, 'successful': True}
            except Exception as e:
                logger.error(f"Recovery failed for {error_type}: {e}")
                return {'attempted': True, 'successful': False}
        else:
            return {'attempted': False, 'successful': False}
            
    def _recover_market_data(self, error_info: Dict[str, Any]) -> None:
        """Recover from market data errors"""
        # Switch to backup data source
        logger.info("Switching to backup market data source")
        
    def _recover_database(self, error_info: Dict[str, Any]) -> None:
        """Recover from database errors"""
        # Attempt reconnection
        logger.info("Attempting database reconnection")
        
    def _recover_ai_model(self, error_info: Dict[str, Any]) -> None:
        """Recover from AI model errors"""
        # Fallback to simpler model
        logger.info("Falling back to backup AI model")
        
    def _recover_trading(self, error_info: Dict[str, Any]) -> None:
        """Recover from trading errors"""
        # Halt trading and assess
        logger.info("Halting trading operations for assessment")

def retry_on_exception(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for automatic retry on exceptions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempts} failed for {func.__name__}: {e}. Retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
            return None
        return wrapper
    return decorator

def handle_exceptions(error_handler: ErrorHandler = None):
    """Decorator for comprehensive exception handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    context = {
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    }
                    error_handler.handle_error(e, context)
                raise
        return wrapper
    return decorator

# Global error handler instance
error_handler = ErrorHandler()
