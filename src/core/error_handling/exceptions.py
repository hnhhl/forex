"""
Custom Exception Classes
Ultimate XAU Super System V4.0

Comprehensive exception hierarchy for specific error handling.
"""

from typing import Optional, Dict, Any
from datetime import datetime

class XAUSystemException(Exception):
    """Base exception for XAU System"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }

class DataSourceException(XAUSystemException):
    """Data source related exceptions"""
    pass

class MarketDataException(DataSourceException):
    """Market data specific exceptions"""
    pass

class FundamentalDataException(DataSourceException):
    """Fundamental data specific exceptions"""
    pass

class AIModelException(XAUSystemException):
    """AI/ML model related exceptions"""
    pass

class NeuralEnsembleException(AIModelException):
    """Neural ensemble specific exceptions"""
    pass

class ReinforcementLearningException(AIModelException):
    """Reinforcement learning specific exceptions"""
    pass

class TradingException(XAUSystemException):
    """Trading operation exceptions"""
    pass

class OrderExecutionException(TradingException):
    """Order execution specific exceptions"""
    pass

class PositionManagementException(TradingException):
    """Position management specific exceptions"""
    pass

class RiskManagementException(TradingException):
    """Risk management specific exceptions"""
    pass

class ConfigurationException(XAUSystemException):
    """Configuration related exceptions"""
    pass

class DatabaseException(XAUSystemException):
    """Database operation exceptions"""
    pass

class ValidationException(XAUSystemException):
    """Data validation exceptions"""
    pass

class IntegrationException(XAUSystemException):
    """System integration exceptions"""
    pass

class SecurityException(XAUSystemException):
    """Security related exceptions"""
    pass

class PerformanceException(XAUSystemException):
    """Performance related exceptions"""
    pass
