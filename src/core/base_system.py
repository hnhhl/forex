"""
Base System Class
Foundation class for all trading and risk management systems
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemConfig:
    """Configuration class for trading systems"""
    
    def __init__(self):
        # Trading configuration
        self.initial_balance = 10000.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_positions = 5
        self.leverage = 1.0
        
        # AI/ML configuration
        self.use_neural_ensemble = True
        self.use_reinforcement_learning = True
        self.ensemble_confidence_threshold = 0.6
        self.rl_exploration_rate = 0.1
        
        # Data configuration
        self.data_timeframe = 'M1'
        self.sequence_length = 60
        self.features_count = 5
        
        # System configuration
        self.enable_logging = True
        self.enable_monitoring = True
        self.enable_alerts = True
        self.enable_kelly_criterion = True
        self.enable_position_sizing = True
        
        # Performance configuration
        self.max_drawdown = 0.15  # 15% max drawdown
        self.profit_target = 0.25  # 25% profit target
        
        # GPU configuration
        self.use_gpu = True
        self.gpu_memory_limit = 3072  # MB
        
        # Timeouts
        self.connection_timeout = 30
        self.trade_timeout = 60
        
        # MT5 Configuration
        self.mt5_login = None
        self.mt5_password = None
        self.mt5_server = None
        self.mt5_timeout = 30
        
        # Monitoring
        self.monitoring_frequency = 1  # seconds
        
        # Alerts
        self.alert_channels = ['console', 'file']
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            attr: getattr(self, attr) 
            for attr in dir(self) 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update config from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BaseSystem(ABC):
    """
    Base class for all trading and risk management systems
    
    Provides common functionality:
    - System identification and configuration
    - Logging and error handling
    - Status tracking
    - Statistics collection
    """
    
    def __init__(self, system_name: str, config: Optional[Dict] = None):
        """Initialize base system"""
        self.system_name = system_name
        self.config = config or {}
        self.created_at = datetime.now()
        self.is_active = False
        self.last_update = None
        self.error_count = 0
        self.operation_count = 0
        
        logger.info(f"{self.system_name} initialized")
    
    def start(self) -> bool:
        """Start the system"""
        try:
            self.is_active = True
            self.last_update = datetime.now()
            logger.info(f"{self.system_name} started")
            return True
        except Exception as e:
            logger.error(f"Error starting {self.system_name}: {e}")
            self.error_count += 1
            return False
    
    def stop(self) -> bool:
        """Stop the system"""
        try:
            self.is_active = False
            self.last_update = datetime.now()
            logger.info(f"{self.system_name} stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping {self.system_name}: {e}")
            self.error_count += 1
            return False
    
    def update_status(self):
        """Update system status"""
        self.last_update = datetime.now()
        self.operation_count += 1
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """Log system error"""
        self.error_count += 1
        if exception:
            logger.error(f"{self.system_name} error: {error_message} - {exception}")
        else:
            logger.error(f"{self.system_name} error: {error_message}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'system_name': self.system_name,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'error_count': self.error_count,
            'operation_count': self.operation_count,
            'uptime_seconds': (datetime.now() - self.created_at).total_seconds()
        }
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get system-specific statistics - must be implemented by subclasses"""
        pass 