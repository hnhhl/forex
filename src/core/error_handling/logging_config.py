"""
Production Logging Configuration
Ultimate XAU Super System V4.0

Structured logging with multiple handlers and formatters.
"""

import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        # Add context if present
        if hasattr(record, 'context'):
            log_data['context'] = record.context
            
        return json.dumps(log_data)

def setup_production_logging(log_level: str = 'INFO', log_dir: str = 'logs') -> None:
    """Setup production logging configuration"""
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    json_formatter = JSONFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, log_level))
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/xau_system.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/xau_system_errors.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    error_handler.setFormatter(json_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    loggers = [
        'xau_system',
        'trading',
        'ai',
        'data',
        'risk',
        'monitoring'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level))
