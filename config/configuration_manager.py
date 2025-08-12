"""
Production Configuration Manager
Ultimate XAU Super System V4.0

Environment-aware configuration management with validation and secrets.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Production configuration management system"""
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path('config')
        self._config_cache = {}
        self._secrets_cache = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration for current environment"""
        if 'config' in self._config_cache:
            return self._config_cache['config']
            
        try:
            # Load base configuration
            base_config_path = self.config_dir / 'system_config.yaml'
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
                
            # Load environment-specific overrides
            env_config_path = self.config_dir / 'environments' / f'{self.environment}.yaml'
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    
                # Merge configurations
                config = self._merge_configs(base_config, env_config)
            else:
                config = base_config
                
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            # Validate configuration
            self._validate_config(config)
            
            self._config_cache['config'] = config
            logger.info(f"Configuration loaded for environment: {self.environment}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'database.host')"""
        config = self.load_config()
        keys = key.split('.')
        
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values"""
        required_sections = ['system', 'database', 'trading']
        
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Required configuration section missing: {section}")
                
        # Validate database configuration
        db_config = config.get('database', {})
        required_db_fields = ['host', 'port', 'database', 'username']
        
        for field in required_db_fields:
            if not db_config.get(field):
                raise ConfigurationError(f"Required database field missing: {field}")
                
        # Validate trading configuration
        trading_config = config.get('trading', {})
        if trading_config.get('initial_balance', 0) <= 0:
            raise ConfigurationError("Initial balance must be positive")
            
        logger.info("Configuration validation passed")

class ConfigurationError(Exception):
    """Configuration-related error"""
    pass

# Global configuration instance
config = ConfigurationManager()
