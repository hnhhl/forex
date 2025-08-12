"""
Test Configuration
Ultimate XAU Super System V4.0
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'database': 'xau_test_db',
        'username': 'test_user',
        'password': 'test_password'
    },
    'trading': {
        'initial_balance': 10000.0,
        'paper_trading': True,
        'max_position_size': 0.1
    },
    'ai': {
        'mock_models': True,
        'fast_training': True
    }
}

@pytest.fixture(scope='session')
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG

@pytest.fixture
def mock_market_data():
    """Mock market data fixture"""
    return {
        'timestamp': '2025-06-17T18:30:00Z',
        'open': 2000.0,
        'high': 2010.0,
        'low': 1990.0,
        'close': 2005.0,
        'volume': 1000000
    }

@pytest.fixture
def mock_ai_model():
    """Mock AI model fixture"""
    model = Mock()
    model.predict.return_value = [0.7, 0.3]  # Buy probability, Sell probability
    model.train.return_value = {'accuracy': 0.85, 'loss': 0.15}
    return model
