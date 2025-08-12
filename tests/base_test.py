"""
Base Test Classes
Ultimate XAU Super System V4.0
"""

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class BaseTestCase(unittest.TestCase):
    """Base test case with common setup"""
    
    def setUp(self):
        """Setup for each test"""
        self.start_time = datetime.now()
        self.mock_data = self.create_mock_data()
        
    def tearDown(self):
        """Cleanup after each test"""
        pass
        
    def create_mock_data(self):
        """Create mock data for testing"""
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        prices = 2000 + np.cumsum(np.random.randn(100) * 10)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 5,
            'high': prices + np.random.randn(100) * 5 + 10,
            'low': prices + np.random.randn(100) * 5 - 10,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
    def assert_close(self, actual, expected, tolerance=1e-6):
        """Assert values are close within tolerance"""
        self.assertTrue(abs(actual - expected) < tolerance,
                       f"Expected {expected}, got {actual}, tolerance {tolerance}")

class AITestCase(BaseTestCase):
    """Base test case for AI components"""
    
    def setUp(self):
        super().setUp()
        self.mock_model = Mock()
        self.mock_model.predict.return_value = [0.7, 0.3]
        
class TradingTestCase(BaseTestCase):
    """Base test case for trading components"""
    
    def setUp(self):
        super().setUp()
        self.initial_balance = 10000.0
        self.mock_broker = Mock()
        
class DataTestCase(BaseTestCase):
    """Base test case for data components"""
    
    def setUp(self):
        super().setUp()
        self.mock_api = Mock()
        self.mock_api.get_data.return_value = self.mock_data
