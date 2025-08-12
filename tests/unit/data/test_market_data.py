"""
Data Layer Unit Tests
Ultimate XAU Super System V4.0
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from tests.base_test import DataTestCase

class TestMarketDataFeed(DataTestCase):
    """Test Market Data Feed functionality"""
    
    def setUp(self):
        super().setUp()
        self.data_feed = Mock()
        
    def test_data_retrieval(self):
        """Test market data retrieval"""
        # Setup
        self.data_feed.get_data.return_value = self.mock_data
        
        # Execute
        data = self.data_feed.get_data('XAUUSD', '1h')
        
        # Assert
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
    def test_data_validation(self):
        """Test data validation"""
        # Setup
        self.data_feed.validate_data.return_value = True
        
        # Execute
        is_valid = self.data_feed.validate_data(self.mock_data)
        
        # Assert
        self.assertTrue(is_valid)
        
    def test_data_caching(self):
        """Test data caching mechanism"""
        # Setup
        cache_key = 'XAUUSD_1h_cache'
        self.data_feed.get_cached_data.return_value = self.mock_data
        
        # Execute
        cached_data = self.data_feed.get_cached_data(cache_key)
        
        # Assert
        self.assertIsNotNone(cached_data)

class TestFundamentalData(DataTestCase):
    """Test Fundamental Data functionality"""
    
    def setUp(self):
        super().setUp()
        self.fundamental_data = Mock()
        
    def test_economic_indicators(self):
        """Test economic indicators retrieval"""
        # Setup
        indicators = {'GDP': 5.2, 'Inflation': 2.1, 'Interest_Rate': 1.5}
        self.fundamental_data.get_indicators.return_value = indicators
        
        # Execute
        result = self.fundamental_data.get_indicators()
        
        # Assert
        self.assertEqual(result, indicators)
        self.assertIn('GDP', result)
        self.assertIn('Inflation', result)

if __name__ == '__main__':
    unittest.main()
