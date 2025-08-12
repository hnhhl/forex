"""
Test Order Manager
Unit tests cho OrderManager system
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.trading.order_types import OrderRequest, OrderType, OrderStatus
from src.core.trading.order_manager import OrderManager


class TestOrderManager(unittest.TestCase):
    """Test cases for OrderManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_concurrent_orders': 5,
            'order_timeout': 10,
            'retry_attempts': 2,
            'retry_delay': 0.1
        }
        
        # Mock MT5 to avoid actual connection
        with patch('core.trading.order_manager.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            self.order_manager = OrderManager(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'order_manager'):
            with patch('core.trading.order_manager.mt5'):
                self.order_manager.stop()
    
    def test_initialization(self):
        """Test OrderManager initialization"""
        self.assertEqual(self.order_manager.name, "OrderManager")
        self.assertEqual(self.order_manager.max_concurrent_orders, 5)
        self.assertEqual(self.order_manager.order_timeout, 10)
        self.assertIsNotNone(self.order_manager.validator)
        self.assertIsNotNone(self.order_manager.executor)
    
    @patch('src.core.trading.order_manager.mt5')
    def test_start_stop(self, mock_mt5):
        """Test start and stop functionality"""
        mock_mt5.initialize.return_value = True
        mock_mt5.shutdown.return_value = None
        
        # Test start
        self.order_manager.start()
        self.assertTrue(self.order_manager.is_active)
        self.assertTrue(self.order_manager.is_monitoring)
        
        # Test stop
        self.order_manager.stop()
        self.assertFalse(self.order_manager.is_active)
        self.assertFalse(self.order_manager.is_monitoring)
    
    def test_order_request_validation(self):
        """Test order request validation"""
        # Valid order request
        valid_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1,
            stop_loss=1900.0,
            take_profit=2100.0,
            comment="Test order"
        )
        
        # Mock validator to return valid
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            success, message, order_id = self.order_manager.submit_order(valid_request)
            
            self.assertTrue(success)
            self.assertIsNotNone(order_id)
            self.assertEqual(message, "Order submitted successfully")
    
    def test_order_validation_failure(self):
        """Test order validation failure"""
        invalid_request = OrderRequest(
            symbol="",  # Invalid empty symbol
            order_type=OrderType.MARKET_BUY,
            volume=0.0  # Invalid zero volume
        )
        
        # Mock validator to return invalid
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (False, "Symbol is required")
            
            success, message, order_id = self.order_manager.submit_order(invalid_request)
            
            self.assertFalse(success)
            self.assertIsNone(order_id)
            self.assertIn("Validation failed", message)
    
    @patch('src.core.trading.order_manager.mt5')
    def test_successful_order_execution(self, mock_mt5):
        """Test successful order execution"""
        # Mock MT5 response
        mock_result = Mock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_result.volume = 0.1
        mock_result.price = 2000.0
        mock_result.comment = "Success"
        
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TIME_GTC = 0
        
        # Create valid order request
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1
        )
        
        # Mock validator
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            # Submit order
            success, message, order_id = self.order_manager.submit_order(order_request)
            
            self.assertTrue(success)
            self.assertIsNotNone(order_id)
            
            # Wait a bit for execution (in real scenario)
            import time
            time.sleep(0.1)
            
            # Check statistics
            stats = self.order_manager.get_statistics()
            self.assertEqual(stats['total_orders'], 1)
    
    @patch('src.core.trading.order_manager.mt5')
    def test_failed_order_execution(self, mock_mt5):
        """Test failed order execution"""
        # Mock MT5 response for failure
        mock_result = Mock()
        mock_result.retcode = 10013  # Invalid request
        mock_result.comment = "Invalid volume"
        
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TIME_GTC = 0
        
        # Create order request
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1
        )
        
        # Mock validator
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            # Submit order
            success, message, order_id = self.order_manager.submit_order(order_request)
            
            self.assertTrue(success)  # Submission successful
            self.assertIsNotNone(order_id)
            
            # Wait for execution to complete
            import time
            time.sleep(0.2)
            
            # Check that order was moved to history with failed status
            history = self.order_manager.get_order_history()
            self.assertTrue(len(history) > 0)
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        # Create a mock order
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.LIMIT_BUY,
            volume=0.1,
            price=1950.0
        )
        
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            # Submit order
            success, message, order_id = self.order_manager.submit_order(order_request)
            self.assertTrue(success)
            
            # Cancel order
            cancel_success, cancel_message = self.order_manager.cancel_order(order_id, "Test cancellation")
            self.assertTrue(cancel_success)
            self.assertEqual(cancel_message, "Order cancelled")
    
    def test_get_active_orders(self):
        """Test getting active orders"""
        # Initially no active orders
        active_orders = self.order_manager.get_active_orders()
        self.assertEqual(len(active_orders), 0)
        
        # Add an order
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.LIMIT_BUY,
            volume=0.1,
            price=1950.0
        )
        
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            success, message, order_id = self.order_manager.submit_order(order_request)
            self.assertTrue(success)
            
            # Check active orders
            active_orders = self.order_manager.get_active_orders()
            self.assertEqual(len(active_orders), 1)
            self.assertEqual(active_orders[0].order_id, order_id)
    
    def test_statistics(self):
        """Test statistics tracking"""
        initial_stats = self.order_manager.get_statistics()
        self.assertEqual(initial_stats['total_orders'], 0)
        self.assertEqual(initial_stats['successful_orders'], 0)
        self.assertEqual(initial_stats['failed_orders'], 0)
        
        # Submit an order to update stats
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1
        )
        
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            success, message, order_id = self.order_manager.submit_order(order_request)
            self.assertTrue(success)
            
            # Check updated stats
            updated_stats = self.order_manager.get_statistics()
            self.assertEqual(updated_stats['total_orders'], 1)
    
    def test_export_orders(self):
        """Test order export functionality"""
        # Create and submit an order
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1
        )
        
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            success, message, order_id = self.order_manager.submit_order(order_request)
            self.assertTrue(success)
            
            # Export orders
            filename = self.order_manager.export_orders("test_export.json")
            self.assertTrue(filename.endswith(".json"))
            
            # Check if file exists
            import os
            self.assertTrue(os.path.exists(filename))
            
            # Clean up
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_callbacks(self):
        """Test event callbacks"""
        callback_called = []
        
        def test_callback(order):
            callback_called.append(order.order_id)
        
        # Add callback
        self.order_manager.add_callback('order_created', test_callback)
        
        # Submit order
        order_request = OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.MARKET_BUY,
            volume=0.1
        )
        
        with patch.object(self.order_manager.validator, 'validate_order') as mock_validate:
            mock_validate.return_value = (True, "Valid order")
            
            success, message, order_id = self.order_manager.submit_order(order_request)
            self.assertTrue(success)
            
            # Check if callback was called
            self.assertEqual(len(callback_called), 1)
            self.assertEqual(callback_called[0], order_id)


if __name__ == '__main__':
    unittest.main() 