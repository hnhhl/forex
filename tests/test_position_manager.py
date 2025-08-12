"""
Unit Tests for Position Management System
Comprehensive testing cho position manager, calculator, và stop loss manager
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid
from datetime import datetime, timedelta
import threading
import time

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.trading.position_manager import PositionManager
from src.core.trading.position_calculator import PositionCalculator, PositionSizingMethod, PnLCalculationType
from src.core.trading.stop_loss_manager import StopLossManager, StopLossRule, StopLossType, TrailingStopMethod
from src.core.trading.position_types import (
    Position, PositionType, PositionStatus, PositionModifyRequest, 
    PositionCloseRequest, PositionSummary
)


class TestPositionManager(unittest.TestCase):
    """Test Position Manager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'update_interval': 0.1,  # Fast for testing
            'max_positions': 10,
            'auto_sync': False  # Disable MT5 sync for testing
        }
        self.manager = PositionManager(self.config)
        
        # Mock MT5
        self.mt5_patcher = patch('src.core.trading.position_manager.mt5')
        self.mock_mt5 = self.mt5_patcher.start()
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.shutdown.return_value = None
        self.mock_mt5.TRADE_RETCODE_DONE = 10009
        self.mock_mt5.TRADE_ACTION_DEAL = 1
        self.mock_mt5.TRADE_ACTION_SLTP = 2
        self.mock_mt5.ORDER_TYPE_BUY = 0
        self.mock_mt5.ORDER_TYPE_SELL = 1
    
    def tearDown(self):
        """Clean up after tests"""
        if self.manager.is_active:
            self.manager.stop()
        self.mt5_patcher.stop()
    
    def test_manager_initialization(self):
        """Test position manager initialization"""
        self.assertEqual(self.manager.name, "PositionManager")
        self.assertFalse(self.manager.is_active)
        self.assertEqual(len(self.manager.positions), 0)
        self.assertEqual(self.manager.stats['total_positions'], 0)
    
    def test_manager_start_stop(self):
        """Test manager start and stop"""
        # Start manager
        self.manager.start()
        self.assertTrue(self.manager.is_active)
        self.assertTrue(self.manager.is_monitoring)
        self.assertIsNotNone(self.manager.monitoring_thread)
        
        # Stop manager
        self.manager.stop()
        self.assertFalse(self.manager.is_active)
        self.assertFalse(self.manager.is_monitoring)
    
    def test_add_position_from_order(self):
        """Test adding position from order"""
        self.manager.start()
        
        # Add position
        position_id = self.manager.add_position_from_order(
            ticket=12345,
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0,
            stop_loss=1990.0,
            take_profit=2020.0,
            comment="Test position"
        )
        
        # Verify position added
        self.assertIsNotNone(position_id)
        self.assertIn(position_id, self.manager.positions)
        
        position = self.manager.positions[position_id]
        self.assertEqual(position.ticket, 12345)
        self.assertEqual(position.symbol, "XAUUSD")
        self.assertEqual(position.position_type, PositionType.BUY)
        self.assertEqual(position.volume, 0.1)
        self.assertEqual(position.open_price, 2000.0)
        
        # Verify statistics updated
        self.assertEqual(self.manager.stats['total_positions'], 1)
        self.assertEqual(self.manager.stats['open_positions'], 1)
        
        self.manager.stop()
    
    def test_close_position_full(self):
        """Test full position close"""
        self.manager.start()
        
        # Add position
        position_id = self.manager.add_position_from_order(
            ticket=12345,
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0
        )
        
        # Mock MT5 close
        self.mock_mt5.order_send.return_value = Mock(retcode=10009, price=2010.0, comment="Position closed")  # TRADE_RETCODE_DONE
        self.mock_mt5.symbol_info_tick.return_value = Mock(bid=2010.0, ask=2010.5)
        
        # Create close request
        close_request = PositionCloseRequest(
            position_id=position_id,
            comment="Test close"
        )
        
        # Close position
        success, message = self.manager.close_position(close_request)
        
        # Debug output
        if not success:
            print(f"Close failed: {message}")
            position = self.manager.positions.get(position_id)
            if position:
                print(f"Position status: {position.status}")
        
        # Verify close
        self.assertTrue(success)
        # Position should be moved to history after full close
        self.assertNotIn(position_id, self.manager.positions)
        self.assertEqual(len(self.manager.position_history), 1)
        closed_position = self.manager.position_history[0]
        self.assertEqual(closed_position.status, PositionStatus.CLOSED)
        self.assertEqual(self.manager.stats['open_positions'], 0)
        self.assertEqual(self.manager.stats['closed_positions'], 1)
        
        self.manager.stop()
    
    def test_close_position_partial(self):
        """Test partial position close"""
        self.manager.start()
        
        # Add position
        position_id = self.manager.add_position_from_order(
            ticket=12345,
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.2,
            open_price=2000.0
        )
        
        # Mock MT5 close
        self.mock_mt5.order_send.return_value = Mock(retcode=10009, price=2010.0, comment="Position partially closed")
        self.mock_mt5.symbol_info_tick.return_value = Mock(bid=2010.0, ask=2010.5)
        
        # Create partial close request
        close_request = PositionCloseRequest(
            position_id=position_id,
            volume=0.1,  # Close half
            comment="Partial close"
        )
        
        # Close position partially
        success, message = self.manager.close_position(close_request)
        
        # Debug output
        if not success:
            print(f"Partial close failed: {message}")
            position = self.manager.positions.get(position_id)
            if position:
                print(f"Position status: {position.status}")
        
        # Verify partial close
        self.assertTrue(success)
        position = self.manager.positions[position_id]
        self.assertEqual(position.status, PositionStatus.PARTIALLY_CLOSED)
        self.assertEqual(position.remaining_volume, 0.1)
        
        self.manager.stop()
    
    def test_modify_position(self):
        """Test position modification"""
        self.manager.start()
        
        # Add position
        position_id = self.manager.add_position_from_order(
            ticket=12345,
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0,
            stop_loss=1990.0,
            take_profit=2020.0
        )
        
        # Mock MT5 modify
        self.mock_mt5.order_send.return_value = Mock(retcode=10009, comment="Position modified")
        
        # Create modify request
        modify_request = PositionModifyRequest(
            position_id=position_id,
            new_stop_loss=1995.0,
            new_take_profit=2025.0,
            comment="Modified SL/TP"
        )
        
        # Modify position
        success, message = self.manager.modify_position(modify_request)
        
        # Debug output
        if not success:
            print(f"Modify failed: {message}")
            position = self.manager.positions.get(position_id)
            if position:
                print(f"Position status: {position.status}")
        
        # Verify modification
        self.assertTrue(success)
        position = self.manager.positions[position_id]
        self.assertEqual(position.stop_loss, 1995.0)
        self.assertEqual(position.take_profit, 2025.0)
        
        self.manager.stop()
    
    def test_position_callbacks(self):
        """Test position event callbacks"""
        self.manager.start()
        
        # Add callback
        callback_called = []
        def test_callback(position):
            callback_called.append(position.position_id)
        
        self.manager.add_callback('position_opened', test_callback)
        
        # Add position
        position_id = self.manager.add_position_from_order(
            ticket=12345,
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0
        )
        
        # Verify callback called
        self.assertIn(position_id, callback_called)
        
        self.manager.stop()
    
    def test_get_positions_by_symbol(self):
        """Test getting positions by symbol"""
        self.manager.start()
        
        # Add positions for different symbols
        pos1 = self.manager.add_position_from_order(12345, "XAUUSD", PositionType.BUY, 0.1, 2000.0)
        pos2 = self.manager.add_position_from_order(12346, "EURUSD", PositionType.SELL, 0.2, 1.1000)
        pos3 = self.manager.add_position_from_order(12347, "XAUUSD", PositionType.SELL, 0.15, 2010.0)
        
        # Get XAUUSD positions
        xau_positions = self.manager.get_positions_by_symbol("XAUUSD")
        self.assertEqual(len(xau_positions), 2)
        
        # Get EURUSD positions
        eur_positions = self.manager.get_positions_by_symbol("EURUSD")
        self.assertEqual(len(eur_positions), 1)
        
        self.manager.stop()
    
    def test_position_summary(self):
        """Test position summary generation"""
        self.manager.start()
        
        # Add positions
        self.manager.add_position_from_order(12345, "XAUUSD", PositionType.BUY, 0.1, 2000.0)
        self.manager.add_position_from_order(12346, "EURUSD", PositionType.SELL, 0.2, 1.1000)
        
        # Get summary
        summary = self.manager.get_position_summary()
        
        # Verify summary
        self.assertIsInstance(summary, PositionSummary)
        self.assertEqual(summary.total_positions, 2)
        self.assertEqual(len(summary.open_positions), 2)
        self.assertEqual(len(summary.symbols), 2)
        
        self.manager.stop()


class TestPositionCalculator(unittest.TestCase):
    """Test Position Calculator functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'default_risk_percentage': 2.0,
            'max_risk_percentage': 5.0,
            'min_position_size': 0.01,
            'max_position_size': 10.0
        }
        self.calculator = PositionCalculator(self.config)
    
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        self.assertEqual(self.calculator.default_risk_percentage, 2.0)
        self.assertEqual(self.calculator.max_risk_percentage, 5.0)
        self.assertEqual(self.calculator.min_position_size, 0.01)
        self.assertEqual(self.calculator.max_position_size, 10.0)
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation"""
        # Create test position
        position = Position(
            position_id="test-1",
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0,
            current_price=2010.0
        )
        
        # Calculate P&L
        pnl = self.calculator.calculate_pnl(position, 2010.0, PnLCalculationType.UNREALIZED)
        
        # For buy position: (current_price - open_price) * volume
        # 2010 - 2000 = 10 * 0.1 = 1.0 (before contract size adjustment)
        self.assertGreater(pnl, 0)  # Should be positive profit
    
    def test_realized_pnl_calculation(self):
        """Test realized P&L calculation"""
        position = Position(
            position_id="test-1",
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0,
            current_price=2010.0
        )
        
        # Set realized profit
        position.realized_profit = 50.0
        
        # Calculate realized P&L
        pnl = self.calculator.calculate_pnl(position, calculation_type=PnLCalculationType.REALIZED)
        self.assertEqual(pnl, 50.0)
    
    def test_risk_based_position_sizing(self):
        """Test risk-based position sizing"""
        # Test parameters
        symbol = "XAUUSD"
        entry_price = 2000.0
        stop_loss = 1990.0
        account_balance = 10000.0
        risk_percentage = 2.0
        
        # Calculate position size
        size = self.calculator.calculate_position_size(
            symbol, entry_price, stop_loss, account_balance, risk_percentage,
            PositionSizingMethod.RISK_BASED
        )
        
        # Should return valid position size
        self.assertGreaterEqual(size, self.calculator.min_position_size)
        self.assertLessEqual(size, self.calculator.max_position_size)
    
    def test_fixed_percentage_sizing(self):
        """Test fixed percentage position sizing"""
        account_balance = 10000.0
        risk_percentage = 2.0
        
        size = self.calculator.calculate_position_size(
            "XAUUSD", 2000.0, 1990.0, account_balance, risk_percentage,
            PositionSizingMethod.FIXED_PERCENTAGE
        )
        
        self.assertGreater(size, 0)
        self.assertLessEqual(size, self.calculator.max_position_size)
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion position sizing"""
        # Set Kelly parameters in config
        self.calculator.config.update({
            'kelly_win_rate': 0.6,
            'kelly_avg_win': 100,
            'kelly_avg_loss': 80,
            'kelly_safety_factor': 0.25
        })
        
        size = self.calculator.calculate_position_size(
            "XAUUSD", 2000.0, 1990.0, 10000.0, 2.0,
            PositionSizingMethod.KELLY_CRITERION
        )
        
        self.assertGreater(size, 0)
        self.assertLessEqual(size, self.calculator.max_position_size)
    
    def test_margin_calculation(self):
        """Test margin requirement calculation"""
        margin = self.calculator.calculate_margin_required("XAUUSD", 0.1, 2000.0)
        self.assertGreaterEqual(margin, 0)
    
    def test_pip_value_calculation(self):
        """Test pip value calculation"""
        pip_value = self.calculator.calculate_pip_value("XAUUSD", 0.1)
        self.assertGreater(pip_value, 0)
    
    def test_break_even_calculation(self):
        """Test break-even price calculation"""
        position = Position(
            position_id="test-1",
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0
        )
        
        break_even = self.calculator.calculate_break_even_price(position, spread=0.5)
        
        # For buy position, break-even should be open_price + spread
        self.assertEqual(break_even, 2000.5)
    
    def test_risk_reward_ratio(self):
        """Test risk-reward ratio calculation"""
        ratio = self.calculator.calculate_risk_reward_ratio(
            entry_price=2000.0,
            stop_loss=1990.0,
            take_profit=2020.0
        )
        
        # Risk = 10, Reward = 20, Ratio = 20/10 = 2.0
        self.assertEqual(ratio, 2.0)
    
    def test_position_metrics(self):
        """Test comprehensive position metrics"""
        position = Position(
            position_id="test-1",
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0,
            current_price=2010.0,
            stop_loss=1990.0,
            take_profit=2020.0
        )
        
        metrics = self.calculator.calculate_position_metrics(position)
        
        # Verify metrics structure
        self.assertIn('unrealized_pnl', metrics)
        self.assertIn('realized_pnl', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('pip_value', metrics)
        self.assertIn('margin_required', metrics)
        self.assertIn('break_even_price', metrics)
        self.assertIn('risk_reward_ratio', metrics)


class TestStopLossManager(unittest.TestCase):
    """Test Stop Loss Manager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'update_interval': 0.1,
            'max_adjustment_frequency': 1
        }
        self.stop_manager = StopLossManager(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        if self.stop_manager.is_monitoring:
            self.stop_manager.stop()
    
    def test_stop_manager_initialization(self):
        """Test stop loss manager initialization"""
        self.assertEqual(len(self.stop_manager.stop_rules), 0)
        self.assertEqual(len(self.stop_manager.position_stops), 0)
        self.assertFalse(self.stop_manager.is_monitoring)
    
    def test_add_stop_rule(self):
        """Test adding stop loss rule"""
        rule = StopLossRule(
            rule_id="test-rule-1",
            stop_type=StopLossType.FIXED,
            distance=0.001
        )
        
        success = self.stop_manager.add_stop_rule(rule)
        
        self.assertTrue(success)
        self.assertIn("test-rule-1", self.stop_manager.stop_rules)
        self.assertEqual(self.stop_manager.stop_rules["test-rule-1"].stop_type, StopLossType.FIXED)
    
    def test_apply_stop_to_position(self):
        """Test applying stop rule to position"""
        # Add rule
        rule = StopLossRule("test-rule-1", StopLossType.FIXED, distance=0.001)
        self.stop_manager.add_stop_rule(rule)
        
        # Apply to position
        success = self.stop_manager.apply_stop_to_position("pos-1", "test-rule-1")
        
        self.assertTrue(success)
        self.assertIn("pos-1", self.stop_manager.position_stops)
        self.assertIn("test-rule-1", self.stop_manager.position_stops["pos-1"])
    
    def test_trailing_stop_rule(self):
        """Test trailing stop rule creation"""
        rule = StopLossRule(
            rule_id="trailing-1",
            stop_type=StopLossType.TRAILING,
            trailing_method=TrailingStopMethod.FIXED_DISTANCE,
            distance=0.001,
            trail_start_profit=0.002,
            trail_step=0.0005
        )
        
        self.stop_manager.add_stop_rule(rule)
        
        # Verify rule properties
        stored_rule = self.stop_manager.stop_rules["trailing-1"]
        self.assertEqual(stored_rule.stop_type, StopLossType.TRAILING)
        self.assertEqual(stored_rule.trailing_method, TrailingStopMethod.FIXED_DISTANCE)
        self.assertEqual(stored_rule.trail_start_profit, 0.002)
    
    def test_breakeven_stop_rule(self):
        """Test breakeven stop rule"""
        rule = StopLossRule(
            rule_id="breakeven-1",
            stop_type=StopLossType.BREAKEVEN,
            breakeven_trigger=0.001,
            breakeven_buffer=0.0002
        )
        
        self.stop_manager.add_stop_rule(rule)
        
        stored_rule = self.stop_manager.stop_rules["breakeven-1"]
        self.assertEqual(stored_rule.stop_type, StopLossType.BREAKEVEN)
        self.assertEqual(stored_rule.breakeven_trigger, 0.001)
    
    def test_atr_based_stop_rule(self):
        """Test ATR-based stop rule"""
        rule = StopLossRule(
            rule_id="atr-1",
            stop_type=StopLossType.ATR_BASED,
            atr_period=14,
            atr_multiplier=2.0
        )
        
        self.stop_manager.add_stop_rule(rule)
        
        stored_rule = self.stop_manager.stop_rules["atr-1"]
        self.assertEqual(stored_rule.stop_type, StopLossType.ATR_BASED)
        self.assertEqual(stored_rule.atr_period, 14)
        self.assertEqual(stored_rule.atr_multiplier, 2.0)
    
    def test_stop_callbacks(self):
        """Test stop loss event callbacks"""
        callback_called = []
        
        def test_callback(position, new_stop=None):
            callback_called.append((position.position_id, new_stop))
        
        self.stop_manager.add_callback('stop_adjusted', test_callback)
        
        # Create test position
        position = Position(
            position_id="test-pos-1",
            symbol="XAUUSD",
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0,
            current_price=2010.0
        )
        
        # Trigger callback
        self.stop_manager._trigger_callbacks('stop_adjusted', position, 1995.0)
        
        # Verify callback called
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0][0], "test-pos-1")
        self.assertEqual(callback_called[0][1], 1995.0)
    
    def test_get_position_stops(self):
        """Test getting stops for position"""
        # Add rules
        rule1 = StopLossRule("rule-1", StopLossType.FIXED, distance=0.001)
        rule2 = StopLossRule("rule-2", StopLossType.TRAILING, distance=0.002)
        
        self.stop_manager.add_stop_rule(rule1)
        self.stop_manager.add_stop_rule(rule2)
        
        # Apply to position
        self.stop_manager.apply_stop_to_position("pos-1", "rule-1")
        self.stop_manager.apply_stop_to_position("pos-1", "rule-2")
        
        # Get position stops
        stops = self.stop_manager.get_position_stops("pos-1")
        
        self.assertEqual(len(stops), 2)
        self.assertIn(rule1, stops)
        self.assertIn(rule2, stops)
    
    def test_remove_position_stops(self):
        """Test removing position stops"""
        # Add rule and apply to position
        rule = StopLossRule("rule-1", StopLossType.FIXED, distance=0.001)
        self.stop_manager.add_stop_rule(rule)
        self.stop_manager.apply_stop_to_position("pos-1", "rule-1")
        
        # Verify applied
        self.assertIn("pos-1", self.stop_manager.position_stops)
        
        # Remove position stops
        self.stop_manager.remove_position_stops("pos-1")
        
        # Verify removed
        self.assertNotIn("pos-1", self.stop_manager.position_stops)
    
    def test_statistics(self):
        """Test stop loss manager statistics"""
        # Add some rules
        rule1 = StopLossRule("rule-1", StopLossType.FIXED, distance=0.001)
        rule2 = StopLossRule("rule-2", StopLossType.TRAILING, distance=0.002)
        
        self.stop_manager.add_stop_rule(rule1)
        self.stop_manager.add_stop_rule(rule2)
        
        # Apply to positions
        self.stop_manager.apply_stop_to_position("pos-1", "rule-1")
        self.stop_manager.apply_stop_to_position("pos-2", "rule-2")
        
        # Get statistics
        stats = self.stop_manager.get_statistics()
        
        # Verify statistics
        self.assertEqual(stats['active_rules'], 2)
        self.assertEqual(stats['total_rules'], 2)
        self.assertEqual(stats['positions_with_stops'], 2)
        self.assertIn('total_adjustments', stats)
        self.assertIn('trailing_activations', stats)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPositionManager))
    test_suite.addTest(unittest.makeSuite(TestPositionCalculator))
    test_suite.addTest(unittest.makeSuite(TestStopLossManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"POSITION MANAGEMENT SYSTEM TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"{'='*60}") 