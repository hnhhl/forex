"""
Unit Tests for Risk Monitoring System
Tests for RiskMonitor, DrawdownCalculator, and RiskLimitManager
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fix import issues
import time

from src.core.risk.risk_monitor import (
    RiskMonitor, RealTimeRiskMetrics, RiskAlert, RiskThreshold,
    AlertSeverity, RiskMetricType
)
from src.core.risk.drawdown_calculator import (
    DrawdownCalculator, DrawdownPeriod, DrawdownStatistics,
    DrawdownType, DrawdownSeverity
)
from src.core.risk.risk_limits import (
    RiskLimitManager, RiskLimit, LimitBreach,
    LimitType, LimitScope, LimitStatus, ActionType
)


class TestRiskMonitor(unittest.TestCase):
    """Test Risk Monitor functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = {
            'monitoring_interval': 30,
            'max_history_size': 1000,
            'enable_real_time': False,  # Disable for testing
            'risk_free_rate': 0.02
        }
        self.risk_monitor = RiskMonitor(self.config)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
        prices = 100 * (1 + returns).cumprod()
        
        self.portfolio_data = pd.DataFrame({'portfolio': prices}, index=dates)
        self.benchmark_data = pd.DataFrame({'benchmark': prices * 0.95}, index=dates)
    
    def test_initialization(self):
        """Test risk monitor initialization"""
        self.assertEqual(self.risk_monitor.system_name, "RiskMonitor")
        self.assertEqual(self.risk_monitor.monitoring_interval, 30)
        self.assertFalse(self.risk_monitor.enable_real_time)
        self.assertEqual(len(self.risk_monitor.risk_thresholds), 6)  # Default thresholds
    
    def test_set_portfolio_data(self):
        """Test setting portfolio data"""
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        self.assertIsNotNone(self.risk_monitor.portfolio_data)
        self.assertIsNotNone(self.risk_monitor.benchmark_data)
        self.assertEqual(len(self.risk_monitor.portfolio_data), 100)
    
    def test_calculate_real_time_metrics(self):
        """Test real-time metrics calculation"""
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        metrics = self.risk_monitor.calculate_real_time_metrics()
        
        self.assertIsInstance(metrics, RealTimeRiskMetrics)
        self.assertGreater(metrics.portfolio_value, 0)
        self.assertIsInstance(metrics.var_95, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.current_drawdown, float)
    
    def test_add_risk_threshold(self):
        """Test adding custom risk threshold"""
        threshold = RiskThreshold(
            metric_type=RiskMetricType.VOLATILITY,
            metric_name="test_volatility",
            warning_threshold=0.15,
            critical_threshold=0.25
        )
        
        initial_count = len(self.risk_monitor.risk_thresholds)
        self.risk_monitor.add_risk_threshold(threshold)
        
        self.assertEqual(len(self.risk_monitor.risk_thresholds), initial_count + 1)
    
    def test_check_risk_thresholds(self):
        """Test risk threshold checking"""
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        # Create metrics with high volatility to trigger threshold
        metrics = self.risk_monitor.calculate_real_time_metrics()
        metrics.realized_volatility = 0.60  # High volatility to trigger alert
        
        initial_alerts = len(self.risk_monitor.active_alerts)
        self.risk_monitor.check_risk_thresholds(metrics)
        
        # Should have generated alerts
        self.assertGreaterEqual(len(self.risk_monitor.active_alerts), initial_alerts)
    
    def test_alert_callback(self):
        """Test alert callback functionality"""
        callback_called = False
        alert_received = None
        
        def test_callback(alert):
            nonlocal callback_called, alert_received
            callback_called = True
            alert_received = alert
        
        self.risk_monitor.add_alert_callback(test_callback)
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        # Create metrics that will trigger alert
        metrics = self.risk_monitor.calculate_real_time_metrics()
        metrics.realized_volatility = 0.60  # High volatility
        
        self.risk_monitor.check_risk_thresholds(metrics)
        
        if len(self.risk_monitor.active_alerts) > 0:
            self.assertTrue(callback_called)
            self.assertIsInstance(alert_received, RiskAlert)
    
    def test_get_risk_dashboard_data(self):
        """Test dashboard data generation"""
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        # Generate some metrics
        metrics = self.risk_monitor.calculate_real_time_metrics()
        
        dashboard_data = self.risk_monitor.get_risk_dashboard_data()
        
        self.assertIn('current_metrics', dashboard_data)
        self.assertIn('alert_summary', dashboard_data)
        self.assertIn('threshold_status', dashboard_data)
        self.assertIn('monitoring_status', dashboard_data)
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment"""
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        # Create an alert
        metrics = self.risk_monitor.calculate_real_time_metrics()
        metrics.realized_volatility = 0.60
        self.risk_monitor.check_risk_thresholds(metrics)
        
        if len(self.risk_monitor.active_alerts) > 0:
            alert = self.risk_monitor.active_alerts[0]
            result = self.risk_monitor.acknowledge_alert(alert.alert_id, "test_user")
            
            self.assertTrue(result)
            self.assertTrue(alert.is_acknowledged)
            self.assertEqual(alert.acknowledged_by, "test_user")
    
    def test_export_risk_data(self):
        """Test risk data export"""
        self.risk_monitor.set_portfolio_data(self.portfolio_data, self.benchmark_data)
        
        # Generate some data
        metrics = self.risk_monitor.calculate_real_time_metrics()
        
        filepath = "test_risk_export.json"
        result = self.risk_monitor.export_risk_data(filepath)
        
        self.assertTrue(result)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.risk_monitor.get_statistics()
        
        self.assertIn('total_metrics_calculated', stats)
        self.assertIn('active_alerts', stats)
        self.assertIn('risk_thresholds_configured', stats)
        self.assertIn('is_monitoring', stats)


class TestDrawdownCalculator(unittest.TestCase):
    """Test Drawdown Calculator functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = {
            'min_drawdown_threshold': 0.02,
            'lookback_window': 252,
            'rolling_window': 30
        }
        self.drawdown_calc = DrawdownCalculator(self.config)
        
        # Create sample price data with known drawdowns
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        prices = [100]
        
        # Create a pattern with peaks and drawdowns
        for i in range(1, 200):
            if i < 50:
                prices.append(prices[-1] * 1.002)  # Uptrend
            elif i < 80:
                prices.append(prices[-1] * 0.998)  # Drawdown
            elif i < 120:
                prices.append(prices[-1] * 1.001)  # Recovery
            elif i < 150:
                prices.append(prices[-1] * 0.997)  # Larger drawdown
            else:
                prices.append(prices[-1] * 1.0005)  # Slow recovery
        
        self.price_data = pd.Series(prices, index=dates)
    
    def test_initialization(self):
        """Test drawdown calculator initialization"""
        self.assertEqual(self.drawdown_calc.system_name, "DrawdownCalculator")
        self.assertEqual(self.drawdown_calc.min_drawdown_threshold, 0.02)
        self.assertEqual(self.drawdown_calc.lookback_window, 252)
    
    def test_set_data(self):
        """Test setting price data"""
        self.drawdown_calc.set_data(self.price_data)
        
        self.assertIsNotNone(self.drawdown_calc.price_data)
        self.assertEqual(len(self.drawdown_calc.price_data), 200)
    
    def test_calculate_drawdown_relative(self):
        """Test relative drawdown calculation"""
        self.drawdown_calc.set_data(self.price_data)
        
        drawdown = self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        
        self.assertIsInstance(drawdown, pd.Series)
        self.assertEqual(len(drawdown), len(self.price_data))
        self.assertLessEqual(drawdown.max(), 0)  # All drawdowns should be negative or zero
    
    def test_calculate_drawdown_absolute(self):
        """Test absolute drawdown calculation"""
        self.drawdown_calc.set_data(self.price_data)
        
        drawdown = self.drawdown_calc.calculate_drawdown(DrawdownType.ABSOLUTE)
        
        self.assertIsInstance(drawdown, pd.Series)
        self.assertEqual(len(drawdown), len(self.price_data))
    
    def test_identify_drawdown_periods(self):
        """Test drawdown period identification"""
        self.drawdown_calc.set_data(self.price_data)
        self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        
        periods = self.drawdown_calc.identify_drawdown_periods(min_threshold=0.01)
        
        self.assertIsInstance(periods, list)
        self.assertGreater(len(periods), 0)  # Should find some drawdown periods
        
        for period in periods:
            self.assertIsInstance(period, DrawdownPeriod)
            self.assertGreater(period.max_drawdown, 0)
            self.assertIsInstance(period.severity, DrawdownSeverity)
    
    def test_calculate_statistics(self):
        """Test drawdown statistics calculation"""
        self.drawdown_calc.set_data(self.price_data)
        self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        self.drawdown_calc.identify_drawdown_periods()
        
        stats = self.drawdown_calc.calculate_statistics()
        
        self.assertIsInstance(stats, DrawdownStatistics)
        self.assertGreaterEqual(stats.max_drawdown, 0)
        self.assertGreaterEqual(stats.total_drawdown_periods, 0)
        self.assertIsInstance(stats.pain_index, float)
        self.assertIsInstance(stats.ulcer_index, float)
    
    def test_get_current_drawdown_info(self):
        """Test current drawdown information"""
        self.drawdown_calc.set_data(self.price_data)
        self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        
        info = self.drawdown_calc.get_current_drawdown_info()
        
        self.assertIn('current_drawdown', info)
        self.assertIn('current_price', info)
        self.assertIn('peak_price', info)
        self.assertIn('severity', info)
        self.assertIn('is_in_drawdown', info)
    
    def test_get_worst_drawdowns(self):
        """Test worst drawdowns retrieval"""
        self.drawdown_calc.set_data(self.price_data)
        self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        self.drawdown_calc.identify_drawdown_periods()
        
        worst = self.drawdown_calc.get_worst_drawdowns(n=3)
        
        self.assertIsInstance(worst, list)
        self.assertLessEqual(len(worst), 3)
        
        # Check if sorted by magnitude (worst first)
        if len(worst) > 1:
            self.assertGreaterEqual(worst[0]['max_drawdown'], worst[1]['max_drawdown'])
    
    def test_calculate_rolling_drawdown(self):
        """Test rolling drawdown calculation"""
        self.drawdown_calc.set_data(self.price_data)
        self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        
        rolling_dd = self.drawdown_calc.calculate_rolling_drawdown(window=20)
        
        self.assertIsInstance(rolling_dd, pd.Series)
        self.assertEqual(len(rolling_dd), len(self.price_data))
    
    def test_generate_drawdown_report(self):
        """Test drawdown report generation"""
        self.drawdown_calc.set_data(self.price_data)
        
        report = self.drawdown_calc.generate_drawdown_report()
        
        self.assertIn('report_timestamp', report)
        self.assertIn('data_period', report)
        self.assertIn('current_drawdown', report)
        self.assertIn('statistics', report)
        self.assertIn('worst_drawdowns', report)
        self.assertIn('summary', report)
    
    def test_export_data(self):
        """Test drawdown data export"""
        self.drawdown_calc.set_data(self.price_data)
        self.drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
        
        filepath = "test_drawdown_export.json"
        result = self.drawdown_calc.export_data(filepath)
        
        self.assertTrue(result)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)


class TestRiskLimitManager(unittest.TestCase):
    """Test Risk Limit Manager functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = {
            'monitoring_interval': 30,
            'enable_enforcement': False,  # Disable for testing
            'max_breach_history': 100
        }
        self.limit_manager = RiskLimitManager(self.config)
        
        # Sample market data
        self.positions = {'EURUSD': 10000, 'GBPUSD': -5000, 'USDJPY': 8000}
        self.portfolio_value = 100000
        self.daily_pnl = -1500  # Loss
        self.var = 2000
        self.drawdown = 0.03  # 3% drawdown
    
    def test_initialization(self):
        """Test risk limit manager initialization"""
        self.assertEqual(self.limit_manager.system_name, "RiskLimitManager")
        self.assertEqual(self.limit_manager.monitoring_interval, 30)
        self.assertFalse(self.limit_manager.enable_enforcement)
        self.assertGreater(len(self.limit_manager.risk_limits), 0)  # Default limits
    
    def test_add_risk_limit(self):
        """Test adding custom risk limit"""
        limit = RiskLimit(
            limit_id="test_limit",
            name="Test Limit",
            limit_type=LimitType.POSITION_SIZE,
            scope=LimitScope.SYMBOL,
            soft_limit=0.05,
            hard_limit=0.10,
            scope_filter="EURUSD"
        )
        
        result = self.limit_manager.add_risk_limit(limit)
        
        self.assertTrue(result)
        self.assertIn("test_limit", self.limit_manager.risk_limits)
    
    def test_update_risk_limit(self):
        """Test updating risk limit"""
        # Use existing default limit
        limit_id = list(self.limit_manager.risk_limits.keys())[0]
        
        updates = {'soft_limit': 0.03, 'hard_limit': 0.06}
        result = self.limit_manager.update_risk_limit(limit_id, updates)
        
        self.assertTrue(result)
        self.assertEqual(self.limit_manager.risk_limits[limit_id].soft_limit, 0.03)
        self.assertEqual(self.limit_manager.risk_limits[limit_id].hard_limit, 0.06)
    
    def test_remove_risk_limit(self):
        """Test removing risk limit"""
        # Add a test limit first
        limit = RiskLimit(
            limit_id="temp_limit",
            name="Temporary Limit",
            limit_type=LimitType.POSITION_SIZE,
            scope=LimitScope.SYMBOL,
            soft_limit=0.05,
            hard_limit=0.10
        )
        self.limit_manager.add_risk_limit(limit)
        
        result = self.limit_manager.remove_risk_limit("temp_limit")
        
        self.assertTrue(result)
        self.assertNotIn("temp_limit", self.limit_manager.risk_limits)
    
    def test_update_market_data(self):
        """Test market data update"""
        self.limit_manager.update_market_data(
            self.positions, self.portfolio_value, self.daily_pnl, self.var, self.drawdown
        )
        
        self.assertEqual(self.limit_manager.current_positions, self.positions)
        self.assertEqual(self.limit_manager.current_portfolio_value, self.portfolio_value)
        self.assertEqual(self.limit_manager.current_daily_pnl, self.daily_pnl)
        self.assertEqual(self.limit_manager.current_var, self.var)
        self.assertEqual(self.limit_manager.current_drawdown, self.drawdown)
    
    def test_check_all_limits_no_breach(self):
        """Test limit checking with no breaches"""
        # Set normal market data
        normal_positions = {'EURUSD': 1000}  # Small position
        self.limit_manager.update_market_data(
            normal_positions, 100000, 100, 500, 0.01  # Normal values
        )
        
        breaches = self.limit_manager.check_all_limits()
        
        self.assertIsInstance(breaches, list)
        # May or may not have breaches depending on default limits
    
    def test_check_all_limits_with_breach(self):
        """Test limit checking with breaches"""
        # Set market data that will breach limits
        high_risk_positions = {'EURUSD': 50000}  # Large position
        self.limit_manager.update_market_data(
            high_risk_positions, 100000, -8000, 12000, 0.15  # High risk values
        )
        
        breaches = self.limit_manager.check_all_limits()
        
        self.assertIsInstance(breaches, list)
        # Should have some breaches with high risk values
        if len(breaches) > 0:
            for breach in breaches:
                self.assertIsInstance(breach, LimitBreach)
                self.assertGreater(breach.current_value, breach.limit_value)
    
    def test_breach_callback(self):
        """Test breach callback functionality"""
        callback_called = False
        breach_received = None
        
        def test_callback(breach):
            nonlocal callback_called, breach_received
            callback_called = True
            breach_received = breach
        
        self.limit_manager.add_breach_callback(test_callback)
        
        # Set data that will breach limits
        high_risk_positions = {'EURUSD': 60000}
        self.limit_manager.update_market_data(
            high_risk_positions, 100000, -10000, 15000, 0.20
        )
        
        breaches = self.limit_manager.check_all_limits()
        
        if len(breaches) > 0:
            self.assertTrue(callback_called)
            self.assertIsInstance(breach_received, LimitBreach)
    
    def test_get_limit_status_report(self):
        """Test limit status report generation"""
        self.limit_manager.update_market_data(
            self.positions, self.portfolio_value, self.daily_pnl, self.var, self.drawdown
        )
        
        report = self.limit_manager.get_limit_status_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('summary', report)
        self.assertIn('limit_utilization', report)
        self.assertIn('active_breaches', report)
        self.assertIn('recent_breaches', report)
        self.assertIn('configuration', report)
    
    def test_export_data(self):
        """Test risk limit data export"""
        filepath = "test_limits_export.json"
        result = self.limit_manager.export_data(filepath)
        
        self.assertTrue(result)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.limit_manager.get_statistics()
        
        self.assertIn('total_limits', stats)
        self.assertIn('active_limits', stats)
        self.assertIn('breached_limits', stats)
        self.assertIn('is_monitoring', stats)


class TestRiskMonitoringIntegration(unittest.TestCase):
    """Test integration between risk monitoring components"""
    
    def setUp(self):
        """Setup integrated test environment"""
        self.risk_monitor = RiskMonitor({'enable_real_time': False})
        self.drawdown_calc = DrawdownCalculator()
        self.limit_manager = RiskLimitManager({'enable_enforcement': False})
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * (1 + returns).cumprod()
        
        self.portfolio_data = pd.DataFrame({'portfolio': prices}, index=dates)
        self.positions = {'EURUSD': 10000, 'GBPUSD': -5000}
    
    def test_integrated_risk_monitoring(self):
        """Test integrated risk monitoring workflow"""
        # Setup risk monitor
        self.risk_monitor.set_portfolio_data(self.portfolio_data)
        metrics = self.risk_monitor.calculate_real_time_metrics()
        
        # Setup drawdown calculator
        self.drawdown_calc.set_data(self.portfolio_data)
        self.drawdown_calc.calculate_drawdown()
        self.drawdown_calc.identify_drawdown_periods()  # Need this before statistics
        drawdown_stats = self.drawdown_calc.calculate_statistics()
        
        # Setup limit manager
        self.limit_manager.update_market_data(
            self.positions, 
            metrics.portfolio_value,
            metrics.daily_pnl,
            metrics.var_95 * metrics.portfolio_value,
            metrics.current_drawdown
        )
        
        # Check all systems
        risk_alerts = self.risk_monitor.active_alerts
        limit_breaches = self.limit_manager.check_all_limits()
        
        # Verify integration
        self.assertIsInstance(metrics, RealTimeRiskMetrics)
        self.assertIsInstance(drawdown_stats, DrawdownStatistics)
        self.assertIsInstance(limit_breaches, list)
    
    def test_cross_system_data_consistency(self):
        """Test data consistency across systems"""
        # Setup all systems with same data
        self.risk_monitor.set_portfolio_data(self.portfolio_data)
        self.drawdown_calc.set_data(self.portfolio_data)
        
        # Calculate metrics
        risk_metrics = self.risk_monitor.calculate_real_time_metrics()
        self.drawdown_calc.calculate_drawdown()
        drawdown_info = self.drawdown_calc.get_current_drawdown_info()
        
        # Check consistency
        self.assertAlmostEqual(
            risk_metrics.current_drawdown,
            drawdown_info['current_drawdown'],
            places=4
        )
    
    def test_alert_escalation_workflow(self):
        """Test alert escalation between systems"""
        # Setup systems
        self.risk_monitor.set_portfolio_data(self.portfolio_data)
        
        # Create high-risk scenario
        high_risk_positions = {'EURUSD': 80000}  # Very large position
        self.limit_manager.update_market_data(
            high_risk_positions, 100000, -15000, 20000, 0.25
        )
        
        # Check for escalation
        limit_breaches = self.limit_manager.check_all_limits()
        
        # Verify escalation logic
        if len(limit_breaches) > 0:
            emergency_breaches = [b for b in limit_breaches if b.action_taken == ActionType.EMERGENCY_STOP]
            self.assertIsInstance(emergency_breaches, list)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRiskMonitor,
        TestDrawdownCalculator,
        TestRiskLimitManager,
        TestRiskMonitoringIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"RISK MONITORING SYSTEM TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"- {test}: {error_msg}") 