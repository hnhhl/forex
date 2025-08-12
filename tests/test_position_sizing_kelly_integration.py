"""
Integration Tests for Position Sizing System with Kelly Criterion Calculator
Tests for advanced Kelly methods and professional calculator integration
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.risk.position_sizer import (
    PositionSizer, SizingMethod, RiskLevel, SizingParameters, SizingResult
)

try:
    from src.core.trading.kelly_criterion import KellyMethod, TradeResult
    KELLY_AVAILABLE = True
except ImportError:
    KELLY_AVAILABLE = False


class TestPositionSizingKellyIntegration(unittest.TestCase):
    """Test cases for Kelly Criterion integration in Position Sizing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sizer = PositionSizer()
        
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # OHLC data
        base_price = 2000.0
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        self.price_data = pd.DataFrame({
            'Open': prices[:-1],
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
            'Close': prices[1:],
        }, index=dates)
        
        # Set data
        self.sizer.set_data(self.price_data, 100000.0)
        self.sizer.set_performance_metrics(0.65, 0.025, -0.015)
        
        # Add sample trade results for Kelly Calculator
        self._add_sample_trades()
    
    def _add_sample_trades(self):
        """Add sample trade results to Kelly Calculator"""
        if not hasattr(self.sizer, 'add_trade_result'):
            return
        
        # Add 50 sample trades with 65% win rate
        np.random.seed(42)
        for i in range(50):
            is_win = np.random.random() < 0.65
            if is_win:
                profit_loss = np.random.uniform(0.01, 0.04)  # 1-4% wins
            else:
                profit_loss = np.random.uniform(-0.03, -0.01)  # 1-3% losses
            
            trade_date = datetime.now() - timedelta(days=50-i)
            entry_price = 2000 + np.random.uniform(-50, 50)
            exit_price = entry_price * (1 + profit_loss)
            
            self.sizer.add_trade_result(
                profit_loss=profit_loss,
                win=is_win,
                trade_date=trade_date,
                symbol="XAUUSD",
                entry_price=entry_price,
                exit_price=exit_price,
                volume=0.1,
                duration_minutes=np.random.randint(30, 240)
            )
    
    @unittest.skipUnless(KELLY_AVAILABLE, "Kelly Criterion Calculator not available")
    def test_kelly_classic_sizing(self):
        """Test Classic Kelly Criterion sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_classic_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_CLASSIC)
        self.assertGreater(result.position_size, 0)
        self.assertIn('kelly_method', result.additional_metrics)
        self.assertEqual(result.additional_metrics['kelly_method'], 'classic')
        self.assertTrue(result.additional_metrics.get('professional_calculator', False))
    
    @unittest.skipUnless(KELLY_AVAILABLE, "Kelly Criterion Calculator not available")
    def test_kelly_fractional_sizing(self):
        """Test Fractional Kelly Criterion sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_fractional_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_FRACTIONAL)
        self.assertGreater(result.position_size, 0)
        self.assertIn('kelly_method', result.additional_metrics)
        self.assertEqual(result.additional_metrics['kelly_method'], 'fractional')
    
    @unittest.skipUnless(KELLY_AVAILABLE, "Kelly Criterion Calculator not available")
    def test_kelly_dynamic_sizing(self):
        """Test Dynamic Kelly Criterion sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_dynamic_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_DYNAMIC)
        self.assertGreater(result.position_size, 0)
        self.assertIn('kelly_method', result.additional_metrics)
        self.assertEqual(result.additional_metrics['kelly_method'], 'dynamic')
    
    @unittest.skipUnless(KELLY_AVAILABLE, "Kelly Criterion Calculator not available")
    def test_kelly_conservative_sizing(self):
        """Test Conservative Kelly Criterion sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_conservative_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_CONSERVATIVE)
        self.assertGreater(result.position_size, 0)
        self.assertIn('kelly_method', result.additional_metrics)
        self.assertEqual(result.additional_metrics['kelly_method'], 'conservative')
    
    @unittest.skipUnless(KELLY_AVAILABLE, "Kelly Criterion Calculator not available")
    def test_kelly_adaptive_sizing(self):
        """Test Adaptive Kelly Criterion sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_adaptive_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_ADAPTIVE)
        self.assertGreater(result.position_size, 0)
        self.assertIn('kelly_method', result.additional_metrics)
        self.assertEqual(result.additional_metrics['kelly_method'], 'adaptive')
    
    @unittest.skipUnless(KELLY_AVAILABLE, "Kelly Criterion Calculator not available")
    def test_kelly_analysis_comprehensive(self):
        """Test comprehensive Kelly analysis"""
        current_price = 2000.0
        
        analysis = self.sizer.get_kelly_analysis(current_price)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('kelly_analysis', analysis)
        self.assertIn('performance_summary', analysis)
        self.assertIn('trade_count', analysis)
        self.assertEqual(analysis['trade_count'], 50)
        
        # Check all Kelly methods are analyzed
        kelly_analysis = analysis['kelly_analysis']
        expected_methods = ['classic', 'fractional', 'dynamic', 'conservative', 'adaptive']
        
        for method in expected_methods:
            self.assertIn(method, kelly_analysis)
            if 'error' not in kelly_analysis[method]:
                self.assertIn('position_size', kelly_analysis[method])
                self.assertIn('kelly_fraction', kelly_analysis[method])
                self.assertIn('confidence_score', kelly_analysis[method])
    
    def test_kelly_fallback_to_basic(self):
        """Test fallback to basic Kelly when professional calculator unavailable"""
        # Mock the professional calculator as unavailable
        original_available = self.sizer.kelly_calculator
        self.sizer.kelly_calculator = None
        
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_criterion_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_CRITERION)
        self.assertGreater(result.position_size, 0)
        self.assertFalse(result.additional_metrics.get('professional_calculator', True))
        
        # Restore original calculator
        self.sizer.kelly_calculator = original_available
    
    def test_kelly_parameters_limits(self):
        """Test Kelly parameters limits are applied"""
        current_price = 2000.0
        
        # Create parameters with strict limits
        parameters = SizingParameters(
            kelly_max_fraction=0.05,  # 5% max
            kelly_min_fraction=0.01   # 1% min
        )
        
        result = self.sizer.calculate_kelly_criterion_size(current_price, parameters)
        
        kelly_fraction = result.additional_metrics.get('kelly_fraction', 0)
        self.assertLessEqual(kelly_fraction, parameters.kelly_max_fraction)
        self.assertGreaterEqual(kelly_fraction, parameters.kelly_min_fraction)
    
    def test_add_trade_result_functionality(self):
        """Test adding trade results to Kelly Calculator"""
        if not hasattr(self.sizer, 'add_trade_result'):
            self.skipTest("add_trade_result method not available")
        
        initial_count = len(self.sizer.trade_history)
        
        # Add a winning trade
        self.sizer.add_trade_result(
            profit_loss=0.02,
            win=True,
            trade_date=datetime.now(),
            symbol="XAUUSD",
            entry_price=2000.0,
            exit_price=2040.0,
            volume=0.1,
            duration_minutes=120
        )
        
        self.assertEqual(len(self.sizer.trade_history), initial_count + 1)
        
        # Check the trade was added correctly
        last_trade = self.sizer.trade_history[-1]
        self.assertEqual(last_trade.profit_loss, 0.02)
        self.assertTrue(last_trade.win)
        self.assertEqual(last_trade.symbol, "XAUUSD")
    
    def test_kelly_confidence_scores(self):
        """Test Kelly confidence scores are reasonable"""
        current_price = 2000.0
        
        if not KELLY_AVAILABLE:
            self.skipTest("Kelly Criterion Calculator not available")
        
        # Test different Kelly methods
        methods = [
            ('classic', self.sizer.calculate_kelly_classic_size),
            ('fractional', self.sizer.calculate_kelly_fractional_size),
            ('conservative', self.sizer.calculate_kelly_conservative_size),
            ('adaptive', self.sizer.calculate_kelly_adaptive_size)
        ]
        
        for method_name, method_func in methods:
            with self.subTest(method=method_name):
                result = method_func(current_price)
                
                # Confidence score should be between 0 and 1
                self.assertGreaterEqual(result.confidence_score, 0.0)
                self.assertLessEqual(result.confidence_score, 1.0)
                
                # Should have reasonable values
                self.assertGreater(result.position_size, 0)
                self.assertGreater(result.risk_amount, 0)
    
    def test_kelly_risk_metrics_integration(self):
        """Test Kelly risk metrics are properly integrated"""
        current_price = 2000.0
        
        if not KELLY_AVAILABLE:
            self.skipTest("Kelly Criterion Calculator not available")
        
        result = self.sizer.calculate_kelly_adaptive_size(current_price)
        
        # Check risk metrics are included
        self.assertIn('risk_metrics', result.additional_metrics)
        risk_metrics = result.additional_metrics['risk_metrics']
        
        # Risk metrics should contain key information
        self.assertIsInstance(risk_metrics, dict)
        
        # Check warnings are included
        self.assertIn('warnings', result.additional_metrics)
        warnings = result.additional_metrics['warnings']
        self.assertIsInstance(warnings, list)
    
    def test_position_size_limits_with_kelly(self):
        """Test position size limits are respected with Kelly methods"""
        current_price = 2000.0
        
        # Create parameters with very restrictive limits
        parameters = SizingParameters(
            max_position_size=0.01,  # 1% max position
            min_position_size=0.005  # 0.5% min position
        )
        
        result = self.sizer.calculate_kelly_criterion_size(current_price, parameters)
        
        # Calculate actual position percentage
        position_value = result.position_size * current_price
        position_percentage = position_value / self.sizer.portfolio_value
        
        # Should respect limits
        self.assertLessEqual(position_percentage, parameters.max_position_size)
        self.assertGreaterEqual(position_percentage, parameters.min_position_size)
    
    def test_kelly_method_comparison(self):
        """Test comparison between different Kelly methods"""
        current_price = 2000.0
        
        if not KELLY_AVAILABLE:
            self.skipTest("Kelly Criterion Calculator not available")
        
        # Calculate with different methods
        classic_result = self.sizer.calculate_kelly_classic_size(current_price)
        conservative_result = self.sizer.calculate_kelly_conservative_size(current_price)
        adaptive_result = self.sizer.calculate_kelly_adaptive_size(current_price)
        
        # Conservative should generally be smaller than classic
        conservative_fraction = conservative_result.additional_metrics.get('kelly_fraction', 0)
        classic_fraction = classic_result.additional_metrics.get('kelly_fraction', 0)
        
        # All should be positive for profitable system
        self.assertGreater(conservative_fraction, 0)
        self.assertGreater(classic_fraction, 0)
        self.assertGreater(adaptive_result.additional_metrics.get('kelly_fraction', 0), 0)
        
        # All should produce reasonable position sizes
        self.assertGreater(classic_result.position_size, 0)
        self.assertGreater(conservative_result.position_size, 0)
        self.assertGreater(adaptive_result.position_size, 0)


if __name__ == '__main__':
    unittest.main() 