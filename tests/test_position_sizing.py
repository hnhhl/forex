"""
Unit Tests for Position Sizing System
Tests for PositionSizer with various sizing methods
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


class TestPositionSizer(unittest.TestCase):
    """Test cases for PositionSizer"""
    
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
        
        # Set performance metrics
        self.sizer.set_performance_metrics(0.6, 0.02, -0.01)
    
    def test_sizer_initialization(self):
        """Test position sizer initialization"""
        sizer = PositionSizer()
        self.assertEqual(sizer.system_name, "PositionSizer")
        self.assertEqual(sizer.portfolio_value, 100000.0)
        self.assertEqual(sizer.win_rate, 0.6)
        self.assertEqual(sizer.avg_win, 0.02)
        self.assertEqual(sizer.avg_loss, -0.01)
    
    def test_set_data(self):
        """Test setting price data"""
        self.assertIsNotNone(self.sizer.price_data)
        self.assertIsNotNone(self.sizer.returns_data)
        self.assertEqual(len(self.sizer.price_data), 100)
        self.assertEqual(self.sizer.portfolio_value, 100000.0)
    
    def test_set_performance_metrics(self):
        """Test setting performance metrics"""
        self.sizer.set_performance_metrics(0.7, 0.03, -0.015)
        
        self.assertEqual(self.sizer.win_rate, 0.7)
        self.assertEqual(self.sizer.avg_win, 0.03)
        self.assertEqual(self.sizer.avg_loss, -0.015)
        
        # Test clamping
        self.sizer.set_performance_metrics(1.5, 0.0, 0.01)  # Invalid values
        self.assertEqual(self.sizer.win_rate, 0.99)  # Clamped to max
        self.assertEqual(self.sizer.avg_win, 0.001)  # Clamped to min
        self.assertEqual(self.sizer.avg_loss, -0.001)  # Clamped to max negative
    
    def test_fixed_amount_sizing(self):
        """Test fixed amount position sizing"""
        current_price = 2000.0
        target_amount = 10000.0
        
        result = self.sizer.calculate_fixed_amount_size(target_amount, current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.FIXED_AMOUNT)
        self.assertAlmostEqual(result.position_size, target_amount / current_price, places=4)
        self.assertEqual(result.confidence_score, 0.8)
        self.assertIn('target_amount', result.additional_metrics)
        self.assertIn('current_price', result.additional_metrics)
    
    def test_fixed_percentage_sizing(self):
        """Test fixed percentage position sizing"""
        current_price = 2000.0
        target_percentage = 0.05  # 5%
        
        result = self.sizer.calculate_fixed_percentage_size(target_percentage, current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.FIXED_PERCENTAGE)
        
        expected_value = self.sizer.portfolio_value * target_percentage
        expected_size = expected_value / current_price
        self.assertAlmostEqual(result.position_size, expected_size, places=4)
        
        self.assertIn('target_percentage', result.additional_metrics)
        self.assertEqual(result.additional_metrics['target_percentage'], target_percentage)
    
    def test_risk_based_sizing(self):
        """Test risk-based position sizing"""
        current_price = 2000.0
        stop_loss_price = 1950.0  # 2.5% stop loss
        
        result = self.sizer.calculate_risk_based_size(current_price, stop_loss_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.RISK_BASED)
        self.assertEqual(result.confidence_score, 0.9)
        
        # Check risk calculation
        risk_per_unit = abs(current_price - stop_loss_price)
        max_risk = self.sizer.portfolio_value * self.sizer.default_parameters.risk_per_trade
        expected_size = max_risk / risk_per_unit
        
        # Should be close (within limits)
        self.assertGreater(result.position_size, 0)
        self.assertIn('risk_per_unit', result.additional_metrics)
        self.assertEqual(result.additional_metrics['risk_per_unit'], risk_per_unit)
    
    def test_risk_based_sizing_invalid_stop_loss(self):
        """Test risk-based sizing with invalid stop loss"""
        current_price = 2000.0
        stop_loss_price = 2000.0  # Same as current price
        
        with self.assertRaises(ValueError):
            self.sizer.calculate_risk_based_size(current_price, stop_loss_price)
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly Criterion position sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_kelly_criterion_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.KELLY_CRITERION)
        self.assertGreater(result.position_size, 0)
        
        # Check Kelly calculation components
        self.assertIn('kelly_fraction', result.additional_metrics)
        self.assertIn('full_kelly', result.additional_metrics)
        self.assertIn('win_rate', result.additional_metrics)
        self.assertIn('odds_ratio', result.additional_metrics)
        self.assertIn('edge', result.additional_metrics)
        
        # Kelly fraction should be positive for profitable system
        self.assertGreaterEqual(result.additional_metrics['kelly_fraction'], 0)
    
    def test_volatility_based_sizing(self):
        """Test volatility-based position sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_volatility_based_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.VOLATILITY_BASED)
        self.assertGreater(result.position_size, 0)
        
        # Check volatility metrics
        self.assertIn('daily_volatility', result.additional_metrics)
        self.assertIn('annual_volatility', result.additional_metrics)
        self.assertIn('volatility_factor', result.additional_metrics)
        
        # Annual volatility should be daily * sqrt(252)
        daily_vol = result.additional_metrics['daily_volatility']
        annual_vol = result.additional_metrics['annual_volatility']
        self.assertAlmostEqual(annual_vol, daily_vol * np.sqrt(252), places=4)
    
    def test_atr_based_sizing(self):
        """Test ATR-based position sizing"""
        current_price = 2000.0
        
        result = self.sizer.calculate_atr_based_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.ATR_BASED)
        self.assertGreater(result.position_size, 0)
        self.assertEqual(result.confidence_score, 0.85)
        
        # Check ATR metrics
        self.assertIn('atr', result.additional_metrics)
        self.assertIn('atr_period', result.additional_metrics)
        self.assertIn('risk_per_unit', result.additional_metrics)
        self.assertIn('atr_multiple', result.additional_metrics)
        
        # Risk per unit should be ATR * multiple
        atr = result.additional_metrics['atr']
        multiple = result.additional_metrics['atr_multiple']
        expected_risk = atr * multiple
        self.assertAlmostEqual(result.additional_metrics['risk_per_unit'], expected_risk, places=4)
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        atr = self.sizer._calculate_atr(14)
        
        self.assertIsInstance(atr, float)
        self.assertGreater(atr, 0)
        
        # ATR should be reasonable for the price range
        current_price = self.price_data['Close'].iloc[-1]
        self.assertLess(atr, current_price * 0.1)  # Less than 10% of price
    
    def test_optimal_sizing(self):
        """Test optimal position sizing"""
        current_price = 2000.0
        stop_loss_price = 1950.0
        
        result = self.sizer.calculate_optimal_size(current_price, stop_loss_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertEqual(result.method, SizingMethod.OPTIMAL_F)
        self.assertGreater(result.position_size, 0)
        
        # Check component metrics
        self.assertIn('component_sizes', result.additional_metrics)
        self.assertIn('component_confidences', result.additional_metrics)
        self.assertIn('weighted_average', result.additional_metrics)
        
        # Should have multiple components
        components = result.additional_metrics['component_sizes']
        self.assertGreaterEqual(len(components), 3)  # At least kelly, volatility, atr
    
    def test_sizing_recommendation(self):
        """Test comprehensive sizing recommendation"""
        current_price = 2000.0
        stop_loss_price = 1950.0
        
        recommendation = self.sizer.get_sizing_recommendation(
            current_price, stop_loss_price, RiskLevel.MODERATE
        )
        
        self.assertIsInstance(recommendation, dict)
        self.assertIn('recommended', recommendation)
        self.assertIn('alternatives', recommendation)
        self.assertIn('risk_level', recommendation)
        self.assertIn('market_conditions', recommendation)
        self.assertIn('sizing_rationale', recommendation)
        
        # Check recommended result
        recommended = recommendation['recommended']
        self.assertIn('method', recommended)
        self.assertIn('position_size', recommended)
        self.assertIn('confidence_score', recommended)
        
        # Check alternatives
        alternatives = recommendation['alternatives']
        self.assertIn('aggressive', alternatives)
        
        # Risk level should match
        self.assertEqual(recommendation['risk_level'], 'moderate')
    
    def test_risk_level_adjustments(self):
        """Test risk level parameter adjustments"""
        current_price = 2000.0
        
        # Conservative
        conservative_rec = self.sizer.get_sizing_recommendation(
            current_price, risk_level=RiskLevel.CONSERVATIVE
        )
        
        # Aggressive
        aggressive_rec = self.sizer.get_sizing_recommendation(
            current_price, risk_level=RiskLevel.AGGRESSIVE
        )
        
        # Conservative should have smaller position
        conservative_size = conservative_rec['recommended']['position_size']
        aggressive_size = aggressive_rec['recommended']['position_size']
        
        self.assertLess(conservative_size, aggressive_size)
    
    def test_market_conditions_assessment(self):
        """Test market conditions assessment"""
        conditions = self.sizer._assess_market_conditions()
        
        self.assertIsInstance(conditions, dict)
        self.assertIn('volatility_condition', conditions)
        self.assertIn('trend_direction', conditions)
        self.assertIn('volatility_value', conditions)
        self.assertIn('trend_value', conditions)
        self.assertIn('recommendation', conditions)
        
        # Values should be reasonable
        self.assertIn(conditions['volatility_condition'], 
                     ['high_volatility', 'low_volatility', 'normal_volatility'])
        self.assertIn(conditions['trend_direction'], ['bullish', 'bearish'])
    
    def test_sizing_rationale_generation(self):
        """Test sizing rationale generation"""
        current_price = 2000.0
        result = self.sizer.calculate_kelly_criterion_size(current_price)
        
        rationale = self.sizer._generate_sizing_rationale(result, RiskLevel.MODERATE)
        
        self.assertIsInstance(rationale, str)
        self.assertGreater(len(rationale), 10)  # Should be meaningful text
        self.assertIn('Kelly Criterion', rationale)
        self.assertIn('moderate', rationale)
    
    def test_position_size_limits(self):
        """Test position size limits are enforced"""
        current_price = 2000.0
        
        # Test with custom parameters
        params = SizingParameters(
            method=SizingMethod.FIXED_PERCENTAGE,
            max_position_size=0.05,  # 5% max
            min_position_size=0.001  # 0.1% min
        )
        
        # Try to get large position (should be limited)
        result = self.sizer.calculate_fixed_percentage_size(0.2, current_price, params)  # 20%
        
        position_value = result.position_size * current_price
        position_pct = position_value / self.sizer.portfolio_value
        
        self.assertLessEqual(position_pct, params.max_position_size)
    
    def test_export_sizing_data(self):
        """Test exporting sizing data"""
        # Generate some results first
        current_price = 2000.0
        self.sizer.calculate_kelly_criterion_size(current_price)
        self.sizer.calculate_volatility_based_size(current_price)
        
        # Test export
        filepath = "test_sizing_export.json"
        result = self.sizer.export_sizing_data(filepath)
        
        self.assertTrue(result)
        
        # Verify file exists and has content
        self.assertTrue(os.path.exists(filepath))
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_statistics(self):
        """Test getting sizing statistics"""
        # Generate some results first
        current_price = 2000.0
        self.sizer.calculate_kelly_criterion_size(current_price)
        self.sizer.calculate_volatility_based_size(current_price)
        self.sizer.calculate_atr_based_size(current_price)
        
        stats = self.sizer.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_calculations', stats)
        self.assertIn('portfolio_value', stats)
        self.assertIn('performance_metrics', stats)
        self.assertIn('position_sizes', stats)
        self.assertIn('risk_amounts', stats)
        self.assertIn('confidence_scores', stats)
        self.assertIn('method_usage', stats)
        
        # Should have 3 calculations
        self.assertEqual(stats['total_calculations'], 3)
        
        # Check performance metrics
        perf_metrics = stats['performance_metrics']
        self.assertEqual(perf_metrics['win_rate'], 0.6)
        self.assertEqual(perf_metrics['avg_win'], 0.02)
        self.assertEqual(perf_metrics['avg_loss'], -0.01)
        
        # Check method usage
        method_usage = stats['method_usage']
        self.assertIn('kelly_criterion', method_usage)
        self.assertIn('volatility_based', method_usage)
        self.assertIn('atr_based', method_usage)
    
    def test_no_data_error_handling(self):
        """Test error handling when no data is set"""
        sizer = PositionSizer()
        current_price = 2000.0
        
        # Should raise error for methods requiring data
        with self.assertRaises(ValueError):
            sizer.calculate_volatility_based_size(current_price)
        
        with self.assertRaises(ValueError):
            sizer.calculate_atr_based_size(current_price)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Create sizer with minimal data
        sizer = PositionSizer()
        
        # Very small dataset
        small_data = pd.DataFrame({
            'Close': [2000, 2010, 2005]
        })
        
        sizer.set_data(small_data, 100000.0)
        
        # Should still work but may use fallbacks
        current_price = 2005.0
        result = sizer.calculate_atr_based_size(current_price)
        
        self.assertIsInstance(result, SizingResult)
        self.assertGreater(result.position_size, 0)
    
    def test_sizing_parameters_to_dict(self):
        """Test SizingParameters to_dict method"""
        params = SizingParameters(
            method=SizingMethod.KELLY_CRITERION,
            risk_per_trade=0.03,
            max_position_size=0.15
        )
        
        params_dict = params.to_dict()
        
        self.assertIsInstance(params_dict, dict)
        self.assertEqual(params_dict['method'], 'kelly_criterion')
        self.assertEqual(params_dict['risk_per_trade'], 0.03)
        self.assertEqual(params_dict['max_position_size'], 0.15)
    
    def test_sizing_result_to_dict(self):
        """Test SizingResult to_dict method"""
        current_price = 2000.0
        result = self.sizer.calculate_kelly_criterion_size(current_price)
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['method'], 'kelly_criterion')
        self.assertIn('position_size', result_dict)
        self.assertIn('risk_amount', result_dict)
        self.assertIn('confidence_score', result_dict)
        self.assertIn('calculation_date', result_dict)
        self.assertIn('parameters_used', result_dict)
        self.assertIn('additional_metrics', result_dict)


if __name__ == '__main__':
    unittest.main() 