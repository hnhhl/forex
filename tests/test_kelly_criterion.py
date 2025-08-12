#!/usr/bin/env python3
"""
Test suite for Kelly Criterion Calculator
Part of Ultimate XAU Super System V4.0
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os
import json

from src.core.trading.kelly_criterion import (
    KellyCriterionCalculator, KellyMethod, TradeResult, 
    KellyParameters, KellyResult
)

class TestKellyCriterionCalculator(unittest.TestCase):
    """Test Kelly Criterion Calculator functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'max_lookback_trades': 100,
            'min_trades_required': 10,
            'confidence_threshold': 0.7
        }
        self.calculator = KellyCriterionCalculator(self.config)
        
        # Sample trade results for testing
        self.sample_trades = self._create_sample_trades()
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def _create_sample_trades(self) -> list:
        """Create sample trade results for testing"""
        trades = []
        base_date = datetime.now() - timedelta(days=100)
        
        # Create 50 trades with 60% win rate and 1.5 profit factor
        for i in range(50):
            trade_date = base_date + timedelta(days=i)
            
            if i < 30:  # 30 winning trades (60%)
                profit_loss = np.random.uniform(100, 300)  # Average win ~200
                win = True
            else:  # 20 losing trades (40%)
                profit_loss = np.random.uniform(-150, -50)  # Average loss ~-100
                win = False
            
            trade = TradeResult(
                profit_loss=profit_loss,
                win=win,
                trade_date=trade_date,
                symbol="XAUUSD",
                entry_price=2000.0 + np.random.uniform(-50, 50),
                exit_price=2000.0 + np.random.uniform(-50, 50),
                volume=0.1,
                duration_minutes=np.random.randint(30, 240)
            )
            trades.append(trade)
        
        return trades
    
    def test_calculator_initialization(self):
        """Test Kelly calculator initialization"""
        self.assertIsInstance(self.calculator, KellyCriterionCalculator)
        self.assertEqual(self.calculator.max_lookback_trades, 100)
        self.assertEqual(self.calculator.min_trades_required, 10)
        self.assertEqual(len(self.calculator.trade_history), 0)
        self.assertEqual(len(self.calculator.calculation_history), 0)
    
    def test_add_trade_result(self):
        """Test adding trade results"""
        trade = self.sample_trades[0]
        
        # Add trade
        self.calculator.add_trade_result(trade)
        
        # Verify trade added
        self.assertEqual(len(self.calculator.trade_history), 1)
        self.assertEqual(self.calculator.trade_history[0], trade)
        
        # Verify performance metrics updated
        self.assertIn('total_trades', self.calculator.performance_metrics)
        self.assertEqual(self.calculator.performance_metrics['total_trades'], 1)
    
    def test_add_multiple_trades(self):
        """Test adding multiple trade results"""
        # Add all sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Verify all trades added
        self.assertEqual(len(self.calculator.trade_history), 50)
        
        # Verify performance metrics
        metrics = self.calculator.performance_metrics
        self.assertEqual(metrics['total_trades'], 50)
        self.assertEqual(metrics['winning_trades'], 30)
        self.assertEqual(metrics['losing_trades'], 20)
        self.assertAlmostEqual(metrics['win_rate'], 0.6, places=2)
        self.assertGreater(metrics['profit_factor'], 1.0)
    
    def test_classic_kelly_calculation(self):
        """Test classic Kelly criterion calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly fraction
        result = self.calculator.calculate_kelly_fraction(KellyMethod.CLASSIC)
        
        # Verify result
        self.assertIsInstance(result, KellyResult)
        self.assertEqual(result.method_used, KellyMethod.CLASSIC)
        self.assertGreater(result.kelly_fraction, 0)
        self.assertLess(result.kelly_fraction, 1)
        self.assertGreater(result.confidence_score, 0)
        self.assertLess(result.confidence_score, 1)
    
    def test_fractional_kelly_calculation(self):
        """Test fractional Kelly calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate fractional Kelly
        result = self.calculator.calculate_kelly_fraction(KellyMethod.FRACTIONAL)
        
        # Verify result
        self.assertEqual(result.method_used, KellyMethod.FRACTIONAL)
        self.assertGreater(result.kelly_fraction, 0)
        
        # Fractional Kelly should be less than classic Kelly
        classic_result = self.calculator.calculate_kelly_fraction(KellyMethod.CLASSIC)
        self.assertLessEqual(result.kelly_fraction, classic_result.kelly_fraction)
    
    def test_conservative_kelly_calculation(self):
        """Test conservative Kelly calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate conservative Kelly
        result = self.calculator.calculate_kelly_fraction(KellyMethod.CONSERVATIVE)
        
        # Verify result
        self.assertEqual(result.method_used, KellyMethod.CONSERVATIVE)
        self.assertGreater(result.kelly_fraction, 0)
        
        # Conservative Kelly should be less than classic Kelly
        classic_result = self.calculator.calculate_kelly_fraction(KellyMethod.CLASSIC)
        self.assertLessEqual(result.kelly_fraction, classic_result.kelly_fraction)
    
    def test_dynamic_kelly_calculation(self):
        """Test dynamic Kelly calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate dynamic Kelly
        result = self.calculator.calculate_kelly_fraction(KellyMethod.DYNAMIC)
        
        # Verify result
        self.assertEqual(result.method_used, KellyMethod.DYNAMIC)
        self.assertGreater(result.kelly_fraction, 0)
    
    def test_adaptive_kelly_calculation(self):
        """Test adaptive Kelly calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate adaptive Kelly
        result = self.calculator.calculate_kelly_fraction(KellyMethod.ADAPTIVE)
        
        # Verify result
        self.assertEqual(result.method_used, KellyMethod.ADAPTIVE)
        self.assertGreater(result.kelly_fraction, 0)
    
    def test_custom_parameters(self):
        """Test Kelly calculation with custom parameters"""
        # Create custom parameters
        custom_params = KellyParameters(
            win_rate=0.65,
            average_win=250.0,
            average_loss=-100.0,
            profit_factor=1.6,
            total_trades=100
        )
        
        # Calculate Kelly with custom parameters
        result = self.calculator.calculate_kelly_fraction(
            KellyMethod.CLASSIC, 
            custom_parameters=custom_params
        )
        
        # Verify result uses custom parameters
        self.assertEqual(result.parameters.win_rate, 0.65)
        self.assertEqual(result.parameters.average_win, 250.0)
        self.assertEqual(result.parameters.average_loss, -100.0)
        self.assertEqual(result.parameters.profit_factor, 1.6)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient trade data"""
        # Add only a few trades (less than minimum required)
        for i in range(5):
            self.calculator.add_trade_result(self.sample_trades[i])
        
        # Should raise ValueError for insufficient data
        with self.assertRaises(ValueError):
            self.calculator.calculate_kelly_fraction()
    
    def test_risk_controls(self):
        """Test risk control mechanisms"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        result = self.calculator.calculate_kelly_fraction()
        
        # Verify risk controls applied
        self.assertLessEqual(result.kelly_fraction, 0.25)  # Max Kelly limit
        self.assertGreaterEqual(result.kelly_fraction, 0.01)  # Min Kelly limit
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        result = self.calculator.calculate_kelly_fraction()
        
        # Verify confidence score
        self.assertGreater(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 1)
        
        # More trades should increase confidence
        more_trades = self._create_sample_trades()
        for trade in more_trades:
            self.calculator.add_trade_result(trade)
        
        result2 = self.calculator.calculate_kelly_fraction()
        self.assertGreaterEqual(result2.confidence_score, result.confidence_score)
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        result = self.calculator.calculate_kelly_fraction()
        
        # Verify risk metrics
        self.assertIn('expected_return_per_trade', result.risk_metrics)
        self.assertIn('return_volatility', result.risk_metrics)
        self.assertIn('max_theoretical_loss', result.risk_metrics)
        
        # Expected return should be positive for profitable strategy
        self.assertGreater(result.risk_metrics['expected_return_per_trade'], 0)
    
    def test_warnings_generation(self):
        """Test warning generation"""
        # Create poor performance trades
        poor_trades = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            trade_date = base_date + timedelta(days=i)
            
            if i < 10:  # Only 33% win rate
                profit_loss = 50.0
                win = True
            else:
                profit_loss = -30.0
                win = False
            
            trade = TradeResult(
                profit_loss=profit_loss,
                win=win,
                trade_date=trade_date,
                symbol="XAUUSD",
                entry_price=2000.0,
                exit_price=2000.0,
                volume=0.1,
                duration_minutes=60
            )
            poor_trades.append(trade)
        
        # Add poor trades
        for trade in poor_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        result = self.calculator.calculate_kelly_fraction()
        
        # Should have warnings for poor performance
        self.assertGreater(len(result.warnings), 0)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly to populate history
        self.calculator.calculate_kelly_fraction()
        
        # Get performance summary
        summary = self.calculator.get_performance_summary()
        
        # Verify summary structure
        self.assertIn('performance_metrics', summary)
        self.assertIn('trade_history_length', summary)
        self.assertIn('calculation_history_length', summary)
        self.assertIn('risk_controls', summary)
        self.assertIn('latest_kelly_result', summary)
        
        # Verify values
        self.assertEqual(summary['trade_history_length'], 50)
        self.assertEqual(summary['calculation_history_length'], 1)
    
    def test_export_data(self):
        """Test data export functionality"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly to populate history
        self.calculator.calculate_kelly_fraction()
        
        # Export data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_filepath = f.name
        
        try:
            # Export data
            success = self.calculator.export_data(temp_filepath)
            self.assertTrue(success)
            
            # Verify file exists and contains data
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            # Verify exported data structure
            self.assertIn('performance_metrics', data)
            self.assertIn('calculation_history', data)
            self.assertIn('trade_history', data)
            self.assertIn('config', data)
            self.assertIn('risk_controls', data)
            self.assertIn('export_timestamp', data)
            
            # Verify data content
            self.assertEqual(len(data['trade_history']), 50)
            self.assertEqual(len(data['calculation_history']), 1)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_statistics(self):
        """Test statistics generation"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        self.calculator.calculate_kelly_fraction()
        
        # Get statistics
        stats = self.calculator.get_statistics()
        
        # Verify statistics structure
        self.assertIn('total_trades_processed', stats)
        self.assertIn('total_calculations_performed', stats)
        self.assertIn('performance_metrics', stats)
        self.assertIn('risk_controls', stats)
        self.assertIn('config', stats)
        self.assertIn('last_calculation', stats)
        
        # Verify values
        self.assertEqual(stats['total_trades_processed'], 50)
        self.assertEqual(stats['total_calculations_performed'], 1)
    
    def test_kelly_result_serialization(self):
        """Test Kelly result serialization"""
        # Add sample trades
        for trade in self.sample_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        result = self.calculator.calculate_kelly_fraction()
        
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Verify dictionary structure
        self.assertIn('kelly_fraction', result_dict)
        self.assertIn('recommended_position_size', result_dict)
        self.assertIn('confidence_score', result_dict)
        self.assertIn('method_used', result_dict)
        self.assertIn('risk_metrics', result_dict)
        self.assertIn('warnings', result_dict)
        self.assertIn('calculation_timestamp', result_dict)
        self.assertIn('parameters', result_dict)
        
        # Verify parameter structure
        params = result_dict['parameters']
        self.assertIn('win_rate', params)
        self.assertIn('average_win', params)
        self.assertIn('average_loss', params)
        self.assertIn('profit_factor', params)
        self.assertIn('total_trades', params)
    
    def test_trade_result_validation(self):
        """Test trade result validation"""
        # Test profit/loss and win flag consistency
        trade1 = TradeResult(
            profit_loss=100.0,
            win=False,  # Should be corrected to True
            trade_date=datetime.now(),
            symbol="XAUUSD",
            entry_price=2000.0,
            exit_price=2100.0,
            volume=0.1,
            duration_minutes=60
        )
        
        # Win flag should be corrected
        self.assertTrue(trade1.win)
        
        trade2 = TradeResult(
            profit_loss=-50.0,
            win=True,  # Should be corrected to False
            trade_date=datetime.now(),
            symbol="XAUUSD",
            entry_price=2000.0,
            exit_price=1950.0,
            volume=0.1,
            duration_minutes=60
        )
        
        # Win flag should be corrected
        self.assertFalse(trade2.win)
    
    def test_kelly_parameters_validation(self):
        """Test Kelly parameters validation"""
        # Test invalid win rate
        with self.assertRaises(ValueError):
            KellyParameters(
                win_rate=1.5,  # Invalid: > 1
                average_win=100.0,
                average_loss=-50.0,
                profit_factor=1.5,
                total_trades=50
            )
        
        # Test invalid average loss (should be negative)
        with self.assertRaises(ValueError):
            KellyParameters(
                win_rate=0.6,
                average_win=100.0,
                average_loss=50.0,  # Invalid: should be negative
                profit_factor=1.5,
                total_trades=50
            )
        
        # Test invalid average win (should be positive)
        with self.assertRaises(ValueError):
            KellyParameters(
                win_rate=0.6,
                average_win=-100.0,  # Invalid: should be positive
                average_loss=-50.0,
                profit_factor=1.5,
                total_trades=50
            )
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test calculation with no data - should raise ValueError
        empty_calculator = KellyCriterionCalculator()
        
        # Should raise ValueError for insufficient data
        with self.assertRaises(ValueError):
            empty_calculator.calculate_kelly_fraction()
        
        # Test with custom parameters that cause other errors
        invalid_params = KellyParameters(
            win_rate=0.5,
            average_win=100.0,
            average_loss=-50.0,
            profit_factor=1.0,
            total_trades=50
        )
        
        # Mock a calculation error
        with patch.object(empty_calculator, '_calculate_classic_kelly', side_effect=Exception("Test error")):
            result = empty_calculator.calculate_kelly_fraction(custom_parameters=invalid_params)
            self.assertEqual(result.kelly_fraction, 0.01)  # Conservative default
            self.assertIn("Calculation failed", result.warnings[0])
    
    def test_consecutive_losses_adjustment(self):
        """Test Kelly adjustment for consecutive losses"""
        # Create trades with consecutive losses at the end
        mixed_trades = []
        base_date = datetime.now() - timedelta(days=30)
        
        # First 20 trades with good performance
        for i in range(20):
            trade_date = base_date + timedelta(days=i)
            profit_loss = 100.0 if i % 2 == 0 else -50.0
            win = profit_loss > 0
            
            trade = TradeResult(
                profit_loss=profit_loss,
                win=win,
                trade_date=trade_date,
                symbol="XAUUSD",
                entry_price=2000.0,
                exit_price=2000.0,
                volume=0.1,
                duration_minutes=60
            )
            mixed_trades.append(trade)
        
        # Last 5 trades are consecutive losses
        for i in range(5):
            trade_date = base_date + timedelta(days=20 + i)
            trade = TradeResult(
                profit_loss=-50.0,
                win=False,
                trade_date=trade_date,
                symbol="XAUUSD",
                entry_price=2000.0,
                exit_price=1950.0,
                volume=0.1,
                duration_minutes=60
            )
            mixed_trades.append(trade)
        
        # Add trades
        for trade in mixed_trades:
            self.calculator.add_trade_result(trade)
        
        # Calculate Kelly
        result = self.calculator.calculate_kelly_fraction()
        
        # Kelly should be reduced due to consecutive losses
        self.assertGreater(result.kelly_fraction, 0)
        # Should have warning about recent performance
        warning_found = any("Recent performance" in warning or "consecutive losses" in warning for warning in result.warnings)
        self.assertTrue(warning_found)


class TestKellyIntegration(unittest.TestCase):
    """Integration tests for Kelly Criterion Calculator"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.calculator = KellyCriterionCalculator()
    
    def test_full_kelly_workflow(self):
        """Test complete Kelly calculation workflow"""
        # Step 1: Add trade history
        trades = self._create_realistic_trade_history()
        for trade in trades:
            self.calculator.add_trade_result(trade)
        
        # Step 2: Calculate Kelly using different methods
        methods = [
            KellyMethod.CLASSIC,
            KellyMethod.FRACTIONAL,
            KellyMethod.CONSERVATIVE,
            KellyMethod.DYNAMIC,
            KellyMethod.ADAPTIVE
        ]
        
        results = {}
        for method in methods:
            result = self.calculator.calculate_kelly_fraction(method)
            results[method] = result
            
            # Verify each result
            self.assertIsInstance(result, KellyResult)
            self.assertGreater(result.kelly_fraction, 0)
            self.assertGreater(result.confidence_score, 0)
        
        # Step 3: Verify method relationships
        # Conservative should be <= Classic
        self.assertLessEqual(
            results[KellyMethod.CONSERVATIVE].kelly_fraction,
            results[KellyMethod.CLASSIC].kelly_fraction
        )
        
        # Fractional should be <= Classic
        self.assertLessEqual(
            results[KellyMethod.FRACTIONAL].kelly_fraction,
            results[KellyMethod.CLASSIC].kelly_fraction
        )
        
        # Step 4: Export and verify data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_filepath = f.name
        
        try:
            success = self.calculator.export_data(temp_filepath)
            self.assertTrue(success)
            
            # Verify exported data
            with open(temp_filepath, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(len(data['calculation_history']), len(methods))
            self.assertEqual(len(data['trade_history']), len(trades))
            
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def _create_realistic_trade_history(self) -> list:
        """Create realistic trade history for integration testing"""
        trades = []
        base_date = datetime.now() - timedelta(days=200)
        
        # Simulate 6 months of trading with varying performance
        for i in range(100):
            trade_date = base_date + timedelta(days=i * 2)
            
            # Simulate market cycles
            if i < 30:  # Good period: 70% win rate
                win_prob = 0.7
            elif i < 60:  # Average period: 55% win rate
                win_prob = 0.55
            else:  # Challenging period: 45% win rate
                win_prob = 0.45
            
            is_win = np.random.random() < win_prob
            
            if is_win:
                profit_loss = np.random.uniform(80, 250)
            else:
                profit_loss = np.random.uniform(-180, -60)
            
            entry_price = 2000.0 + np.random.uniform(-100, 100)
            exit_price = entry_price + (profit_loss / 0.1)  # Assuming 0.1 lot size
            
            trade = TradeResult(
                profit_loss=profit_loss,
                win=is_win,
                trade_date=trade_date,
                symbol="XAUUSD",
                entry_price=entry_price,
                exit_price=exit_price,
                volume=0.1,
                duration_minutes=np.random.randint(30, 300)
            )
            trades.append(trade)
        
        return trades


if __name__ == '__main__':
    unittest.main()
