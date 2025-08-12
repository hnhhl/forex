"""
Unit Tests for VaR System
Tests for VaRCalculator, MonteCarloSimulator, and VaRBacktester
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.risk.var_calculator import (
    VaRCalculator, VaRMethod, DistributionType, VaRResult
)
from src.core.risk.monte_carlo_simulator import (
    MonteCarloSimulator, SimulationMethod, DistributionModel, 
    SimulationConfig, SimulationResult, StressTestScenario
)
from src.core.risk.var_backtester import (
    VaRBacktester, BacktestType, BacktestResult, BacktestConfig, VaRBacktestReport
)


class TestVaRCalculator(unittest.TestCase):
    """Test cases for VaRCalculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.var_calculator = VaRCalculator()
        
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        returns_data = pd.DataFrame({
            'XAUUSD': np.random.normal(0.001, 0.02, 300),
            'EURUSD': np.random.normal(0.0005, 0.015, 300),
            'GBPUSD': np.random.normal(0.0008, 0.018, 300)
        }, index=dates)
        
        # Create portfolio values
        portfolio_values = pd.Series(
            100000 * (1 + returns_data.mean(axis=1)).cumprod(),
            index=dates
        )
        
        self.var_calculator.set_data(returns_data, portfolio_values)
    
    def test_var_calculator_initialization(self):
        """Test VaR calculator initialization"""
        calculator = VaRCalculator()
        self.assertEqual(calculator.system_name, "VaRCalculator")
        self.assertEqual(len(calculator.default_confidence_levels), 3)
        self.assertEqual(calculator.lookback_period, 252)
        self.assertEqual(calculator.monte_carlo_simulations, 10000)
    
    def test_set_data(self):
        """Test setting returns data"""
        self.assertIsNotNone(self.var_calculator.returns_data)
        self.assertIsNotNone(self.var_calculator.portfolio_values)
        self.assertEqual(len(self.var_calculator.returns_data), 300)
        self.assertEqual(len(self.var_calculator.returns_data.columns), 3)
    
    def test_historical_var_calculation(self):
        """Test historical VaR calculation"""
        result = self.var_calculator.calculate_historical_var(confidence_level=0.95)
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.HISTORICAL)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertIsNotNone(result.var_value)
        self.assertIsNotNone(result.cvar_value)
        self.assertLess(result.cvar_value, result.var_value)  # CVaR should be more negative
    
    def test_parametric_var_normal(self):
        """Test parametric VaR with normal distribution"""
        result = self.var_calculator.calculate_parametric_var(
            confidence_level=0.95,
            distribution='normal'
        )
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.PARAMETRIC_NORMAL)
        self.assertIsNotNone(result.additional_metrics)
        self.assertIn('mean_return', result.additional_metrics)
        self.assertIn('std_return', result.additional_metrics)
    
    def test_parametric_var_t_student(self):
        """Test parametric VaR with t-Student distribution"""
        result = self.var_calculator.calculate_parametric_var(
            confidence_level=0.99,
            distribution='t'
        )
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.PARAMETRIC_T)
        self.assertIn('degrees_freedom', result.additional_metrics)
    
    def test_monte_carlo_var(self):
        """Test Monte Carlo VaR calculation"""
        result = self.var_calculator.calculate_monte_carlo_var(
            confidence_level=0.95,
            simulations=1000
        )
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.MONTE_CARLO)
        self.assertEqual(result.additional_metrics['simulations'], 1000)
        self.assertIsNotNone(result.var_value)
        self.assertIsNotNone(result.cvar_value)
    
    def test_cornish_fisher_var(self):
        """Test Cornish-Fisher VaR calculation"""
        result = self.var_calculator.calculate_cornish_fisher_var(confidence_level=0.95)
        
        self.assertIsInstance(result, VaRResult)
        self.assertEqual(result.method, VaRMethod.CORNISH_FISHER)
        self.assertIn('skewness', result.additional_metrics)
        self.assertIn('kurtosis', result.additional_metrics)
    
    def test_calculate_all_var_methods(self):
        """Test calculating VaR using all methods"""
        results = self.var_calculator.calculate_all_var_methods(confidence_level=0.95)
        
        self.assertIsInstance(results, dict)
        self.assertIn('historical', results)
        self.assertIn('parametric_normal', results)
        self.assertIn('parametric_t', results)
        self.assertIn('monte_carlo', results)
        self.assertIn('cornish_fisher', results)
        
        for method, result in results.items():
            self.assertIsInstance(result, VaRResult)
            self.assertEqual(result.confidence_level, 0.95)
    
    def test_var_summary(self):
        """Test VaR summary generation"""
        summary = self.var_calculator.get_var_summary([0.95, 0.99])
        
        self.assertIsInstance(summary, dict)
        self.assertIn('95.0%', summary)
        self.assertIn('99.0%', summary)
        self.assertEqual(len(summary), 2)
    
    def test_export_var_data(self):
        """Test VaR data export"""
        # Run some calculations first
        self.var_calculator.calculate_historical_var(0.95)
        self.var_calculator.calculate_monte_carlo_var(0.99)
        
        # Test export
        filepath = "test_var_export.json"
        result = self.var_calculator.export_var_data(filepath)
        
        self.assertTrue(result)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_statistics(self):
        """Test getting VaR calculator statistics"""
        # Run some calculations
        self.var_calculator.calculate_historical_var(0.95)
        
        stats = self.var_calculator.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('var_calculations', stats)
        self.assertIn('data_points', stats)
        self.assertIn('mean_return', stats)
        self.assertGreaterEqual(stats['var_calculations'], 1)
    
    def test_var_with_insufficient_data(self):
        """Test VaR calculation with insufficient data"""
        # Create small dataset
        small_data = pd.DataFrame({
            'XAUUSD': np.random.normal(0, 0.01, 50)
        })
        
        calculator = VaRCalculator({'min_observations': 100})
        portfolio_values = pd.Series(100000 * (1 + small_data.iloc[:, 0]).cumprod())
        calculator.set_data(small_data, portfolio_values)
        
        # Should still work but with warning
        result = calculator.calculate_historical_var()
        self.assertIsInstance(result, VaRResult)


class TestMonteCarloSimulator(unittest.TestCase):
    """Test cases for MonteCarloSimulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = MonteCarloSimulator()
        
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns_data = pd.DataFrame({
            'XAUUSD': np.random.normal(0.001, 0.02, 200),
            'EURUSD': np.random.normal(0.0005, 0.015, 200)
        }, index=dates)
        
        self.simulator.set_data(returns_data)
    
    def test_simulator_initialization(self):
        """Test Monte Carlo simulator initialization"""
        simulator = MonteCarloSimulator()
        self.assertEqual(simulator.system_name, "MonteCarloSimulator")
        self.assertEqual(simulator.default_config['simulations'], 10000)
        self.assertEqual(simulator.default_simulations, 10000)
    
    def test_set_data(self):
        """Test setting returns data"""
        self.assertIsNotNone(self.simulator.returns_data)
        self.assertIsNotNone(self.simulator.correlation_matrix)
        self.assertEqual(len(self.simulator.returns_data), 200)
        self.assertEqual(len(self.simulator.fitted_distributions), 2)
    
    def test_standard_simulation(self):
        """Test standard Monte Carlo simulation"""
        config = SimulationConfig(n_simulations=1000, method=SimulationMethod.STANDARD)
        result = self.simulator.simulate_returns(config)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.method, SimulationMethod.STANDARD)
        self.assertEqual(len(result.simulated_returns), 1000)
        self.assertIn(0.95, result.var_estimates)
        self.assertIn(0.95, result.cvar_estimates)
    
    def test_antithetic_simulation(self):
        """Test antithetic variates simulation"""
        config = SimulationConfig(n_simulations=1000, method=SimulationMethod.ANTITHETIC)
        result = self.simulator.simulate_returns(config)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.method, SimulationMethod.ANTITHETIC)
        self.assertEqual(len(result.simulated_returns), 1000)
    
    def test_control_variate_simulation(self):
        """Test control variate simulation"""
        config = SimulationConfig(n_simulations=1000, method=SimulationMethod.CONTROL_VARIATE)
        result = self.simulator.simulate_returns(config)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.method, SimulationMethod.CONTROL_VARIATE)
    
    def test_quasi_random_simulation(self):
        """Test quasi-random simulation"""
        config = SimulationConfig(n_simulations=1000, method=SimulationMethod.QUASI_RANDOM)
        result = self.simulator.simulate_returns(config)
        
        self.assertIsInstance(result, SimulationResult)
        # Should work even if scipy.stats.qmc is not available (fallback to standard)
    
    def test_t_student_distribution(self):
        """Test simulation with t-Student distribution"""
        config = SimulationConfig(
            n_simulations=1000, 
            distribution=DistributionModel.T_STUDENT
        )
        result = self.simulator.simulate_returns(config)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.distribution, DistributionModel.T_STUDENT)
    
    def test_stress_testing(self):
        """Test stress testing functionality"""
        # Create stress scenarios
        scenario1 = StressTestScenario(
            name="Market Crash",
            shock_type="volatility",
            shock_magnitude=0.5,
            description="Severe market downturn"
        )
        
        scenario2 = StressTestScenario(
            name="Volatility Spike",
            shock_type="volatility",
            shock_magnitude=1.0,
            description="High volatility period"
        )
        
        self.simulator.add_stress_scenario(scenario1)
        self.simulator.add_stress_scenario(scenario2)
        
        # Run stress tests
        config = SimulationConfig(n_simulations=500)
        results = self.simulator.run_stress_test([scenario1, scenario2], config)
        
        self.assertIsInstance(results, dict)
        self.assertIn("Market Crash", results)
        self.assertIn("Volatility Spike", results)
        
        for scenario_name, result in results.items():
            self.assertIsInstance(result, SimulationResult)
    
    def test_simulation_statistics(self):
        """Test simulation statistics calculation"""
        config = SimulationConfig(n_simulations=1000)
        result = self.simulator.simulate_returns(config)
        
        stats = result.statistics
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('skewness', stats)
        self.assertIn('kurtosis', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
    
    def test_simulation_summary(self):
        """Test simulation summary generation"""
        # Run some simulations
        config = SimulationConfig(n_simulations=500)
        self.simulator.simulate_returns(config)
        
        summary = self.simulator.get_simulation_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_simulations_run', summary)
        self.assertIn('recent_results', summary)
        self.assertIn('performance_metrics', summary)
    
    def test_export_simulation_data(self):
        """Test simulation data export"""
        # Run simulation
        config = SimulationConfig(n_simulations=500)
        self.simulator.simulate_returns(config)
        
        # Test export
        filepath = "test_simulation_export.json"
        result = self.simulator.export_simulation_data(filepath)
        
        self.assertTrue(result)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_statistics(self):
        """Test getting simulator statistics"""
        stats = self.simulator.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_simulations_run', stats)
        self.assertIn('default_n_simulations', stats)
        self.assertIn('methods_available', stats)
        self.assertIn('distributions_available', stats)


class TestVaRBacktester(unittest.TestCase):
    """Test cases for VaRBacktester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backtester = VaRBacktester()
        
        # Create sample VaR forecasts
        np.random.seed(42)
        n_obs = 250
        
        # Generate sample returns
        returns = np.random.normal(0.001, 0.02, n_obs)
        dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
        self.actual_returns = pd.Series(returns, index=dates)
        
        # Generate VaR forecasts (should have some violations)
        var_values = np.random.normal(-0.03, 0.005, n_obs)  # VaR estimates
        
        self.var_forecasts = []
        for i, var_val in enumerate(var_values):
            var_result = VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=0.95,
                var_value=var_val,
                cvar_value=var_val * 1.2,  # CVaR more negative
                portfolio_value=100000.0,
                var_absolute=abs(var_val * 100000.0),
                var_percentage=abs(var_val) * 100,
                calculation_date=dates[i],
                lookback_period=252
            )
            self.var_forecasts.append(var_result)
    
    def test_backtester_initialization(self):
        """Test VaR backtester initialization"""
        backtester = VaRBacktester()
        self.assertEqual(backtester.system_name, "VaRBacktester")
        self.assertEqual(backtester.default_config.confidence_level, 0.95)
        self.assertEqual(len(backtester.default_config.tests_to_run), 3)
    
    def test_backtest_var_model(self):
        """Test comprehensive VaR model backtesting"""
        report = self.backtester.backtest_var_model(
            self.var_forecasts,
            self.actual_returns,
            model_name="Test_Model"
        )
        
        self.assertIsInstance(report, VaRBacktestReport)
        self.assertEqual(report.model_name, "Test_Model")
        self.assertEqual(report.var_method, VaRMethod.HISTORICAL)
        self.assertEqual(report.confidence_level, 0.95)
        self.assertEqual(report.total_observations, 250)
        
        # Check test results
        self.assertIsNotNone(report.kupiec_test)
        self.assertIsNotNone(report.christoffersen_test)
        self.assertIn(report.kupiec_test.result, [BacktestResult.ACCEPT, BacktestResult.REJECT])
        self.assertIn(report.christoffersen_test.result, [BacktestResult.ACCEPT, BacktestResult.REJECT])
    
    def test_kupiec_test(self):
        """Test Kupiec test implementation"""
        # Create violations array with known properties
        violations = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0] * 25)  # 50 violations out of 250
        
        kupiec_result = self.backtester._kupiec_test(violations, 0.95, 0.05)
        
        self.assertEqual(kupiec_result.test_name, "Kupiec Test (Unconditional Coverage)")
        self.assertIsInstance(kupiec_result.test_statistic, float)
        self.assertIsInstance(kupiec_result.p_value, float)
        self.assertIn(kupiec_result.result, [BacktestResult.ACCEPT, BacktestResult.REJECT])
    
    def test_christoffersen_test(self):
        """Test Christoffersen test implementation"""
        # Create violations with some clustering
        violations = np.zeros(250)
        violations[10:15] = 1  # Cluster of violations
        violations[50:52] = 1  # Another cluster
        violations[100] = 1    # Isolated violation
        
        christoffersen_result = self.backtester._christoffersen_test(violations, 0.95, 0.05)
        
        self.assertEqual(christoffersen_result.test_name, "Christoffersen Test (Independence)")
        self.assertIsInstance(christoffersen_result.test_statistic, float)
        self.assertIsInstance(christoffersen_result.p_value, float)
    
    def test_traffic_light_classification(self):
        """Test traffic light system classification"""
        # Test different scenarios
        green_result = self.backtester._traffic_light_classification(10, 12)  # Within expected
        yellow_result = self.backtester._traffic_light_classification(15, 10)  # Moderate excess
        red_result = self.backtester._traffic_light_classification(25, 10)  # High excess
        
        self.assertEqual(green_result, "Green")
        self.assertEqual(yellow_result, "Yellow")
        self.assertEqual(red_result, "Red")
    
    def test_backtest_with_no_violations(self):
        """Test backtesting with no violations"""
        # Create VaR forecasts that are always exceeded (no violations)
        conservative_forecasts = []
        for i, ret in enumerate(self.actual_returns):
            var_result = VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=0.95,
                var_value=ret - 0.01,  # Always more negative than actual return
                cvar_value=ret - 0.015,
                portfolio_value=100000.0,
                var_absolute=abs((ret - 0.01) * 100000.0),
                var_percentage=abs(ret - 0.01) * 100,
                calculation_date=self.actual_returns.index[i],
                lookback_period=252
            )
            conservative_forecasts.append(var_result)
        
        report = self.backtester.backtest_var_model(
            conservative_forecasts,
            self.actual_returns,
            model_name="Conservative_Model"
        )
        
        self.assertEqual(report.total_violations, 0)
        self.assertEqual(report.traffic_light_zone, "Green")
    
    def test_backtest_with_many_violations(self):
        """Test backtesting with excessive violations"""
        # Create VaR forecasts that are frequently exceeded
        aggressive_forecasts = []
        for i, ret in enumerate(self.actual_returns):
            var_result = VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=0.95,
                var_value=ret + 0.01,  # Often less negative than actual return
                cvar_value=ret + 0.005,
                portfolio_value=100000.0,
                var_absolute=abs((ret + 0.01) * 100000.0),
                var_percentage=abs(ret + 0.01) * 100,
                calculation_date=self.actual_returns.index[i],
                lookback_period=252
            )
            aggressive_forecasts.append(var_result)
        
        report = self.backtester.backtest_var_model(
            aggressive_forecasts,
            self.actual_returns,
            model_name="Aggressive_Model"
        )
        
        self.assertGreater(report.total_violations, report.expected_violations)
        # Should likely be Red or Yellow
        self.assertIn(report.traffic_light_zone, ["Yellow", "Red"])
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        # Run backtests for multiple models
        self.backtester.backtest_var_model(
            self.var_forecasts,
            self.actual_returns,
            model_name="Model_A"
        )
        
        # Create second model with different characteristics
        var_forecasts_b = []
        for var_result in self.var_forecasts:
            var_result_b = VaRResult(
                method=VaRMethod.MONTE_CARLO,
                confidence_level=0.95,
                var_value=var_result.var_value * 1.1,  # Slightly more conservative
                cvar_value=var_result.cvar_value * 1.1,
                portfolio_value=var_result.portfolio_value,
                var_absolute=abs(var_result.var_value * 1.1 * var_result.portfolio_value),
                var_percentage=abs(var_result.var_value * 1.1) * 100,
                calculation_date=var_result.calculation_date,
                lookback_period=var_result.lookback_period
            )
            var_forecasts_b.append(var_result_b)
        
        self.backtester.backtest_var_model(
            var_forecasts_b,
            self.actual_returns,
            model_name="Model_B"
        )
        
        # Compare models
        comparison = self.backtester.compare_models(["Model_A", "Model_B"])
        
        self.assertIsInstance(comparison, dict)
        self.assertIn('models', comparison)
        self.assertIn('comparison_metrics', comparison)
        self.assertIn('rankings', comparison)
        self.assertEqual(len(comparison['models']), 2)
    
    def test_backtest_summary(self):
        """Test backtest summary generation"""
        # Run some backtests
        self.backtester.backtest_var_model(
            self.var_forecasts,
            self.actual_returns,
            model_name="Summary_Test_Model"
        )
        
        summary = self.backtester.get_backtest_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_backtests', summary)
        self.assertIn('unique_models', summary)
        self.assertIn('recent_results', summary)
        self.assertIn('performance_summary', summary)
        self.assertGreaterEqual(summary['total_backtests'], 1)
    
    def test_export_backtest_data(self):
        """Test backtest data export"""
        # Run backtest
        self.backtester.backtest_var_model(
            self.var_forecasts,
            self.actual_returns,
            model_name="Export_Test_Model"
        )
        
        # Test export
        filepath = "test_backtest_export.json"
        result = self.backtester.export_backtest_data(filepath)
        
        self.assertTrue(result)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def test_get_statistics(self):
        """Test getting backtester statistics"""
        stats = self.backtester.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_backtests_performed', stats)
        self.assertIn('unique_models_tested', stats)
        self.assertIn('available_tests', stats)
        self.assertIn('default_tests', stats)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Test with very few observations
        short_forecasts = self.var_forecasts[:50]
        short_returns = self.actual_returns.iloc[:50]
        
        config = BacktestConfig(min_observations=100)
        
        with self.assertRaises(ValueError):
            self.backtester.backtest_var_model(
                short_forecasts,
                short_returns,
                model_name="Short_Data_Model",
                config=config
            )


class TestVaRSystemIntegration(unittest.TestCase):
    """Integration tests for the complete VaR System"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.var_calculator = VaRCalculator()
        self.monte_carlo_simulator = MonteCarloSimulator()
        self.var_backtester = VaRBacktester()
        
        # Create comprehensive test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        returns_data = pd.DataFrame({
            'XAUUSD': np.random.normal(0.001, 0.02, 500),
            'EURUSD': np.random.normal(0.0005, 0.015, 500),
            'GBPUSD': np.random.normal(0.0008, 0.018, 500)
        }, index=dates)
        
        portfolio_values = pd.Series(
            100000 * (1 + returns_data.mean(axis=1)).cumprod(),
            index=dates
        )
        
        self.var_calculator.set_data(returns_data, portfolio_values)
        self.monte_carlo_simulator.set_data(returns_data)
        
        # Split data for backtesting
        self.train_data = returns_data.iloc[:300]
        self.test_data = returns_data.iloc[300:]
    
    def test_full_var_workflow(self):
        """Test complete VaR workflow from calculation to backtesting"""
        # 1. Calculate VaR using multiple methods
        var_results = self.var_calculator.calculate_all_var_methods(confidence_level=0.95)
        
        self.assertIsInstance(var_results, dict)
        self.assertGreater(len(var_results), 0)
        
        # 2. Run Monte Carlo simulation
        from src.core.risk.monte_carlo_simulator import SimulationConfig
        config = SimulationConfig(n_simulations=1000)
        simulation_result = self.monte_carlo_simulator.simulate_returns(config)
        
        self.assertIsInstance(simulation_result, SimulationResult)
        
        # 3. Create VaR forecasts for backtesting
        var_forecasts = []
        for i in range(len(self.test_data)):
            # Use historical VaR as example
            var_result = VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=0.95,
                var_value=var_results['historical'].var_value,
                cvar_value=var_results['historical'].cvar_value,
                portfolio_value=100000.0,
                var_absolute=abs(var_results['historical'].var_value * 100000.0),
                var_percentage=abs(var_results['historical'].var_value) * 100,
                calculation_date=datetime.now(),
                lookback_period=252
            )
            var_forecasts.append(var_result)
        
        # 4. Backtest the VaR model
        test_returns = self.test_data.mean(axis=1)  # Portfolio returns
        backtest_report = self.var_backtester.backtest_var_model(
            var_forecasts,
            test_returns,
            model_name="Integrated_Test_Model"
        )
        
        self.assertIsInstance(backtest_report, VaRBacktestReport)
        self.assertEqual(backtest_report.model_name, "Integrated_Test_Model")
    
    def test_stress_testing_integration(self):
        """Test stress testing integration with VaR calculation"""
        # Create stress scenarios
        from src.core.risk.monte_carlo_simulator import StressTestScenario
        
        market_crash = StressTestScenario(
            name="Market Crash 2008",
            shock_type="volatility",
            shock_magnitude=1.0,
            description="Severe market downturn similar to 2008"
        )
        
        # Run stress test
        from src.core.risk.monte_carlo_simulator import SimulationConfig
        config = SimulationConfig(n_simulations=1000)
        stress_results = self.monte_carlo_simulator.run_stress_test([market_crash], config)
        
        self.assertIn("Market Crash 2008", stress_results)
        
        # Compare stressed VaR with normal VaR
        normal_result = self.monte_carlo_simulator.simulate_returns(config)
        stressed_result = stress_results["Market Crash 2008"]
        
        # Stressed VaR should be more negative (higher risk)
        self.assertLess(
            stressed_result.var_estimates[0.95],
            normal_result.var_estimates[0.95]
        )
    
    def test_model_comparison_workflow(self):
        """Test comparing multiple VaR models"""
        # Generate forecasts for different models
        models_data = {
            "Historical_VaR": [],
            "Monte_Carlo_VaR": [],
            "Parametric_VaR": []
        }
        
        # Create different VaR forecasts
        for i in range(len(self.test_data)):
            # Historical VaR
            var_val = np.random.normal(-0.03, 0.005)
            hist_result = VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=0.95,
                var_value=var_val,
                cvar_value=np.random.normal(-0.04, 0.005),
                portfolio_value=100000.0,
                var_absolute=abs(var_val * 100000.0),
                var_percentage=abs(var_val) * 100,
                calculation_date=datetime.now(),
                lookback_period=252
            )
            models_data["Historical_VaR"].append(hist_result)
            
            # Monte Carlo VaR
            mc_var_val = np.random.normal(-0.032, 0.006)
            mc_result = VaRResult(
                method=VaRMethod.MONTE_CARLO,
                confidence_level=0.95,
                var_value=mc_var_val,
                cvar_value=np.random.normal(-0.042, 0.006),
                portfolio_value=100000.0,
                var_absolute=abs(mc_var_val * 100000.0),
                var_percentage=abs(mc_var_val) * 100,
                calculation_date=datetime.now(),
                lookback_period=252
            )
            models_data["Monte_Carlo_VaR"].append(mc_result)
            
            # Parametric VaR
            param_var_val = np.random.normal(-0.028, 0.004)
            param_result = VaRResult(
                method=VaRMethod.PARAMETRIC_NORMAL,
                confidence_level=0.95,
                var_value=param_var_val,
                cvar_value=np.random.normal(-0.038, 0.004),
                portfolio_value=100000.0,
                var_absolute=abs(param_var_val * 100000.0),
                var_percentage=abs(param_var_val) * 100,
                calculation_date=datetime.now(),
                lookback_period=252
            )
            models_data["Parametric_VaR"].append(param_result)
        
        # Backtest all models
        test_returns = self.test_data.mean(axis=1)
        
        for model_name, forecasts in models_data.items():
            self.var_backtester.backtest_var_model(
                forecasts,
                test_returns,
                model_name=model_name
            )
        
        # Compare models
        comparison = self.var_backtester.compare_models(list(models_data.keys()))
        
        self.assertIsInstance(comparison, dict)
        self.assertEqual(len(comparison['models']), 3)
        self.assertIn('rankings', comparison)
        self.assertIn('overall', comparison['rankings'])
    
    def test_performance_monitoring(self):
        """Test performance monitoring across all components"""
        # Get statistics from all components
        var_stats = self.var_calculator.get_statistics()
        sim_stats = self.monte_carlo_simulator.get_statistics()
        backtest_stats = self.var_backtester.get_statistics()
        
        # Verify all components are working
        self.assertIsInstance(var_stats, dict)
        self.assertIsInstance(sim_stats, dict)
        self.assertIsInstance(backtest_stats, dict)
        
        # Check key metrics
        self.assertIn('total_var_calculations', var_stats)
        self.assertIn('total_simulations_run', sim_stats)
        self.assertIn('total_backtests_performed', backtest_stats)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestVaRCalculator))
    test_suite.addTest(unittest.makeSuite(TestMonteCarloSimulator))
    test_suite.addTest(unittest.makeSuite(TestVaRBacktester))
    test_suite.addTest(unittest.makeSuite(TestVaRSystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VAR SYSTEM TEST SUMMARY")
    print(f"{'='*60}")
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
    
    print(f"{'='*60}") 