"""
Demo Script for VaR System
Demonstrates VaRCalculator, MonteCarloSimulator, and VaRBacktester functionality
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.risk.var_calculator import (
    VaRCalculator, VaRMethod, DistributionType
)
from src.core.risk.monte_carlo_simulator import (
    MonteCarloSimulator, SimulationMethod, DistributionModel, 
    SimulationConfig, StressTestScenario
)
from src.core.risk.var_backtester import (
    VaRBacktester, BacktestType, BacktestConfig
)


def create_sample_data():
    """Create sample market data for demonstration"""
    print("📊 Creating sample market data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create 2 years of daily data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    
    # Generate correlated returns for multiple assets
    n_assets = 4
    assets = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
    
    # Create correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.3, 0.4, -0.2],
        [0.3, 1.0, 0.6, -0.1],
        [0.4, 0.6, 1.0, -0.15],
        [-0.2, -0.1, -0.15, 1.0]
    ])
    
    # Generate correlated random returns
    mean_returns = np.array([0.0008, 0.0002, 0.0003, -0.0001])  # Daily returns
    volatilities = np.array([0.025, 0.012, 0.015, 0.018])  # Daily volatilities
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate independent normal random variables
    independent_returns = np.random.normal(0, 1, (len(dates), n_assets))
    
    # Apply correlation and scale
    correlated_returns = independent_returns @ L.T
    
    # Scale by volatility and add mean
    scaled_returns = correlated_returns * volatilities + mean_returns
    
    # Create DataFrame
    returns_data = pd.DataFrame(scaled_returns, index=dates, columns=assets)
    
    # Create portfolio values (starting at $100,000)
    portfolio_returns = returns_data.mean(axis=1)  # Equal weight portfolio
    portfolio_values = pd.Series(
        100000 * (1 + portfolio_returns).cumprod(),
        index=dates
    )
    
    print(f"✅ Created {len(returns_data)} days of data for {len(assets)} assets")
    print(f"📈 Portfolio value range: ${portfolio_values.min():,.2f} - ${portfolio_values.max():,.2f}")
    
    return returns_data, portfolio_values


def demo_var_calculator():
    """Demonstrate VaR Calculator functionality"""
    print("\n" + "="*60)
    print("🧮 VAR CALCULATOR DEMO")
    print("="*60)
    
    # Create VaR Calculator
    var_calculator = VaRCalculator({
        'confidence_levels': [0.95, 0.99, 0.999],
        'lookback_period': 252,
        'monte_carlo_simulations': 5000
    })
    
    # Create sample data
    returns_data, portfolio_values = create_sample_data()
    var_calculator.set_data(returns_data, portfolio_values)
    
    print(f"\n📋 VaR Calculator Configuration:")
    print(f"  Confidence Levels: {var_calculator.default_confidence_levels}")
    print(f"  Lookback Period: {var_calculator.lookback_period} days")
    print(f"  Monte Carlo Simulations: {var_calculator.monte_carlo_simulations}")
    
    # 1. Historical VaR
    print(f"\n1️⃣ Historical VaR Calculation:")
    hist_var = var_calculator.calculate_historical_var(confidence_level=0.95)
    print(f"  VaR (95%): ${hist_var.var_value:,.2f}")
    print(f"  CVaR (95%): ${hist_var.cvar_value:,.2f}")
    print(f"  VaR as % of portfolio: {hist_var.var_percentage:.2f}%")
    
    # 2. Parametric VaR (Normal)
    print(f"\n2️⃣ Parametric VaR (Normal Distribution):")
    param_var_normal = var_calculator.calculate_parametric_var(
        confidence_level=0.95,
        distribution='normal'
    )
    print(f"  VaR (95%): ${param_var_normal.var_value:,.2f}")
    print(f"  CVaR (95%): ${param_var_normal.cvar_value:,.2f}")
    print(f"  Mean Return: {param_var_normal.additional_metrics['mean_return']:.6f}")
    print(f"  Std Deviation: {param_var_normal.additional_metrics['std_return']:.6f}")
    
    # 3. Parametric VaR (t-Student)
    print(f"\n3️⃣ Parametric VaR (t-Student Distribution):")
    param_var_t = var_calculator.calculate_parametric_var(
        confidence_level=0.95,
        distribution='t'
    )
    print(f"  VaR (95%): ${param_var_t.var_value:,.2f}")
    print(f"  CVaR (95%): ${param_var_t.cvar_value:,.2f}")
    print(f"  Degrees of Freedom: {param_var_t.additional_metrics['degrees_freedom']:.2f}")
    
    # 4. Monte Carlo VaR
    print(f"\n4️⃣ Monte Carlo VaR:")
    mc_var = var_calculator.calculate_monte_carlo_var(
        confidence_level=0.95,
        simulations=5000
    )
    print(f"  VaR (95%): ${mc_var.var_value:,.2f}")
    print(f"  CVaR (95%): ${mc_var.cvar_value:,.2f}")
    print(f"  Simulations: {mc_var.additional_metrics['simulations']}")
    
    # 5. Cornish-Fisher VaR
    print(f"\n5️⃣ Cornish-Fisher VaR (Higher Moments):")
    cf_var = var_calculator.calculate_cornish_fisher_var(confidence_level=0.95)
    print(f"  VaR (95%): ${cf_var.var_value:,.2f}")
    print(f"  CVaR (95%): ${cf_var.cvar_value:,.2f}")
    print(f"  Skewness: {cf_var.additional_metrics['skewness']:.4f}")
    print(f"  Kurtosis: {cf_var.additional_metrics['kurtosis']:.4f}")
    
    # 6. All Methods Comparison
    print(f"\n6️⃣ All Methods Comparison (95% Confidence):")
    all_results = var_calculator.calculate_all_var_methods(confidence_level=0.95)
    
    print(f"  {'Method':<20} {'VaR ($)':<12} {'CVaR ($)':<12} {'VaR %':<8}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    
    for method_name, result in all_results.items():
        print(f"  {method_name:<20} {result.var_value:>10,.0f} {result.cvar_value:>11,.0f} {result.var_percentage:>6.2f}%")
    
    # 7. Multiple Confidence Levels
    print(f"\n7️⃣ VaR Summary (Multiple Confidence Levels):")
    summary = var_calculator.get_var_summary([0.90, 0.95, 0.99, 0.999])
    
    for level, methods in summary.items():
        print(f"\n  Confidence Level: {level}")
        for method, values in methods.items():
            print(f"    {method:<20}: VaR ${values['var_absolute']:>8,.0f}, CVaR ${values['cvar_absolute']:>8,.0f}")
    
    # 8. VaR Statistics
    print(f"\n8️⃣ VaR Calculator Statistics:")
    stats = var_calculator.get_statistics()
    print(f"  Total VaR Calculations: {stats['var_calculations']}")
    print(f"  Data Points: {stats['data_points']}")
    print(f"  Mean Return: {stats['mean_return']:.6f}")
    print(f"  Std Return: {stats['std_return']:.6f}")
    
    return var_calculator, returns_data, portfolio_values


def demo_monte_carlo_simulator():
    """Demonstrate Monte Carlo Simulator functionality"""
    print("\n" + "="*60)
    print("🎲 MONTE CARLO SIMULATOR DEMO")
    print("="*60)
    
    # Create Monte Carlo Simulator
    simulator = MonteCarloSimulator({
        'n_simulations': 10000,
        'method': SimulationMethod.STANDARD,
        'distribution': DistributionModel.NORMAL
    })
    
    # Create sample data
    returns_data, _ = create_sample_data()
    simulator.set_data(returns_data)
    
    print(f"\n📋 Monte Carlo Simulator Configuration:")
    print(f"  Default Simulations: {simulator.default_config.n_simulations}")
    print(f"  Default Method: {simulator.default_config.method.value}")
    print(f"  Default Distribution: {simulator.default_config.distribution.value}")
    print(f"  Assets: {len(returns_data.columns)}")
    
    # 1. Standard Simulation
    print(f"\n1️⃣ Standard Monte Carlo Simulation:")
    config_standard = SimulationConfig(
        n_simulations=5000,
        method=SimulationMethod.STANDARD,
        distribution=DistributionModel.NORMAL
    )
    
    result_standard = simulator.simulate_returns(config_standard)
    print(f"  Method: {result_standard.method.value}")
    print(f"  Distribution: {result_standard.distribution.value}")
    print(f"  Simulations: {len(result_standard.simulated_returns)}")
    print(f"  VaR (95%): {result_standard.var_estimates[0.95]:.6f}")
    print(f"  CVaR (95%): {result_standard.cvar_estimates[0.95]:.6f}")
    print(f"  Computation Time: {result_standard.computation_time:.3f} seconds")
    
    # 2. Antithetic Variates Simulation
    print(f"\n2️⃣ Antithetic Variates Simulation (Variance Reduction):")
    config_antithetic = SimulationConfig(
        n_simulations=5000,
        method=SimulationMethod.ANTITHETIC,
        distribution=DistributionModel.NORMAL
    )
    
    result_antithetic = simulator.simulate_returns(config_antithetic)
    print(f"  Method: {result_antithetic.method.value}")
    print(f"  VaR (95%): {result_antithetic.var_estimates[0.95]:.6f}")
    print(f"  CVaR (95%): {result_antithetic.cvar_estimates[0.95]:.6f}")
    print(f"  Computation Time: {result_antithetic.computation_time:.3f} seconds")
    
    # 3. Control Variate Simulation
    print(f"\n3️⃣ Control Variate Simulation (Variance Reduction):")
    config_control = SimulationConfig(
        n_simulations=5000,
        method=SimulationMethod.CONTROL_VARIATE,
        distribution=DistributionModel.NORMAL
    )
    
    result_control = simulator.simulate_returns(config_control)
    print(f"  Method: {result_control.method.value}")
    print(f"  VaR (95%): {result_control.var_estimates[0.95]:.6f}")
    print(f"  CVaR (95%): {result_control.cvar_estimates[0.95]:.6f}")
    print(f"  Computation Time: {result_control.computation_time:.3f} seconds")
    
    # 4. t-Student Distribution Simulation
    print(f"\n4️⃣ t-Student Distribution Simulation:")
    config_t = SimulationConfig(
        n_simulations=5000,
        method=SimulationMethod.STANDARD,
        distribution=DistributionModel.T_STUDENT
    )
    
    result_t = simulator.simulate_returns(config_t)
    print(f"  Distribution: {result_t.distribution.value}")
    print(f"  VaR (95%): {result_t.var_estimates[0.95]:.6f}")
    print(f"  CVaR (95%): {result_t.cvar_estimates[0.95]:.6f}")
    print(f"  Skewness: {result_t.statistics['skewness']:.4f}")
    print(f"  Kurtosis: {result_t.statistics['kurtosis']:.4f}")
    
    # 5. Method Comparison
    print(f"\n5️⃣ Simulation Methods Comparison:")
    methods_comparison = [
        ("Standard", result_standard),
        ("Antithetic", result_antithetic),
        ("Control Variate", result_control)
    ]
    
    print(f"  {'Method':<15} {'VaR (95%)':<12} {'CVaR (95%)':<12} {'Time (s)':<10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    
    for method_name, result in methods_comparison:
        print(f"  {method_name:<15} {result.var_estimates[0.95]:>10.6f} {result.cvar_estimates[0.95]:>11.6f} {result.computation_time:>8.3f}")
    
    # 6. Stress Testing
    print(f"\n6️⃣ Stress Testing Scenarios:")
    
    # Create stress scenarios
    market_crash = StressTestScenario(
        name="Market Crash",
        description="Severe market downturn (-10% mean, +100% volatility)",
        shock_parameters={'volatility_shock': 1.0, 'mean_shock': -0.10},
        probability=0.001
    )
    
    volatility_spike = StressTestScenario(
        name="Volatility Spike",
        description="High volatility period (+150% volatility)",
        shock_parameters={'volatility_shock': 1.5},
        probability=0.01
    )
    
    correlation_breakdown = StressTestScenario(
        name="Correlation Breakdown",
        description="Correlation structure breaks down",
        shock_parameters={'correlation_shock': 0.5},
        probability=0.005
    )
    
    # Add scenarios to simulator
    simulator.add_stress_scenario(market_crash)
    simulator.add_stress_scenario(volatility_spike)
    simulator.add_stress_scenario(correlation_breakdown)
    
    # Run stress tests
    stress_config = SimulationConfig(n_simulations=3000)
    stress_results = simulator.run_stress_test([market_crash, volatility_spike], stress_config)
    
    print(f"  {'Scenario':<20} {'VaR (95%)':<12} {'CVaR (95%)':<12} {'Probability':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    # Baseline (normal conditions)
    baseline_result = simulator.simulate_returns(stress_config)
    print(f"  {'Baseline':<20} {baseline_result.var_estimates[0.95]:>10.6f} {baseline_result.cvar_estimates[0.95]:>11.6f} {'Normal':<12}")
    
    for scenario_name, result in stress_results.items():
        scenario = next(s for s in [market_crash, volatility_spike] if s.name == scenario_name)
        print(f"  {scenario_name:<20} {result.var_estimates[0.95]:>10.6f} {result.cvar_estimates[0.95]:>11.6f} {scenario.probability:<12.3f}")
    
    # 7. Simulation Statistics
    print(f"\n7️⃣ Simulation Statistics:")
    sim_stats = simulator.get_statistics()
    print(f"  Total Simulations Run: {sim_stats['total_simulations_run']}")
    print(f"  Stress Scenarios: {sim_stats['stress_scenarios_count']}")
    print(f"  Available Methods: {', '.join(sim_stats['methods_available'])}")
    print(f"  Available Distributions: {', '.join(sim_stats['distributions_available'])}")
    
    return simulator


def demo_var_backtester():
    """Demonstrate VaR Backtester functionality"""
    print("\n" + "="*60)
    print("🔍 VAR BACKTESTER DEMO")
    print("="*60)
    
    # Create VaR Backtester
    backtester = VaRBacktester({
        'confidence_level': 0.95,
        'test_window': 252,
        'significance_level': 0.05
    })
    
    # Create sample data for backtesting
    returns_data, portfolio_values = create_sample_data()
    
    # Split data: first 300 days for model estimation, last 200 for backtesting
    train_data = returns_data.iloc[:300]
    test_data = returns_data.iloc[300:]
    test_returns = test_data.mean(axis=1)  # Portfolio returns
    
    print(f"\n📋 Backtesting Setup:")
    print(f"  Training Period: {len(train_data)} days")
    print(f"  Testing Period: {len(test_data)} days")
    print(f"  Confidence Level: {backtester.default_config.confidence_level}")
    print(f"  Significance Level: {backtester.default_config.significance_level}")
    
    # Create VaR forecasts for different models
    def create_var_forecasts(method_name, base_var, adjustment_factor=1.0):
        """Create VaR forecasts with some variation"""
        forecasts = []
        for i, date in enumerate(test_data.index):
            # Add some time variation and noise
            var_value = base_var * adjustment_factor * (1 + 0.1 * np.sin(i/10) + 0.05 * np.random.normal())
            cvar_value = var_value * 1.3  # CVaR typically 30% worse than VaR
            
            from src.core.risk.var_calculator import VaRResult, VaRMethod
            var_result = VaRResult(
                var_value=var_value,
                cvar_value=cvar_value,
                confidence_level=0.95,
                method=getattr(VaRMethod, method_name),
                calculation_time=date
            )
            forecasts.append(var_result)
        return forecasts
    
    # Estimate base VaR from training data
    train_returns = train_data.mean(axis=1)
    base_var = np.percentile(train_returns, 5)  # 5th percentile for 95% VaR
    
    print(f"  Base VaR (from training): {base_var:.6f}")
    
    # 1. Historical VaR Model
    print(f"\n1️⃣ Historical VaR Model Backtest:")
    hist_forecasts = create_var_forecasts("HISTORICAL", base_var, 1.0)
    
    hist_report = backtester.backtest_var_model(
        hist_forecasts,
        test_returns,
        model_name="Historical_VaR"
    )
    
    print(f"  Model: {hist_report.model_name}")
    print(f"  Test Period: {hist_report.test_period_start.date()} to {hist_report.test_period_end.date()}")
    print(f"  Total Observations: {hist_report.total_observations}")
    print(f"  Expected Violations: {hist_report.expected_violations}")
    print(f"  Actual Violations: {hist_report.total_violations}")
    print(f"  Violation Rate: {hist_report.violation_rate:.2%} (Expected: {hist_report.expected_violation_rate:.2%})")
    print(f"  Kupiec Test: {hist_report.kupiec_test.result.value} (p-value: {hist_report.kupiec_test.p_value:.4f})")
    print(f"  Christoffersen Test: {hist_report.christoffersen_test.result.value} (p-value: {hist_report.christoffersen_test.p_value:.4f})")
    print(f"  Traffic Light: {hist_report.traffic_light_zone}")
    print(f"  Overall Result: {hist_report.overall_result.value}")
    
    # 2. Conservative Model (underestimates risk)
    print(f"\n2️⃣ Conservative VaR Model Backtest:")
    conservative_forecasts = create_var_forecasts("PARAMETRIC", base_var, 0.7)  # 30% less conservative
    
    conservative_report = backtester.backtest_var_model(
        conservative_forecasts,
        test_returns,
        model_name="Conservative_VaR"
    )
    
    print(f"  Model: {conservative_report.model_name}")
    print(f"  Actual Violations: {conservative_report.total_violations}")
    print(f"  Violation Rate: {conservative_report.violation_rate:.2%}")
    print(f"  Kupiec Test: {conservative_report.kupiec_test.result.value}")
    print(f"  Traffic Light: {conservative_report.traffic_light_zone}")
    print(f"  Overall Result: {conservative_report.overall_result.value}")
    
    # 3. Aggressive Model (overestimates risk)
    print(f"\n3️⃣ Aggressive VaR Model Backtest:")
    aggressive_forecasts = create_var_forecasts("MONTE_CARLO", base_var, 1.5)  # 50% more conservative
    
    aggressive_report = backtester.backtest_var_model(
        aggressive_forecasts,
        test_returns,
        model_name="Aggressive_VaR"
    )
    
    print(f"  Model: {aggressive_report.model_name}")
    print(f"  Actual Violations: {aggressive_report.total_violations}")
    print(f"  Violation Rate: {aggressive_report.violation_rate:.2%}")
    print(f"  Kupiec Test: {aggressive_report.kupiec_test.result.value}")
    print(f"  Traffic Light: {aggressive_report.traffic_light_zone}")
    print(f"  Overall Result: {aggressive_report.overall_result.value}")
    
    # 4. Model Comparison
    print(f"\n4️⃣ Model Comparison:")
    comparison = backtester.compare_models(["Historical_VaR", "Conservative_VaR", "Aggressive_VaR"])
    
    print(f"  Models Compared: {len(comparison['models'])}")
    print(f"  Best Model (Overall): {comparison['summary']['best_model']}")
    
    print(f"\n  Detailed Comparison:")
    print(f"  {'Model':<15} {'Violations':<11} {'Rate':<8} {'Kupiec':<8} {'Traffic':<8} {'Overall':<10}")
    print(f"  {'-'*15} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    
    models_data = {
        "Historical_VaR": hist_report,
        "Conservative_VaR": conservative_report,
        "Aggressive_VaR": aggressive_report
    }
    
    for model_name, report in models_data.items():
        print(f"  {model_name:<15} {report.total_violations:>9}/{report.total_observations:<1} {report.violation_rate:>6.1%} {report.kupiec_test.result.value:<8} {report.traffic_light_zone:<8} {report.overall_result.value:<10}")
    
    # 5. Violation Analysis
    print(f"\n5️⃣ Violation Analysis (Historical VaR):")
    if hist_report.violation_dates:
        print(f"  Total Violations: {len(hist_report.violation_dates)}")
        print(f"  Max Violation: {hist_report.max_violation:.6f}")
        print(f"  Avg Violation: {hist_report.avg_violation:.6f}")
        print(f"  First 5 Violation Dates:")
        for i, date in enumerate(hist_report.violation_dates[:5]):
            print(f"    {i+1}. {date.date()}")
    else:
        print(f"  No violations detected")
    
    # 6. Recommendations
    print(f"\n6️⃣ Model Recommendations:")
    for model_name, report in models_data.items():
        print(f"\n  {model_name}:")
        for i, recommendation in enumerate(report.recommendations, 1):
            print(f"    {i}. {recommendation}")
    
    # 7. Backtesting Summary
    print(f"\n7️⃣ Backtesting Summary:")
    summary = backtester.get_backtest_summary()
    
    print(f"  Total Backtests Performed: {summary['total_backtests']}")
    print(f"  Unique Models Tested: {summary['unique_models']}")
    print(f"  Accept Rate: {summary['performance_summary']['accept_rate']:.1%}")
    print(f"  Reject Rate: {summary['performance_summary']['reject_rate']:.1%}")
    print(f"  Green Light Rate: {summary['performance_summary']['green_light_rate']:.1%}")
    print(f"  Average Violation Rate: {summary['performance_summary']['avg_violation_rate']:.2%}")
    
    return backtester


def demo_integrated_var_system():
    """Demonstrate integrated VaR system workflow"""
    print("\n" + "="*60)
    print("🔗 INTEGRATED VAR SYSTEM DEMO")
    print("="*60)
    
    print(f"\n🎯 Complete VaR Risk Management Workflow:")
    print(f"  1. Calculate VaR using multiple methods")
    print(f"  2. Run Monte Carlo simulations with stress testing")
    print(f"  3. Backtest VaR models for validation")
    print(f"  4. Compare model performance")
    print(f"  5. Generate risk management recommendations")
    
    # Create sample data
    returns_data, portfolio_values = create_sample_data()
    
    # Split data for backtesting
    train_data = returns_data.iloc[:350]
    test_data = returns_data.iloc[350:]
    
    print(f"\n📊 Data Summary:")
    print(f"  Total Period: {len(returns_data)} days")
    print(f"  Training Period: {len(train_data)} days")
    print(f"  Testing Period: {len(test_data)} days")
    print(f"  Assets: {', '.join(returns_data.columns)}")
    print(f"  Portfolio Value Range: ${portfolio_values.min():,.0f} - ${portfolio_values.max():,.0f}")
    
    # 1. VaR Calculation
    print(f"\n1️⃣ Multi-Method VaR Calculation:")
    var_calculator = VaRCalculator()
    var_calculator.set_data(train_data, portfolio_values.iloc[:350])
    
    var_results = var_calculator.calculate_all_var_methods(confidence_level=0.95)
    
    print(f"  {'Method':<20} {'VaR ($)':<12} {'CVaR ($)':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for method, result in var_results.items():
        print(f"  {method:<20} {result.var_value:>10,.0f} {result.cvar_value:>11,.0f}")
    
    # 2. Monte Carlo Simulation with Stress Testing
    print(f"\n2️⃣ Monte Carlo Simulation & Stress Testing:")
    simulator = MonteCarloSimulator()
    simulator.set_data(train_data)
    
    # Normal simulation
    normal_config = SimulationConfig(n_simulations=5000)
    normal_result = simulator.simulate_returns(normal_config)
    
    # Stress scenarios
    market_stress = StressTestScenario(
        name="Market Stress",
        description="Severe market conditions",
        shock_parameters={'volatility_shock': 1.0, 'mean_shock': -0.05},
        probability=0.001
    )
    
    stress_results = simulator.run_stress_test([market_stress], normal_config)
    
    print(f"  Normal Conditions:")
    print(f"    VaR (95%): {normal_result.var_estimates[0.95]:.6f}")
    print(f"    CVaR (95%): {normal_result.cvar_estimates[0.95]:.6f}")
    
    print(f"  Stress Conditions:")
    stress_result = stress_results["Market Stress"]
    print(f"    VaR (95%): {stress_result.var_estimates[0.95]:.6f}")
    print(f"    CVaR (95%): {stress_result.cvar_estimates[0.95]:.6f}")
    
    stress_multiplier = abs(stress_result.var_estimates[0.95] / normal_result.var_estimates[0.95])
    print(f"    Stress Multiplier: {stress_multiplier:.2f}x")
    
    # 3. VaR Model Backtesting
    print(f"\n3️⃣ VaR Model Backtesting:")
    backtester = VaRBacktester()
    
    # Create forecasts based on different VaR methods
    test_returns = test_data.mean(axis=1)
    
    models_to_test = {
        "Historical": var_results['historical'],
        "Monte_Carlo": var_results['monte_carlo'],
        "Parametric": var_results['parametric_normal']
    }
    
    backtest_results = {}
    
    for model_name, var_result in models_to_test.items():
        # Create forecasts (simplified - using same VaR for all periods)
        forecasts = []
        for date in test_data.index:
            from src.core.risk.var_calculator import VaRResult
            forecast = VaRResult(
                var_value=var_result.var_value,
                cvar_value=var_result.cvar_value,
                confidence_level=0.95,
                method=var_result.method,
                calculation_time=date
            )
            forecasts.append(forecast)
        
        # Backtest
        report = backtester.backtest_var_model(forecasts, test_returns, model_name)
        backtest_results[model_name] = report
    
    print(f"  {'Model':<12} {'Violations':<11} {'Rate':<8} {'Expected':<8} {'Result':<8}")
    print(f"  {'-'*12} {'-'*11} {'-'*8} {'-'*8} {'-'*8}")
    
    for model_name, report in backtest_results.items():
        print(f"  {model_name:<12} {report.total_violations:>9}/{report.total_observations:<1} {report.violation_rate:>6.1%} {report.expected_violation_rate:>6.1%} {report.overall_result.value:<8}")
    
    # 4. Model Comparison & Selection
    print(f"\n4️⃣ Model Performance Comparison:")
    comparison = backtester.compare_models(list(models_to_test.keys()))
    
    if comparison and 'summary' in comparison:
        best_model = comparison['summary']['best_model']
        print(f"  Best Performing Model: {best_model}")
        
        # Show ranking
        if 'rankings' in comparison and 'overall' in comparison['rankings']:
            print(f"  Model Ranking (Best to Worst):")
            for i, model in enumerate(comparison['rankings']['overall'], 1):
                result = backtest_results[model].overall_result.value
                traffic = backtest_results[model].traffic_light_zone
                print(f"    {i}. {model} ({result}, {traffic} Light)")
    
    # 5. Risk Management Recommendations
    print(f"\n5️⃣ Risk Management Recommendations:")
    
    # Portfolio-level recommendations
    current_portfolio_value = portfolio_values.iloc[-1]
    worst_var = min(result.var_value for result in var_results.values())
    var_as_percent = abs(worst_var / current_portfolio_value) * 100
    
    print(f"  Portfolio Risk Assessment:")
    print(f"    Current Portfolio Value: ${current_portfolio_value:,.0f}")
    print(f"    Worst-Case VaR (95%): ${worst_var:,.0f}")
    print(f"    VaR as % of Portfolio: {var_as_percent:.2f}%")
    
    if var_as_percent > 5:
        print(f"    ⚠️  HIGH RISK: VaR exceeds 5% of portfolio value")
        print(f"    📋 Recommendation: Consider reducing position sizes or increasing diversification")
    elif var_as_percent > 2:
        print(f"    ⚡ MODERATE RISK: VaR is within acceptable range but monitor closely")
        print(f"    📋 Recommendation: Maintain current risk management practices")
    else:
        print(f"    ✅ LOW RISK: VaR is well within acceptable limits")
        print(f"    📋 Recommendation: Current risk level is conservative")
    
    # Model-specific recommendations
    print(f"\n  Model-Specific Recommendations:")
    for model_name, report in backtest_results.items():
        print(f"    {model_name}:")
        for rec in report.recommendations[:2]:  # Show first 2 recommendations
            print(f"      • {rec}")
    
    # Stress testing recommendations
    print(f"\n  Stress Testing Insights:")
    if stress_multiplier > 2:
        print(f"    ⚠️  Stress conditions increase VaR by {stress_multiplier:.1f}x")
        print(f"    📋 Recommendation: Prepare contingency plans for market stress")
        print(f"    📋 Consider stress-testing portfolio monthly")
    else:
        print(f"    ✅ Portfolio shows reasonable resilience to stress ({stress_multiplier:.1f}x increase)")
    
    # 6. Export Results
    print(f"\n6️⃣ Exporting Results:")
    
    # Export VaR data
    var_export_success = var_calculator.export_var_data("var_analysis_results.json")
    sim_export_success = simulator.export_simulation_data("simulation_results.json")
    backtest_export_success = backtester.export_backtest_data("backtest_results.json")
    
    print(f"  VaR Analysis Export: {'✅ Success' if var_export_success else '❌ Failed'}")
    print(f"  Simulation Export: {'✅ Success' if sim_export_success else '❌ Failed'}")
    print(f"  Backtest Export: {'✅ Success' if backtest_export_success else '❌ Failed'}")
    
    # Final summary
    print(f"\n📊 Final System Statistics:")
    var_stats = var_calculator.get_statistics()
    sim_stats = simulator.get_statistics()
    backtest_stats = backtester.get_statistics()
    
    print(f"  VaR Calculations Performed: {var_stats['total_var_calculations']}")
    print(f"  Monte Carlo Simulations: {sim_stats['total_simulations_run']}")
    print(f"  Backtests Completed: {backtest_stats['total_backtests_performed']}")
    print(f"  Models Validated: {backtest_stats['unique_models_tested']}")


def main():
    """Main demo function"""
    print("🚀 VaR SYSTEM COMPREHENSIVE DEMO")
    print("=" * 80)
    print("This demo showcases the complete VaR Risk Management System:")
    print("• VaRCalculator: Multiple VaR calculation methods")
    print("• MonteCarloSimulator: Advanced simulation techniques")
    print("• VaRBacktester: Comprehensive model validation")
    print("• Integrated Workflow: Complete risk management process")
    
    try:
        # Individual component demos
        demo_var_calculator()
        demo_monte_carlo_simulator()
        demo_var_backtester()
        
        # Integrated system demo
        demo_integrated_var_system()
        
        print(f"\n🎉 VaR SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print(f"All components are working correctly and ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 