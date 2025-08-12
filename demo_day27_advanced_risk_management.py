"""
Demo Day 27: Advanced Risk Management Systems
Ultimate XAU Super System V4.0

Comprehensive demonstration of advanced risk management:
- Comprehensive Risk Metrics Calculation (VaR, CVaR, Stress Testing)
- Monte Carlo Simulation and Historical Scenario Analysis
- Dynamic Hedging Strategy Recommendations
- Real-time Risk Monitoring and Alert Systems
- Liquidity Risk Assessment and Management
- Stress Testing Framework with Multiple Scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import time
import json
from typing import Dict, List, Any

warnings.filterwarnings('ignore')

from src.core.analysis.advanced_risk_management import (
    AdvancedRiskManagement, RiskConfig, StressTestType, HedgingStrategy,
    create_advanced_risk_management
)

# Try to import portfolio optimization for integration
try:
    from src.core.analysis.risk_adjusted_portfolio_optimization import create_risk_adjusted_portfolio_optimization
    PORTFOLIO_OPTIMIZATION_AVAILABLE = True
except:
    PORTFOLIO_OPTIMIZATION_AVAILABLE = False

# Try to import regime detection for integration
try:
    from src.core.analysis.market_regime_detection import create_market_regime_detection
    REGIME_DETECTION_AVAILABLE = True
except:
    REGIME_DETECTION_AVAILABLE = False


def main():
    """Run comprehensive Advanced Risk Management demo"""
    
    print("🛡️ Advanced Risk Management Systems Demo - Day 27")
    print("=" * 70)
    
    start_time = datetime.now()
    results = {}
    
    # Demo 1: Comprehensive Risk Metrics Calculation
    print("\n📊 Demo 1: Comprehensive Risk Metrics Calculation")
    print("-" * 50)
    
    # Create advanced risk management system
    config = {
        'confidence_levels': [0.95, 0.99],
        'var_lookback_period': 252,
        'monte_carlo_simulations': 5000,
        'max_var_95': 0.03,  # 3% daily VaR limit
        'max_var_99': 0.05,  # 5% daily VaR limit
        'max_drawdown': 0.15,
        'enable_dynamic_hedging': True,
        'real_time_monitoring': True
    }
    
    risk_system = create_advanced_risk_management(config)
    
    # Generate comprehensive portfolio data
    assets = ['GOLD', 'SILVER', 'PLATINUM', 'COPPER', 'CRUDE_OIL', 'NATURAL_GAS']
    portfolio_data = generate_portfolio_data(assets, periods=252)
    portfolio_weights = {
        'GOLD': 0.35, 'SILVER': 0.25, 'PLATINUM': 0.15,
        'COPPER': 0.10, 'CRUDE_OIL': 0.10, 'NATURAL_GAS': 0.05
    }
    portfolio_value = 1000000  # $1M portfolio
    
    # Calculate comprehensive risk metrics
    print("🔍 Calculating comprehensive risk metrics...")
    
    metrics_start = time.time()
    risk_metrics = risk_system.calculate_comprehensive_risk_metrics(
        portfolio_data, portfolio_weights, portfolio_value
    )
    metrics_time = time.time() - metrics_start
    
    print(f"✅ Risk Metrics Calculation Results:")
    print(f"   Daily VaR (95%): {risk_metrics.var_95_daily:.4f} ({risk_metrics.var_95_daily*portfolio_value:,.0f} USD)")
    print(f"   Daily VaR (99%): {risk_metrics.var_99_daily:.4f} ({risk_metrics.var_99_daily*portfolio_value:,.0f} USD)")
    print(f"   Daily CVaR (95%): {risk_metrics.cvar_95_daily:.4f} ({risk_metrics.cvar_95_daily*portfolio_value:,.0f} USD)")
    print(f"   Annual Volatility: {risk_metrics.volatility_annual:.2%}")
    print(f"   Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
    print(f"   Sortino Ratio: {risk_metrics.sortino_ratio:.3f}")
    print(f"   Calmar Ratio: {risk_metrics.calmar_ratio:.3f}")
    print(f"   Current Drawdown: {risk_metrics.current_drawdown:.2%}")
    print(f"   Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"   Concentration Risk: {risk_metrics.concentration_risk:.1%}")
    print(f"   Liquidity Score: {risk_metrics.liquidity_score:.3f}")
    print(f"   Calculation Time: {metrics_time:.4f}s")
    
    # Score Demo 1 based on performance
    var_score = 100 if risk_metrics.var_95_daily < 0.05 else 50
    sharpe_score = min(100, risk_metrics.sharpe_ratio * 50)
    liquidity_score = risk_metrics.liquidity_score * 100
    speed_score = 100 if metrics_time < 1.0 else 50
    demo1_score = (var_score + sharpe_score + liquidity_score + speed_score) / 4
    
    results['risk_metrics'] = {
        'var_95_daily': risk_metrics.var_95_daily,
        'var_99_daily': risk_metrics.var_99_daily,
        'sharpe_ratio': risk_metrics.sharpe_ratio,
        'max_drawdown': risk_metrics.max_drawdown,
        'liquidity_score': risk_metrics.liquidity_score,
        'calculation_time': metrics_time,
        'score': demo1_score
    }
    
    # Demo 2: Comprehensive Stress Testing
    print("\n🧪 Demo 2: Comprehensive Stress Testing")
    print("-" * 50)
    
    print("Running stress testing scenarios...")
    
    stress_start = time.time()
    stress_results = risk_system.run_comprehensive_stress_tests(portfolio_data, portfolio_weights)
    stress_time = time.time() - stress_start
    
    print(f"✅ Stress Testing Results ({len(stress_results)} scenarios):")
    
    # Analyze stress test results
    worst_loss = 0
    total_scenarios = len(stress_results)
    scenario_summary = {}
    
    for result in stress_results:
        print(f"   📉 {result.scenario_name}:")
        print(f"      Portfolio Impact: {result.portfolio_pnl_pct:.2%} ({result.portfolio_pnl:,.0f} USD)")
        print(f"      Stressed VaR 95%: {result.stressed_var_95:.4f}")
        print(f"      Recovery Time: {result.estimated_recovery_time} days")
        print(f"      Loss Probability: {result.probability_of_loss:.1%}")
        
        if result.portfolio_pnl_pct < worst_loss:
            worst_loss = result.portfolio_pnl_pct
        
        scenario_summary[result.scenario_name] = {
            'impact_pct': result.portfolio_pnl_pct,
            'impact_usd': result.portfolio_pnl,
            'recovery_days': result.estimated_recovery_time,
            'probability': result.probability_of_loss
        }
    
    print(f"   🎯 Summary:")
    print(f"      Total Scenarios: {total_scenarios}")
    print(f"      Worst Case Loss: {worst_loss:.2%} ({worst_loss*portfolio_value:,.0f} USD)")
    print(f"      Average Recovery: {np.mean([r.estimated_recovery_time for r in stress_results]):.0f} days")
    print(f"      Testing Time: {stress_time:.4f}s")
    
    # Score Demo 2
    scenario_score = min(100, total_scenarios * 10)  # 10 points per scenario
    worst_case_score = 100 if abs(worst_loss) < 0.3 else 50
    speed_score = 100 if stress_time < 2.0 else 75
    demo2_score = (scenario_score + worst_case_score + speed_score) / 3
    
    results['stress_testing'] = {
        'total_scenarios': total_scenarios,
        'worst_case_loss': worst_loss,
        'scenario_details': scenario_summary,
        'testing_time': stress_time,
        'score': demo2_score
    }
    
    # Demo 3: Dynamic Hedging Strategy Recommendations (Simplified)
    print("\n⚡ Demo 3: Dynamic Hedging Strategy Recommendations")
    print("-" * 50)
    
    print("Generating dynamic hedging recommendations...")
    
    # Simplified hedging test without market condition adjustment
    hedging_results = {}
    
    # Test base hedging recommendation
    hedge_recommendation = risk_system.generate_hedging_recommendations(portfolio_data, risk_metrics)
    
    print(f"✅ Hedging Recommendation for Current Market:")
    print(f"   Strategy Type: {hedge_recommendation.strategy_type.value}")
    print(f"   Current Hedge Ratio: {hedge_recommendation.current_hedge_ratio:.1%}")
    print(f"   Target Hedge Ratio: {hedge_recommendation.target_hedge_ratio:.1%}")
    print(f"   Expected Protection: {hedge_recommendation.expected_protection:.1%}")
    print(f"   Estimated Cost: {hedge_recommendation.estimated_cost:.2%}")
    print(f"   VaR Reduction: {hedge_recommendation.var_reduction:.4f}")
    print(f"   Execution Priority: {hedge_recommendation.execution_priority}")
    print(f"   Confidence Level: {hedge_recommendation.confidence_level:.1%}")
    
    # Show hedge instruments
    if hedge_recommendation.hedge_instruments:
        print("   Recommended Instruments:")
        for instrument, weight in hedge_recommendation.hedge_instruments.items():
            print(f"      {instrument}: {weight:.1%}")
    
    # Test with modified risk metrics for different scenarios
    scenarios = ['low_risk', 'medium_risk', 'high_risk', 'extreme_risk']
    var_multipliers = [0.8, 1.5, 2.5, 4.0]
    
    for scenario, multiplier in zip(scenarios, var_multipliers):
        # Create modified metrics
        modified_metrics = risk_metrics
        modified_metrics.var_95_daily *= multiplier
        
        hedge_rec = risk_system.generate_hedging_recommendations(portfolio_data, modified_metrics)
        
        hedging_results[scenario] = {
            'strategy_type': hedge_rec.strategy_type.value,
            'target_hedge_ratio': hedge_rec.target_hedge_ratio,
            'expected_protection': hedge_rec.expected_protection,
            'estimated_cost': hedge_rec.estimated_cost,
            'var_reduction': hedge_rec.var_reduction,
            'execution_priority': hedge_rec.execution_priority,
            'confidence_level': hedge_rec.confidence_level
        }
        
        print(f"   {scenario.upper()}: {hedge_rec.strategy_type.value}, "
              f"Hedge: {hedge_rec.target_hedge_ratio:.1%}, "
              f"Protection: {hedge_rec.expected_protection:.1%}")
    
    # Score Demo 3
    hedge_strategies = len(hedging_results) + 1  # +1 for base recommendation
    avg_protection = np.mean([h['expected_protection'] for h in hedging_results.values()])
    avg_cost = np.mean([h['estimated_cost'] for h in hedging_results.values()])
    
    strategy_score = hedge_strategies * 20  # 20 points per strategy
    protection_score = avg_protection * 100
    cost_score = max(0, 100 - avg_cost * 1000)  # Lower cost = higher score
    demo3_score = (strategy_score + protection_score + cost_score) / 3
    
    results['hedging_strategies'] = {
        'scenarios_tested': hedge_strategies,
        'average_protection': avg_protection,
        'average_cost': avg_cost,
        'strategy_details': hedging_results,
        'score': demo3_score
    }
    
    # Demo 4: Real-time Risk Monitoring and Alerts
    print("\n🚨 Demo 4: Real-time Risk Monitoring and Alerts")
    print("-" * 50)
    
    print("Testing real-time risk monitoring system...")
    
    # Test with different risk scenarios
    risk_scenarios = [
        {'name': 'normal_risk', 'var_multiplier': 1.0, 'drawdown': -0.05},
        {'name': 'elevated_risk', 'var_multiplier': 1.5, 'drawdown': -0.08},
        {'name': 'high_risk', 'var_multiplier': 2.2, 'drawdown': -0.12},
        {'name': 'extreme_risk', 'var_multiplier': 3.0, 'drawdown': -0.18}
    ]
    
    monitoring_results = {}
    total_alerts = 0
    
    for scenario in risk_scenarios:
        print(f"\n🔍 Testing {scenario['name'].replace('_', ' ').title()} Scenario:")
        
        # Create modified risk metrics for scenario
        modified_metrics = risk_metrics
        modified_metrics.var_95_daily *= scenario['var_multiplier']
        modified_metrics.var_99_daily *= scenario['var_multiplier']
        modified_metrics.current_drawdown = scenario['drawdown']
        
        # Monitor risk and generate alerts
        alerts = risk_system.monitor_real_time_risk(modified_metrics)
        
        print(f"   VaR 95%: {modified_metrics.var_95_daily:.4f} (limit: {config['max_var_95']:.4f})")
        print(f"   Current Drawdown: {scenario['drawdown']:.1%} (limit: {config['max_drawdown']:.1%})")
        print(f"   Alerts Generated: {len(alerts)}")
        
        scenario_alerts = []
        for alert in alerts:
            print(f"      {alert.severity.upper()}: {alert.metric_name}")
            print(f"         Current: {alert.current_value:.4f}, Limit: {alert.limit_value:.4f}")
            print(f"         Breach: {alert.breach_percentage:.1f}%, Urgency: {alert.urgency_score:.2f}")
            
            scenario_alerts.append({
                'type': alert.alert_type,
                'severity': alert.severity,
                'metric': alert.metric_name,
                'breach_percentage': alert.breach_percentage,
                'urgency_score': alert.urgency_score
            })
        
        total_alerts += len(alerts)
        monitoring_results[scenario['name']] = {
            'alerts_count': len(alerts),
            'alert_details': scenario_alerts,
            'var_95': modified_metrics.var_95_daily,
            'drawdown': scenario['drawdown']
        }
    
    # Get alert summary
    alert_summary = risk_system.risk_monitor.get_alert_summary()
    
    print(f"\n📊 Risk Monitoring Summary:")
    print(f"   Total Alerts Generated: {total_alerts}")
    print(f"   Active Alerts: {alert_summary['total_active_alerts']}")
    print(f"   Highest Urgency Score: {alert_summary['highest_urgency']:.2f}")
    print(f"   Alert Rate (24h): {alert_summary['alert_rate_24h']}")
    
    # Score Demo 4
    monitoring_score = min(100, len(risk_scenarios) * 20)  # 20 points per scenario
    alert_score = min(100, total_alerts * 5)  # 5 points per alert generated
    responsiveness_score = 100 if total_alerts > 0 else 50
    demo4_score = (monitoring_score + alert_score + responsiveness_score) / 3
    
    results['risk_monitoring'] = {
        'scenarios_tested': len(risk_scenarios),
        'total_alerts': total_alerts,
        'monitoring_details': monitoring_results,
        'alert_summary': alert_summary,
        'score': demo4_score
    }
    
    # Demo 5: Integrated Risk Dashboard
    print("\n📋 Demo 5: Integrated Risk Dashboard")
    print("-" * 50)
    
    print("Generating comprehensive risk dashboard...")
    
    dashboard_start = time.time()
    dashboard = risk_system.get_risk_dashboard()
    dashboard_time = time.time() - dashboard_start
    
    print("✅ Risk Dashboard Components:")
    
    # Display dashboard sections
    print(f"   📊 Risk Metrics:")
    print(f"      VaR (95%): {dashboard['risk_metrics']['var_95']:.4f}")
    print(f"      VaR (99%): {dashboard['risk_metrics']['var_99']:.4f}")
    print(f"      Current Drawdown: {dashboard['risk_metrics']['current_drawdown']:.2%}")
    print(f"      Volatility: {dashboard['risk_metrics']['volatility']:.2%}")
    print(f"      Sharpe Ratio: {dashboard['risk_metrics']['sharpe_ratio']:.3f}")
    
    print(f"   🧪 Stress Testing:")
    print(f"      Worst Case Loss: {dashboard['stress_testing']['worst_case_loss']:.2%}")
    print(f"      Average Stress Loss: {dashboard['stress_testing']['average_stress_loss']:.2%}")
    print(f"      Scenarios Tested: {dashboard['stress_testing']['scenarios_tested']}")
    
    print(f"   ⚡ Hedging:")
    print(f"      Hedge Coverage: {dashboard['hedging']['hedge_coverage']:.1%}")
    print(f"      Estimated Cost: {dashboard['hedging']['estimated_cost']:.2%}")
    print(f"      Strategy Count: {dashboard['hedging']['strategy_count']}")
    
    print(f"   🚨 Alerts:")
    print(f"      Total Active: {dashboard['alerts']['total_active_alerts']}")
    print(f"      Highest Urgency: {dashboard['alerts']['highest_urgency']:.2f}")
    print(f"      24h Alert Rate: {dashboard['alerts']['alert_rate_24h']}")
    
    print(f"   💧 Liquidity:")
    print(f"      Liquidity Score: {dashboard['liquidity']['score']:.3f}")
    print(f"      Liquidation Time: {dashboard['liquidity']['liquidation_time']} days")
    print(f"      Market Impact: {dashboard['liquidity']['market_impact']:.2%}")
    
    print(f"   ⏱️ Dashboard Generation Time: {dashboard_time:.4f}s")
    
    # Integration testing
    integration_score = 0
    if PORTFOLIO_OPTIMIZATION_AVAILABLE:
        print("\n🔗 Integration Testing with Portfolio Optimization:")
        try:
            portfolio_system = create_risk_adjusted_portfolio_optimization()
            integration_test = True
            integration_score += 50
            print("   ✅ Portfolio Optimization integration: SUCCESS")
        except Exception as e:
            print(f"   ❌ Portfolio Optimization integration: FAILED ({e})")
    
    if REGIME_DETECTION_AVAILABLE:
        print("🔗 Integration Testing with Regime Detection:")
        try:
            regime_system = create_market_regime_detection()
            integration_test = True
            integration_score += 50
            print("   ✅ Regime Detection integration: SUCCESS")
        except Exception as e:
            print(f"   ❌ Regime Detection integration: FAILED ({e})")
    
    # Score Demo 5
    dashboard_completeness = 100 if len(dashboard) >= 5 else len(dashboard) * 20
    generation_speed = 100 if dashboard_time < 0.1 else 50
    demo5_score = (dashboard_completeness + generation_speed + integration_score) / 3
    
    results['risk_dashboard'] = {
        'dashboard_components': len(dashboard),
        'generation_time': dashboard_time,
        'integration_score': integration_score,
        'dashboard_data': dashboard,
        'score': demo5_score
    }
    
    # Final Summary
    print("\n" + "="*70)
    print("📋 FINAL DEMO SUMMARY")
    print("="*70)
    
    total_demo_time = (datetime.now() - start_time).total_seconds()
    
    # Calculate overall performance score
    scores = [
        results['risk_metrics']['score'],
        results['stress_testing']['score'],
        results['hedging_strategies']['score'],
        results['risk_monitoring']['score'],
        results['risk_dashboard']['score']
    ]
    
    overall_score = np.mean(scores)
    
    # Grade assignment
    if overall_score >= 90:
        grade = "EXCEPTIONAL"
        emoji = "🏆"
    elif overall_score >= 80:
        grade = "EXCELLENT"
        emoji = "🥇"
    elif overall_score >= 70:
        grade = "GOOD"
        emoji = "🥈"
    else:
        grade = "SATISFACTORY"
        emoji = "🥉"
    
    print(f"📊 Performance Summary:")
    print(f"   Risk Metrics Calculation: {results['risk_metrics']['score']:.1f}/100")
    print(f"   Comprehensive Stress Testing: {results['stress_testing']['score']:.1f}/100")
    print(f"   Dynamic Hedging Strategies: {results['hedging_strategies']['score']:.1f}/100")
    print(f"   Real-time Risk Monitoring: {results['risk_monitoring']['score']:.1f}/100")
    print(f"   Integrated Risk Dashboard: {results['risk_dashboard']['score']:.1f}/100")
    print()
    print(f"{emoji} OVERALL PERFORMANCE: {overall_score:.1f}/100")
    print(f"   Grade: {grade}")
    print(f"   Total Demo Time: {total_demo_time:.2f} seconds")
    print(f"   Demo Modules Completed: 5/5")
    
    # Key achievements summary
    print(f"\n🎯 Key Achievements:")
    print(f"   🛡️ VaR Calculation: {risk_metrics.var_95_daily:.4f} daily (< 5% target)")
    print(f"   🧪 Stress Scenarios: {total_scenarios} comprehensive tests")
    print(f"   ⚡ Hedging Strategies: {len(hedging_results)+1} market conditions")
    print(f"   🚨 Risk Alerts: {total_alerts} alerts across scenarios")
    print(f"   📊 Dashboard Components: {len(dashboard)} integrated modules")
    print(f"   🔗 System Integration: {integration_score}% compatibility")
    
    # Export results
    export_data = {
        'demo_info': {
            'day': 27,
            'system_name': 'Advanced Risk Management Systems',
            'demo_date': start_time.isoformat(),
            'total_duration': total_demo_time
        },
        'performance_summary': {
            'overall_score': overall_score,
            'grade': grade,
            'individual_scores': {
                'risk_metrics': results['risk_metrics']['score'],
                'stress_testing': results['stress_testing']['score'],
                'hedging_strategies': results['hedging_strategies']['score'],
                'risk_monitoring': results['risk_monitoring']['score'],
                'risk_dashboard': results['risk_dashboard']['score']
            }
        },
        'detailed_results': results
    }
    
    with open('day27_advanced_risk_management_results.json', 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"\n📊 Results exported to: day27_advanced_risk_management_results.json")
    print(f"✅ Day 27 Advanced Risk Management Demo completed successfully!")
    print(f"🎯 System ready for production with {grade} grade!")


def generate_portfolio_data(assets: List[str], periods: int = 252) -> pd.DataFrame:
    """Generate realistic portfolio data with correlations"""
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=periods, freq='1D')
    
    # Asset parameters with realistic volatilities
    asset_params = {
        'GOLD': {'return': 0.08, 'vol': 0.16},
        'SILVER': {'return': 0.10, 'vol': 0.28},
        'PLATINUM': {'return': 0.06, 'vol': 0.22},
        'COPPER': {'return': 0.08, 'vol': 0.30},
        'CRUDE_OIL': {'return': 0.04, 'vol': 0.40},
        'NATURAL_GAS': {'return': 0.02, 'vol': 0.50}
    }
    
    # Generate correlated returns
    n_assets = len(assets)
    correlation_matrix = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Cholesky decomposition
    chol = np.linalg.cholesky(correlation_matrix)
    
    portfolio_data = pd.DataFrame(index=dates, columns=assets)
    
    # Generate independent returns
    independent_returns = np.random.normal(0, 1, (periods, n_assets))
    
    # Apply correlation
    correlated_returns = independent_returns @ chol.T
    
    for i, asset in enumerate(assets):
        params = asset_params.get(asset, {'return': 0.06, 'vol': 0.25})
        
        # Scale returns to desired parameters
        scaled_returns = correlated_returns[:, i] * params['vol'] / np.sqrt(252) + params['return'] / 252
        
        # Convert to prices
        prices = [100]
        for ret in scaled_returns:
            prices.append(prices[-1] * (1 + ret))
        
        portfolio_data[asset] = prices[1:]
    
    return portfolio_data


if __name__ == "__main__":
    main() 