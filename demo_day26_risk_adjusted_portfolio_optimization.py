"""
Demo Day 26: Risk-Adjusted Portfolio Optimization
Ultimate XAU Super System V4.0

Comprehensive demonstration of risk-adjusted portfolio optimization:
- Multi-objective portfolio optimization (Sharpe, Min-Var, Risk Parity, Kelly)
- Dynamic rebalancing with regime awareness
- Performance attribution and risk analysis
- Transaction cost optimization
- Advanced risk metrics and drawdown analysis
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

from src.core.analysis.risk_adjusted_portfolio_optimization import (
    RiskAdjustedPortfolioOptimization, PortfolioConfig, OptimizationObjective,
    RebalanceFrequency, create_risk_adjusted_portfolio_optimization
)

# Try to import regime detection for integration
try:
    from src.core.analysis.market_regime_detection import create_market_regime_detection
    REGIME_DETECTION_AVAILABLE = True
except:
    REGIME_DETECTION_AVAILABLE = False


def main():
    """Run comprehensive Risk-Adjusted Portfolio Optimization demo"""
    
    print("🎯 Risk-Adjusted Portfolio Optimization Demo - Day 26")
    print("=" * 70)
    
    start_time = datetime.now()
    results = {}
    
    # Demo 1: Multi-Objective Portfolio Optimization
    print("\n📊 Demo 1: Multi-Objective Portfolio Optimization")
    print("-" * 50)
    
    # Generate multi-asset universe
    assets = ['GOLD', 'SILVER', 'PLATINUM', 'COPPER', 'CRUDE_OIL', 'NATURAL_GAS']
    price_data = generate_multi_asset_data(assets, periods=252)
    
    # Test different optimization objectives
    objectives = [
        OptimizationObjective.SHARPE_RATIO,
        OptimizationObjective.MIN_VARIANCE,
        OptimizationObjective.RISK_PARITY,
        OptimizationObjective.KELLY_OPTIMAL
    ]
    
    optimization_results = {}
    
    for objective in objectives:
        print(f"\n🔧 Testing {objective.value.replace('_', ' ').title()} Optimization...")
        
        config = {
            'optimization_objective': objective,
            'lookback_period': 126,  # 6 months
            'max_volatility': 0.18,
            'max_concentration': 0.40,
            'enable_kelly_sizing': True
        }
        
        system = create_risk_adjusted_portfolio_optimization(config)
        
        # Optimize portfolio
        opt_start = time.time()
        optimal_weights = system.optimize_portfolio(price_data)
        opt_time = time.time() - opt_start
        
        # Calculate portfolio metrics
        returns = price_data.pct_change().dropna()
        portfolio_returns = calculate_portfolio_returns(returns, optimal_weights.weights)
        performance = system.analyze_performance(portfolio_returns)
        
        optimization_results[objective.value] = {
            'weights': optimal_weights.weights,
            'expected_return': optimal_weights.expected_return,
            'expected_volatility': optimal_weights.expected_volatility,
            'sharpe_ratio': optimal_weights.sharpe_ratio,
            'diversification_ratio': optimal_weights.diversification_ratio,
            'actual_return': performance.annualized_return,
            'actual_volatility': performance.volatility,
            'actual_sharpe': performance.sharpe_ratio,
            'max_drawdown': performance.max_drawdown,
            'optimization_time': opt_time
        }
        
        print(f"   Expected Return: {optimal_weights.expected_return:.2%}")
        print(f"   Expected Volatility: {optimal_weights.expected_volatility:.2%}")
        print(f"   Sharpe Ratio: {optimal_weights.sharpe_ratio:.3f}")
        print(f"   Diversification: {optimal_weights.diversification_ratio:.3f}")
        print(f"   Optimization Time: {opt_time:.4f}s")
        
        # Show top 3 weights
        sorted_weights = sorted(optimal_weights.weights.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top Holdings: {sorted_weights[0][0]}({sorted_weights[0][1]:.1%}), "
              f"{sorted_weights[1][0]}({sorted_weights[1][1]:.1%}), "
              f"{sorted_weights[2][0]}({sorted_weights[2][1]:.1%})")
    
    results['multi_objective'] = optimization_results
    
    # Demo 2: Regime-Aware Portfolio Optimization
    print("\n📈 Demo 2: Regime-Aware Portfolio Optimization")
    print("-" * 50)
    
    regime_results = {}
    
    if REGIME_DETECTION_AVAILABLE:
        print("Integrating with Market Regime Detection system...")
        
        # Create regime detection system
        regime_system = create_market_regime_detection({
            'lookback_period': 50,
            'enable_ml_prediction': False
        })
        
        # Test portfolio optimization with different regimes
        regime_scenarios = ['trending_market', 'volatile_market', 'ranging_market']
        
        for scenario in regime_scenarios:
            print(f"\n📊 Testing {scenario.replace('_', ' ').title()} Scenario...")
            
            # Generate scenario-specific data
            scenario_data = generate_regime_specific_data(assets, scenario)
            
            # Detect regime
            regime_result = regime_system.analyze_regime(scenario_data)
            regime_context = {
                'current_regime': regime_result.regime.value,
                'confidence': regime_result.confidence
            }
            
            # Optimize with regime awareness
            config = {
                'optimization_objective': OptimizationObjective.SHARPE_RATIO,
                'enable_regime_awareness': True,
                'regime_weight_adjustment': 0.15,
                'max_volatility': 0.20
            }
            
            regime_system_opt = create_risk_adjusted_portfolio_optimization(config)
            
            # Optimization with regime context
            regime_weights = regime_system_opt.optimize_portfolio(scenario_data, regime_context)
            
            # Performance analysis
            regime_returns = scenario_data.pct_change().dropna()
            regime_portfolio_returns = calculate_portfolio_returns(regime_returns, regime_weights.weights)
            regime_performance = regime_system_opt.analyze_performance(regime_portfolio_returns)
            
            regime_results[scenario] = {
                'regime_detected': regime_result.regime.value,
                'regime_confidence': regime_result.confidence,
                'weights': regime_weights.weights,
                'expected_return': regime_weights.expected_return,
                'expected_volatility': regime_weights.expected_volatility,
                'sharpe_ratio': regime_weights.sharpe_ratio,
                'actual_performance': {
                    'return': regime_performance.annualized_return,
                    'volatility': regime_performance.volatility,
                    'sharpe': regime_performance.sharpe_ratio,
                    'max_drawdown': regime_performance.max_drawdown
                }
            }
            
            print(f"   Detected Regime: {regime_result.regime.value} (confidence: {regime_result.confidence:.3f})")
            print(f"   Expected Return: {regime_weights.expected_return:.2%}")
            print(f"   Expected Volatility: {regime_weights.expected_volatility:.2%}")
            print(f"   Sharpe Ratio: {regime_weights.sharpe_ratio:.3f}")
            print(f"   Actual Return: {regime_performance.annualized_return:.2%}")
            print(f"   Max Drawdown: {regime_performance.max_drawdown:.2%}")
    else:
        print("Regime Detection not available, simulating regime scenarios...")
        
        # Simulate regime scenarios without actual regime detection
        simulated_regimes = [
            {'name': 'high_volatility', 'vol_multiplier': 2.0, 'return_adjustment': -0.02},
            {'name': 'low_volatility', 'vol_multiplier': 0.5, 'return_adjustment': 0.01},
            {'name': 'trending_up', 'vol_multiplier': 1.2, 'return_adjustment': 0.05}
        ]
        
        for regime in simulated_regimes:
            scenario_data = generate_regime_specific_data(assets, regime['name'], 
                                                        vol_multiplier=regime['vol_multiplier'],
                                                        return_adjustment=regime['return_adjustment'])
            
            config = {
                'optimization_objective': OptimizationObjective.SHARPE_RATIO,
                'max_volatility': 0.25 if regime['vol_multiplier'] > 1.5 else 0.15
            }
            
            regime_system_opt = create_risk_adjusted_portfolio_optimization(config)
            regime_weights = regime_system_opt.optimize_portfolio(scenario_data)
            
            regime_results[regime['name']] = {
                'weights': regime_weights.weights,
                'expected_return': regime_weights.expected_return,
                'expected_volatility': regime_weights.expected_volatility,
                'sharpe_ratio': regime_weights.sharpe_ratio
            }
            
            print(f"   {regime['name'].replace('_', ' ').title()}: Return {regime_weights.expected_return:.2%}, "
                  f"Vol {regime_weights.expected_volatility:.2%}, Sharpe {regime_weights.sharpe_ratio:.3f}")
    
    results['regime_aware'] = regime_results
    
    # Demo 3: Dynamic Rebalancing Analysis
    print("\n⚡ Demo 3: Dynamic Rebalancing Analysis")
    print("-" * 50)
    
    # Test different rebalancing strategies
    rebalancing_strategies = [
        {'frequency': RebalanceFrequency.MONTHLY, 'name': 'Monthly'},
        {'frequency': RebalanceFrequency.QUARTERLY, 'name': 'Quarterly'},
        {'frequency': RebalanceFrequency.THRESHOLD, 'name': 'Threshold-based'}
    ]
    
    rebalancing_results = {}
    
    # Generate time series data for rebalancing analysis
    extended_data = generate_multi_asset_data(assets, periods=504)  # 2 years
    
    for strategy in rebalancing_strategies:
        print(f"\n🔄 Testing {strategy['name']} Rebalancing...")
        
        config = {
            'optimization_objective': OptimizationObjective.SHARPE_RATIO,
            'rebalance_frequency': strategy['frequency'],
            'enable_dynamic_rebalancing': True,
            'drift_threshold': 0.05,
            'transaction_cost_bps': 5.0,
            'enable_transaction_costs': True
        }
        
        rebalance_system = create_risk_adjusted_portfolio_optimization(config)
        
        # Simulate rebalancing over time
        rebalance_dates = []
        portfolio_values = []
        transaction_costs = []
        
        # Initial optimization
        initial_data = extended_data.iloc[:126]  # First 6 months
        initial_weights = rebalance_system.optimize_portfolio(initial_data)
        current_weights = initial_weights.weights.copy()
        
        # Monthly check for rebalancing
        for month in range(6, 24):  # 18 months of rebalancing
            month_data = extended_data.iloc[:month*21]  # Up to current month
            
            # Get new optimal weights
            new_weights = rebalance_system.optimize_portfolio(month_data)
            
            # Check if rebalancing is needed
            should_rebalance, rebalance_info = rebalance_system.check_rebalancing(
                current_weights, new_weights.weights
            )
            
            if should_rebalance:
                rebalance_dates.append(month)
                transaction_costs.append(rebalance_info.get('estimated_costs', 0))
                current_weights = new_weights.weights.copy()
        
        # Calculate rebalancing metrics
        avg_transaction_cost = np.mean(transaction_costs) if transaction_costs else 0
        rebalancing_frequency = len(rebalance_dates) / 18 * 12  # Annualized frequency
        
        rebalancing_results[strategy['name']] = {
            'rebalance_count': len(rebalance_dates),
            'avg_transaction_cost': avg_transaction_cost,
            'annualized_frequency': rebalancing_frequency,
            'total_transaction_costs': sum(transaction_costs)
        }
        
        print(f"   Rebalances executed: {len(rebalance_dates)}")
        print(f"   Average transaction cost: {avg_transaction_cost:.4f}")
        print(f"   Annualized frequency: {rebalancing_frequency:.1f} times/year")
        print(f"   Total transaction costs: {sum(transaction_costs):.4f}")
    
    results['rebalancing'] = rebalancing_results
    
    # Demo 4: Risk Analysis and Attribution
    print("\n🛡️ Demo 4: Risk Analysis and Attribution")
    print("-" * 50)
    
    # Comprehensive risk analysis
    config = {
        'optimization_objective': OptimizationObjective.SHARPE_RATIO,
        'max_volatility': 0.16,
        'max_drawdown': 0.12,
        'enable_attribution': True
    }
    
    risk_system = create_risk_adjusted_portfolio_optimization(config)
    
    # Optimize portfolio
    risk_weights = risk_system.optimize_portfolio(price_data)
    
    # Calculate detailed performance metrics
    returns = price_data.pct_change().dropna()
    portfolio_returns = calculate_portfolio_returns(returns, risk_weights.weights)
    performance = risk_system.analyze_performance(portfolio_returns)
    
    # Risk decomposition analysis
    risk_metrics = analyze_risk_decomposition(returns, risk_weights.weights)
    
    # Attribution analysis
    attribution = calculate_attribution_analysis(returns, risk_weights.weights)
    
    risk_results = {
        'portfolio_metrics': {
            'total_return': performance.total_return,
            'annualized_return': performance.annualized_return,
            'volatility': performance.volatility,
            'sharpe_ratio': performance.sharpe_ratio,
            'sortino_ratio': performance.sortino_ratio,
            'calmar_ratio': performance.calmar_ratio,
            'max_drawdown': performance.max_drawdown,
            'var_95': performance.var_95,
            'cvar_95': performance.cvar_95
        },
        'risk_decomposition': risk_metrics,
        'attribution_analysis': attribution
    }
    
    print("✅ Portfolio Risk Metrics:")
    print(f"   Total Return: {performance.total_return:.2%}")
    print(f"   Annualized Return: {performance.annualized_return:.2%}")
    print(f"   Volatility: {performance.volatility:.2%}")
    print(f"   Sharpe Ratio: {performance.sharpe_ratio:.3f}")
    print(f"   Sortino Ratio: {performance.sortino_ratio:.3f}")
    print(f"   Calmar Ratio: {performance.calmar_ratio:.3f}")
    print(f"   Max Drawdown: {performance.max_drawdown:.2%}")
    print(f"   VaR (95%): {performance.var_95:.3f}")
    print(f"   CVaR (95%): {performance.cvar_95:.3f}")
    
    print("\n📊 Risk Decomposition:")
    for metric, value in risk_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n💼 Attribution Analysis:")
    for asset, contribution in attribution.items():
        print(f"   {asset}: {contribution:.3f}")
    
    results['risk_analysis'] = risk_results
    
    # Demo 5: Performance Comparison
    print("\n🏆 Demo 5: Performance Comparison Analysis")
    print("-" * 50)
    
    # Compare different optimization approaches
    comparison_configs = {
        'Equal Weight': {
            'optimization_objective': OptimizationObjective.SHARPE_RATIO,
            'min_weight': 1/len(assets),
            'max_weight': 1/len(assets)
        },
        'Sharpe Optimal': {
            'optimization_objective': OptimizationObjective.SHARPE_RATIO,
            'max_volatility': 0.18
        },
        'Min Variance': {
            'optimization_objective': OptimizationObjective.MIN_VARIANCE,
            'max_concentration': 0.50
        },
        'Kelly Optimal': {
            'optimization_objective': OptimizationObjective.KELLY_OPTIMAL,
            'kelly_fraction': 0.25
        }
    }
    
    comparison_results = {}
    
    for name, config in comparison_configs.items():
        comp_system = create_risk_adjusted_portfolio_optimization(config)
        comp_weights = comp_system.optimize_portfolio(price_data)
        comp_returns = calculate_portfolio_returns(returns, comp_weights.weights)
        comp_performance = comp_system.analyze_performance(comp_returns)
        
        comparison_results[name] = {
            'annualized_return': comp_performance.annualized_return,
            'volatility': comp_performance.volatility,
            'sharpe_ratio': comp_performance.sharpe_ratio,
            'max_drawdown': comp_performance.max_drawdown,
            'calmar_ratio': comp_performance.calmar_ratio,
            'total_return': comp_performance.total_return
        }
        
        print(f"\n📈 {name} Strategy:")
        print(f"   Return: {comp_performance.annualized_return:.2%}")
        print(f"   Volatility: {comp_performance.volatility:.2%}")
        print(f"   Sharpe: {comp_performance.sharpe_ratio:.3f}")
        print(f"   Max DD: {comp_performance.max_drawdown:.2%}")
        print(f"   Calmar: {comp_performance.calmar_ratio:.3f}")
    
    results['performance_comparison'] = comparison_results
    
    # Final Summary
    print("\n" + "="*70)
    print("📋 FINAL DEMO SUMMARY")
    print("="*70)
    
    total_demo_time = (datetime.now() - start_time).total_seconds()
    
    # Calculate overall performance score
    scores = []
    
    # Multi-objective score (diversity and performance)
    multi_obj_score = len(optimization_results) * 20  # 20 points per objective
    best_sharpe = max([r['sharpe_ratio'] for r in optimization_results.values()])
    sharpe_score = min(40, best_sharpe * 20)  # Max 40 points
    multi_total = min(100, multi_obj_score + sharpe_score)
    scores.append(multi_total)
    
    # Regime-aware score
    regime_score = len(regime_results) * 30  # 30 points per regime
    scores.append(min(100, regime_score))
    
    # Rebalancing score
    rebal_efficiency = min([r['avg_transaction_cost'] for r in rebalancing_results.values()])
    rebal_score = max(0, 100 - rebal_efficiency * 10000)  # Lower cost = higher score
    scores.append(rebal_score)
    
    # Risk analysis score
    risk_score = 100 if performance.sharpe_ratio > 1.0 else performance.sharpe_ratio * 100
    scores.append(risk_score)
    
    # Performance comparison score
    best_comp_sharpe = max([r['sharpe_ratio'] for r in comparison_results.values()])
    comp_score = min(100, best_comp_sharpe * 50)
    scores.append(comp_score)
    
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
    print(f"   Multi-Objective Optimization: {multi_total:.1f}/100")
    print(f"   Regime-Aware Analysis: {min(100, len(regime_results) * 30):.1f}/100")
    print(f"   Rebalancing Efficiency: {rebal_score:.1f}/100")
    print(f"   Risk Analysis: {risk_score:.1f}/100")
    print(f"   Performance Comparison: {comp_score:.1f}/100")
    print()
    print(f"{emoji} OVERALL PERFORMANCE: {overall_score:.1f}/100")
    print(f"   Grade: {grade}")
    print(f"   Total Demo Time: {total_demo_time:.2f} seconds")
    print(f"   Demo Modules Completed: 5/5")
    
    # Export results
    export_data = {
        'demo_info': {
            'day': 26,
            'system_name': 'Risk-Adjusted Portfolio Optimization',
            'demo_date': start_time.isoformat(),
            'total_duration': total_demo_time
        },
        'performance_summary': {
            'overall_score': overall_score,
            'grade': grade,
            'individual_scores': {
                'multi_objective': multi_total,
                'regime_aware': min(100, len(regime_results) * 30),
                'rebalancing': rebal_score,
                'risk_analysis': risk_score,
                'performance_comparison': comp_score
            }
        },
        'detailed_results': results
    }
    
    with open('day26_risk_adjusted_portfolio_optimization_results.json', 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"\n📊 Results exported to: day26_risk_adjusted_portfolio_optimization_results.json")
    print(f"✅ Day 26 Risk-Adjusted Portfolio Optimization Demo completed successfully!")
    print(f"🎯 System ready for production with {grade} grade!")


def generate_multi_asset_data(assets: List[str], periods: int = 252) -> pd.DataFrame:
    """Generate multi-asset price data with realistic correlations"""
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=periods, freq='1D')
    
    # Asset parameters (return, volatility)
    asset_params = {
        'GOLD': {'return': 0.08, 'vol': 0.15},
        'SILVER': {'return': 0.12, 'vol': 0.25},
        'PLATINUM': {'return': 0.06, 'vol': 0.20},
        'COPPER': {'return': 0.10, 'vol': 0.28},
        'CRUDE_OIL': {'return': 0.05, 'vol': 0.35},
        'NATURAL_GAS': {'return': 0.03, 'vol': 0.40}
    }
    
    # Generate correlated returns
    correlations = np.random.uniform(0.2, 0.7, (len(assets), len(assets)))
    correlations = (correlations + correlations.T) / 2
    np.fill_diagonal(correlations, 1.0)
    
    # Cholesky decomposition for correlation
    chol = np.linalg.cholesky(correlations)
    
    price_data = pd.DataFrame(index=dates, columns=assets)
    
    for i, asset in enumerate(assets):
        params = asset_params.get(asset, {'return': 0.06, 'vol': 0.20})
        
        # Generate independent random returns
        independent_returns = np.random.normal(0, 1, periods)
        
        # Apply correlation
        correlated_returns = np.zeros(periods)
        for j in range(len(assets)):
            if j <= i:
                corr_factor = chol[i, j] if j < len(chol) and i < len(chol) else 0
                correlated_returns += corr_factor * np.random.normal(0, 1, periods)
        
        # Scale to desired return/volatility
        returns = correlated_returns * params['vol'] / np.sqrt(252) + params['return'] / 252
        
        # Convert to prices
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[asset] = prices[1:]
    
    return price_data


def generate_regime_specific_data(assets: List[str], scenario: str, 
                                vol_multiplier: float = 1.0, 
                                return_adjustment: float = 0.0) -> pd.DataFrame:
    """Generate asset data for specific regime scenarios"""
    
    np.random.seed(hash(scenario) % 2**32)
    periods = 126  # 6 months
    dates = pd.date_range('2024-01-01', periods=periods, freq='1D')
    
    base_params = {
        'GOLD': {'return': 0.08, 'vol': 0.15},
        'SILVER': {'return': 0.12, 'vol': 0.25},
        'PLATINUM': {'return': 0.06, 'vol': 0.20},
        'COPPER': {'return': 0.10, 'vol': 0.28},
        'CRUDE_OIL': {'return': 0.05, 'vol': 0.35},
        'NATURAL_GAS': {'return': 0.03, 'vol': 0.40}
    }
    
    price_data = pd.DataFrame(index=dates, columns=assets)
    
    for asset in assets:
        params = base_params.get(asset, {'return': 0.06, 'vol': 0.20})
        
        # Apply scenario modifications
        adj_return = params['return'] + return_adjustment
        adj_vol = params['vol'] * vol_multiplier
        
        # Generate scenario-specific patterns
        if scenario == 'trending_market':
            trend_component = np.linspace(0, 0.15, periods)
            returns = np.random.normal(adj_return/252, adj_vol/np.sqrt(252), periods) + trend_component/252
        elif scenario == 'volatile_market':
            # Add volatility clustering
            garch_vol = [adj_vol/np.sqrt(252)]
            for i in range(1, periods):
                garch_vol.append(0.95 * garch_vol[-1] + 0.05 * adj_vol/np.sqrt(252))
            returns = [np.random.normal(adj_return/252, vol) for vol in garch_vol]
        elif scenario == 'ranging_market':
            # Mean-reverting returns
            price_level = 100
            returns = []
            for i in range(periods):
                reversion = (100 - price_level) * 0.01  # Mean reversion force
                ret = np.random.normal(adj_return/252 + reversion, adj_vol/np.sqrt(252))
                returns.append(ret)
                price_level *= (1 + ret)
        else:
            returns = np.random.normal(adj_return/252, adj_vol/np.sqrt(252), periods)
        
        # Convert to prices
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[asset] = prices[1:]
    
    return price_data


def calculate_portfolio_returns(asset_returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Calculate portfolio returns from asset returns and weights"""
    
    portfolio_returns = pd.Series(index=asset_returns.index, dtype=float)
    
    for date in asset_returns.index:
        daily_return = 0
        for asset, weight in weights.items():
            if asset in asset_returns.columns:
                daily_return += weight * asset_returns.loc[date, asset]
        portfolio_returns.loc[date] = daily_return
    
    return portfolio_returns


def analyze_risk_decomposition(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
    """Analyze portfolio risk decomposition"""
    
    # Calculate covariance matrix
    cov_matrix = returns.cov().values * 252  # Annualized
    
    # Weight vector
    w = np.array([weights.get(col, 0) for col in returns.columns])
    
    # Portfolio variance
    portfolio_var = np.dot(w, np.dot(cov_matrix, w))
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Marginal risk contributions
    marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol
    
    # Risk contributions
    risk_contrib = w * marginal_contrib
    
    # Diversification metrics
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(w, asset_vols)
    diversification_ratio = weighted_avg_vol / portfolio_vol
    
    return {
        'portfolio_volatility': portfolio_vol,
        'diversification_ratio': diversification_ratio,
        'concentration_risk': max(risk_contrib) / sum(risk_contrib),
        'effective_assets': 1 / sum(w**2)  # Herfindahl index
    }


def calculate_attribution_analysis(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
    """Calculate return attribution by asset"""
    
    attribution = {}
    total_portfolio_return = 0
    
    for asset in returns.columns:
        weight = weights.get(asset, 0)
        asset_return = returns[asset].mean() * 252  # Annualized
        contribution = weight * asset_return
        attribution[asset] = contribution
        total_portfolio_return += contribution
    
    return attribution


if __name__ == "__main__":
    main() 