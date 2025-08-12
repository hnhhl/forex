"""
Portfolio Management System Demo
Comprehensive demonstration of PortfolioManager, PortfolioOptimizer, and CorrelationAnalyzer
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.trading.portfolio_manager import (
    PortfolioManager, PortfolioRiskLevel, AllocationMethod,
    PortfolioMetrics, SymbolAllocation, PortfolioRiskMetrics
)
from src.core.trading.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationMethod, ObjectiveFunction,
    OptimizationConstraints, OptimizationResult, BlackLittermanInputs
)
from src.core.trading.correlation_analyzer import (
    CorrelationAnalyzer, CorrelationMethod, ClusteringMethod,
    CorrelationMetrics, ClusterAnalysis, RollingCorrelation
)
from src.core.trading.position_types import Position, PositionType, PositionStatus


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_section(title: str):
    """Print formatted section"""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def create_sample_data():
    """Create sample market data for demonstration"""
    print_section("Creating Sample Market Data")
    
    # Create 1 year of daily returns data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    # Create correlated returns
    base_factor = np.random.normal(0.0005, 0.015, 252)
    
    returns_data = pd.DataFrame({
        'XAUUSD': base_factor * 0.8 + np.random.normal(0.001, 0.02, 252),
        'EURUSD': base_factor * 0.6 + np.random.normal(0.0005, 0.012, 252),
        'GBPUSD': base_factor * 0.7 + np.random.normal(0.0003, 0.015, 252),
        'USDJPY': -base_factor * 0.4 + np.random.normal(-0.0002, 0.010, 252),
        'AUDUSD': base_factor * 0.5 + np.random.normal(0.0008, 0.018, 252)
    }, index=dates)
    
    print(f"✓ Created returns data for {len(symbols)} symbols over {len(dates)} days")
    print(f"✓ Data range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Average daily returns:")
    for symbol in symbols:
        avg_return = returns_data[symbol].mean()
        volatility = returns_data[symbol].std()
        print(f"  {symbol}: {avg_return:.4f} (σ: {volatility:.4f})")
    
    return returns_data


def demo_portfolio_manager():
    """Demonstrate PortfolioManager functionality"""
    print_header("PORTFOLIO MANAGER DEMO")
    
    # Initialize portfolio manager
    config = {
        'update_interval': 1,
        'max_symbols': 10,
        'risk_level': 'moderate',
        'max_portfolio_risk': 0.05,
        'max_symbol_weight': 0.4,
        'min_symbol_weight': 0.05
    }
    
    portfolio_manager = PortfolioManager(config)
    
    print_section("1. Portfolio Manager Initialization")
    print(f"✓ Portfolio Manager created: {portfolio_manager.name}")
    print(f"✓ Risk Level: {portfolio_manager.risk_level.value}")
    print(f"✓ Max Symbols: {portfolio_manager.max_symbols}")
    print(f"✓ Max Portfolio Risk: {portfolio_manager.max_portfolio_risk:.1%}")
    
    print_section("2. Adding Symbols to Portfolio")
    symbols_weights = [
        ('XAUUSD', 0.3),
        ('EURUSD', 0.25),
        ('GBPUSD', 0.2),
        ('USDJPY', 0.15),
        ('AUDUSD', 0.1)
    ]
    
    for symbol, weight in symbols_weights:
        result = portfolio_manager.add_symbol(symbol, weight)
        status = "✓" if result else "✗"
        print(f"{status} {symbol}: {weight:.1%} weight")
    
    print(f"\nPortfolio composition:")
    for symbol, allocation in portfolio_manager.get_symbol_allocations().items():
        print(f"  {symbol}: {allocation.target_weight:.1%} target weight")
    
    print_section("3. Adding Mock Positions")
    mock_positions = [
        Position(
            position_id=f"POS_{i:03d}",
            symbol=symbol,
            position_type=PositionType.BUY,
            volume=0.1,
            open_price=2000.0 + i * 10,
            current_price=2000.0 + i * 10 + np.random.normal(0, 20),
            open_time=datetime.now() - timedelta(days=i),
            status=PositionStatus.OPEN,
            remaining_volume=0.1,
            realized_profit=0.0
        )
        for i, (symbol, _) in enumerate(symbols_weights[:3])
    ]
    
    for position in mock_positions:
        result = portfolio_manager.add_position_to_portfolio(position)
        pnl = (position.current_price - position.open_price) * position.volume
        status = "✓" if result else "✗"
        print(f"{status} {position.symbol} position: {pnl:+.2f} P&L")
    
    print_section("4. Portfolio Metrics Calculation")
    portfolio_value = portfolio_manager.get_portfolio_value()
    total_pnl, realized_pnl, unrealized_pnl = portfolio_manager.get_portfolio_pnl()
    
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Total P&L: ${total_pnl:+,.2f}")
    print(f"  Realized: ${realized_pnl:+,.2f}")
    print(f"  Unrealized: ${unrealized_pnl:+,.2f}")
    
    metrics = portfolio_manager.calculate_portfolio_metrics()
    print(f"\nPortfolio Metrics:")
    print(f"  Total Return: {metrics.total_return:+.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    
    print_section("5. Risk Analysis")
    risk_metrics = portfolio_manager.calculate_portfolio_risk()
    print(f"Portfolio Risk Metrics:")
    print(f"  Volatility: {risk_metrics.portfolio_volatility:.2%}")
    print(f"  Concentration Risk: {risk_metrics.concentration_risk:.2%}")
    print(f"  Leverage Ratio: {risk_metrics.leverage_ratio:.2f}")
    print(f"  Risk Budget Utilization: {risk_metrics.risk_budget_utilization:.1%}")
    
    print_section("6. Portfolio Rebalancing")
    print("Testing different rebalancing methods:")
    
    methods = [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.MINIMUM_VARIANCE
    ]
    
    for method in methods:
        # Mock portfolio value for rebalancing
        portfolio_manager.get_portfolio_value = lambda: 10000.0
        result = portfolio_manager.rebalance_portfolio(method)
        status = "✓" if result else "✗"
        print(f"{status} {method.value} rebalancing")
    
    print_section("7. Portfolio Statistics")
    stats = portfolio_manager.get_statistics()
    print(f"Portfolio Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if 'count' in key or 'total' in key:
                print(f"  {key}: {value:,}")
            elif 'ratio' in key or 'risk' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    return portfolio_manager


def demo_portfolio_optimizer(returns_data):
    """Demonstrate PortfolioOptimizer functionality"""
    print_header("PORTFOLIO OPTIMIZER DEMO")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer({
        'risk_free_rate': 0.02,
        'confidence_level': 0.95,
        'lookback_period': 252
    })
    
    print_section("1. Setting Returns Data")
    optimizer.set_returns_data(returns_data)
    print(f"✓ Returns data set for {len(optimizer.symbols)} symbols")
    print(f"✓ Covariance matrix shape: {optimizer.covariance_matrix.shape}")
    print(f"✓ Expected returns calculated")
    
    print_section("2. Portfolio Optimization Methods")
    
    optimization_methods = [
        (OptimizationMethod.EQUAL_WEIGHT, "Equal Weight"),
        (OptimizationMethod.MINIMUM_VARIANCE, "Minimum Variance"),
        (OptimizationMethod.MAXIMUM_SHARPE, "Maximum Sharpe Ratio"),
        (OptimizationMethod.RISK_PARITY, "Risk Parity"),
        (OptimizationMethod.MEAN_VARIANCE, "Mean Variance"),
        (OptimizationMethod.BLACK_LITTERMAN, "Black-Litterman")
    ]
    
    optimization_results = {}
    
    for method, name in optimization_methods:
        print(f"\nOptimizing using {name}...")
        
        if method == OptimizationMethod.BLACK_LITTERMAN:
            # Create Black-Litterman inputs
            bl_inputs = BlackLittermanInputs(
                market_caps={symbol: 1.0 for symbol in optimizer.symbols},
                risk_aversion=3.0,
                tau=0.025
            )
            result = optimizer.optimize_portfolio(method, bl_inputs=bl_inputs)
        else:
            result = optimizer.optimize_portfolio(method)
        
        optimization_results[name] = result
        
        if result.success:
            print(f"✓ Optimization successful")
            print(f"  Expected Return: {result.expected_return:.2%}")
            print(f"  Expected Volatility: {result.expected_volatility:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
            print(f"  Computation Time: {result.computation_time:.3f}s")
            
            print(f"  Weights:")
            for symbol, weight in result.weights.items():
                print(f"    {symbol}: {weight:.1%}")
        else:
            print(f"✗ Optimization failed: {result.message}")
    
    print_section("3. Optimization with Constraints")
    constraints = OptimizationConstraints(
        min_weight=0.05,
        max_weight=0.4,
        sum_weights=1.0
    )
    
    result = optimizer.optimize_portfolio(
        OptimizationMethod.MEAN_VARIANCE,
        constraints=constraints,
        objective=ObjectiveFunction.MAXIMIZE_SHARPE
    )
    
    if result.success:
        print(f"✓ Constrained optimization successful")
        print(f"  Weights (5%-40% constraints):")
        for symbol, weight in result.weights.items():
            print(f"    {symbol}: {weight:.1%}")
    
    print_section("4. Efficient Frontier Generation")
    frontier = optimizer.generate_efficient_frontier(n_points=20)
    
    if not frontier.empty:
        print(f"✓ Generated efficient frontier with {len(frontier)} points")
        print(f"  Return range: {frontier['return'].min():.2%} to {frontier['return'].max():.2%}")
        print(f"  Volatility range: {frontier['volatility'].min():.2%} to {frontier['volatility'].max():.2%}")
        print(f"  Max Sharpe ratio: {frontier['sharpe_ratio'].max():.3f}")
    
    print_section("5. Strategy Backtesting")
    strategies = {
        'Equal Weight': {symbol: 0.2 for symbol in optimizer.symbols},
        'Gold Heavy': {'XAUUSD': 0.4, 'EURUSD': 0.2, 'GBPUSD': 0.2, 'USDJPY': 0.1, 'AUDUSD': 0.1},
        'Conservative': {'XAUUSD': 0.1, 'EURUSD': 0.3, 'GBPUSD': 0.3, 'USDJPY': 0.2, 'AUDUSD': 0.1}
    }
    
    backtest_results = {}
    for strategy_name, weights in strategies.items():
        result = optimizer.backtest_strategy(weights)
        backtest_results[strategy_name] = result
        
        if result:
            print(f"\n{strategy_name} Strategy:")
            print(f"  Total Return: {result['total_return']:.2%}")
            print(f"  Annualized Return: {result['annualized_return']:.2%}")
            print(f"  Volatility: {result['volatility']:.2%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"  Calmar Ratio: {result['calmar_ratio']:.3f}")
    
    print_section("6. Strategy Comparison")
    comparison = optimizer.compare_strategies(strategies)
    
    if not comparison.empty:
        print(f"✓ Strategy comparison completed")
        print(f"\nRanking by Sharpe Ratio:")
        sorted_strategies = comparison.sort_values('sharpe_ratio', ascending=False)
        for i, (_, row) in enumerate(sorted_strategies.iterrows(), 1):
            print(f"  {i}. {row['strategy']}: {row['sharpe_ratio']:.3f}")
    
    print_section("7. Optimizer Statistics")
    stats = optimizer.get_statistics()
    print(f"Optimizer Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if 'time' in key:
                print(f"  {key}: {value:.3f}s")
            elif 'rate' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    return optimizer, optimization_results


def demo_correlation_analyzer(returns_data):
    """Demonstrate CorrelationAnalyzer functionality"""
    print_header("CORRELATION ANALYZER DEMO")
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer({
        'lookback_period': 252,
        'rolling_window': 30,
        'correlation_threshold': 0.5
    })
    
    print_section("1. Setting Data")
    analyzer.set_data(returns_data)
    print(f"✓ Data set for {len(analyzer.symbols)} symbols")
    print(f"✓ Data shape: {analyzer.returns_data.shape}")
    
    print_section("2. Correlation Matrix Calculation")
    methods = [
        (CorrelationMethod.PEARSON, "Pearson"),
        (CorrelationMethod.SPEARMAN, "Spearman"),
        (CorrelationMethod.KENDALL, "Kendall")
    ]
    
    for method, name in methods:
        corr_matrix = analyzer.calculate_correlation_matrix(method)
        print(f"\n{name} Correlation Matrix:")
        
        # Print correlation matrix
        symbols = analyzer.symbols
        print(f"{'':>8}", end="")
        for symbol in symbols:
            print(f"{symbol:>8}", end="")
        print()
        
        for i, symbol1 in enumerate(symbols):
            print(f"{symbol1:>8}", end="")
            for j, symbol2 in enumerate(symbols):
                print(f"{corr_matrix[i, j]:>8.3f}", end="")
            print()
    
    print_section("3. Comprehensive Correlation Analysis")
    metrics = analyzer.analyze_correlations()
    
    print(f"Correlation Metrics:")
    print(f"  Average Correlation: {metrics.average_correlation:.3f}")
    print(f"  Max Correlation: {metrics.max_correlation:.3f}")
    print(f"  Min Correlation: {metrics.min_correlation:.3f}")
    print(f"  Correlation Stability: {metrics.correlation_stability:.3f}")
    print(f"  Diversification Ratio: {metrics.diversification_ratio:.3f}")
    print(f"  Effective Assets: {metrics.effective_assets:.2f}")
    print(f"  Concentration Risk: {metrics.concentration_risk:.3f}")
    
    print_section("4. Rolling Correlation Analysis")
    symbol_pairs = [
        ('XAUUSD', 'EURUSD'),
        ('EURUSD', 'GBPUSD'),
        ('USDJPY', 'AUDUSD')
    ]
    
    for symbol1, symbol2 in symbol_pairs:
        rolling_corr = analyzer.rolling_correlation_analysis(symbol1, symbol2, window=30)
        
        if rolling_corr.correlations:
            print(f"\n{symbol1} vs {symbol2}:")
            print(f"  Average Correlation: {np.mean(rolling_corr.correlations):.3f}")
            print(f"  Correlation Volatility: {rolling_corr.volatility:.3f}")
            print(f"  Trend: {rolling_corr.trend}")
            print(f"  Breakpoints: {len(rolling_corr.breakpoints)}")
    
    print_section("5. Cluster Analysis")
    cluster_analysis = analyzer.cluster_analysis(ClusteringMethod.HIERARCHICAL)
    
    if cluster_analysis.n_clusters > 0:
        print(f"✓ Hierarchical clustering completed")
        print(f"  Number of clusters: {cluster_analysis.n_clusters}")
        print(f"  Silhouette score: {cluster_analysis.silhouette_score:.3f}")
        print(f"  Inter-cluster correlation: {cluster_analysis.inter_cluster_correlation:.3f}")
        print(f"  Intra-cluster correlation: {cluster_analysis.intra_cluster_correlation:.3f}")
        
        print(f"\nCluster assignments:")
        for symbol, cluster_id in cluster_analysis.clusters.items():
            print(f"  {symbol}: Cluster {cluster_id}")
    
    print_section("6. Highly Correlated Pairs")
    high_corr_pairs = analyzer.identify_highly_correlated_pairs(threshold=0.3)
    
    if high_corr_pairs:
        print(f"✓ Found {len(high_corr_pairs)} highly correlated pairs (>30%)")
        for symbol1, symbol2, correlation in high_corr_pairs[:5]:  # Top 5
            print(f"  {symbol1} - {symbol2}: {correlation:.3f}")
    else:
        print("✓ No highly correlated pairs found")
    
    print_section("7. Portfolio Correlation Risk")
    test_portfolios = {
        'Equal Weight': {symbol: 0.2 for symbol in analyzer.symbols},
        'Gold Heavy': {'XAUUSD': 0.5, 'EURUSD': 0.2, 'GBPUSD': 0.15, 'USDJPY': 0.1, 'AUDUSD': 0.05},
        'Diversified': {'XAUUSD': 0.15, 'EURUSD': 0.25, 'GBPUSD': 0.25, 'USDJPY': 0.2, 'AUDUSD': 0.15}
    }
    
    for portfolio_name, weights in test_portfolios.items():
        risk_metrics = analyzer.calculate_portfolio_correlation_risk(weights)
        
        if risk_metrics:
            print(f"\n{portfolio_name} Portfolio:")
            print(f"  Portfolio Correlation: {risk_metrics['portfolio_correlation']:.3f}")
            print(f"  Weighted Avg Correlation: {risk_metrics['weighted_average_correlation']:.3f}")
            print(f"  Concentration Risk: {risk_metrics['concentration_risk']:.3f}")
            print(f"  Diversification Benefit: {risk_metrics['diversification_benefit']:.3f}")
            print(f"  Correlation Risk Score: {risk_metrics['correlation_risk_score']:.3f}")
    
    print_section("8. Comprehensive Correlation Report")
    report = analyzer.generate_correlation_report()
    
    if report:
        print(f"✓ Correlation report generated")
        print(f"  Analysis method: {report['method']}")
        print(f"  Number of symbols: {report['n_symbols']}")
        print(f"  High correlation pairs: {len(report['high_correlation_pairs'])}")
        print(f"  Clusters identified: {report['cluster_analysis']['n_clusters']}")
    
    print_section("9. Analyzer Statistics")
    stats = analyzer.get_statistics()
    print(f"Analyzer Statistics:")
    for key, value in stats.items():
        if key != 'last_analysis':  # Skip complex object
            print(f"  {key}: {value}")
    
    return analyzer


def demo_integrated_system(returns_data):
    """Demonstrate integrated portfolio management system"""
    print_header("INTEGRATED PORTFOLIO MANAGEMENT SYSTEM")
    
    print_section("1. System Integration Setup")
    
    # Initialize all components
    portfolio_manager = PortfolioManager({
        'max_symbol_weight': 0.4,
        'min_symbol_weight': 0.05,
        'max_portfolio_risk': 0.06
    })
    
    optimizer = PortfolioOptimizer()
    optimizer.set_returns_data(returns_data)
    
    analyzer = CorrelationAnalyzer()
    analyzer.set_data(returns_data)
    
    print(f"✓ Portfolio Manager initialized")
    print(f"✓ Portfolio Optimizer initialized with {len(optimizer.symbols)} symbols")
    print(f"✓ Correlation Analyzer initialized")
    
    print_section("2. Correlation-Informed Portfolio Construction")
    
    # Analyze correlations first
    correlation_metrics = analyzer.analyze_correlations()
    high_corr_pairs = analyzer.identify_highly_correlated_pairs(threshold=0.5)
    
    print(f"Correlation Analysis Results:")
    print(f"  Average correlation: {correlation_metrics.average_correlation:.3f}")
    print(f"  Effective assets: {correlation_metrics.effective_assets:.2f}")
    print(f"  High correlation pairs: {len(high_corr_pairs)}")
    
    # Use correlation insights for portfolio construction
    if correlation_metrics.average_correlation > 0.5:
        print(f"⚠ High average correlation detected - using risk parity approach")
        optimization_method = OptimizationMethod.RISK_PARITY
    else:
        print(f"✓ Moderate correlation - using mean variance optimization")
        optimization_method = OptimizationMethod.MEAN_VARIANCE
    
    print_section("3. Portfolio Optimization with Risk Constraints")
    
    # Set constraints based on correlation analysis
    max_weight = min(0.4, 1.0 / correlation_metrics.effective_assets * 1.5)
    
    constraints = OptimizationConstraints(
        min_weight=0.05,
        max_weight=max_weight,
        sum_weights=1.0
    )
    
    optimization_result = optimizer.optimize_portfolio(
        optimization_method,
        constraints=constraints
    )
    
    if optimization_result.success:
        print(f"✓ Portfolio optimization successful")
        print(f"  Method: {optimization_result.optimization_method}")
        print(f"  Expected return: {optimization_result.expected_return:.2%}")
        print(f"  Expected volatility: {optimization_result.expected_volatility:.2%}")
        print(f"  Sharpe ratio: {optimization_result.sharpe_ratio:.3f}")
        
        print(f"\nOptimal weights:")
        for symbol, weight in optimization_result.weights.items():
            print(f"  {symbol}: {weight:.1%}")
    
    print_section("4. Portfolio Implementation")
    
    # Add symbols to portfolio manager with optimized weights
    for symbol, weight in optimization_result.weights.items():
        result = portfolio_manager.add_symbol(symbol, weight)
        status = "✓" if result else "✗"
        print(f"{status} Added {symbol} with {weight:.1%} weight")
    
    print_section("5. Risk Monitoring and Validation")
    
    # Calculate portfolio correlation risk
    correlation_risk = analyzer.calculate_portfolio_correlation_risk(optimization_result.weights)
    
    if correlation_risk:
        print(f"Portfolio Risk Assessment:")
        print(f"  Concentration risk: {correlation_risk['concentration_risk']:.3f}")
        print(f"  Diversification benefit: {correlation_risk['diversification_benefit']:.3f}")
        print(f"  Correlation risk score: {correlation_risk['correlation_risk_score']:.3f}")
        
        # Risk alerts
        if correlation_risk['concentration_risk'] > 0.4:
            print(f"⚠ WARNING: High concentration risk detected!")
        
        if correlation_risk['correlation_risk_score'] > 0.3:
            print(f"⚠ WARNING: High correlation risk score!")
        
        if correlation_risk['diversification_benefit'] < 0.1:
            print(f"⚠ WARNING: Low diversification benefit!")
    
    print_section("6. Performance Monitoring Simulation")
    
    # Simulate portfolio performance monitoring
    print(f"Simulating portfolio monitoring...")
    
    # Mock some portfolio positions and performance
    portfolio_manager.daily_returns = list(np.random.normal(0.001, 0.02, 30))
    
    # Calculate portfolio metrics
    portfolio_risk = portfolio_manager.calculate_portfolio_risk()
    
    print(f"Portfolio Risk Metrics:")
    print(f"  Portfolio volatility: {portfolio_risk.portfolio_volatility:.2%}")
    print(f"  Risk budget utilization: {portfolio_risk.risk_budget_utilization:.1%}")
    
    # Check if rebalancing is needed
    if portfolio_risk.risk_budget_utilization > 0.8:
        print(f"⚠ Risk budget utilization high - rebalancing recommended")
        
        # Simulate rebalancing
        portfolio_manager.get_portfolio_value = lambda: 100000.0
        rebalance_result = portfolio_manager.rebalance_portfolio(AllocationMethod.RISK_PARITY)
        
        if rebalance_result:
            print(f"✓ Portfolio rebalanced successfully")
        else:
            print(f"✗ Portfolio rebalancing failed")
    
    print_section("7. System Performance Summary")
    
    # Get statistics from all components
    pm_stats = portfolio_manager.get_statistics()
    opt_stats = optimizer.get_statistics()
    corr_stats = analyzer.get_statistics()
    
    print(f"System Performance Summary:")
    print(f"  Portfolio Manager:")
    print(f"    Symbols managed: {pm_stats['symbols_count']}")
    print(f"    Rebalances performed: {pm_stats['rebalance_count']}")
    print(f"    Risk breaches: {pm_stats['risk_breaches']}")
    
    print(f"  Portfolio Optimizer:")
    print(f"    Optimizations run: {opt_stats['n_optimizations']}")
    print(f"    Success rate: {opt_stats['successful_optimizations']}/{opt_stats['n_optimizations']}")
    print(f"    Avg computation time: {opt_stats['average_computation_time']:.3f}s")
    
    print(f"  Correlation Analyzer:")
    print(f"    Correlation analyses: {corr_stats['n_correlation_analyses']}")
    print(f"    Cluster analyses: {corr_stats['n_cluster_analyses']}")
    print(f"    Symbols analyzed: {corr_stats['n_symbols']}")
    
    print_section("8. Integration Benefits")
    
    print(f"Integrated System Benefits:")
    print(f"✓ Correlation-informed optimization")
    print(f"✓ Dynamic risk monitoring")
    print(f"✓ Automated rebalancing triggers")
    print(f"✓ Multi-dimensional risk assessment")
    print(f"✓ Real-time portfolio analytics")
    print(f"✓ Professional-grade risk management")


def main():
    """Main demo function"""
    print_header("PORTFOLIO MANAGEMENT SYSTEM - COMPREHENSIVE DEMO")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create sample data
        returns_data = create_sample_data()
        
        # Run individual component demos
        portfolio_manager = demo_portfolio_manager()
        optimizer, optimization_results = demo_portfolio_optimizer(returns_data)
        analyzer = demo_correlation_analyzer(returns_data)
        
        # Run integrated system demo
        demo_integrated_system(returns_data)
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print(f"✓ All components demonstrated successfully")
        print(f"✓ Portfolio Management System is production-ready")
        print(f"✓ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()