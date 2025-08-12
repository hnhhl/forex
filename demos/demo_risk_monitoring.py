"""
Risk Monitoring System Demo
Comprehensive demonstration of RiskMonitor, DrawdownCalculator, and RiskLimitManager
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.risk.risk_monitor import (
    RiskMonitor, RiskThreshold, AlertSeverity, RiskMetricType
)
from src.core.risk.drawdown_calculator import (
    DrawdownCalculator, DrawdownType, DrawdownSeverity
)
from src.core.risk.risk_limits import (
    RiskLimitManager, RiskLimit, LimitType, LimitScope, ActionType
)


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def print_subheader(title):
    """Print formatted subheader"""
    print(f"\n{'-'*40}")
    print(f"{title}")
    print(f"{'-'*40}")


def generate_sample_data():
    """Generate sample market data for demonstration"""
    print("📊 Generating sample market data...")
    
    # Create 2 years of daily data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate correlated returns for multiple assets
    base_return = 0.0005  # 0.05% daily base return
    volatility = 0.015    # 1.5% daily volatility
    
    # Generate returns with some correlation
    returns_eurusd = np.random.normal(base_return, volatility, n_days)
    returns_gbpusd = returns_eurusd * 0.7 + np.random.normal(0, volatility * 0.8, n_days)
    returns_usdjpy = returns_eurusd * -0.3 + np.random.normal(0, volatility * 1.2, n_days)
    
    # Add some market stress periods
    stress_periods = [
        (100, 120),  # First stress period
        (300, 330),  # Second stress period
        (500, 540),  # Third stress period
    ]
    
    for start, end in stress_periods:
        if end < n_days:
            # Increase volatility and add negative bias during stress
            returns_eurusd[start:end] += np.random.normal(-0.002, 0.03, end-start)
            returns_gbpusd[start:end] += np.random.normal(-0.0015, 0.025, end-start)
            returns_usdjpy[start:end] += np.random.normal(-0.001, 0.035, end-start)
    
    # Convert to prices
    prices_eurusd = 1.2000 * (1 + returns_eurusd).cumprod()
    prices_gbpusd = 1.3500 * (1 + returns_gbpusd).cumprod()
    prices_usdjpy = 110.00 * (1 + returns_usdjpy).cumprod()
    
    # Create portfolio (equal weighted)
    portfolio_values = (prices_eurusd + prices_gbpusd + prices_usdjpy/100) * 33333.33
    
    # Create DataFrames
    portfolio_data = pd.DataFrame({
        'EURUSD': prices_eurusd * 33333.33,
        'GBPUSD': prices_gbpusd * 33333.33,
        'USDJPY': prices_usdjpy * 333.33,
        'Portfolio': portfolio_values
    }, index=dates)
    
    # Benchmark (market index)
    benchmark_data = pd.DataFrame({
        'Benchmark': portfolio_values * 0.95  # Slightly underperforming benchmark
    }, index=dates)
    
    print(f"✅ Generated {len(dates)} days of market data")
    print(f"   Portfolio value range: ${portfolio_values.min():,.0f} - ${portfolio_values.max():,.0f}")
    print(f"   Total return: {(portfolio_values[-1]/portfolio_values[0] - 1)*100:.2f}%")
    
    return portfolio_data, benchmark_data


def demo_risk_monitor():
    """Demonstrate Risk Monitor functionality"""
    print_header("RISK MONITOR DEMO")
    
    # Generate sample data
    portfolio_data, benchmark_data = generate_sample_data()
    
    # Initialize Risk Monitor
    config = {
        'monitoring_interval': 60,
        'max_history_size': 1000,
        'enable_real_time': False,  # Disable for demo
        'risk_free_rate': 0.02
    }
    
    risk_monitor = RiskMonitor(config)
    print(f"✅ Risk Monitor initialized with {len(risk_monitor.risk_thresholds)} default thresholds")
    
    # Set portfolio data
    print_subheader("Setting Portfolio Data")
    risk_monitor.set_portfolio_data(portfolio_data, benchmark_data)
    print("✅ Portfolio and benchmark data loaded")
    
    # Calculate real-time metrics
    print_subheader("Real-Time Risk Metrics")
    metrics = risk_monitor.calculate_real_time_metrics()
    
    print(f"📊 Current Risk Metrics:")
    print(f"   Portfolio Value: ${metrics.portfolio_value:,.2f}")
    print(f"   Daily P&L: ${metrics.daily_pnl:,.2f}")
    print(f"   Daily Return: {metrics.daily_return:.4f} ({metrics.daily_return*100:.2f}%)")
    print(f"   VaR 95%: {metrics.var_95:.4f} ({metrics.var_95*100:.2f}%)")
    print(f"   VaR 99%: {metrics.var_99:.4f} ({metrics.var_99*100:.2f}%)")
    print(f"   CVaR 95%: {metrics.cvar_95:.4f} ({metrics.cvar_95*100:.2f}%)")
    print(f"   Current Drawdown: {metrics.current_drawdown:.4f} ({metrics.current_drawdown*100:.2f}%)")
    print(f"   Max Drawdown: {metrics.max_drawdown:.4f} ({metrics.max_drawdown*100:.2f}%)")
    print(f"   Realized Volatility: {metrics.realized_volatility:.4f} ({metrics.realized_volatility*100:.2f}%)")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"   Sortino Ratio: {metrics.sortino_ratio:.4f}")
    print(f"   Calmar Ratio: {metrics.calmar_ratio:.4f}")
    
    # Add custom risk threshold
    print_subheader("Custom Risk Thresholds")
    custom_threshold = RiskThreshold(
        metric_type=RiskMetricType.VOLATILITY,
        metric_name="realized_volatility",
        warning_threshold=0.15,   # 15% annual volatility
        critical_threshold=0.25,  # 25% annual volatility
        emergency_threshold=0.40  # 40% annual volatility
    )
    
    risk_monitor.add_risk_threshold(custom_threshold)
    print(f"✅ Added custom volatility threshold")
    
    # Check risk thresholds
    print_subheader("Risk Threshold Monitoring")
    initial_alerts = len(risk_monitor.active_alerts)
    risk_monitor.check_risk_thresholds(metrics)
    
    print(f"📊 Risk Threshold Check Results:")
    print(f"   Active Alerts: {len(risk_monitor.active_alerts)}")
    print(f"   New Alerts: {len(risk_monitor.active_alerts) - initial_alerts}")
    
    if risk_monitor.active_alerts:
        print(f"   Alert Details:")
        for alert in risk_monitor.active_alerts[-3:]:  # Show last 3 alerts
            print(f"     - {alert.severity.value.upper()}: {alert.message}")
    
    # Test alert callback
    print_subheader("Alert Callback System")
    alert_count = 0
    
    def alert_callback(alert):
        global alert_count
        alert_count += 1
        print(f"🚨 ALERT CALLBACK: {alert.severity.value} - {alert.metric_name}")
    
    risk_monitor.add_alert_callback(alert_callback)
    
    # Create high volatility scenario to trigger alerts
    high_vol_metrics = risk_monitor.calculate_real_time_metrics()
    high_vol_metrics.realized_volatility = 0.30  # 30% volatility to trigger alerts
    
    risk_monitor.check_risk_thresholds(high_vol_metrics)
    print(f"✅ Alert callback system tested")
    
    # Generate dashboard data
    print_subheader("Risk Dashboard Data")
    dashboard_data = risk_monitor.get_risk_dashboard_data()
    
    print(f"📊 Dashboard Data Summary:")
    print(f"   Current Metrics Available: {'Yes' if 'current_metrics' in dashboard_data else 'No'}")
    print(f"   Historical Data Points: {len(dashboard_data.get('historical_data', []))}")
    print(f"   Alert Summary Available: {'Yes' if 'alert_summary' in dashboard_data else 'No'}")
    print(f"   Threshold Status Items: {len(dashboard_data.get('threshold_status', []))}")
    
    # Export risk data
    print_subheader("Data Export")
    export_file = "risk_monitor_export.json"
    success = risk_monitor.export_risk_data(export_file)
    print(f"✅ Risk data exported: {success}")
    
    if success and os.path.exists(export_file):
        file_size = os.path.getsize(export_file)
        print(f"   Export file size: {file_size:,} bytes")
        os.remove(export_file)  # Cleanup
    
    # Statistics
    print_subheader("Risk Monitor Statistics")
    stats = risk_monitor.get_statistics()
    print(f"📊 Risk Monitor Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return risk_monitor


def demo_drawdown_calculator():
    """Demonstrate Drawdown Calculator functionality"""
    print_header("DRAWDOWN CALCULATOR DEMO")
    
    # Generate sample data with known drawdowns
    portfolio_data, _ = generate_sample_data()
    
    # Initialize Drawdown Calculator
    config = {
        'min_drawdown_threshold': 0.02,  # 2% minimum drawdown
        'lookback_window': 252,
        'rolling_window': 30
    }
    
    drawdown_calc = DrawdownCalculator(config)
    print(f"✅ Drawdown Calculator initialized")
    
    # Set data
    print_subheader("Setting Price Data")
    drawdown_calc.set_data(portfolio_data['Portfolio'])
    print(f"✅ Price data loaded: {len(drawdown_calc.price_data)} observations")
    
    # Calculate different types of drawdowns
    print_subheader("Drawdown Calculations")
    
    # Relative drawdown
    relative_dd = drawdown_calc.calculate_drawdown(DrawdownType.RELATIVE)
    print(f"📊 Relative Drawdown:")
    print(f"   Current: {abs(relative_dd.iloc[-1]):.4f} ({abs(relative_dd.iloc[-1])*100:.2f}%)")
    print(f"   Maximum: {abs(relative_dd.min()):.4f} ({abs(relative_dd.min())*100:.2f}%)")
    print(f"   Average: {abs(relative_dd.mean()):.4f} ({abs(relative_dd.mean())*100:.2f}%)")
    
    # Absolute drawdown
    absolute_dd = drawdown_calc.calculate_drawdown(DrawdownType.ABSOLUTE)
    print(f"📊 Absolute Drawdown:")
    print(f"   Current: ${absolute_dd.iloc[-1]:,.2f}")
    print(f"   Maximum: ${absolute_dd.min():,.2f}")
    
    # Rolling drawdown
    rolling_dd = drawdown_calc.calculate_rolling_drawdown(window=30)
    print(f"📊 Rolling 30-Day Max Drawdown:")
    print(f"   Current: {rolling_dd.iloc[-1]:.4f} ({rolling_dd.iloc[-1]*100:.2f}%)")
    print(f"   Average: {rolling_dd.mean():.4f} ({rolling_dd.mean()*100:.2f}%)")
    
    # Identify drawdown periods
    print_subheader("Drawdown Period Analysis")
    periods = drawdown_calc.identify_drawdown_periods(min_threshold=0.01)
    
    print(f"📊 Drawdown Periods Identified: {len(periods)}")
    
    if periods:
        # Severity distribution
        severity_counts = {}
        for period in periods:
            severity = period.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        print(f"   Severity Distribution:")
        for severity, count in severity_counts.items():
            print(f"     {severity.title()}: {count}")
        
        # Show worst drawdowns
        worst_periods = sorted(periods, key=lambda x: x.max_drawdown, reverse=True)[:3]
        print(f"   Top 3 Worst Drawdowns:")
        for i, period in enumerate(worst_periods, 1):
            print(f"     {i}. {period.max_drawdown:.2%} over {period.duration_days} days")
            print(f"        From {period.start_date.strftime('%Y-%m-%d')} to {period.end_date.strftime('%Y-%m-%d') if period.end_date else 'Ongoing'}")
            print(f"        Recovery: {'Yes' if period.is_recovered else 'No'}")
    
    # Calculate comprehensive statistics
    print_subheader("Drawdown Statistics")
    stats = drawdown_calc.calculate_statistics()
    
    print(f"📊 Comprehensive Drawdown Statistics:")
    print(f"   Current Drawdown: {stats.current_drawdown:.4f} ({stats.current_drawdown*100:.2f}%)")
    print(f"   Maximum Drawdown: {stats.max_drawdown:.4f} ({stats.max_drawdown*100:.2f}%)")
    print(f"   Average Drawdown: {stats.average_drawdown:.4f} ({stats.average_drawdown*100:.2f}%)")
    print(f"   Max Drawdown Duration: {stats.max_drawdown_duration} days")
    print(f"   Average Duration: {stats.average_duration:.1f} days")
    print(f"   Recovery Factor: {stats.recovery_factor:.2f}")
    print(f"   Pain Index: {stats.pain_index:.4f} ({stats.pain_index*100:.2f}%)")
    print(f"   Ulcer Index: {stats.ulcer_index:.4f}")
    print(f"   Recovery Rate: {stats.recovery_rate:.2%}")
    print(f"   Drawdowns per Year: {stats.drawdowns_per_year:.1f}")
    
    # Current drawdown information
    print_subheader("Current Drawdown Status")
    current_info = drawdown_calc.get_current_drawdown_info()
    
    print(f"📊 Current Drawdown Information:")
    print(f"   Current Drawdown: {current_info['current_drawdown']:.4f} ({current_info['current_drawdown']*100:.2f}%)")
    print(f"   Current Price: ${current_info['current_price']:,.2f}")
    print(f"   Peak Price: ${current_info['peak_price']:,.2f}")
    print(f"   Peak Date: {current_info['peak_date']}")
    print(f"   Duration: {current_info['duration_days']} days")
    print(f"   Severity: {current_info['severity'].title()}")
    print(f"   In Drawdown: {'Yes' if current_info['is_in_drawdown'] else 'No'}")
    print(f"   Recovery Needed: {current_info['recovery_needed']:.2%}")
    
    # Generate comprehensive report
    print_subheader("Drawdown Report Generation")
    report = drawdown_calc.generate_drawdown_report()
    
    print(f"📊 Drawdown Report Generated:")
    print(f"   Report Timestamp: {report['report_timestamp']}")
    print(f"   Data Period: {report['data_period']['start_date']} to {report['data_period']['end_date']}")
    print(f"   Total Observations: {report['data_period']['total_observations']:,}")
    print(f"   Summary Available: {'Yes' if 'summary' in report else 'No'}")
    
    # Export data
    print_subheader("Data Export")
    export_file = "drawdown_export.json"
    success = drawdown_calc.export_data(export_file)
    print(f"✅ Drawdown data exported: {success}")
    
    if success and os.path.exists(export_file):
        file_size = os.path.getsize(export_file)
        print(f"   Export file size: {file_size:,} bytes")
        os.remove(export_file)  # Cleanup
    
    return drawdown_calc


def demo_risk_limit_manager():
    """Demonstrate Risk Limit Manager functionality"""
    print_header("RISK LIMIT MANAGER DEMO")
    
    # Initialize Risk Limit Manager
    config = {
        'monitoring_interval': 30,
        'enable_enforcement': False,  # Disable actual enforcement for demo
        'max_breach_history': 100
    }
    
    limit_manager = RiskLimitManager(config)
    print(f"✅ Risk Limit Manager initialized with {len(limit_manager.risk_limits)} default limits")
    
    # Show default limits
    print_subheader("Default Risk Limits")
    print(f"📊 Default Risk Limits:")
    for limit_id, limit in limit_manager.risk_limits.items():
        print(f"   {limit.name}:")
        print(f"     Type: {limit.limit_type.value}")
        print(f"     Scope: {limit.scope.value}")
        print(f"     Soft Limit: {limit.soft_limit}")
        print(f"     Hard Limit: {limit.hard_limit}")
        if limit.emergency_limit:
            print(f"     Emergency Limit: {limit.emergency_limit}")
        print()
    
    # Add custom risk limit
    print_subheader("Custom Risk Limits")
    custom_limit = RiskLimit(
        limit_id="custom_eurusd_position",
        name="EURUSD Position Limit",
        limit_type=LimitType.POSITION_SIZE,
        scope=LimitScope.SYMBOL,
        soft_limit=0.08,    # 8% of portfolio
        hard_limit=0.15,    # 15% of portfolio
        emergency_limit=0.25,  # 25% of portfolio
        scope_filter="EURUSD",
        is_percentage=True,
        soft_action=ActionType.ALERT_ONLY,
        hard_action=ActionType.REDUCE_POSITION,
        emergency_action=ActionType.CLOSE_POSITION
    )
    
    success = limit_manager.add_risk_limit(custom_limit)
    print(f"✅ Custom EURUSD limit added: {success}")
    
    # Test different market scenarios
    print_subheader("Market Scenario Testing")
    
    scenarios = [
        {
            'name': 'Normal Trading',
            'positions': {'EURUSD': 5000, 'GBPUSD': -3000, 'USDJPY': 2000},
            'portfolio_value': 100000,
            'daily_pnl': 250,
            'var': 1500,
            'drawdown': 0.01
        },
        {
            'name': 'High Risk Scenario',
            'positions': {'EURUSD': 25000, 'GBPUSD': -15000, 'USDJPY': 20000},
            'portfolio_value': 100000,
            'daily_pnl': -3500,
            'var': 8000,
            'drawdown': 0.08
        },
        {
            'name': 'Emergency Scenario',
            'positions': {'EURUSD': 45000, 'GBPUSD': -30000, 'USDJPY': 35000},
            'portfolio_value': 100000,
            'daily_pnl': -12000,
            'var': 18000,
            'drawdown': 0.22
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 Testing Scenario: {scenario['name']}")
        
        # Update market data
        limit_manager.update_market_data(
            scenario['positions'],
            scenario['portfolio_value'],
            scenario['daily_pnl'],
            scenario['var'],
            scenario['drawdown']
        )
        
        # Check limits
        breaches = limit_manager.check_all_limits()
        
        print(f"   Portfolio Value: ${scenario['portfolio_value']:,}")
        print(f"   Daily P&L: ${scenario['daily_pnl']:,}")
        print(f"   VaR: ${scenario['var']:,}")
        print(f"   Drawdown: {scenario['drawdown']:.2%}")
        print(f"   Position Exposure: ${sum(abs(p) for p in scenario['positions'].values()):,}")
        print(f"   Limit Breaches: {len(breaches)}")
        
        if breaches:
            print(f"   Breach Details:")
            for breach in breaches:
                print(f"     - {breach.limit_name}: {breach.current_value:.4f} > {breach.limit_value:.4f}")
                print(f"       Action: {breach.action_taken.value}")
                print(f"       Magnitude: {breach.breach_percentage:.1f}% over limit")
    
    # Test breach callbacks
    print_subheader("Breach Callback System")
    breach_count = 0
    
    def breach_callback(breach):
        global breach_count
        breach_count += 1
        print(f"🚨 BREACH CALLBACK: {breach.limit_name} - Action: {breach.action_taken.value}")
    
    def action_callback(breach):
        print(f"⚡ ACTION CALLBACK: Executed {breach.action_taken.value} for {breach.limit_name}")
    
    limit_manager.add_breach_callback(breach_callback)
    limit_manager.add_action_callback(action_callback)
    
    # Trigger callbacks with high-risk scenario
    limit_manager.update_market_data(
        {'EURUSD': 50000, 'GBPUSD': -40000},
        100000, -15000, 20000, 0.25
    )
    
    breaches = limit_manager.check_all_limits()
    print(f"✅ Callback system tested with {len(breaches)} breaches")
    
    # Generate limit status report
    print_subheader("Limit Status Report")
    report = limit_manager.get_limit_status_report()
    
    print(f"📊 Limit Status Report:")
    print(f"   Timestamp: {report['timestamp']}")
    print(f"   Total Limits: {report['summary']['total_limits']}")
    print(f"   Active Limits: {report['summary']['active_limits']}")
    print(f"   Breached Limits: {report['summary']['breached_limits']}")
    print(f"   Breaches Today: {report['summary']['total_breaches_today']}")
    print(f"   Monitoring Status: {'Active' if report['summary']['monitoring_status'] else 'Inactive'}")
    
    # Show limit utilization
    if report['limit_utilization']:
        print(f"\n   Top Limit Utilizations:")
        utilizations = sorted(
            report['limit_utilization'].items(),
            key=lambda x: x[1]['utilization_pct'],
            reverse=True
        )[:5]
        
        for limit_id, util in utilizations:
            print(f"     {util['name']}: {util['utilization_pct']:.1f}% ({util['status']})")
    
    # Test limit management operations
    print_subheader("Limit Management Operations")
    
    # Update existing limit
    update_success = limit_manager.update_risk_limit(
        "custom_eurusd_position",
        {'soft_limit': 0.06, 'hard_limit': 0.12}
    )
    print(f"✅ Limit update: {update_success}")
    
    # Remove limit
    remove_success = limit_manager.remove_risk_limit("custom_eurusd_position")
    print(f"✅ Limit removal: {remove_success}")
    
    # Export data
    print_subheader("Data Export")
    export_file = "risk_limits_export.json"
    success = limit_manager.export_data(export_file)
    print(f"✅ Risk limits data exported: {success}")
    
    if success and os.path.exists(export_file):
        file_size = os.path.getsize(export_file)
        print(f"   Export file size: {file_size:,} bytes")
        os.remove(export_file)  # Cleanup
    
    return limit_manager


def demo_integrated_system():
    """Demonstrate integrated risk monitoring system"""
    print_header("INTEGRATED RISK MONITORING SYSTEM")
    
    # Initialize all components
    print_subheader("System Initialization")
    
    risk_monitor = RiskMonitor({'enable_real_time': False})
    drawdown_calc = DrawdownCalculator()
    limit_manager = RiskLimitManager({'enable_enforcement': False})
    
    print("✅ All risk monitoring components initialized")
    
    # Generate and set data
    portfolio_data, benchmark_data = generate_sample_data()
    
    risk_monitor.set_portfolio_data(portfolio_data, benchmark_data)
    drawdown_calc.set_data(portfolio_data['Portfolio'])
    
    print("✅ Market data loaded into all systems")
    
    # Integrated risk assessment
    print_subheader("Integrated Risk Assessment")
    
    # Calculate metrics from all systems
    risk_metrics = risk_monitor.calculate_real_time_metrics()
    
    drawdown_calc.calculate_drawdown()
    drawdown_stats = drawdown_calc.calculate_statistics()
    current_dd_info = drawdown_calc.get_current_drawdown_info()
    
    # Simulate current positions
    current_positions = {
        'EURUSD': 15000,
        'GBPUSD': -8000,
        'USDJPY': 12000
    }
    
    limit_manager.update_market_data(
        current_positions,
        risk_metrics.portfolio_value,
        risk_metrics.daily_pnl,
        risk_metrics.var_95 * risk_metrics.portfolio_value,
        risk_metrics.current_drawdown
    )
    
    limit_breaches = limit_manager.check_all_limits()
    
    print(f"📊 Integrated Risk Assessment:")
    print(f"   Portfolio Value: ${risk_metrics.portfolio_value:,.2f}")
    print(f"   Daily Return: {risk_metrics.daily_return:.2%}")
    print(f"   VaR 95%: {risk_metrics.var_95:.2%}")
    print(f"   Current Drawdown: {risk_metrics.current_drawdown:.2%}")
    print(f"   Max Drawdown: {drawdown_stats.max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"   Active Risk Alerts: {len(risk_monitor.active_alerts)}")
    print(f"   Limit Breaches: {len(limit_breaches)}")
    print(f"   Drawdown Periods: {drawdown_stats.total_drawdown_periods}")
    
    # Risk scoring
    print_subheader("Risk Scoring System")
    
    # Calculate composite risk score
    risk_score = 0
    risk_factors = []
    
    # VaR component (0-25 points)
    var_score = min(25, risk_metrics.var_95 * 500)  # Scale VaR
    risk_score += var_score
    risk_factors.append(f"VaR: {var_score:.1f}/25")
    
    # Drawdown component (0-25 points)
    dd_score = min(25, risk_metrics.current_drawdown * 250)  # Scale drawdown
    risk_score += dd_score
    risk_factors.append(f"Drawdown: {dd_score:.1f}/25")
    
    # Volatility component (0-25 points)
    vol_score = min(25, risk_metrics.realized_volatility * 100)  # Scale volatility
    risk_score += vol_score
    risk_factors.append(f"Volatility: {vol_score:.1f}/25")
    
    # Limit breach component (0-25 points)
    breach_score = min(25, len(limit_breaches) * 5)  # 5 points per breach
    risk_score += breach_score
    risk_factors.append(f"Breaches: {breach_score:.1f}/25")
    
    # Risk level classification
    if risk_score < 20:
        risk_level = "LOW"
        risk_color = "🟢"
    elif risk_score < 40:
        risk_level = "MEDIUM"
        risk_color = "🟡"
    elif risk_score < 70:
        risk_level = "HIGH"
        risk_color = "🟠"
    else:
        risk_level = "CRITICAL"
        risk_color = "🔴"
    
    print(f"📊 Composite Risk Score: {risk_score:.1f}/100 {risk_color}")
    print(f"   Risk Level: {risk_level}")
    print(f"   Risk Factors:")
    for factor in risk_factors:
        print(f"     - {factor}")
    
    # Risk recommendations
    print_subheader("Risk Management Recommendations")
    
    recommendations = []
    
    if risk_metrics.var_95 > 0.03:
        recommendations.append("🔸 Consider reducing position sizes (VaR > 3%)")
    
    if risk_metrics.current_drawdown > 0.05:
        recommendations.append("🔸 Monitor drawdown closely (Current DD > 5%)")
    
    if risk_metrics.realized_volatility > 0.25:
        recommendations.append("🔸 High volatility detected - consider hedging")
    
    if len(limit_breaches) > 0:
        recommendations.append(f"🔸 Address {len(limit_breaches)} limit breach(es)")
    
    if risk_metrics.sharpe_ratio < 0.5:
        recommendations.append("🔸 Poor risk-adjusted returns - review strategy")
    
    if drawdown_stats.recovery_rate < 0.8:
        recommendations.append("🔸 Low recovery rate - improve risk management")
    
    if not recommendations:
        recommendations.append("✅ Risk profile appears healthy")
    
    print(f"📋 Risk Management Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    # System performance summary
    print_subheader("System Performance Summary")
    
    risk_stats = risk_monitor.get_statistics()
    dd_stats = drawdown_calc.get_statistics()
    limit_stats = limit_manager.get_statistics()
    
    print(f"📊 System Performance:")
    print(f"   Risk Monitor:")
    print(f"     - Metrics Calculated: {risk_stats['total_metrics_calculated']}")
    print(f"     - Thresholds Configured: {risk_stats['risk_thresholds_configured']}")
    print(f"     - Active Alerts: {risk_stats['active_alerts']}")
    
    print(f"   Drawdown Calculator:")
    print(f"     - Price Data Length: {dd_stats['price_data_length']:,}")
    print(f"     - Drawdown Periods: {dd_stats['drawdown_periods_identified']}")
    print(f"     - Series Calculated: {'Yes' if dd_stats['drawdown_series_calculated'] else 'No'}")
    
    print(f"   Limit Manager:")
    print(f"     - Total Limits: {limit_stats['total_limits']}")
    print(f"     - Active Limits: {limit_stats['active_limits']}")
    print(f"     - Total Breaches: {limit_stats['total_breaches']}")
    print(f"     - Breaches Today: {limit_stats['breaches_today']}")
    
    print(f"\n✅ Integrated risk monitoring system demonstration completed!")
    
    return {
        'risk_monitor': risk_monitor,
        'drawdown_calc': drawdown_calc,
        'limit_manager': limit_manager,
        'risk_score': risk_score,
        'risk_level': risk_level
    }


def main():
    """Main demo function"""
    print_header("RISK MONITORING SYSTEM COMPREHENSIVE DEMO")
    print("🚀 Starting comprehensive demonstration of Risk Monitoring System")
    print("   Components: RiskMonitor, DrawdownCalculator, RiskLimitManager")
    
    try:
        # Individual component demos
        print("\n" + "="*60)
        print("PHASE 1: INDIVIDUAL COMPONENT DEMONSTRATIONS")
        print("="*60)
        
        risk_monitor = demo_risk_monitor()
        drawdown_calc = demo_drawdown_calculator()
        limit_manager = demo_risk_limit_manager()
        
        # Integrated system demo
        print("\n" + "="*60)
        print("PHASE 2: INTEGRATED SYSTEM DEMONSTRATION")
        print("="*60)
        
        integrated_results = demo_integrated_system()
        
        # Final summary
        print_header("DEMO COMPLETION SUMMARY")
        print("🎉 Risk Monitoring System Demo Completed Successfully!")
        print(f"   Final Risk Score: {integrated_results['risk_score']:.1f}/100")
        print(f"   Risk Level: {integrated_results['risk_level']}")
        print("\n📊 All components demonstrated:")
        print("   ✅ Risk Monitor - Real-time risk metrics and alerting")
        print("   ✅ Drawdown Calculator - Comprehensive drawdown analysis")
        print("   ✅ Risk Limit Manager - Automated limit enforcement")
        print("   ✅ Integrated System - Unified risk management")
        
        print(f"\n🔧 Key Features Demonstrated:")
        print("   • Real-time risk metrics calculation")
        print("   • Multi-level alert system")
        print("   • Comprehensive drawdown analysis")
        print("   • Automated limit monitoring")
        print("   • Risk scoring and recommendations")
        print("   • Data export capabilities")
        print("   • Callback systems")
        print("   • Integration workflows")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 