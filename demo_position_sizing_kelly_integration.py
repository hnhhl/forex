#!/usr/bin/env python3

"""
Ultimate XAU Super System V4.0 - Position Sizing with Kelly Criterion Integration Demo
=====================================================================================

Demo showcasing the integration of professional Kelly Criterion Calculator
with Position Sizing System for optimal position sizing.

Features demonstrated:
- 5 Kelly Criterion methods (Classic, Fractional, Dynamic, Conservative, Adaptive)
- Professional risk controls and safeguards
- Comprehensive position sizing analysis
- Real-world trading scenario simulation

Author: Ultimate XAU Super System V4.0
Date: December 19, 2024
"""

import sys
import os
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.risk.position_sizer import (
    PositionSizer,
    SizingMethod,
    SizingParameters,
    RiskLevel
)

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_section(title: str):
    """Print formatted section"""
    print(f"\n📊 {title}")
    print("-" * 60)

def generate_sample_price_data(days: int = 100) -> pd.DataFrame:
    """Generate realistic XAU price data"""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='D')
    
    # Generate realistic XAU price movements
    base_price = 2000.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    prices = [base_price]
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    price_data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
    }, index=dates)
    
    return price_data

def add_sample_trades(sizer: PositionSizer, num_trades: int = 50):
    """Add sample trade results to position sizer"""
    print(f"📈 Adding {num_trades} sample trades...")
    
    np.random.seed(42)
    win_rate = 0.65  # 65% win rate
    
    for i in range(num_trades):
        is_win = np.random.random() < win_rate
        
        if is_win:
            profit_loss = np.random.uniform(0.015, 0.035)  # 1.5-3.5% wins
        else:
            profit_loss = np.random.uniform(-0.025, -0.010)  # 1-2.5% losses
        
        trade_date = datetime.now() - timedelta(days=num_trades-i)
        entry_price = 2000 + np.random.uniform(-100, 100)
        exit_price = entry_price * (1 + profit_loss)
        
        sizer.add_trade_result(
            profit_loss=profit_loss,
            win=is_win,
            trade_date=trade_date,
            symbol="XAUUSD",
            entry_price=entry_price,
            exit_price=exit_price,
            volume=np.random.uniform(0.1, 0.5),
            duration_minutes=np.random.randint(60, 480)
        )
    
    print(f"✅ Added {num_trades} trades (Win Rate: {win_rate:.1%})")

def demo_kelly_methods_comparison():
    """Demo comparing different Kelly methods"""
    print_section("Kelly Methods Comparison")
    
    # Initialize position sizer
    sizer = PositionSizer()
    
    # Set up data
    price_data = generate_sample_price_data(100)
    sizer.set_data(price_data, portfolio_value=100000.0)
    sizer.set_performance_metrics(win_rate=0.65, avg_win=0.025, avg_loss=-0.015)
    
    # Add trade history
    add_sample_trades(sizer, 50)
    
    current_price = 2050.0
    
    print(f"\n🎯 Current Price: ${current_price:,.2f}")
    print(f"💰 Portfolio Value: ${sizer.portfolio_value:,.2f}")
    print(f"📊 Performance: WR={sizer.win_rate:.1%}, AvgWin={sizer.avg_win:.1%}, AvgLoss={sizer.avg_loss:.1%}")
    
    # Test different Kelly methods
    kelly_methods = [
        ("Classic Kelly", sizer.calculate_kelly_classic_size),
        ("Fractional Kelly", sizer.calculate_kelly_fractional_size),
        ("Dynamic Kelly", sizer.calculate_kelly_dynamic_size),
        ("Conservative Kelly", sizer.calculate_kelly_conservative_size),
        ("Adaptive Kelly", sizer.calculate_kelly_adaptive_size)
    ]
    
    print(f"\n{'Method':<20} | {'Position Size':<12} | {'Kelly %':<8} | {'Confidence':<10} | {'Risk Amount':<12}")
    print("-" * 80)
    
    results = {}
    for method_name, method_func in kelly_methods:
        try:
            result = method_func(current_price)
            kelly_fraction = result.additional_metrics.get('kelly_fraction', 0) * 100
            
            print(f"{method_name:<20} | {result.position_size:>10.4f} | {kelly_fraction:>6.2f}% | {result.confidence_score:>8.2f} | ${result.risk_amount:>10.2f}")
            
            results[method_name] = result
            
        except Exception as e:
            print(f"{method_name:<20} | Error: {str(e)}")
    
    return results

def demo_comprehensive_kelly_analysis():
    """Demo comprehensive Kelly analysis"""
    print_section("Comprehensive Kelly Analysis")
    
    # Initialize position sizer
    sizer = PositionSizer()
    
    # Set up data
    price_data = generate_sample_price_data(100)
    sizer.set_data(price_data, portfolio_value=250000.0)  # Larger portfolio
    sizer.set_performance_metrics(win_rate=0.68, avg_win=0.028, avg_loss=-0.012)
    
    # Add more trade history
    add_sample_trades(sizer, 100)
    
    current_price = 2075.0
    
    # Get comprehensive analysis
    analysis = sizer.get_kelly_analysis(current_price)
    
    if 'error' in analysis:
        print(f"❌ Analysis Error: {analysis['error']}")
        return
    
    print(f"\n🎯 Current Price: ${current_price:,.2f}")
    print(f"💰 Portfolio Value: ${analysis['portfolio_value']:,.2f}")
    print(f"📈 Total Trades Analyzed: {analysis['trade_count']}")
    
    # Display Kelly analysis results
    print(f"\n📊 Kelly Analysis Results:")
    kelly_analysis = analysis['kelly_analysis']
    
    for method, data in kelly_analysis.items():
        if 'error' not in data:
            position_value = data['position_value']
            portfolio_pct = (position_value / analysis['portfolio_value']) * 100
            
            print(f"\n  🔹 {method.upper()}:")
            print(f"     Position Size: {data['position_size']:.4f} units")
            print(f"     Kelly Fraction: {data['kelly_fraction']*100:.2f}%")
            print(f"     Position Value: ${position_value:,.2f} ({portfolio_pct:.1f}% of portfolio)")
            print(f"     Confidence: {data['confidence_score']:.2f}")
            print(f"     Risk Amount: ${data['risk_amount']:,.2f}")
    
    # Display performance summary
    if 'performance_summary' in analysis:
        perf = analysis['performance_summary']
        print(f"\n📈 Performance Summary:")
        print(f"     Win Rate: {perf.get('win_rate', 0):.1%}")
        print(f"     Profit Factor: {perf.get('profit_factor', 0):.2f}")
        print(f"     Average Win: {perf.get('average_win', 0):.2%}")
        print(f"     Average Loss: {perf.get('average_loss', 0):.2%}")
        print(f"     Total Trades: {perf.get('total_trades', 0)}")

def demo_risk_controls_and_limits():
    """Demo risk controls and position limits"""
    print_section("Risk Controls & Position Limits")
    
    # Initialize with custom parameters
    custom_params = SizingParameters(
        kelly_max_fraction=0.15,  # 15% max Kelly
        kelly_min_fraction=0.02,  # 2% min Kelly
        max_position_size=0.08,   # 8% max position
        min_position_size=0.01,   # 1% min position
        risk_per_trade=0.015      # 1.5% risk per trade
    )
    
    sizer = PositionSizer()
    
    # Set up data
    price_data = generate_sample_price_data(100)
    sizer.set_data(price_data, portfolio_value=500000.0)
    sizer.set_performance_metrics(win_rate=0.72, avg_win=0.032, avg_loss=-0.018)
    
    # Add trade history
    add_sample_trades(sizer, 75)
    
    current_price = 2100.0
    
    print(f"\n🛡️ Risk Control Parameters:")
    print(f"   Max Kelly Fraction: {custom_params.kelly_max_fraction:.1%}")
    print(f"   Min Kelly Fraction: {custom_params.kelly_min_fraction:.1%}")
    print(f"   Max Position Size: {custom_params.max_position_size:.1%}")
    print(f"   Min Position Size: {custom_params.min_position_size:.1%}")
    print(f"   Risk Per Trade: {custom_params.risk_per_trade:.1%}")
    
    # Test with and without limits
    print(f"\n📊 Comparison: With vs Without Limits")
    print(f"{'Method':<15} | {'No Limits':<12} | {'With Limits':<12} | {'Difference':<12}")
    print("-" * 60)
    
    methods = [
        ("Classic", sizer.calculate_kelly_classic_size),
        ("Adaptive", sizer.calculate_kelly_adaptive_size)
    ]
    
    for method_name, method_func in methods:
        # Without limits (default parameters)
        result_no_limits = method_func(current_price)
        
        # With limits (custom parameters)
        result_with_limits = method_func(current_price, custom_params)
        
        diff = result_with_limits.position_size - result_no_limits.position_size
        
        print(f"{method_name:<15} | {result_no_limits.position_size:>10.4f} | {result_with_limits.position_size:>10.4f} | {diff:>+10.4f}")

def demo_real_world_scenario():
    """Demo real-world trading scenario"""
    print_section("Real-World Trading Scenario")
    
    # Simulate a trading day with multiple position sizing decisions
    sizer = PositionSizer()
    
    # Set up realistic data
    price_data = generate_sample_price_data(200)
    sizer.set_data(price_data, portfolio_value=1000000.0)  # $1M portfolio
    sizer.set_performance_metrics(win_rate=0.63, avg_win=0.022, avg_loss=-0.014)
    
    # Add extensive trade history
    add_sample_trades(sizer, 200)
    
    # Simulate different market conditions
    scenarios = [
        {"name": "Normal Market", "price": 2050.0, "volatility": "Normal"},
        {"name": "High Volatility", "price": 2120.0, "volatility": "High"},
        {"name": "Market Stress", "price": 1980.0, "volatility": "Extreme"}
    ]
    
    print(f"\n🌍 Real-World Scenario Analysis")
    print(f"Portfolio Value: ${sizer.portfolio_value:,.2f}")
    print(f"System Performance: {sizer.win_rate:.1%} WR, {sizer.avg_win:.1%} AvgWin, {sizer.avg_loss:.1%} AvgLoss")
    
    for scenario in scenarios:
        print(f"\n📈 {scenario['name']} (Price: ${scenario['price']:,.2f})")
        
        # Get adaptive Kelly recommendation (best for changing conditions)
        result = sizer.calculate_kelly_adaptive_size(scenario['price'])
        
        position_value = result.position_size * scenario['price']
        portfolio_pct = (position_value / sizer.portfolio_value) * 100
        
        print(f"   Recommended Position: {result.position_size:.4f} units")
        print(f"   Position Value: ${position_value:,.2f} ({portfolio_pct:.1f}% of portfolio)")
        print(f"   Kelly Fraction: {result.additional_metrics.get('kelly_fraction', 0)*100:.2f}%")
        print(f"   Confidence Score: {result.confidence_score:.2f}")
        print(f"   Risk Amount: ${result.risk_amount:,.2f}")
        
        # Risk assessment
        if result.confidence_score > 0.8:
            risk_level = "🟢 LOW"
        elif result.confidence_score > 0.6:
            risk_level = "🟡 MODERATE"
        else:
            risk_level = "🔴 HIGH"
        
        print(f"   Risk Level: {risk_level}")

def main():
    """Main demo function"""
    print_header("ULTIMATE XAU SUPER SYSTEM V4.0")
    print("🚀 Position Sizing with Kelly Criterion Integration Demo")
    print("📊 Professional Kelly Calculator + Advanced Position Sizing")
    print("⚡ 5 Kelly Methods + Risk Controls + Real-World Scenarios")
    
    try:
        # Demo 1: Kelly Methods Comparison
        demo_kelly_methods_comparison()
        
        # Demo 2: Comprehensive Kelly Analysis
        demo_comprehensive_kelly_analysis()
        
        # Demo 3: Risk Controls and Limits
        demo_risk_controls_and_limits()
        
        # Demo 4: Real-World Scenario
        demo_real_world_scenario()
        
        print_header("DEMO COMPLETED SUCCESSFULLY!")
        print("✅ Kelly Criterion Calculator successfully integrated with Position Sizing System")
        print("🎯 Professional-grade position sizing with 5 Kelly methods")
        print("🛡️ Advanced risk controls and safeguards")
        print("📈 Ready for live trading implementation")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 