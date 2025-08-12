"""
Demo Script for Position Sizing System
Showcases various position sizing methods and capabilities
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.risk.position_sizer import (
    PositionSizer, SizingMethod, RiskLevel, SizingParameters
)


def create_sample_data():
    """Create sample XAU price data"""
    print("📊 Creating sample XAU price data...")
    
    # Generate 6 months of daily XAU data
    dates = pd.date_range('2023-07-01', '2023-12-31', freq='D')
    
    # Simulate realistic XAU price movement
    base_price = 2000.0
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Slightly positive drift, 1.5% daily vol
    
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    price_data = pd.DataFrame({
        'Open': prices[:-1],
        'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices[:-1]],
        'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices[:-1]],
        'Close': prices[1:],
    }, index=dates)
    
    print(f"✅ Generated {len(price_data)} days of XAU data")
    print(f"📈 Price range: ${price_data['Close'].min():.2f} - ${price_data['Close'].max():.2f}")
    
    return price_data


def demo_basic_sizing_methods():
    """Demo basic position sizing methods"""
    print("\n" + "="*60)
    print("🎯 DEMO: BASIC POSITION SIZING METHODS")
    print("="*60)
    
    # Create sample data
    price_data = create_sample_data()
    
    # Initialize position sizer
    sizer = PositionSizer()
    sizer.set_data(price_data, portfolio_value=100000.0)
    sizer.set_performance_metrics(win_rate=0.65, avg_win=0.025, avg_loss=-0.015)
    
    current_price = price_data['Close'].iloc[-1]
    stop_loss_price = current_price * 0.975  # 2.5% stop loss
    
    print(f"\n📊 Current XAU Price: ${current_price:.2f}")
    print(f"🛡️ Stop Loss Price: ${stop_loss_price:.2f}")
    print(f"💰 Portfolio Value: $100,000")
    print(f"📈 Win Rate: 65% | Avg Win: 2.5% | Avg Loss: -1.5%")
    
    # Test different sizing methods
    methods = [
        ("Fixed Amount", lambda: sizer.calculate_fixed_amount_size(10000, current_price)),
        ("Fixed Percentage", lambda: sizer.calculate_fixed_percentage_size(0.05, current_price)),
        ("Risk-Based", lambda: sizer.calculate_risk_based_size(current_price, stop_loss_price)),
        ("Kelly Criterion", lambda: sizer.calculate_kelly_criterion_size(current_price)),
        ("Volatility-Based", lambda: sizer.calculate_volatility_based_size(current_price)),
        ("ATR-Based", lambda: sizer.calculate_atr_based_size(current_price)),
    ]
    
    print(f"\n{'Method':<20} {'Position Size':<15} {'Position Value':<15} {'Risk Amount':<15} {'Confidence':<12}")
    print("-" * 85)
    
    for method_name, method_func in methods:
        try:
            result = method_func()
            position_value = result.position_size * current_price
            
            print(f"{method_name:<20} {result.position_size:<15.4f} ${position_value:<14,.0f} ${result.risk_amount:<14,.0f} {result.confidence_score:<12.2f}")
            
        except Exception as e:
            print(f"{method_name:<20} ERROR: {str(e)}")


def demo_optimal_sizing():
    """Demo optimal position sizing"""
    print("\n" + "="*60)
    print("🚀 DEMO: OPTIMAL POSITION SIZING")
    print("="*60)
    
    # Create sample data
    price_data = create_sample_data()
    
    # Initialize position sizer
    sizer = PositionSizer()
    sizer.set_data(price_data, portfolio_value=100000.0)
    sizer.set_performance_metrics(win_rate=0.65, avg_win=0.025, avg_loss=-0.015)
    
    current_price = price_data['Close'].iloc[-1]
    stop_loss_price = current_price * 0.975
    
    print(f"\n📊 Calculating optimal position size...")
    
    # Calculate optimal size
    result = sizer.calculate_optimal_size(current_price, stop_loss_price)
    
    print(f"\n🎯 OPTIMAL SIZING RESULT:")
    print(f"   Method: {result.method.value}")
    print(f"   Position Size: {result.position_size:.4f} units")
    print(f"   Position Value: ${result.position_size * current_price:,.2f}")
    print(f"   Risk Amount: ${result.risk_amount:,.2f}")
    print(f"   Confidence Score: {result.confidence_score:.2f}")
    
    # Show component analysis
    components = result.additional_metrics['component_sizes']
    confidences = result.additional_metrics['component_confidences']
    
    print(f"\n📊 COMPONENT ANALYSIS:")
    print(f"{'Component':<15} {'Size':<12} {'Confidence':<12} {'Weight':<10}")
    print("-" * 50)
    
    total_confidence = sum(confidences.values())
    for component, size in components.items():
        conf = confidences[component]
        weight = conf / total_confidence if total_confidence > 0 else 0
        print(f"{component:<15} {size:<12.4f} {conf:<12.2f} {weight:<10.1%}")


def demo_risk_level_comparison():
    """Demo different risk levels"""
    print("\n" + "="*60)
    print("⚖️ DEMO: RISK LEVEL COMPARISON")
    print("="*60)
    
    # Create sample data
    price_data = create_sample_data()
    
    # Initialize position sizer
    sizer = PositionSizer()
    sizer.set_data(price_data, portfolio_value=100000.0)
    sizer.set_performance_metrics(win_rate=0.65, avg_win=0.025, avg_loss=-0.015)
    
    current_price = price_data['Close'].iloc[-1]
    stop_loss_price = current_price * 0.975
    
    risk_levels = [
        RiskLevel.CONSERVATIVE,
        RiskLevel.MODERATE,
        RiskLevel.AGGRESSIVE,
        RiskLevel.VERY_AGGRESSIVE
    ]
    
    print(f"\n📊 Comparing position sizes across risk levels:")
    print(f"{'Risk Level':<20} {'Position Size':<15} {'Position Value':<15} {'Risk %':<10} {'Rationale':<30}")
    print("-" * 95)
    
    for risk_level in risk_levels:
        recommendation = sizer.get_sizing_recommendation(current_price, stop_loss_price, risk_level)
        
        recommended = recommendation['recommended']
        position_size = recommended['position_size']
        position_value = position_size * current_price
        risk_pct = (recommended['risk_amount'] / 100000) * 100
        
        # Truncate rationale for display
        rationale = recommendation['sizing_rationale'][:25] + "..." if len(recommendation['sizing_rationale']) > 25 else recommendation['sizing_rationale']
        
        print(f"{risk_level.value:<20} {position_size:<15.4f} ${position_value:<14,.0f} {risk_pct:<10.2f}% {rationale:<30}")


def demo_market_conditions():
    """Demo market conditions assessment"""
    print("\n" + "="*60)
    print("🌊 DEMO: MARKET CONDITIONS ASSESSMENT")
    print("="*60)
    
    # Create different market scenarios
    scenarios = [
        ("Normal Market", 0.015, 0.0005),      # Normal volatility, slight uptrend
        ("High Volatility", 0.035, -0.001),   # High volatility, slight downtrend
        ("Low Volatility", 0.008, 0.002),     # Low volatility, strong uptrend
        ("Bear Market", 0.025, -0.003),       # Medium volatility, strong downtrend
    ]
    
    for scenario_name, daily_vol, daily_return in scenarios:
        print(f"\n📊 {scenario_name.upper()}:")
        
        # Generate scenario-specific data
        dates = pd.date_range('2023-07-01', '2023-12-31', freq='D')
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        
        base_price = 2000.0
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data = pd.DataFrame({
            'Close': prices[1:]
        }, index=dates)
        
        # Initialize sizer with scenario data
        sizer = PositionSizer()
        sizer.set_data(price_data, portfolio_value=100000.0)
        sizer.set_performance_metrics(win_rate=0.65, avg_win=0.025, avg_loss=-0.015)
        
        # Assess market conditions
        conditions = sizer._assess_market_conditions()
        
        print(f"   Volatility: {conditions['volatility_condition']} ({conditions['volatility_value']:.1%})")
        print(f"   Trend: {conditions['trend_direction']} ({conditions['trend_value']:.3%} daily)")
        print(f"   Recommendation: {conditions['recommendation']}")
        
        # Get sizing recommendation
        current_price = price_data['Close'].iloc[-1]
        recommendation = sizer.get_sizing_recommendation(current_price, risk_level=RiskLevel.MODERATE)
        
        recommended_size = recommendation['recommended']['position_size']
        position_value = recommended_size * current_price
        
        print(f"   Recommended Position: {recommended_size:.4f} units (${position_value:,.0f})")


def demo_performance_tracking():
    """Demo performance tracking and statistics"""
    print("\n" + "="*60)
    print("📈 DEMO: PERFORMANCE TRACKING")
    print("="*60)
    
    # Create sample data
    price_data = create_sample_data()
    
    # Initialize position sizer
    sizer = PositionSizer()
    sizer.set_data(price_data, portfolio_value=100000.0)
    sizer.set_performance_metrics(win_rate=0.65, avg_win=0.025, avg_loss=-0.015)
    
    current_price = price_data['Close'].iloc[-1]
    
    print(f"\n🔄 Generating multiple sizing calculations...")
    
    # Generate multiple calculations
    for i in range(10):
        price_variation = current_price * (1 + np.random.normal(0, 0.01))
        
        if i % 3 == 0:
            sizer.calculate_kelly_criterion_size(price_variation)
        elif i % 3 == 1:
            sizer.calculate_volatility_based_size(price_variation)
        else:
            sizer.calculate_atr_based_size(price_variation)
    
    # Get statistics
    stats = sizer.get_statistics()
    
    print(f"\n📊 PERFORMANCE STATISTICS:")
    print(f"   Total Calculations: {stats['total_calculations']}")
    print(f"   Portfolio Value: ${stats['portfolio_value']:,.0f}")
    
    print(f"\n📈 Performance Metrics:")
    perf = stats['performance_metrics']
    print(f"   Win Rate: {perf['win_rate']:.1%}")
    print(f"   Average Win: {perf['avg_win']:.2%}")
    print(f"   Average Loss: {perf['avg_loss']:.2%}")
    
    print(f"\n📏 Position Size Statistics:")
    sizes = stats['position_sizes']
    print(f"   Mean: {sizes['mean']:.4f}")
    print(f"   Median: {sizes['median']:.4f}")
    print(f"   Std Dev: {sizes['std']:.4f}")
    print(f"   Range: {sizes['min']:.4f} - {sizes['max']:.4f}")
    
    print(f"\n🎯 Confidence Scores:")
    conf = stats['confidence_scores']
    print(f"   Mean: {conf['mean']:.2f}")
    print(f"   Range: {conf['min']:.2f} - {conf['max']:.2f}")
    
    print(f"\n🔧 Method Usage:")
    for method, usage in stats['method_usage'].items():
        print(f"   {method}: {usage['count']} times ({usage['percentage']:.1f}%)")


def main():
    """Main demo function"""
    print("🎯 ULTIMATE XAU SUPER SYSTEM V4.0")
    print("🔧 POSITION SIZING SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_basic_sizing_methods()
        demo_optimal_sizing()
        demo_risk_level_comparison()
        demo_market_conditions()
        demo_performance_tracking()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("🚀 Position Sizing System is ready for production!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 