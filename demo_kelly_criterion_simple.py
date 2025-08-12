#!/usr/bin/env python3
"""
Ultimate XAU Super System V4.0 - Kelly Criterion Calculator Simple Demo
======================================================================

Simple demo showcasing the Kelly Criterion Calculator functionality.

Author: Ultimate XAU Super System V4.0
Date: December 19, 2024
"""

import sys
import os
from datetime import datetime, timedelta
import random

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.trading.kelly_criterion import (
    KellyCriterionCalculator,
    TradeResult,
    KellyMethod
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

def generate_sample_trades(num_trades: int = 100) -> list[TradeResult]:
    """Generate realistic sample trading data"""
    trades = []
    base_date = datetime.now() - timedelta(days=num_trades)
    
    # Simulate realistic trading patterns
    win_rate = 0.55  # 55% win rate
    avg_win = 150.0  # Average win $150
    avg_loss = 100.0  # Average loss $100
    
    for i in range(num_trades):
        trade_date = base_date + timedelta(days=i)
        
        # Determine if trade is winning
        is_win = random.random() < win_rate
        
        if is_win:
            # Winning trade with some variance
            pnl = avg_win * random.uniform(0.5, 2.0)
        else:
            # Losing trade with some variance
            pnl = -avg_loss * random.uniform(0.3, 1.5)
        
        entry_price = random.uniform(1900, 2100)
        exit_price = entry_price + (pnl / random.uniform(0.1, 1.0))
        
        trades.append(TradeResult(
            profit_loss=pnl,
            win=pnl > 0,
            trade_date=trade_date,
            symbol="XAUUSD",
            entry_price=entry_price,
            exit_price=exit_price,
            volume=random.uniform(0.1, 1.0),
            duration_minutes=random.randint(60, 1440)
        ))
    
    return trades

def main():
    """Main demo function"""
    print_header("Ultimate XAU Super System V4.0 - Kelly Criterion Calculator Demo")
    
    print("""
🎯 This demo showcases the Kelly Criterion Calculator with:

✅ 5 Kelly Calculation Methods
✅ Advanced Risk Controls
✅ Performance Analytics
✅ Professional Implementation

Let's test with realistic trading data...
    """)
    
    try:
        # Initialize calculator
        print_section("Initializing Kelly Calculator")
        calculator = KellyCriterionCalculator()
        print("✅ Kelly Criterion Calculator initialized successfully")
        
        # Generate and add sample trades
        print_section("Adding Sample Trading Data")
        trades = generate_sample_trades(100)
        
        for trade in trades:
            calculator.add_trade_result(trade)
        
        print(f"📈 Added {len(trades)} sample trades")
        print(f"💰 Total P&L: ${sum(t.profit_loss for t in trades):.2f}")
        
        winning_trades = [t for t in trades if t.win]
        losing_trades = [t for t in trades if not t.win]
        
        print(f"🎯 Win Rate: {len(winning_trades)/len(trades):.1%}")
        print(f"📊 Avg Win: ${sum(t.profit_loss for t in winning_trades)/len(winning_trades):.2f}")
        print(f"📉 Avg Loss: ${sum(t.profit_loss for t in losing_trades)/len(losing_trades):.2f}")
        
        # Test different Kelly methods
        print_section("Kelly Calculation Results")
        
        methods = [
            (KellyMethod.CLASSIC, "Classic Kelly"),
            (KellyMethod.FRACTIONAL, "Fractional Kelly (Conservative)"),
            (KellyMethod.DYNAMIC, "Dynamic Kelly (Market-Adaptive)"),
            (KellyMethod.CONSERVATIVE, "Conservative Kelly (Risk-Averse)"),
            (KellyMethod.ADAPTIVE, "Adaptive Kelly (ML-Enhanced)")
        ]
        
        print("🎯 Kelly Calculation Results:")
        for method, name in methods:
            try:
                result = calculator.calculate_kelly_fraction(method)
                print(f"  {name:35} | {result.kelly_fraction*100:6.2f}% | Confidence: {result.confidence_score:.2f}")
                
                if result.warnings:
                    print(f"    ⚠️ Warnings: {len(result.warnings)} active")
                    
            except Exception as e:
                print(f"  {name:35} | Error: {str(e)}")
        
        # Show detailed analysis for Adaptive method
        print_section("Detailed Risk Analysis (Adaptive Method)")
        
        try:
            result = calculator.calculate_kelly_fraction(KellyMethod.ADAPTIVE)
            
            print(f"🎯 Recommended Position Size: {result.kelly_fraction*100:.2f}%")
            print(f"🔍 Confidence Score: {result.confidence_score:.2f}")
            print(f"📊 Method Used: {result.method_used.value}")
            
            print(f"\n📈 Risk Metrics:")
            for metric, value in result.risk_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric:25} | {value:.4f}")
                else:
                    print(f"  {metric:25} | {value}")
            
            if result.warnings:
                print(f"\n⚠️ Risk Warnings ({len(result.warnings)}):")
                for i, warning in enumerate(result.warnings, 1):
                    print(f"  {i}. {warning}")
            else:
                print(f"\n✅ No risk warnings - Safe to proceed")
                
        except Exception as e:
            print(f"❌ Error in detailed analysis: {e}")
        
        # Show performance statistics
        print_section("Performance Statistics")
        
        try:
            stats = calculator.get_statistics()
            
            print(f"📊 Trading Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    if 'rate' in key.lower() or 'ratio' in key.lower():
                        print(f"  {key:25} | {value:.1%}")
                    else:
                        print(f"  {key:25} | {value:.2f}")
                else:
                    print(f"  {key:25} | {value}")
                    
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
        
        print_header("Demo Completed Successfully! 🎉")
        
        print("""
✅ Kelly Criterion Calculator Demo Complete!

Key Features Demonstrated:
• 5 different Kelly calculation methods
• Advanced risk controls and warnings
• Comprehensive performance analytics
• Professional-grade implementation
• Real-time confidence scoring

The Kelly Criterion Calculator is ready for integration with the
Ultimate XAU Super System V4.0 trading platform!

Next Steps:
• Integrate with Position Sizing System
• Connect to real-time market data
• Implement automated position sizing
• Add machine learning enhancements (Phase 2)
        """)
        
    except Exception as e:
        print(f"\n❌ Demo Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Please check the implementation and try again.")

if __name__ == "__main__":
    # Set random seed for reproducible demo
    random.seed(42)
    main() 