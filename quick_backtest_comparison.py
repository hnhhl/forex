#!/usr/bin/env python3
"""
QUICK BACKTEST COMPARISON
So sánh hiệu suất hệ thống trước và sau khi sửa lỗi
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append('src/core')

def run_quick_backtest():
    print("⚡ QUICK BACKTEST COMPARISON")
    print("=" * 60)
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Initialize system
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized for backtest")
        
        # Get historical data
        data = system._get_comprehensive_market_data("XAUUSDc")
        if data.empty:
            print("❌ No historical data available")
            return None
        
        print(f"📊 Historical data: {len(data)} candles")
        print(f"   Date range: {data['time'].min()} to {data['time'].max()}")
        
        # Simulate trading over last 100 candles
        test_data = data.tail(100).copy()
        
        # Trading simulation
        balance = 10000.0
        positions = []
        trades = []
        
        print(f"\n🔄 Running backtest simulation...")
        
        for i in range(len(test_data) - 1):
            current_row = test_data.iloc[i:i+1]
            next_row = test_data.iloc[i+1]
            
            # Generate signal
            signal = system.generate_signal("XAUUSDc")
            
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0)
            price = current_row['close'].iloc[0]
            next_price = next_row['close']
            
            # Simple trading logic
            if action == 'BUY' and confidence > 0.6:
                # Enter long position
                position = {
                    'type': 'LONG',
                    'entry_price': price,
                    'entry_time': current_row['time'].iloc[0],
                    'confidence': confidence
                }
                positions.append(position)
                
            elif action == 'SELL' and confidence > 0.6:
                # Enter short position
                position = {
                    'type': 'SHORT',
                    'entry_price': price,
                    'entry_time': current_row['time'].iloc[0],
                    'confidence': confidence
                }
                positions.append(position)
            
            # Close positions (simple 1-candle holding)
            for pos in positions[:]:
                if pos['type'] == 'LONG':
                    pnl = next_price - pos['entry_price']
                elif pos['type'] == 'SHORT':
                    pnl = pos['entry_price'] - next_price
                
                # Close position
                trade = {
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': next_price,
                    'pnl': pnl,
                    'confidence': pos['confidence'],
                    'entry_time': pos['entry_time'],
                    'exit_time': next_row['time']
                }
                trades.append(trade)
                positions.remove(pos)
                balance += pnl
            
            # Progress indicator
            if i % 20 == 0:
                print(f"   Processed {i+1}/100 candles... Balance: ${balance:.2f}")
        
        # Analysis
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 60)
        
        total_trades = len(trades)
        if total_trades == 0:
            print("❌ No trades executed (confidence threshold not met)")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_confidence': 0
            }
        
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = (balance - 10000) / 10000 * 100
        avg_confidence = np.mean([t['confidence'] for t in trades])
        
        print(f"💰 PERFORMANCE METRICS:")
        print(f"   • Total trades: {total_trades}")
        print(f"   • Winning trades: {winning_trades}")
        print(f"   • Losing trades: {losing_trades}")
        print(f"   • Win rate: {win_rate:.1%}")
        print(f"   • Total return: {total_return:+.2f}%")
        print(f"   • Final balance: ${balance:.2f}")
        print(f"   • Average confidence: {avg_confidence:.1%}")
        
        # Quality assessment
        print(f"\n🎯 ASSESSMENT:")
        
        if total_trades == 0:
            print("❌ CRITICAL: No trades executed - system too conservative")
            quality_score = 0
        elif win_rate >= 0.6:
            print("✅ EXCELLENT: High win rate")
            quality_score = 90
        elif win_rate >= 0.5:
            print("✅ GOOD: Decent win rate")
            quality_score = 70
        elif win_rate >= 0.4:
            print("🟡 MODERATE: Acceptable win rate")
            quality_score = 50
        else:
            print("❌ POOR: Low win rate")
            quality_score = 20
        
        if total_return > 5:
            print("✅ PROFITABLE: Strong positive returns")
            quality_score += 10
        elif total_return > 0:
            print("✅ PROFITABLE: Positive returns")
            quality_score += 5
        else:
            print("❌ UNPROFITABLE: Negative returns")
            quality_score -= 20
        
        if avg_confidence > 0.7:
            print("✅ HIGH CONFIDENCE: Strong signal quality")
        elif avg_confidence > 0.6:
            print("✅ GOOD CONFIDENCE: Acceptable signal quality")
        else:
            print("❌ LOW CONFIDENCE: Poor signal quality")
            quality_score -= 10
        
        print(f"\n🏆 OVERALL QUALITY SCORE: {max(0, quality_score)}/100")
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': balance,
            'avg_confidence': avg_confidence,
            'quality_score': max(0, quality_score),
            'trades': trades
        }
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_expected():
    """So sánh với kết quả mong đợi"""
    print(f"\n📋 COMPARISON WITH EXPECTATIONS")
    print("=" * 60)
    
    print("📊 BEFORE FIXES (from previous analysis):")
    print("   • Total trades: 1 (in 18 months)")
    print("   • Win rate: 25%")
    print("   • Average confidence: ~45%")
    print("   • Return: +12.68% (but only 1 trade)")
    print("   • Issue: Too passive, inconsistent")
    
    print(f"\n📊 EXPECTED AFTER FIXES:")
    print("   • Total trades: Should increase significantly")
    print("   • Win rate: Should improve to >50%")
    print("   • Average confidence: Should be >60%")
    print("   • Return: Should be consistent and positive")
    print("   • Behavior: More active but selective")

def main():
    """Main comparison function"""
    print("🧪 QUICK BACKTEST COMPARISON")
    print("=" * 70)
    print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Show expectations
    compare_with_expected()
    
    # Run actual backtest
    results = run_quick_backtest()
    
    # Save results
    if results:
        results['test_time'] = datetime.now().isoformat()
        
        filename = f"quick_backtest_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if results['quality_score'] >= 70:
            print("🏆 SUCCESS: Fixes have significantly improved the system!")
        elif results['quality_score'] >= 50:
            print("✅ IMPROVEMENT: Some fixes are working, but more work needed")
        elif results['quality_score'] >= 30:
            print("🟡 PARTIAL: Minor improvements, major issues remain")
        else:
            print("❌ FAILURE: Fixes have not resolved the core issues")
    
    print(f"\n🏁 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 