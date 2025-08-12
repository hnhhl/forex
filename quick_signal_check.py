# -*- coding: utf-8 -*-
"""Quick Signal Check - 10 signals to check diversity"""

import sys
import os
sys.path.append('src')

import time
from datetime import datetime
from collections import defaultdict

def quick_signal_check():
    print("⚡ QUICK SIGNAL CHECK - 10 SIGNALS")
    print("="*50)
    
    # Initialize system
    try:
        from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Generate 10 signals
    signals = []
    signal_counts = defaultdict(int)
    confidences = []
    
    print(f"\n🔄 Generating 10 signals...")
    
    for i in range(10):
        try:
            signal = system.generate_signal("XAUUSDc")
            
            action = signal.get('action', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            price = signal.get('current_price', 0)
            
            signal_data = {
                'num': i + 1,
                'action': action,
                'confidence': confidence,
                'price': price
            }
            
            signals.append(signal_data)
            signal_counts[action] += 1
            confidences.append(confidence)
            
            print(f"   Signal {i+1}: {action} ({confidence:.1%}) - ${price:.2f}")
            
            # Small delay
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error generating signal {i+1}: {e}")
    
    # Analysis
    print("\n" + "="*50)
    print("📊 SIGNAL ANALYSIS")
    print("="*50)
    
    if signals:
        # Distribution
        print("📈 SIGNAL DISTRIBUTION:")
        total = len(signals)
        for action in ['BUY', 'SELL', 'HOLD']:
            count = signal_counts[action]
            pct = (count / total) * 100
            print(f"   {action}: {count} signals ({pct:.1f}%)")
        
        # Diversity
        unique_actions = len(set([s['action'] for s in signals]))
        print(f"\n🎲 SIGNAL DIVERSITY:")
        print(f"   Types: {unique_actions}/3")
        
        if unique_actions == 3:
            print("   ✅ EXCELLENT - All 3 signal types")
        elif unique_actions == 2:
            print("   ⚡ GOOD - 2 signal types")
        else:
            print("   ⚠️ LIMITED - Only 1 signal type")
        
        # Confidence
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\n📈 CONFIDENCE ANALYSIS:")
        print(f"   Average: {avg_confidence:.1%}")
        print(f"   Range: {min(confidences):.1%} - {max(confidences):.1%}")
        
        if avg_confidence >= 0.4:
            print("   ✅ HIGH confidence level")
        elif avg_confidence >= 0.25:
            print("   ⚡ GOOD confidence level")
        else:
            print("   ⚠️ LOW confidence level")
        
        # Market Assessment
        print(f"\n💹 MARKET ASSESSMENT:")
        buy_pct = (signal_counts['BUY'] / total) * 100
        sell_pct = (signal_counts['SELL'] / total) * 100
        hold_pct = (signal_counts['HOLD'] / total) * 100
        
        if buy_pct > 60:
            print("   🚀 STRONG BULLISH TREND")
        elif sell_pct > 60:
            print("   📉 STRONG BEARISH TREND")
        elif hold_pct > 60:
            print("   ⚖️ SIDEWAYS/UNCERTAIN MARKET")
        else:
            print("   🔄 MIXED SIGNALS - BALANCED MARKET")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        if unique_actions >= 2 and avg_confidence >= 0.25:
            print("   ✅ SYSTEM WORKING WELL")
            print("   ✅ Good diversity and confidence")
        elif unique_actions >= 2:
            print("   ⚡ SYSTEM MOSTLY GOOD")
            print("   ⚡ Good diversity, confidence could be higher")
        elif avg_confidence >= 0.25:
            print("   ⚡ SYSTEM MOSTLY GOOD")
            print("   ⚡ Good confidence, diversity limited")
        else:
            print("   ⚠️ SYSTEM NEEDS IMPROVEMENT")
            print("   ⚠️ Both diversity and confidence are limited")
    
    else:
        print("❌ No signals generated")
    
    print("\n" + "="*50)
    print("✅ Quick check completed!")

if __name__ == "__main__":
    quick_signal_check() 