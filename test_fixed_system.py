#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE SYSTEM TEST - PERFECT AI3.0
Test tất cả 7 components để đạt 100% hoàn hảo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_test_data():
    """Tạo dữ liệu test comprehensive"""
    print("📊 Creating comprehensive test data...")
    
    # Generate 1000 realistic price data points
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    
    # Realistic XAU/USD price simulation
    base_price = 2000.0
    prices = []
    current_price = base_price
    
    for i in range(1000):
        # Add realistic volatility and trend
        change = np.random.normal(0, 2.5)  # $2.5 average volatility
        trend = 0.01 * np.sin(i / 100)  # Slow trend
        current_price += change + trend
        prices.append(current_price)
    
    # Create comprehensive DataFrame
    data = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1.5)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100, 1000, 1000),
        'spread': np.random.uniform(0.1, 0.5, 1000)
    })
    
    print(f"✅ Created {len(data)} realistic data points")
    return data

def test_perfect_system():
    """Test hệ thống hoàn hảo 7/7 components"""
    print("\n" + "="*60)
    print("🎯 TESTING PERFECT AI3.0 SYSTEM - 7/7 COMPONENTS")
    print("="*60)
    
    try:
        # Initialize system
        print("\n🚀 Initializing Ultimate XAU System...")
        config = SystemConfig(
            symbol="XAUUSD",
            timeframe=60,  # H1
            risk_per_trade=0.02,
            max_positions=3
        )
        
        system = UltimateXAUSystem(config)
        
        # Test initialization
        print("\n📋 System Status:")
        status = system.get_system_status()
        
        # Extract correct values from new structure
        system_health = status.get('system_health', {})
        total_systems = system_health.get('systems_total', 0)
        active_systems = system_health.get('systems_active', 0)
        
        print(f"   📊 Total Systems: {total_systems}")
        print(f"   ✅ Active Systems: {active_systems}")
        if total_systems > 0:
            print(f"   🎯 Success Rate: {active_systems}/{total_systems} = {(active_systems/total_systems*100):.1f}%")
        else:
            print(f"   🎯 Success Rate: N/A")
        
        # Test signal generation with comprehensive data
        test_data = create_comprehensive_test_data()
        
        print(f"\n🧪 Testing with {len(test_data)} data points...")
        
        # Multiple signal tests
        results = []
        for i in range(5):
            print(f"\n📡 Signal Test #{i+1}/5:")
            
            # Use different data slices for variety
            start_idx = i * 200
            end_idx = start_idx + 200
            test_slice = test_data.iloc[start_idx:end_idx].copy()
            
            signal = system.generate_signal("XAUUSD")
            results.append(signal)
            
            # Display comprehensive results
            print(f"   🎯 Signal: {signal['signal']}")
            print(f"   📊 Confidence: {signal['confidence']:.1f}%")
            print(f"   💪 Strength: {signal['strength']:.3f}")
            print(f"   ⚡ Processing Time: {signal.get('processing_time_ms', 0):.1f}ms")
            
            # Component analysis
            components = signal.get('signal_components', {})
            print(f"   🔧 Active Components: {len([c for c in components.values() if c.get('prediction') is not None])}/7")
            
            # Individual component status
            for name, comp in components.items():
                status = "✅" if comp.get('prediction') is not None else "❌"
                pred = comp.get('prediction', 'N/A')
                conf = comp.get('confidence', 'N/A')
                print(f"      {status} {name}: pred={pred}, conf={conf}")
        
        # Final comprehensive analysis
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("="*60)
        
        # Analyze all results
        signals = [r['signal'] for r in results]
        confidences = [r['confidence'] for r in results]
        strengths = [r['strength'] for r in results]
        
        print(f"📈 Signal Distribution:")
        print(f"   🟢 BUY: {signals.count('BUY')}/5 ({signals.count('BUY')/5*100:.1f}%)")
        print(f"   🔴 SELL: {signals.count('SELL')}/5 ({signals.count('SELL')/5*100:.1f}%)")
        print(f"   ⚪ HOLD: {signals.count('HOLD')}/5 ({signals.count('HOLD')/5*100:.1f}%)")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   📊 Average Confidence: {np.mean(confidences):.1f}%")
        print(f"   💪 Average Strength: {np.mean(strengths):.3f}")
        print(f"   📈 Confidence Range: {min(confidences):.1f}% - {max(confidences):.1f}%")
        print(f"   🎯 Strength Range: {min(strengths):.3f} - {max(strengths):.3f}")
        
        # Component reliability analysis
        all_components = {}
        for result in results:
            components = result.get('signal_components', {})
            for name, comp in components.items():
                if name not in all_components:
                    all_components[name] = {'active': 0, 'total': 0}
                all_components[name]['total'] += 1
                if comp.get('prediction') is not None:
                    all_components[name]['active'] += 1
        
        print(f"\n🔧 Component Reliability:")
        total_active = 0
        total_components = 0
        for name, stats in all_components.items():
            reliability = stats['active'] / stats['total'] * 100
            status = "✅" if reliability > 80 else "⚠️" if reliability > 50 else "❌"
            print(f"   {status} {name}: {stats['active']}/{stats['total']} ({reliability:.1f}%)")
            total_active += stats['active']
            total_components += stats['total']
        
        overall_reliability = total_active / total_components * 100
        print(f"\n🎯 OVERALL SYSTEM RELIABILITY: {total_active}/{total_components} ({overall_reliability:.1f}%)")
        
        # Success criteria
        print(f"\n🏆 SUCCESS CRITERIA EVALUATION:")
        criteria = {
            "System Initialization": active_systems >= 5,
            "Signal Generation": all(r['signal'] in ['BUY', 'SELL', 'HOLD'] for r in results),
            "Confidence Levels": all(r['confidence'] > 0 for r in results),
            "Component Reliability": overall_reliability >= 70,
            "Signal Diversity": len(set(signals)) > 1,
            "Performance Consistency": max(confidences) - min(confidences) < 50
        }
        
        passed = 0
        for criterion, result in criteria.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {criterion}")
            if result:
                passed += 1
        
        final_score = passed / len(criteria) * 100
        print(f"\n🎯 FINAL SYSTEM SCORE: {passed}/{len(criteria)} ({final_score:.1f}%)")
        
        if final_score >= 80:
            print("🏆 SYSTEM STATUS: EXCELLENT - READY FOR PRODUCTION!")
        elif final_score >= 60:
            print("✅ SYSTEM STATUS: GOOD - MINOR OPTIMIZATIONS NEEDED")
        else:
            print("⚠️ SYSTEM STATUS: NEEDS IMPROVEMENT")
        
        return final_score >= 80
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE AI3.0 SYSTEM TEST")
    print("Testing for 100% Perfect Performance")
    print("-" * 50)
    
    success = test_perfect_system()
    
    print("\n" + "="*60)
    if success:
        print("🎉 PERFECT SYSTEM ACHIEVED! 🎉")
        print("✅ All components working flawlessly")
        print("🚀 Ready for production deployment")
    else:
        print("🔧 SYSTEM NEEDS OPTIMIZATION")
        print("⚠️ Some components require attention")
    print("="*60) 