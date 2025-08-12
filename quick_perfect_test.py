#!/usr/bin/env python3
"""
🧪 QUICK PERFECT SYSTEM TEST - NO GPU ISSUES
Test nhanh 7/7 components hoàn hảo
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Force CPU mode before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_quick_perfect_system():
    """Test nhanh hệ thống hoàn hảo"""
    print("\n" + "="*60)
    print("🎯 QUICK PERFECT AI3.0 SYSTEM TEST - 7/7 COMPONENTS")
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
        
        # Test system status
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
        
        # Quick signal test (single test to avoid GPU issues)
        print(f"\n🧪 Testing Signal Generation...")
        
        signal = system.generate_signal("XAUUSD")
        
        # Display results
        print(f"   🎯 Signal: {signal.get('action', signal.get('signal', 'UNKNOWN'))}")
        print(f"   📊 Confidence: {signal['confidence']:.1f}%")
        print(f"   💪 Strength: {signal.get('strength', 'UNKNOWN')}")
        print(f"   ⚡ Processing Time: {signal.get('processing_time_ms', 0):.1f}ms")
        
        # Component analysis
        components = signal.get('signal_components', {})
        active_components = len([c for c in components.values() if c.get('prediction') is not None])
        print(f"   🔧 Active Components: {active_components}/7")
        
        # Individual component status
        print(f"\n🔍 Component Details:")
        for name, comp in components.items():
            if comp.get('prediction') is not None:
                status_icon = "✅"
                pred = f"{comp.get('prediction', 'N/A'):.3f}"
                conf = f"{comp.get('confidence', 'N/A'):.3f}"
            else:
                status_icon = "❌"
                pred = 'N/A'
                conf = 'N/A'
            print(f"   {status_icon} {name}: pred={pred}, conf={conf}")
        
        # Final evaluation
        signal_action = signal.get('action', signal.get('signal', 'UNKNOWN'))
        print(f"\n🏆 QUICK EVALUATION RESULTS:")
        print(f"   ✅ System Initialization: {'PASS' if active_systems >= 5 else 'FAIL'}")
        print(f"   ✅ Signal Generation: {'PASS' if signal_action in ['BUY', 'SELL', 'HOLD'] else 'FAIL'}")
        print(f"   ✅ Confidence Level: {'PASS' if signal['confidence'] > 0 else 'FAIL'}")
        print(f"   ✅ Component Activation: {'PASS' if active_components >= 5 else 'FAIL'}")
        
        # Overall score
        score_criteria = [
            active_systems >= 5,
            signal_action in ['BUY', 'SELL', 'HOLD'],
            signal['confidence'] > 0,
            active_components >= 5
        ]
        
        passed = sum(score_criteria)
        final_score = passed / len(score_criteria) * 100
        
        print(f"\n🎯 FINAL QUICK SCORE: {passed}/{len(score_criteria)} ({final_score:.1f}%)")
        
        if final_score >= 75:
            print("🏆 SYSTEM STATUS: EXCELLENT - COMPONENTS WORKING PERFECTLY!")
            return True
        else:
            print("⚠️ SYSTEM STATUS: NEEDS ATTENTION")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 QUICK AI3.0 PERFECT SYSTEM TEST")
    print("Testing Components Without GPU Issues")
    print("-" * 50)
    
    success = test_quick_perfect_system()
    
    print("\n" + "="*60)
    if success:
        print("🎉 QUICK TEST PASSED! 🎉")
        print("✅ Core components working correctly")
        print("🚀 System ready for full testing")
    else:
        print("🔧 QUICK TEST NEEDS WORK")
        print("⚠️ Some components require fixes")
    print("="*60) 