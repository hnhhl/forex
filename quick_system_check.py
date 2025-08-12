# -*- coding: utf-8 -*-
"""Quick AI3.0 System Check"""

import sys
import os
sys.path.append('src')

def quick_system_check():
    print("QUICK AI3.0 SYSTEM CHECK")
    print("="*50)
    
    issues = []
    
    try:
        # 1. Basic Import Test
        print("1. Testing basic imports...")
        from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        print("   ✅ Core imports successful")
        
        # 2. System Initialization
        print("\n2. Testing system initialization...")
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        active_systems = len([s for s in system.system_manager.systems.keys()])
        print(f"   ✅ System initialized: {active_systems}/8 systems")
        
        # 3. Signal Generation Test
        print("\n3. Testing signal generation...")
        signal = system.generate_signal("XAUUSDc")
        
        action = signal.get('action', 'UNKNOWN')
        confidence = signal.get('confidence', 0)
        method = signal.get('ensemble_method', 'unknown')
        
        print(f"   ✅ Signal generated: {action}")
        print(f"   📊 Confidence: {confidence:.1%}")
        print(f"   🔧 Method: {method}")
        
        # 4. Check Hybrid Logic
        print("\n4. Checking hybrid logic...")
        if method == 'hybrid_ai2_ai3_consensus':
            print("   ✅ Hybrid logic active")
        else:
            print(f"   ⚠️ Unexpected method: {method}")
            issues.append(f"Wrong method: {method}")
        
        # 5. Check Confidence Level
        print("\n5. Checking confidence level...")
        if confidence > 0.25:
            print(f"   ✅ Good confidence: {confidence:.1%}")
        elif confidence > 0.15:
            print(f"   ⚠️ Low confidence: {confidence:.1%}")
            issues.append(f"Low confidence: {confidence:.1%}")
        else:
            print(f"   ❌ Very low confidence: {confidence:.1%}")
            issues.append(f"Very low confidence: {confidence:.1%}")
        
        # 6. Check Signal Diversity
        print("\n6. Checking signal diversity...")
        actions = []
        for i in range(5):
            sig = system.generate_signal("XAUUSDc")
            actions.append(sig.get('action', 'UNKNOWN'))
        
        unique_actions = set(actions)
        print(f"   📈 Actions: {list(unique_actions)}")
        
        if len(unique_actions) > 1:
            print("   ✅ Good signal diversity")
        else:
            print("   ⚠️ Limited diversity")
            issues.append(f"Limited diversity: {unique_actions}")
        
        # 7. Check Consensus Quality
        print("\n7. Checking consensus quality...")
        try:
            metrics = signal.get('hybrid_metrics', {})
            consensus = metrics.get('hybrid_consensus', 0)
            
            if consensus > 0.6:
                print(f"   ✅ Good consensus: {consensus:.1%}")
            elif consensus > 0.4:
                print(f"   ⚠️ Moderate consensus: {consensus:.1%}")
                issues.append(f"Moderate consensus: {consensus:.1%}")
            else:
                print(f"   ❌ Poor consensus: {consensus:.1%}")
                issues.append(f"Poor consensus: {consensus:.1%}")
        except:
            print("   ⚠️ Consensus data not available")
            issues.append("Consensus data missing")
        
        # 8. Check Neural Network Status
        print("\n8. Checking neural networks...")
        try:
            neural_system = system.system_manager.systems.get('NeuralNetworkSystem')
            if neural_system and hasattr(neural_system, 'models'):
                model_count = len(neural_system.models)
                print(f"   ✅ Neural models: {model_count} loaded")
            else:
                print("   ⚠️ Neural system not found")
                issues.append("Neural system missing")
        except Exception as e:
            print(f"   ❌ Neural check failed: {e}")
            issues.append("Neural system error")
        
        # Summary
        print("\n" + "="*50)
        print("QUICK CHECK SUMMARY")
        print("="*50)
        
        if not issues:
            print("🎉 NO MAJOR ISSUES FOUND!")
            print("✅ AI3.0 system appears to be working well")
            status = "GOOD"
        elif len(issues) <= 2:
            print("⚡ MINOR ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            print("⚡ System is mostly functional")
            status = "FAIR"
        else:
            print("⚠️ MULTIPLE ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            print("⚠️ System needs attention")
            status = "NEEDS_WORK"
        
        # Current Status
        print(f"\n📊 CURRENT STATUS:")
        print(f"   🎯 Action: {action}")
        print(f"   📈 Confidence: {confidence:.1%}")
        print(f"   🔧 Method: {method}")
        print(f"   🏛️ Active Systems: {active_systems}/8")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if status == "GOOD":
            print("   ✅ System ready for use")
            print("   ✅ Monitor performance regularly")
        elif status == "FAIR":
            print("   ⚡ Address minor issues above")
            print("   ⚡ Continue monitoring")
        else:
            print("   ⚠️ Fix issues before production use")
            print("   ⚠️ Consider additional debugging")
        
        return status, issues
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        return "ERROR", [str(e)]

if __name__ == "__main__":
    print("Starting AI3.0 system check...\n")
    status, issues = quick_system_check()
    
    print(f"\n🏁 FINAL STATUS: {status}")
    if status == "GOOD":
        print("🚀 AI3.0 is ready!")
    elif status == "FAIR":
        print("⚡ AI3.0 needs minor fixes")
    else:
        print("⚠️ AI3.0 needs attention") 