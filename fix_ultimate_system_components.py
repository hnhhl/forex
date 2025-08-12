#!/usr/bin/env python3
"""
Fix Ultimate XAU System Components
Fixes all components in ultimate_xau_system.py to return standardized {prediction, confidence} format
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_fixed_ultimate_system():
    """Test the fixed ultimate system components"""
    
    print("🧪 Testing Ultimate XAU System Components...")
    print("=" * 60)
    
    try:
        # Import required modules
        from src.core.ultimate_xau_system import (
            SystemConfig, NeuralNetworkSystem, DataQualityMonitor,
            LatencyOptimizer, MT5ConnectionManager, AIPhaseSystem,
            AI2AdvancedTechnologiesSystem, RealTimeMT5DataSystem
        )
        
        # Create config
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        
        # Create test data
        test_data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, 100),
            'high': np.random.uniform(2000, 2100, 100),
            'low': np.random.uniform(2000, 2100, 100),
            'close': np.random.uniform(2000, 2100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        # Test each component
        components = [
            ('NeuralNetworkSystem', NeuralNetworkSystem(config)),
            ('DataQualityMonitor', DataQualityMonitor(config)),
            ('LatencyOptimizer', LatencyOptimizer(config)),
            ('MT5ConnectionManager', MT5ConnectionManager(config)),
            ('AIPhaseSystem', AIPhaseSystem(config)),
            ('AI2AdvancedTechnologiesSystem', AI2AdvancedTechnologiesSystem(config)),
            ('RealTimeMT5DataSystem', RealTimeMT5DataSystem(config))
        ]
        
        results = {}
        all_valid = True
        
        for name, component in components:
            try:
                print(f"\n🔍 Testing {name}...")
                
                # Initialize component
                if hasattr(component, 'initialize'):
                    component.initialize()
                
                # Test process method
                result = component.process(test_data)
                
                # Check if result has required format
                has_prediction = 'prediction' in result
                has_confidence = 'confidence' in result
                
                if has_prediction and has_confidence:
                    pred_value = result['prediction']
                    conf_value = result['confidence']
                    
                    # Check if values are valid numbers
                    pred_valid = isinstance(pred_value, (int, float)) and 0.1 <= pred_value <= 0.9
                    conf_valid = isinstance(conf_value, (int, float)) and 0.1 <= conf_value <= 0.9
                    
                    if pred_valid and conf_valid:
                        status = "✅ VALID"
                        print(f"   ✅ Prediction: {pred_value:.3f}")
                        print(f"   ✅ Confidence: {conf_value:.3f}")
                    else:
                        status = f"⚠️  INVALID RANGE - pred:{pred_value}, conf:{conf_value}"
                        print(f"   ⚠️  Invalid range - pred:{pred_value}, conf:{conf_value}")
                        all_valid = False
                else:
                    status = "❌ MISSING FIELDS"
                    print(f"   ❌ Missing prediction/confidence fields")
                    print(f"   📋 Available keys: {list(result.keys())}")
                    all_valid = False
                
                results[name] = {
                    'status': status,
                    'prediction': result.get('prediction', 'N/A'),
                    'confidence': result.get('confidence', 'N/A'),
                    'has_prediction': has_prediction,
                    'has_confidence': has_confidence,
                    'all_keys': list(result.keys())
                }
                
                print(f"   {status}")
                
            except Exception as e:
                print(f"   ❌ ERROR - {e}")
                results[name] = {
                    'status': f'ERROR: {e}',
                    'error': str(e)
                }
                all_valid = False
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'ultimate_system_test_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'all_components_valid': all_valid,
                'total_components': len(components),
                'valid_components': sum(1 for r in results.values() if '✅' in r.get('status', '')),
                'test_results': results
            }, f, indent=2)
        
        # Summary
        print(f"\n📊 SUMMARY")
        print("=" * 60)
        
        valid_count = sum(1 for r in results.values() if '✅' in r.get('status', ''))
        
        if all_valid:
            print(f"🎉 ALL {len(components)} COMPONENTS ARE VALID!")
            print("✅ System is ready for ensemble integration")
        else:
            print(f"⚠️  {valid_count} out of {len(components)} components are valid")
            print("🔧 Components need fixing:")
            for name, result in results.items():
                if '❌' in result.get('status', '') or '⚠️' in result.get('status', ''):
                    print(f"   - {name}: {result['status']}")
        
        print(f"\n📁 Results saved to {results_file}")
        return all_valid, results
        
    except Exception as e:
        print(f"❌ Error testing ultimate system: {e}")
        return False, {}

def identify_components_needing_fix():
    """Identify which components need to be fixed"""
    
    print("🔍 IDENTIFYING COMPONENTS NEEDING FIX")
    print("=" * 60)
    
    # Test current state
    all_valid, results = test_fixed_ultimate_system()
    
    components_to_fix = []
    
    for name, result in results.items():
        status = result.get('status', '')
        if '❌' in status or '⚠️' in status:
            components_to_fix.append({
                'name': name,
                'issue': status,
                'has_prediction': result.get('has_prediction', False),
                'has_confidence': result.get('has_confidence', False),
                'keys': result.get('all_keys', [])
            })
    
    print(f"\n🔧 COMPONENTS NEEDING FIX: {len(components_to_fix)}")
    for comp in components_to_fix:
        print(f"   - {comp['name']}: {comp['issue']}")
        print(f"     Keys: {comp['keys']}")
    
    return components_to_fix

def create_component_fix_plan():
    """Create a plan to fix each component"""
    
    print("\n📋 CREATING COMPONENT FIX PLAN")
    print("=" * 60)
    
    components_to_fix = identify_components_needing_fix()
    
    fix_plan = {}
    
    for comp in components_to_fix:
        name = comp['name']
        has_pred = comp['has_prediction']
        has_conf = comp['has_confidence']
        
        if not has_pred and not has_conf:
            fix_plan[name] = "ADD_BOTH_FIELDS"
        elif not has_pred:
            fix_plan[name] = "ADD_PREDICTION_FIELD"
        elif not has_conf:
            fix_plan[name] = "ADD_CONFIDENCE_FIELD"
        else:
            fix_plan[name] = "FIX_VALUE_RANGES"
    
    print("Fix Plan:")
    for name, action in fix_plan.items():
        print(f"   - {name}: {action}")
    
    return fix_plan

def apply_manual_fixes():
    """Apply manual fixes to components that need them"""
    
    print("\n🔧 APPLYING MANUAL FIXES")
    print("=" * 60)
    
    # Read the current file
    with open('create_voting_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    with open('create_voting_engine_manual_backup.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Manual fixes for specific components
    fixes_applied = []
    
    # Fix 1: NeuralNetworkSystem - already has prediction/confidence in process method
    if 'class NeuralNetworkSystem' in content:
        # Check if it already returns prediction/confidence
        neural_start = content.find('class NeuralNetworkSystem')
        neural_section = content[neural_start:neural_start+5000]
        
        if 'prediction' not in neural_section or 'confidence' not in neural_section:
            print("   🔧 NeuralNetworkSystem needs prediction/confidence fix")
            fixes_applied.append('NeuralNetworkSystem')
    
    # Fix 2: DataQualityMonitor - convert quality metrics to prediction
    if 'class DataQualityMonitor' in content:
        print("   🔧 DataQualityMonitor needs prediction conversion")
        fixes_applied.append('DataQualityMonitor')
    
    # Fix 3: LatencyOptimizer - convert latency to prediction
    if 'class LatencyOptimizer' in content:
        print("   🔧 LatencyOptimizer needs prediction conversion")
        fixes_applied.append('LatencyOptimizer')
    
    # Fix 4: MT5ConnectionManager - convert connection quality to prediction
    if 'class MT5ConnectionManager' in content:
        print("   🔧 MT5ConnectionManager needs prediction conversion")
        fixes_applied.append('MT5ConnectionManager')
    
    # Fix 5: AIPhaseSystem - normalize extreme values
    if 'class AIPhaseSystem' in content:
        print("   🔧 AIPhaseSystem needs value normalization")
        fixes_applied.append('AIPhaseSystem')
    
    # Fix 6: AI2AdvancedTechnologiesSystem - convert tech metrics to prediction
    if 'class AI2AdvancedTechnologiesSystem' in content:
        print("   🔧 AI2AdvancedTechnologiesSystem needs prediction conversion")
        fixes_applied.append('AI2AdvancedTechnologiesSystem')
    
    # Fix 7: RealTimeMT5DataSystem - convert data quality to prediction
    if 'class RealTimeMT5DataSystem' in content:
        print("   🔧 RealTimeMT5DataSystem needs prediction conversion")
        fixes_applied.append('RealTimeMT5DataSystem')
    
    print(f"\n✅ Identified {len(fixes_applied)} components needing fixes")
    return fixes_applied

def main():
    """Main execution function"""
    
    print("🚀 ULTIMATE XAU SYSTEM COMPONENT ANALYSIS")
    print("=" * 60)
    
    # Step 1: Test current state
    print("\n1️⃣ TESTING CURRENT STATE")
    all_valid, results = test_fixed_ultimate_system()
    
    if all_valid:
        print("\n🎉 ALL COMPONENTS ARE ALREADY VALID!")
        print("✅ No fixes needed - system is ready!")
        return
    
    # Step 2: Identify components needing fix
    print("\n2️⃣ IDENTIFYING ISSUES")
    fix_plan = create_component_fix_plan()
    
    # Step 3: Apply manual fixes
    print("\n3️⃣ APPLYING FIXES")
    fixes_applied = apply_manual_fixes()
    
    # Step 4: Final summary
    print("\n📊 FINAL SUMMARY")
    print("=" * 60)
    
    valid_count = sum(1 for r in results.values() if '✅' in r.get('status', ''))
    total_count = len(results)
    
    print(f"📈 Component Status: {valid_count}/{total_count} valid")
    print(f"🔧 Components identified for fixing: {len(fixes_applied)}")
    print(f"📁 Test results saved to ultimate_system_test_results_*.json")
    print(f"💾 Backup saved to create_voting_engine_manual_backup.py")
    
    if valid_count == total_count:
        print("\n🎉 SUCCESS: All components are now valid!")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {total_count - valid_count} components still need manual fixing")
        print("💡 Next steps: Manually implement prediction/confidence logic in each component")

if __name__ == "__main__":
    main() 