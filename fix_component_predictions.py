#!/usr/bin/env python3
"""
FIX COMPONENT PREDICTIONS
Sửa các component không trả về prediction hợp lệ
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append('src/core')

def fix_component_predictions():
    print("🔧 FIXING COMPONENT PREDICTIONS")
    print("=" * 50)
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Initialize system
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        # Get data
        data = system._get_comprehensive_market_data("XAUUSDc")
        
        # Test each component individually
        print("\n🧪 Testing Each Component:")
        
        component_fixes = []
        
        for system_name, subsystem in system.system_manager.systems.items():
            try:
                print(f"\n--- Testing {system_name} ---")
                
                # Process data
                result = subsystem.process(data)
                
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")
                
                if isinstance(result, dict):
                    prediction = result.get('prediction')
                    confidence = result.get('confidence')
                    
                    print(f"Prediction: {prediction}")
                    print(f"Confidence: {confidence}")
                    
                    # Check if needs fixing
                    needs_fix = False
                    fix_reason = []
                    
                    if prediction is None:
                        needs_fix = True
                        fix_reason.append("No prediction returned")
                    
                    if confidence is None:
                        needs_fix = True
                        fix_reason.append("No confidence returned")
                    
                    if prediction is not None:
                        try:
                            pred_val = float(prediction)
                            if abs(pred_val) > 10:
                                needs_fix = True
                                fix_reason.append(f"Extreme prediction value: {pred_val}")
                        except:
                            needs_fix = True
                            fix_reason.append("Invalid prediction format")
                    
                    if needs_fix:
                        print(f"❌ NEEDS FIX: {', '.join(fix_reason)}")
                        component_fixes.append({
                            'component': system_name,
                            'issues': fix_reason,
                            'current_result': result
                        })
                    else:
                        print("✅ Component working correctly")
                
                else:
                    print(f"❌ NEEDS FIX: Invalid result type {type(result)}")
                    component_fixes.append({
                        'component': system_name,
                        'issues': ['Invalid result type'],
                        'current_result': str(result)[:100]
                    })
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                component_fixes.append({
                    'component': system_name,
                    'issues': [f'Exception: {str(e)}'],
                    'current_result': None
                })
        
        # Summary
        print(f"\n📋 COMPONENT FIX SUMMARY")
        print("=" * 50)
        
        print(f"Total components: {len(system.system_manager.systems)}")
        print(f"Components needing fixes: {len(component_fixes)}")
        
        if component_fixes:
            print(f"\n❌ Components to fix:")
            for fix in component_fixes:
                print(f"   • {fix['component']}: {', '.join(fix['issues'])}")
        
        # Generate specific fixes
        print(f"\n🔧 SPECIFIC FIXES NEEDED:")
        
        # Fix 1: NeuralNetworkSystem
        if any(fix['component'] == 'NeuralNetworkSystem' for fix in component_fixes):
            print(f"\n1. NeuralNetworkSystem Fix:")
            print(f"   • Issue: Not returning prediction/confidence")
            print(f"   • Fix: Ensure _ensemble_predict returns dict with prediction & confidence")
            print(f"   • Location: Line ~2327 in ultimate_xau_system.py")
        
        # Fix 2: DataQualityMonitor
        if any(fix['component'] == 'DataQualityMonitor' for fix in component_fixes):
            print(f"\n2. DataQualityMonitor Fix:")
            print(f"   • Issue: Returns quality assessment, not prediction")
            print(f"   • Fix: Add prediction based on data quality score")
            print(f"   • Location: Line ~902 in ultimate_xau_system.py")
        
        # Fix 3: Other components
        for fix in component_fixes:
            if fix['component'] not in ['NeuralNetworkSystem', 'DataQualityMonitor']:
                print(f"\n3. {fix['component']} Fix:")
                print(f"   • Issues: {', '.join(fix['issues'])}")
                print(f"   • Fix: Ensure process() returns {{prediction: float, confidence: float}}")
        
        # Save analysis
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'total_components': len(system.system_manager.systems),
            'broken_components': len(component_fixes),
            'component_fixes': component_fixes
        }
        
        filename = f"component_fix_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to: {filename}")
        
        return component_fixes
        
    except Exception as e:
        print(f"❌ Fix analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    fixes_needed = fix_component_predictions()
    
    if fixes_needed:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Fix {len(fixes_needed)} broken components")
        print(f"2. Ensure all components return {{prediction: float, confidence: float}}")
        print(f"3. Test ensemble calculation with all components working")
        print(f"4. Verify confidence calculation improvement")
    else:
        print(f"\n✅ All components working correctly") 