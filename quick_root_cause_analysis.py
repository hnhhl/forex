#!/usr/bin/env python3
"""
QUICK ROOT CAUSE ANALYSIS
Tìm nguyên nhân gốc rễ của vấn đề hệ thống một cách nhanh chóng
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append('src/core')

def analyze_root_cause():
    print("🔍 QUICK ROOT CAUSE ANALYSIS")
    print("=" * 50)
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Initialize system
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized")
        
        # Test 1: Single signal generation
        print("\n🎯 TEST 1: Single Signal Analysis")
        signal = system.generate_signal("XAUUSDc")
        
        print(f"Action: {signal.get('action')}")
        print(f"Confidence: {signal.get('confidence'):.3f}")
        print(f"Prediction: {signal.get('prediction'):.3f}")
        
        # Test 2: Check individual components
        print("\n🧩 TEST 2: Component Check")
        data = system._get_comprehensive_market_data("XAUUSDc")
        signal_components = system._process_all_systems(data)
        
        print(f"Total components: {len(signal_components)}")
        
        predictions = []
        for comp_name, comp_result in signal_components.items():
            if isinstance(comp_result, dict):
                pred = comp_result.get('prediction')
                if pred is not None:
                    try:
                        predictions.append(float(pred))
                        print(f"{comp_name}: {pred:.3f}")
                    except:
                        print(f"{comp_name}: Invalid prediction")
        
        # Test 3: Check ensemble calculation
        print("\n🔄 TEST 3: Ensemble Calculation")
        
        if predictions:
            # Manual calculation
            weights = [system._get_system_weight(name) for name in signal_components.keys()]
            valid_weights = [w for w in weights if w > 0]
            
            print(f"Predictions: {[f'{p:.3f}' for p in predictions[:5]]}")
            print(f"Weights: {[f'{w:.3f}' for w in valid_weights[:5]]}")
            
            if len(predictions) == len(valid_weights):
                manual_weighted = np.average(predictions, weights=valid_weights)
                print(f"Manual weighted avg: {manual_weighted:.3f}")
                
                # Check against system result
                system_prediction = signal.get('prediction', 0)
                print(f"System prediction: {system_prediction:.3f}")
                
                if abs(manual_weighted - system_prediction) > 0.01:
                    print("❌ MISMATCH: Manual vs System prediction")
                else:
                    print("✅ Calculation matches")
        
        # Test 4: Check threshold logic
        print("\n⚖️ TEST 4: Threshold Logic")
        
        hybrid_metrics = signal.get('hybrid_metrics', {})
        if hybrid_metrics:
            signal_strength = hybrid_metrics.get('signal_strength', 0)
            consensus = hybrid_metrics.get('hybrid_consensus', 0)
            
            print(f"Signal strength: {signal_strength:.3f}")
            print(f"Consensus: {consensus:.3f}")
            
            # Check threshold logic manually
            action = signal.get('action')
            
            if signal_strength > 0.15 and consensus >= 0.6:
                expected = "BUY STRONG"
            elif signal_strength > 0.08 and consensus >= 0.55:
                expected = "BUY MODERATE"
            elif signal_strength < -0.15 and consensus >= 0.6:
                expected = "SELL STRONG"
            elif signal_strength < -0.08 and consensus >= 0.55:
                expected = "SELL MODERATE"
            else:
                expected = "HOLD"
            
            actual = f"{action} {signal.get('strength', '')}"
            
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            
            if expected.split()[0] != action:
                print("❌ THRESHOLD LOGIC ERROR")
            else:
                print("✅ Threshold logic correct")
        
        # Test 5: Check for stuck values
        print("\n🔒 TEST 5: Stuck Values Check")
        
        signals_sample = []
        for i in range(5):
            s = system.generate_signal("XAUUSDc")
            signals_sample.append({
                'action': s.get('action'),
                'confidence': s.get('confidence'),
                'prediction': s.get('prediction')
            })
        
        # Check variance
        confidences = [s['confidence'] for s in signals_sample]
        predictions = [s['prediction'] for s in signals_sample]
        actions = [s['action'] for s in signals_sample]
        
        conf_variance = np.var(confidences)
        pred_variance = np.var(predictions)
        unique_actions = len(set(actions))
        
        print(f"Confidence variance: {conf_variance:.6f}")
        print(f"Prediction variance: {pred_variance:.6f}")
        print(f"Unique actions: {unique_actions}/5")
        
        # Identify issues
        issues = []
        
        if conf_variance < 0.0001:
            issues.append("Confidence values stuck")
        
        if pred_variance < 0.0001:
            issues.append("Prediction values stuck")
        
        if unique_actions == 1:
            issues.append(f"All actions are {actions[0]}")
        
        if np.mean(confidences) < 0.5:
            issues.append("Low confidence levels")
        
        # Summary
        print("\n📋 ROOT CAUSE SUMMARY")
        print("=" * 50)
        
        if issues:
            print("❌ ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            
            # Specific recommendations
            print("\n💡 RECOMMENDED FIXES:")
            
            if "stuck" in str(issues):
                print("   • Check confidence calculation formula")
                print("   • Verify ensemble weight calculation")
                
            if "All actions" in str(issues):
                print("   • Check threshold values")
                print("   • Verify signal strength calculation")
                
            if "Low confidence" in str(issues):
                print("   • Review confidence boost logic")
                print("   • Check base confidence calculation")
        
        else:
            print("✅ No obvious issues detected")
            print("   System may have subtle logic problems")
        
        return {
            'issues_found': issues,
            'sample_signal': signal,
            'component_count': len(signal_components),
            'confidence_variance': conf_variance,
            'prediction_variance': pred_variance
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = analyze_root_cause()
    
    if result:
        # Save results
        filename = f"root_cause_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Analysis saved to: {filename}") 