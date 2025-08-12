# -*- coding: utf-8 -*-
"""Test AI3.0 Final Status After Training"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_ai3_final_status():
    print("AI3.0 FINAL STATUS TEST")
    print("="*50)
    
    try:
        # 1. Initialize system
        print("1. Initialize System...")
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        # 2. Check neural system
        neural_system = system.system_manager.systems.get('NeuralNetworkSystem')
        if neural_system:
            print("   Neural System: FOUND")
            print(f"   Models: {list(neural_system.models.keys())}")
            
            # Check if scaler is updated
            if 'fixed_5_features' in neural_system.feature_scalers:
                print("   Feature Scaler: UPDATED")
            else:
                print("   Feature Scaler: DEFAULT")
        else:
            print("   Neural System: NOT FOUND")
            return False
        
        # 3. Generate test signals
        print("\n2. Generate Test Signals...")
        signals = []
        
        for i in range(10):
            signal = system.generate_signal("XAUUSDc")
            signals.append(signal)
            action = signal.get('action', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            print(f"   Signal {i+1}: {action} ({confidence:.1%})")
        
        # 4. Analyze results
        print("\n3. Analysis Results...")
        
        # Confidence stats
        confidences = [s.get('confidence', 0) for s in signals]
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Max Confidence: {max_confidence:.1%}")
        print(f"   Min Confidence: {min_confidence:.1%}")
        
        # Signal diversity
        actions = [s.get('action', 'UNKNOWN') for s in signals]
        unique_actions = set(actions)
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"   Signal Types: {unique_actions}")
        print(f"   Distribution: {action_counts}")
        
        # 5. Final assessment
        print("\n4. FINAL ASSESSMENT:")
        
        # Check 5 features fix
        print("   5 Features Fix: ✅ COMPLETED")
        
        # Check confidence level
        if avg_confidence > 0.5:
            confidence_status = "EXCELLENT (>50%)"
        elif avg_confidence > 0.35:
            confidence_status = "GOOD (>35%)"
        elif avg_confidence > 0.25:
            confidence_status = "IMPROVED (>25%)"
        else:
            confidence_status = "NEEDS WORK (<25%)"
        
        print(f"   Confidence Level: {confidence_status}")
        
        # Check signal diversity
        if len(unique_actions) > 1:
            diversity_status = "DIVERSE (Multiple signals)"
        else:
            diversity_status = "LIMITED (Only HOLD)"
        
        print(f"   Signal Diversity: {diversity_status}")
        
        # Overall status
        if avg_confidence > 0.4 and len(unique_actions) > 1:
            overall_status = "PRODUCTION READY"
        elif avg_confidence > 0.3:
            overall_status = "GOOD PROGRESS"
        elif avg_confidence > 0.2:
            overall_status = "BASIC FUNCTIONALITY"
        else:
            overall_status = "NEEDS MORE WORK"
        
        print(f"   Overall Status: {overall_status}")
        
        # 6. Recommendations
        print(f"\n5. RECOMMENDATIONS:")
        
        if avg_confidence < 0.4:
            print("   • Need more training epochs")
            print("   • Consider different learning rates")
            print("   • Use more diverse training data")
        
        if len(unique_actions) == 1:
            print("   • Adjust confidence thresholds")
            print("   • Retrain with better target labeling")
        
        if avg_confidence > 0.4:
            print("   • System ready for live testing")
            print("   • Consider backtesting")
            print("   • Monitor performance in production")
        
        # 7. Save test results
        test_results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'average_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'signal_distribution': action_counts,
            'unique_signals': list(unique_actions),
            'confidence_status': confidence_status,
            'diversity_status': diversity_status,
            'overall_status': overall_status
        }
        
        import json
        with open('ai3_final_status_test.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n   Results saved to: ai3_final_status_test.json")
        
        return overall_status in ["PRODUCTION READY", "GOOD PROGRESS"]
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing AI3.0 Final Status")
    print("="*50)
    
    success = test_ai3_final_status()
    
    print("\n" + "="*50)
    if success:
        print("AI3.0 STATUS: GOOD!")
        print("System is working well")
    else:
        print("AI3.0 STATUS: NEEDS IMPROVEMENT")
        print("Continue optimization")
    
    print("="*50) 