"""
Test script để kiểm tra tất cả 7 components đã có prediction/confidence
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime
from src.core.ultimate_xau_system import *

def test_all_components():
    print("🧪 Testing all 7 components for prediction/confidence...")
    
    # Tạo config
    config = SystemConfig(
        symbol="XAUUSD",
        timeframe=mt5.TIMEFRAME_H1 if 'mt5' in globals() else 60,
        mt5_login=0,
        mt5_password="",
        mt5_server=""
    )
    
    # Tạo test data
    test_data = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'open': np.random.uniform(2000, 2100, 100),
        'high': np.random.uniform(2050, 2150, 100),
        'low': np.random.uniform(1950, 2050, 100),
        'close': np.random.uniform(2000, 2100, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    components = [
        ("DataQualityMonitor", DataQualityMonitor),
        ("LatencyOptimizer", LatencyOptimizer),
        ("MT5ConnectionManager", MT5ConnectionManager),
        ("AIPhaseSystem", AIPhaseSystem),
        ("AI2AdvancedTechnologiesSystem", AI2AdvancedTechnologiesSystem),
        ("RealTimeMT5DataSystem", RealTimeMT5DataSystem),
        ("NeuralNetworkSystem", NeuralNetworkSystem)
    ]
    
    results = {}
    
    for name, component_class in components:
        try:
            print(f"\n🔍 Testing {name}...")
            
            # Khởi tạo component
            component = component_class(config)
            
            # Khởi tạo (có thể fail, không sao)
            try:
                component.initialize()
            except:
                pass
            
            # Test process method
            result = component.process(test_data)
            
            # Kiểm tra có prediction và confidence không
            has_prediction = 'prediction' in result
            has_confidence = 'confidence' in result
            
            if has_prediction and has_confidence:
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Kiểm tra giá trị hợp lệ
                valid_prediction = isinstance(prediction, (int, float)) and 0.0 <= prediction <= 1.0
                valid_confidence = isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                
                if valid_prediction and valid_confidence:
                    status = "✅ PASS"
                    results[name] = {
                        'status': 'PASS',
                        'prediction': prediction,
                        'confidence': confidence
                    }
                else:
                    status = f"❌ FAIL - Invalid values (pred={prediction}, conf={confidence})"
                    results[name] = {
                        'status': 'FAIL',
                        'reason': 'Invalid values',
                        'prediction': prediction,
                        'confidence': confidence
                    }
            else:
                missing = []
                if not has_prediction:
                    missing.append('prediction')
                if not has_confidence:
                    missing.append('confidence')
                
                status = f"❌ FAIL - Missing: {', '.join(missing)}"
                results[name] = {
                    'status': 'FAIL',
                    'reason': f'Missing: {", ".join(missing)}',
                    'result_keys': list(result.keys())
                }
            
            print(f"   {status}")
            
        except Exception as e:
            print(f"   ❌ ERROR - {str(e)}")
            results[name] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    # Tổng kết
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL COMPONENTS FIXED SUCCESSFULLY!")
        print("   All 7 components now return prediction/confidence")
    else:
        print("\n⚠️  Some components still need fixing:")
        for name, result in results.items():
            if result.get('status') != 'PASS':
                print(f"   - {name}: {result.get('reason', result.get('error', 'Unknown'))}")
    
    return results

if __name__ == "__main__":
    test_all_components() 