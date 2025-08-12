#!/usr/bin/env python3
"""
TEST MODEL INTEGRATION
Kiểm tra xem các models đã được tích hợp vào hệ thống chưa
"""

import os
import sys
import tensorflow as tf
import pickle
import numpy as np
from datetime import datetime

def test_model_loading():
    """Test loading individual models"""
    print("🔍 TESTING MODEL LOADING...")
    print("=" * 50)
    
    models_loaded = 0
    
    # Test Neural Models
    neural_models = [
        "neural_ensemble_y_direction_2_lstm.keras",
        "neural_ensemble_y_direction_2_dense.keras",
        "production_lstm_20250620_140549.keras",
        "gpu_lstm_model.keras"
    ]
    
    for model_name in neural_models:
        model_path = f"trained_models/{model_name}"
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print(f"✅ Neural Model: {model_name}")
                print(f"   Parameters: {model.count_params():,}")
                models_loaded += 1
            else:
                print(f"❌ Not found: {model_name}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)[:100]}")
    
    # Test ML Models
    ml_models = [
        "random_forest_y_direction_2.pkl",
        "gradient_boost_y_direction_2.pkl",
        "lightgbm_y_direction_2.pkl",
        "enhanced_random_forest_20250619_130809.pkl"
    ]
    
    for model_name in ml_models:
        model_path = f"trained_models/{model_name}"
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ ML Model: {model_name}")
                print(f"   Type: {type(model).__name__}")
                models_loaded += 1
            else:
                print(f"❌ Not found: {model_name}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)[:100]}")
    
    # Test Scalers
    scalers = [
        "scaler_y_direction_2.pkl",
        "scaler_y_direction_4.pkl",
        "scaler_y_direction_8.pkl"
    ]
    
    for scaler_name in scalers:
        scaler_path = f"trained_models/{scaler_name}"
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Scaler: {scaler_name}")
                print(f"   Type: {type(scaler).__name__}")
                models_loaded += 1
            else:
                print(f"❌ Not found: {scaler_name}")
        except Exception as e:
            print(f"❌ Error loading {scaler_name}: {str(e)[:100]}")
    
    print(f"\n📊 SUMMARY: {models_loaded} models loaded successfully")
    return models_loaded > 0

def test_system_integration():
    """Test system integration with models"""
    print("\n🔧 TESTING SYSTEM INTEGRATION...")
    print("=" * 50)
    
    try:
        # Test Ultimate XAU System
        print("Testing Ultimate XAU System...")
        sys.path.append('src/core')
        from ultimate_xau_system import UltimateXAUSystem
        
        system = UltimateXAUSystem()
        print("✅ Ultimate XAU System created")
        
        # Test Neural Network System
        if hasattr(system, 'neural_system'):
            print("✅ Neural Network System available")
            if hasattr(system.neural_system, 'models'):
                print(f"   Models count: {len(system.neural_system.models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration error: {str(e)[:200]}")
        return False

def test_master_integration():
    """Test master integration system"""
    print("\n🚀 TESTING MASTER INTEGRATION...")
    print("=" * 50)
    
    try:
        sys.path.append('src/core/integration')
        from master_system import create_development_system
        
        system = create_development_system()
        print("✅ Master Integration System created")
        
        # Check components
        if hasattr(system, 'components'):
            print(f"✅ Components available: {len(system.components)}")
            for name, component in system.components.items():
                print(f"   • {name}: {'✅ Active' if component else '❌ Inactive'}")
        
        # Check system state
        if hasattr(system, 'state'):
            print("✅ System state available")
            if hasattr(system.state, 'components_status'):
                print(f"   Status tracking: {len(system.state.components_status)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ Master integration error: {str(e)[:200]}")
        return False

def test_production_models():
    """Test production model loader"""
    print("\n🏭 TESTING PRODUCTION MODELS...")
    print("=" * 50)
    
    try:
        from production_model_loader import ProductionModelLoader
        
        loader = ProductionModelLoader()
        print("✅ Production Model Loader created")
        
        # Check active models
        active_models = loader.get_active_models()
        print(f"✅ Active models: {len(active_models)}")
        
        # Try to load best model
        best_model = loader.load_best_model()
        if best_model:
            print("✅ Best model loaded successfully")
        else:
            print("⚠️ No best model available")
        
        return True
        
    except Exception as e:
        print(f"❌ Production models error: {str(e)[:200]}")
        return False

def main():
    """Main test function"""
    print("🧪 MODEL INTEGRATION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    results = {}
    
    # Test individual model loading
    results['model_loading'] = test_model_loading()
    
    # Test system integration
    results['system_integration'] = test_system_integration()
    
    # Test master integration
    results['master_integration'] = test_master_integration()
    
    # Test production models
    results['production_models'] = test_production_models()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Models are properly integrated!")
    else:
        print("⚠️ SOME TESTS FAILED - Models may not be fully integrated")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main() 