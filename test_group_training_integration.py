#!/usr/bin/env python3
"""
🧪 TEST GROUP TRAINING INTEGRATION
Kiểm tra tích hợp Group Training models với main system
"""

import sys
import os
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any

# Test imports
def test_imports():
    """Test import Group Training loader"""
    print("🧪 TESTING IMPORTS")
    print("-" * 50)
    
    try:
        from group_training_production_loader import group_training_loader
        print("✅ Group Training Loader imported successfully")
        
        # Test loader initialization
        print(f"✅ Model info count: {len(group_training_loader.model_info)}")
        print(f"✅ Device: {group_training_loader.device}")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\n🧪 TESTING PREDICTION")
    print("-" * 50)
    
    try:
        from group_training_production_loader import group_training_loader
        
        # Create dummy features (20 features)
        features = np.random.randn(20)
        
        # Test prediction
        result = group_training_loader.predict_ensemble(features)
        
        print(f"✅ Prediction result:")
        print(f"   • Prediction: {result['prediction']:.4f}")
        print(f"   • Confidence: {result['confidence']:.4f}")
        print(f"   • Signal: {result['signal']}")
        print(f"   • Model Count: {result['model_count']}")
        print(f"   • Method: {result['method']}")
        
        # Validate result structure
        required_keys = ['prediction', 'confidence', 'signal', 'model_count', 'method']
        for key in required_keys:
            if key not in result:
                print(f"❌ Missing key: {key}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_main_system_integration():
    """Test integration with main system"""
    print("\n🧪 TESTING MAIN SYSTEM INTEGRATION")
    print("-" * 50)
    
    try:
        # Test import from main system
        sys.path.append('src')
        from core.integration.master_system import MasterIntegrationSystem, SystemConfig
        
        print("✅ Main system imported successfully")
        
        # Check if GROUP_TRAINING_MODELS_AVAILABLE is True
        from core.integration.master_system import GROUP_TRAINING_MODELS_AVAILABLE
        print(f"✅ Group Training Available: {GROUP_TRAINING_MODELS_AVAILABLE}")
        
        if not GROUP_TRAINING_MODELS_AVAILABLE:
            print("❌ Group Training models not available in main system")
            return False
        
        # Test system initialization
        config = SystemConfig()
        system = MasterIntegrationSystem(config)
        
        print("✅ Master system initialized")
        
        # Test if _process_group_training_models method exists
        if hasattr(system, '_process_group_training_models'):
            print("✅ _process_group_training_models method found")
        else:
            print("❌ _process_group_training_models method not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Main system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_generation():
    """Test signal generation through main system"""
    print("\n🧪 TESTING SIGNAL GENERATION")
    print("-" * 50)
    
    try:
        sys.path.append('src')
        from core.integration.master_system import MasterIntegrationSystem, SystemConfig, MarketData
        
        # Create system
        config = SystemConfig()
        system = MasterIntegrationSystem(config)
        
        # Create dummy market data
        @dataclass
        class DummyMarketData:
            timestamp: datetime = datetime.now()
            symbol: str = "XAUUSD"
            price: float = 2000.0
            open_price: float = 1999.0
            high_price: float = 2001.0
            low_price: float = 1998.0
            volume: float = 1000.0
            technical_indicators: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.technical_indicators is None:
                    self.technical_indicators = {
                        'ma_5': 2000.5,
                        'ma_10': 2000.0,
                        'ma_20': 1999.5,
                        'ma_50': 1999.0,
                        'ma_100': 1998.5,
                        'ma_200': 1998.0,
                        'rsi': 55.0,
                        'bb_upper': 2002.0,
                        'bb_middle': 2000.0,
                        'bb_lower': 1998.0,
                        'macd': 0.5,
                        'macd_signal': 0.3,
                        'macd_histogram': 0.2,
                        'volume_ma': 1000.0,
                        'volatility': 0.5
                    }
        
        market_data = DummyMarketData()
        
        # Test Group Training signal generation
        signal = system._process_group_training_models(market_data)
        
        if signal:
            print("✅ Signal generated successfully:")
            print(f"   • Type: {signal.signal_type}")
            print(f"   • Confidence: {signal.confidence:.4f}")
            print(f"   • Source: {signal.source}")
            print(f"   • Risk Score: {signal.risk_score:.4f}")
            print(f"   • Metadata: {signal.metadata}")
            return True
        else:
            print("❌ No signal generated")
            return False
            
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_production_readiness():
    """Test production readiness"""
    print("\n🧪 TESTING PRODUCTION READINESS")
    print("-" * 50)
    
    try:
        # Check required files exist
        required_files = [
            'group_training_production_loader.py',
            'group_training_scaler.pkl',
            'group_training_config.json',
            'group_training_registry.json'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                print(f"✅ {file} exists")
            else:
                print(f"❌ {file} missing")
                return False
        
        # Test configuration
        import json
        with open('group_training_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"✅ Config loaded:")
        print(f"   • Enabled: {config['group_training_integration']['enabled']}")
        print(f"   • Model Count: {config['group_training_integration']['model_count']}")
        print(f"   • Best Accuracy: {config['group_training_integration']['best_accuracy']}")
        
        # Test registry
        with open('group_training_registry.json', 'r') as f:
            registry = json.load(f)
        
        print(f"✅ Registry loaded:")
        print(f"   • Total Models Trained: {registry['total_models_trained']}")
        print(f"   • Production Models: {registry['production_models']}")
        print(f"   • Registry Version: {registry['registry_version']}")
        
        return True
    except Exception as e:
        print(f"❌ Production readiness failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 GROUP TRAINING INTEGRATION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Prediction", test_prediction),
        ("Main System Integration", test_main_system_integration),
        ("Signal Generation", test_signal_generation),
        ("Production Readiness", test_production_readiness)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Group Training models are successfully integrated with main system!")
        print("✅ Ready for production deployment!")
    else:
        print(f"\n⚠️ {total-passed} TESTS FAILED")
        print("❌ Integration not complete - needs fixing")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 