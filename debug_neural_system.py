#!/usr/bin/env python3
"""
Debug Neural Network System trong AI3.0
"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np

def debug_neural_system():
    """Debug neural network system"""
    print("🔍 DEBUG NEURAL NETWORK SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize system
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized")
        
        # Check system manager
        if hasattr(system, 'system_manager'):
            print(f"✅ System manager found")
            print(f"📊 Registered systems: {list(system.system_manager.systems.keys())}")
            
            # Check neural network system specifically
            neural_system = system.system_manager.systems.get('neural_network')
            if neural_system:
                print(f"✅ Neural network system found!")
                print(f"📊 Type: {type(neural_system)}")
                
                # Check models
                if hasattr(neural_system, 'models'):
                    print(f"📊 Models attribute exists: {bool(neural_system.models)}")
                    if neural_system.models:
                        print(f"📊 Available models: {list(neural_system.models.keys())}")
                    else:
                        print(f"❌ Models dict is empty")
                else:
                    print(f"❌ No models attribute")
                
                # Test feature preparation
                print(f"\n🧪 Testing feature preparation:")
                
                # Get sample data
                market_data = system._get_comprehensive_market_data(config.symbol)
                if market_data is not None:
                    print(f"📊 Market data shape: {market_data.shape}")
                    print(f"📊 Columns: {list(market_data.columns)}")
                    
                    # Test _prepare_features
                    if hasattr(neural_system, '_prepare_features'):
                        features = neural_system._prepare_features(market_data)
                        if features is not None:
                            print(f"✅ Features prepared: {features.shape}")
                            
                            # Test each model if available
                            if neural_system.models:
                                for model_name, model in neural_system.models.items():
                                    try:
                                        prediction = neural_system._predict_with_model(model_name, model, market_data)
                                        print(f"✅ {model_name}: prediction={prediction.get('prediction', 0):.3f}")
                                    except Exception as e:
                                        print(f"❌ {model_name}: {str(e)[:100]}")
                        else:
                            print(f"❌ Feature preparation failed")
                    else:
                        print(f"❌ No _prepare_features method")
                else:
                    print(f"❌ No market data")
                    
            else:
                print(f"❌ Neural network system NOT found")
                print(f"📊 Available systems: {list(system.system_manager.systems.keys())}")
        else:
            print(f"❌ No system manager")
        
        # Direct test of signal generation
        print(f"\n🎯 Direct signal generation test:")
        signal = system.generate_signal(config.symbol)
        print(f"📊 Signal: {signal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_neural_system() 