#!/usr/bin/env python3
"""
Retrain Neural Models với 5 features để có confidence cao
"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime

def retrain_neural_models_5_features():
    """Retrain neural models với 5 features"""
    print("🚀 RETRAIN NEURAL MODELS VỚI 5 FEATURES")
    print("=" * 70)
    
    try:
        # 1. Initialize system
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized")
        
        # 2. Get neural network system
        neural_system = system.system_manager.systems.get('NeuralNetworkSystem')
        if not neural_system:
            print("❌ Neural network system not found")
            return False
        
        print("✅ Neural network system found")
        
        # 3. Prepare training data với 5 features
        print("\n📊 PREPARING TRAINING DATA")
        print("-" * 50)
        
        # Get comprehensive data
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        
        # Get more data for training
        rates = mt5.copy_rates_from_pos("XAUUSDc", mt5.TIMEFRAME_M1, 0, 5000)
        if rates is None:
            print("❌ Cannot get training data")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Ensure 5 features
        if 'volume' not in df.columns:
            df['volume'] = df['tick_volume']
        
        print(f"📊 Training data shape: {df.shape}")
        print(f"📊 Features: {['open', 'high', 'low', 'close', 'volume']}")
        
        # 4. Create training sequences
        print("\n🔧 CREATING TRAINING SEQUENCES")
        print("-" * 50)
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        sequence_length = 60
        
        # Prepare sequences
        X, y = [], []
        
        for i in range(sequence_length, len(df) - 1):
            # Input sequence (60 timesteps, 5 features)
            sequence = df[feature_columns].iloc[i-sequence_length:i].values
            X.append(sequence)
            
            # Target (future price direction)
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + 1]
            direction = 1 if future_price > current_price else 0
            y.append(direction)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Training sequences created")
        print(f"📊 X shape: {X.shape}")  # Should be (samples, 60, 5)
        print(f"📊 y shape: {y.shape}")
        
        # Normalize features
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        # Reshape for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        print(f"✅ Features normalized")
        
        # 5. Train models
        print(f"\n🤖 TRAINING NEURAL MODELS")
        print("-" * 50)
        
        # Split data
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train each model
        trained_models = {}
        
        for model_name in ['lstm', 'cnn', 'transformer']:
            try:
                print(f"\n🔧 Training {model_name.upper()}...")
                
                # Get model
                model = neural_system.models.get(model_name)
                if model is None:
                    print(f"❌ {model_name} model not found")
                    continue
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,  # Quick training
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate
                train_loss = history.history['loss'][-1]
                val_loss = history.history['val_loss'][-1]
                
                # Test prediction
                test_pred = model.predict(X_test[:10], verbose=0)
                test_accuracy = np.mean((test_pred > 0.5) == y_test[:10])
                
                trained_models[model_name] = {
                    'model': model,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'test_accuracy': test_accuracy
                }
                
                print(f"✅ {model_name}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={test_accuracy:.1%}")
                
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
        
        # 6. Update neural system
        print(f"\n🔄 UPDATING NEURAL SYSTEM")
        print("-" * 50)
        
        # Update models in neural system
        for model_name, model_info in trained_models.items():
            neural_system.models[model_name] = model_info['model']
        
        # Update scaler
        neural_system.feature_scalers['fixed_5_features'] = scaler
        
        print(f"✅ Neural system updated with trained models")
        
        # 7. Test new system performance
        print(f"\n🧪 TEST NEW SYSTEM PERFORMANCE")
        print("-" * 50)
        
        # Generate signals
        signals = []
        for i in range(5):
            signal = system.generate_signal("XAUUSDc")
            signals.append(signal)
            print(f"Signal {i+1}: {signal.get('action', 'N/A')} (confidence: {signal.get('confidence', 0):.1%})")
        
        # Calculate improvements
        confidences = [s.get('confidence', 0) for s in signals]
        avg_confidence = np.mean(confidences)
        
        print(f"\n📊 RESULTS:")
        print(f"   • Average Confidence: {avg_confidence:.1%}")
        print(f"   • Trained Models: {list(trained_models.keys())}")
        print(f"   • Training Data: {len(X)} sequences with 5 features")
        
        if avg_confidence > 0.5:
            print(f"✅ SUCCESS: High confidence achieved!")
        elif avg_confidence > 0.3:
            print(f"⚠️ PARTIAL SUCCESS: Moderate improvement")
        else:
            print(f"❌ NEEDS MORE TRAINING: Low confidence")
        
        mt5.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Retrain failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = retrain_neural_models_5_features()
    if success:
        print(f"\n🎉 NEURAL MODELS RETRAINED SUCCESSFULLY!")
        print(f"🎯 AI3.0 neural models now trained with proper 5 features")
        print(f"🎯 Expected: Higher confidence, better BUY/SELL signals")
    else:
        print(f"\n❌ RETRAINING FAILED") 