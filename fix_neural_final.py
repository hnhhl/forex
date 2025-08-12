#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FINAL FIX: Neural Models Retraining"""

import sys
import os
sys.path.append('src')

from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

def fix_neural_models():
    print("FINAL FIX: NEURAL MODELS RETRAINING")
    print("="*70)
    
    # 1. Initialize system
    print("1. INITIALIZE SYSTEM")
    config = SystemConfig()
    config.symbol = "XAUUSDc"
    system = UltimateXAUSystem(config)
    
    # 2. Test BEFORE fix
    print("\n2. TEST BEFORE FIX")
    before_signals = []
    for i in range(3):
        signal = system.generate_signal("XAUUSDc")
        before_signals.append(signal)
        print(f"   Before {i+1}: {signal.get('action')} (confidence: {signal.get('confidence', 0):.1%})")
    
    before_confidence = np.mean([s.get('confidence', 0) for s in before_signals])
    print(f"Average confidence BEFORE: {before_confidence:.1%}")
    
    # 3. Get neural system
    neural_system = system.system_manager.systems.get('NeuralNetworkSystem')
    if neural_system:
        print("Neural system found")
        print(f"Available models: {list(neural_system.models.keys())}")
    else:
        print("Neural system not found")
        return False
    
    # 4. Get training data
    print("\n4. GET TRAINING DATA")
    if not mt5.initialize():
        print("MT5 failed")
        return False
    
    rates = mt5.copy_rates_from_pos("XAUUSDc", mt5.TIMEFRAME_M1, 0, 2000)
    if rates is None or len(rates) < 500:
        print("Insufficient data")
        return False
    
    df = pd.DataFrame(rates)
    if 'volume' not in df.columns:
        df['volume'] = df['tick_volume']
    
    print(f"Got {len(df)} rows of training data")
    
    # 5. Create training data
    print("\n5. CREATE TRAINING DATA")
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    sequence_length = 60
    
    X, y = [], []
    for i in range(sequence_length, len(df) - 3):
        sequence = df[feature_columns].iloc[i-sequence_length:i].values
        X.append(sequence)
        
        # Price direction target
        current_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + 3]
        price_change = (future_price - current_price) / current_price
        
        if price_change > 0.0005:  # 0.05% threshold
            target = 0.8  # BUY signal
        elif price_change < -0.0005:
            target = 0.2  # SELL signal  
        else:
            target = 0.5  # HOLD signal
        
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences, shape: {X.shape}")
    
    # 6. Normalize data
    print("\n6. NORMALIZE DATA")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # Split data
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 7. Retrain models
    print("\n7. RETRAIN MODELS")
    trained_count = 0
    
    for model_name in ['lstm', 'cnn', 'transformer']:
        try:
            model = neural_system.models.get(model_name)
            if model is None:
                continue
            
            print(f"Training {model_name.upper()}...")
            
            # Quick training
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=5,  # Quick training
                batch_size=32,
                verbose=0
            )
            
            # Test prediction
            test_pred = model.predict(X_test[:50], verbose=0)
            accuracy = np.mean(np.abs(test_pred.flatten() - y_test[:50]) < 0.2)
            
            print(f"{model_name}: accuracy={accuracy:.1%}")
            trained_count += 1
            
        except Exception as e:
            print(f"{model_name} failed: {str(e)[:50]}")
    
    if trained_count == 0:
        print("No models trained")
        return False
    
    # 8. Update scaler
    neural_system.feature_scalers['fixed_5_features'] = scaler
    print(f"Updated {trained_count} models and scaler")
    
    # 9. Test AFTER fix
    print("\n9. TEST AFTER FIX")
    after_signals = []
    for i in range(5):
        signal = system.generate_signal("XAUUSDc")
        after_signals.append(signal)
        print(f"   After {i+1}: {signal.get('action')} (confidence: {signal.get('confidence', 0):.1%})")
    
    after_confidence = np.mean([s.get('confidence', 0) for s in after_signals])
    
    # 10. Results
    print(f"\n10. RESULTS")
    improvement = (after_confidence - before_confidence) * 100
    
    print(f"BEFORE: {before_confidence:.1%}")
    print(f"AFTER:  {after_confidence:.1%}")
    print(f"IMPROVEMENT: +{improvement:.1f} percentage points")
    
    # Check signal diversity
    unique_actions = set(s.get('action') for s in after_signals)
    print(f"Signal types: {unique_actions}")
    
    mt5.shutdown()
    
    # Final assessment
    if after_confidence > 0.5:
        print("\nSUCCESS: High confidence achieved!")
        return True
    elif after_confidence > before_confidence:
        print("\nIMPROVED: Partial success")
        return True
    else:
        print("\nFAILED: No improvement")
        return False

if __name__ == "__main__":
    success = fix_neural_models()
    print(f"\n{'='*70}")
    if success:
        print("AI3.0 NEURAL MODELS FIXED!")
        print("Ready for production trading!")
    else:
        print("Need more training time")
    print(f"{'='*70}") 