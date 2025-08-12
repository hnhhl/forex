#!/usr/bin/env python3
"""
🚀 TEST GPU TRAINING - AI3.0 TRADING SYSTEM
Test GPU training với data thực và tích hợp vào system
"""

import os
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from datetime import datetime
import json

# Import GPU Neural System
from core.gpu_neural_system import GPUNeuralNetworkSystem

def load_training_data():
    """Load M1 data cho training"""
    print("📊 LOADING TRAINING DATA...")
    
    data_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return None
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} records")
    
    # Use last 10,000 records for quick training
    df = df.tail(10000).copy()
    print(f"📊 Using last {len(df):,} records for training")
    
    # Prepare features (OHLCV)
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    return features

def create_sequences(data, sequence_length=60):
    """Tạo sequences cho training"""
    print(f"🔄 CREATING SEQUENCES (length={sequence_length})...")
    
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # Input sequence
        X.append(data[i-sequence_length:i])
        
        # Target: price direction
        current_close = data[i, 3]  # close price
        previous_close = data[i-1, 3]
        y.append(1 if current_close > previous_close else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"✅ Created {len(X)} sequences: X={X.shape}, y={y.shape}")
    return X, y

def normalize_data(X):
    """Normalize features"""
    print("🔄 NORMALIZING DATA...")
    
    # Simple min-max normalization
    X_norm = np.zeros_like(X)
    
    for i in range(X.shape[2]):  # For each feature
        feature_data = X[:, :, i]
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)
        
        if max_val > min_val:
            X_norm[:, :, i] = (feature_data - min_val) / (max_val - min_val)
        else:
            X_norm[:, :, i] = feature_data
    
    print("✅ Data normalized")
    return X_norm

def test_gpu_training():
    """Test GPU training end-to-end"""
    print("🚀 GPU TRAINING TEST")
    print("=" * 50)
    print(f"🕒 Start Time: {datetime.now()}")
    
    # Load data
    data = load_training_data()
    if data is None:
        return False
    
    # Create sequences
    X, y = create_sequences(data)
    
    # Normalize
    X = normalize_data(X)
    
    # Train/validation split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Validation set: {X_val.shape}")
    
    # Initialize GPU Neural System
    print(f"\n🧠 INITIALIZING GPU NEURAL SYSTEM...")
    
    # Mock config
    config = type('Config', (), {})()
    
    gpu_system = GPUNeuralNetworkSystem(config)
    
    if not gpu_system.initialize():
        print("❌ GPU Neural System initialization failed!")
        return False
    
    # Train models
    print(f"\n🔥 TRAINING GPU MODELS...")
    training_results = gpu_system.train_gpu(X_train, y_train, X_val, y_val)
    
    if not training_results:
        print("❌ GPU training failed!")
        return False
    
    # Test predictions
    print(f"\n🔮 TESTING PREDICTIONS...")
    test_sample = X_val[:1]  # Single prediction
    
    ensemble_pred, confidence = gpu_system.get_ensemble_prediction(test_sample)
    
    print(f"   🎯 Ensemble Prediction: {ensemble_pred:.4f}")
    print(f"   📊 Confidence: {confidence:.4f}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': X.shape,
        'training_results': training_results,
        'test_prediction': {
            'prediction': float(ensemble_pred) if ensemble_pred else None,
            'confidence': float(confidence)
        },
        'gpu_available': gpu_system.is_gpu_available,
        'models_trained': len(training_results)
    }
    
    results_file = f"gpu_training_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    gpu_system.cleanup()
    
    print(f"\n✅ GPU TRAINING TEST COMPLETED!")
    print(f"📊 Models trained: {len(training_results)}")
    print(f"📄 Results saved: {results_file}")
    print(f"🕒 End Time: {datetime.now()}")
    
    return True

if __name__ == "__main__":
    success = test_gpu_training()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n❌ TESTS FAILED!") 