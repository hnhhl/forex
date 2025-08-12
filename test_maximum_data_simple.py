#!/usr/bin/env python3
"""
🔍 SIMPLE TEST FOR MAXIMUM DATA LOADING
Debug script để tìm lỗi trong việc load dữ liệu lớn
"""

import os
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test basic data loading"""
    print("📊 TESTING DATA LOADING...")
    print("=" * 50)
    
    try:
        # Test với sample nhỏ trước
        print("   🔄 Loading sample data (1000 rows)...")
        df_sample = pd.read_csv('data/working_free_data/XAUUSD_M1_realistic.csv', nrows=1000)
        print(f"   ✅ Sample loaded: {len(df_sample)} rows")
        print(f"   📊 Columns: {list(df_sample.columns)}")
        
        # Test với dữ liệu lớn hơn
        print("   🔄 Loading larger sample (10,000 rows)...")
        df_large = pd.read_csv('data/working_free_data/XAUUSD_M1_realistic.csv', nrows=10000)
        print(f"   ✅ Large sample loaded: {len(df_large)} rows")
        
        # Test toàn bộ file
        print("   🔄 Loading full dataset...")
        df_full = pd.read_csv('data/working_free_data/XAUUSD_M1_realistic.csv')
        print(f"   ✅ Full dataset loaded: {len(df_full):,} rows")
        print(f"   💾 Memory usage: {df_full.memory_usage().sum() / 1024 / 1024:.1f} MB")
        
        return df_full
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None

def test_data_preprocessing(df, max_rows=5000):
    """Test data preprocessing với sample nhỏ"""
    print(f"\n🔧 TESTING DATA PREPROCESSING...")
    print("=" * 50)
    
    try:
        # Limit rows để test
        if len(df) > max_rows:
            df = df.tail(max_rows).copy()
            print(f"   📊 Using last {max_rows:,} rows for testing")
        
        # Extract features
        print("   🔄 Extracting features...")
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        print(f"   ✅ Features shape: {features.shape}")
        
        # Normalize
        print("   🔄 Normalizing data...")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        print(f"   ✅ Data normalized")
        
        # Create sequences
        print("   🔄 Creating sequences...")
        sequence_length = 60
        X, y = [], []
        
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            
            current_close = features_scaled[i-1, 3]
            next_close = features_scaled[i, 3]
            y.append(1 if next_close > current_close else 0)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"   ✅ Sequences created:")
        print(f"      X shape: {X.shape}")
        print(f"      y shape: {y.shape}")
        print(f"      Memory: {(X.nbytes + y.nbytes) / 1024 / 1024:.1f} MB")
        
        return X, y, scaler
        
    except Exception as e:
        print(f"   ❌ Error in preprocessing: {e}")
        return None, None, None

def test_gpu_setup():
    """Test GPU setup"""
    print(f"\n🚀 TESTING GPU SETUP...")
    print("=" * 50)
    
    try:
        # Check GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ GPU Available: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu}")
            
            # Configure GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Test mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f"   ✅ Mixed precision enabled")
            
            return True
        else:
            print(f"   ⚠️  No GPU available, using CPU")
            return False
            
    except Exception as e:
        print(f"   ❌ GPU setup error: {e}")
        return False

def test_simple_model(X, y):
    """Test simple model training"""
    print(f"\n🧠 TESTING SIMPLE MODEL...")
    print("=" * 50)
    
    try:
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"   📊 Training samples: {len(X_train):,}")
        print(f"   📊 Validation samples: {len(X_val):,}")
        
        # Create simple model
        print("   🔄 Creating simple model...")
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(60, 5)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✅ Model created: {model.count_params():,} parameters")
        
        # Train
        print("   🔄 Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"   ✅ Training completed:")
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Val Accuracy: {val_acc:.4f}")
        
        return model, val_acc
        
    except Exception as e:
        print(f"   ❌ Model training error: {e}")
        return None, 0

def test_signal_generation(model, X, y):
    """Test signal generation"""
    print(f"\n🔮 TESTING SIGNAL GENERATION...")
    print("=" * 50)
    
    try:
        # Test trên sample nhỏ
        test_samples = X[-10:]
        test_actual = y[-10:]
        
        print(f"   🔄 Testing on {len(test_samples)} samples...")
        
        predictions = model.predict(test_samples, verbose=0)
        
        correct_count = 0
        for i, (pred, actual) in enumerate(zip(predictions, test_actual)):
            signal = "BUY" if pred[0] > 0.5 else "SELL"
            actual_move = "UP" if actual == 1 else "DOWN"
            correct = (pred[0] > 0.5) == (actual == 1)
            
            if correct:
                correct_count += 1
            
            status = "✅" if correct else "❌"
            print(f"      Sample {i+1}: {signal} ({pred[0]:.3f}) | Actual: {actual_move} {status}")
        
        accuracy = correct_count / len(test_samples)
        print(f"   🎯 Signal Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return accuracy
        
    except Exception as e:
        print(f"   ❌ Signal generation error: {e}")
        return 0

def main():
    """Main test execution"""
    print("🔍 SIMPLE MAXIMUM DATA TEST")
    print("=" * 60)
    print(f"🕒 Started: {datetime.now()}")
    print()
    
    # Test 1: Data Loading
    df = test_data_loading()
    if df is None:
        print("❌ Data loading failed - stopping tests")
        return
    
    # Test 2: Data Preprocessing
    X, y, scaler = test_data_preprocessing(df, max_rows=5000)
    if X is None:
        print("❌ Data preprocessing failed - stopping tests")
        return
    
    # Test 3: GPU Setup
    gpu_available = test_gpu_setup()
    
    # Test 4: Simple Model
    model, val_acc = test_simple_model(X, y)
    if model is None:
        print("❌ Model training failed - stopping tests")
        return
    
    # Test 5: Signal Generation
    signal_acc = test_signal_generation(model, X, y)
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 60)
    print(f"   Data Loading: ✅ SUCCESS ({len(df):,} records)")
    print(f"   Data Preprocessing: ✅ SUCCESS ({X.shape[0]:,} sequences)")
    print(f"   GPU Setup: {'✅ SUCCESS' if gpu_available else '⚠️ CPU ONLY'}")
    print(f"   Model Training: ✅ SUCCESS ({val_acc:.4f} accuracy)")
    print(f"   Signal Generation: ✅ SUCCESS ({signal_acc:.4f} accuracy)")
    
    overall_score = (val_acc + signal_acc) / 2
    
    if overall_score >= 0.6:
        status = "🟢 GOOD - Ready for full training"
    elif overall_score >= 0.5:
        status = "🟡 OK - Needs optimization"
    else:
        status = "🔴 POOR - Needs significant work"
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Score: {overall_score:.4f}")
    print(f"   Status: {status}")
    
    print(f"\n✅ All tests completed successfully!")
    print(f"🕒 Finished: {datetime.now()}")

if __name__ == "__main__":
    main() 