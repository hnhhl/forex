#!/usr/bin/env python3
"""
🚀 GPU NEURAL TRAINING - AI3.0 TRADING SYSTEM
Train neural models sử dụng GPU NVIDIA GeForce GTX 1650 Ti
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('src')

import tensorflow as tf
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure GPU
def setup_gpu():
    """Setup GPU configuration tối ưu"""
    print("🔧 SETTING UP GPU...")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("❌ NO GPU FOUND!")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    try:
        # Enable memory growth để tránh OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth: Enabled")
        
        # Set memory limit (3GB từ 4GB total)
        tf.config.experimental.set_memory_limit(gpus[0], 3072)
        print("✅ GPU Memory Limit: 3GB")
        
    except RuntimeError as e:
        print(f"⚠️  GPU Config Warning: {e}")
    
    # Mixed precision để tăng tốc
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision: Enabled")
    except:
        print("⚠️  Mixed Precision: Disabled")
    
    # Test GPU
    with tf.device('/GPU:0'):
        try:
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("✅ GPU Test: PASSED")
            return True
        except Exception as e:
            print(f"❌ GPU Test Failed: {e}")
            return False

def load_training_data():
    """Load 3-year M1 data cho training"""
    print("📊 LOADING TRAINING DATA...")
    
    data_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return None
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} records")
    print(f"📅 Date range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    
    # Prepare features
    features = ['open', 'high', 'low', 'close', 'tick_volume']
    
    # Check columns
    if not all(col in df.columns for col in features):
        print(f"❌ Missing columns. Available: {list(df.columns)}")
        return None
    
    # Get features
    data = df[features].values.astype(np.float32)
    print(f"✅ Features shape: {data.shape}")
    
    return data

def create_sequences(data, sequence_length=60):
    """Tạo sequences cho training"""
    print(f"🔄 CREATING SEQUENCES (length={sequence_length})...")
    
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # Input sequence (60 timesteps, 5 features)
        X.append(data[i-sequence_length:i])
        
        # Target: price direction (1 if price goes up, 0 if down)
        current_close = data[i, 3]  # close price
        previous_close = data[i-1, 3]
        y.append(1 if current_close > previous_close else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"✅ Sequences created: X={X.shape}, y={y.shape}")
    return X, y

def normalize_data(X):
    """Normalize features"""
    print("🔄 NORMALIZING DATA...")
    
    # Normalize each feature independently
    X_norm = np.zeros_like(X)
    
    for i in range(X.shape[2]):  # For each feature
        feature_data = X[:, :, i]
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        
        if std > 0:
            X_norm[:, :, i] = (feature_data - mean) / std
        else:
            X_norm[:, :, i] = feature_data
    
    print("✅ Data normalized")
    return X_norm

def create_neural_models():
    """Tạo 3 neural models tối ưu cho GPU"""
    print("🧠 CREATING NEURAL MODELS...")
    
    models = {}
    
    # 1. LSTM Model
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 5)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')  # Mixed precision
    ])
    
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    models['lstm'] = lstm_model
    print("✅ LSTM Model created")
    
    # 2. CNN Model
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(60, 5)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    models['cnn'] = cnn_model
    print("✅ CNN Model created")
    
    # 3. Dense Model (Simple but effective)
    dense_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(60, 5)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    dense_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    models['dense'] = dense_model
    print("✅ Dense Model created")
    
    return models

def train_models_gpu(models, X_train, y_train, X_val, y_val):
    """Train models sử dụng GPU"""
    print("🚀 STARTING GPU TRAINING...")
    
    results = {}
    
    # Training parameters
    epochs = 50  # Tăng epochs với GPU
    batch_size = 64  # Tăng batch size với GPU
    
    for name, model in models.items():
        print(f"\n🔥 Training {name.upper()} Model...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6
            )
        ]
        
        # Train với GPU
        with tf.device('/GPU:0'):
            start_time = datetime.now()
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        results[name] = {
            'training_time': training_time,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"✅ {name.upper()} Training completed:")
        print(f"   ⏱️  Time: {training_time:.1f}s")
        print(f"   📊 Train Acc: {train_acc:.4f}")
        print(f"   📊 Val Acc: {val_acc:.4f}")
        
        # Save model
        model_path = f"trained_models/gpu_{name}_model.keras"
        model.save(model_path)
        print(f"   💾 Saved: {model_path}")
    
    return results

def main():
    """Main training function"""
    print("🚀 AI3.0 GPU NEURAL TRAINING")
    print("=" * 50)
    print(f"🕒 Start Time: {datetime.now()}")
    
    # 1. Setup GPU
    if not setup_gpu():
        print("❌ GPU Setup failed!")
        return
    
    # 2. Load data
    data = load_training_data()
    if data is None:
        print("❌ Data loading failed!")
        return
    
    # 3. Create sequences
    X, y = create_sequences(data)
    
    # 4. Normalize
    X = normalize_data(X)
    
    # 5. Train/validation split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Validation set: {X_val.shape}")
    
    # 6. Create models
    models = create_neural_models()
    
    # 7. Train models
    results = train_models_gpu(models, X_train, y_train, X_val, y_val)
    
    # 8. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gpu_training_results/gpu_training_{timestamp}.json"
    
    os.makedirs("gpu_training_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'gpu_info': 'NVIDIA GeForce GTX 1650 Ti',
            'data_shape': X.shape,
            'training_results': results
        }, f, indent=2)
    
    print(f"\n✅ GPU TRAINING COMPLETED!")
    print(f"📊 Results saved: {results_file}")
    print(f"🕒 End Time: {datetime.now()}")

if __name__ == "__main__":
    main() 