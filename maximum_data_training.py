#!/usr/bin/env python3
"""
🚀 MAXIMUM DATA TRAINING & SIGNAL GENERATION
Training hệ thống AI3.0 trên dữ liệu tối đa (1.1M+ records) và test tín hiệu
"""

import os
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def load_maximum_data():
    """Load dữ liệu tối đa từ tất cả sources"""
    print("📊 LOADING MAXIMUM DATA...")
    print("=" * 60)
    
    # Load file M1 chính (1.1M+ records)
    main_file = 'data/working_free_data/XAUUSD_M1_realistic.csv'
    
    if not os.path.exists(main_file):
        print(f"❌ Main data file not found: {main_file}")
        return None
    
    try:
        print(f"   🔄 Loading main file: {main_file}")
        df = pd.read_csv(main_file)
        print(f"   ✅ Loaded: {len(df):,} records")
        print(f"   📊 Columns: {list(df.columns)}")
        print(f"   💾 Memory: {df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        
        # Kiểm tra data quality
        print(f"   📊 Data Quality Check:")
        print(f"      Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"      Missing values: {df.isnull().sum().sum()}")
        print(f"      Duplicates: {df.duplicated().sum()}")
        
        # Create datetime column
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            print(f"   ✅ Datetime column created")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None

def prepare_large_dataset(df: pd.DataFrame, sequence_length: int = 60, max_samples: int = None):
    """Chuẩn bị dataset lớn với optimization"""
    print(f"\n🔧 PREPARING LARGE DATASET...")
    print("=" * 60)
    
    # Limit samples nếu cần (để tránh memory issues)
    if max_samples and len(df) > max_samples:
        print(f"   ⚠️  Limiting to {max_samples:,} samples (from {len(df):,})")
        df = df.tail(max_samples).copy()
    
    # Extract features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = df[feature_cols].values.astype(np.float32)
    print(f"   📊 Features shape: {features.shape}")
    
    # Normalize data
    print(f"   🔄 Normalizing data...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"   ✅ Data normalized")
    
    # Create sequences in batches để tránh memory overflow
    print(f"   🔄 Creating sequences (length={sequence_length})...")
    
    batch_size = 10000  # Process in batches
    X_batches, y_batches = [], []
    
    total_sequences = len(features_scaled) - sequence_length
    processed = 0
    
    for start_idx in range(sequence_length, len(features_scaled), batch_size):
        end_idx = min(start_idx + batch_size, len(features_scaled))
        
        batch_X, batch_y = [], []
        
        for i in range(max(start_idx, sequence_length), end_idx):
            batch_X.append(features_scaled[i-sequence_length:i])
            
            # Target: price direction (1 if up, 0 if down)
            current_close = features_scaled[i-1, 3]
            next_close = features_scaled[i, 3]
            batch_y.append(1 if next_close > current_close else 0)
        
        if batch_X:
            X_batches.append(np.array(batch_X, dtype=np.float32))
            y_batches.append(np.array(batch_y, dtype=np.float32))
            
            processed += len(batch_X)
            print(f"      Progress: {processed:,}/{total_sequences:,} ({processed/total_sequences*100:.1f}%)")
    
    # Combine all batches
    print(f"   🔄 Combining batches...")
    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    
    print(f"   ✅ Final dataset:")
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Memory usage: {(X.nbytes + y.nbytes) / 1024 / 1024:.1f} MB")
    print(f"      Positive samples: {np.sum(y):.0f} ({np.mean(y)*100:.1f}%)")
    
    # Train/validation split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Data split:")
    print(f"      Training: {X_train.shape[0]:,} samples")
    print(f"      Validation: {X_val.shape[0]:,} samples")
    
    return X_train, X_val, y_train, y_val, scaler

def create_optimized_gpu_models():
    """Tạo models GPU optimized cho dataset lớn"""
    print(f"\n🚀 CREATING OPTIMIZED GPU MODELS...")
    print("=" * 60)
    
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"   🎯 GPU Available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"   ⚡ Mixed precision enabled")
    
    models = {}
    
    # 1. Efficient LSTM Model
    print(f"   🧠 Creating Efficient LSTM Model...")
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(60, 5)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['lstm'] = lstm_model
    print(f"      ✅ LSTM: {lstm_model.count_params():,} parameters")
    
    # 2. Efficient CNN Model
    print(f"   🔍 Creating Efficient CNN Model...")
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(60, 5)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['cnn'] = cnn_model
    print(f"      ✅ CNN: {cnn_model.count_params():,} parameters")
    
    # 3. Dense Model (fast baseline)
    print(f"   ⚡ Creating Dense Model...")
    dense_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(60, 5)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    dense_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['dense'] = dense_model
    print(f"      ✅ Dense: {dense_model.count_params():,} parameters")
    
    total_params = sum(model.count_params() for model in models.values())
    print(f"\n   📊 Total Models: {len(models)}")
    print(f"   📊 Total Parameters: {total_params:,}")
    
    return models

def train_on_large_dataset(models: Dict, X_train, X_val, y_train, y_val):
    """Training trên dataset lớn với optimization"""
    print(f"\n🎯 TRAINING ON LARGE DATASET...")
    print("=" * 60)
    
    training_results = {}
    
    for model_name, model in models.items():
        print(f"\n   🚀 Training {model_name.upper()} Model...")
        print(f"   {'-' * 50}")
        
        start_time = time.time()
        
        # Callbacks for large dataset
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'trained_models/checkpoint_{model_name}.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Training với batch size tối ưu cho dataset lớn
        batch_size = 64 if model_name == 'lstm' else 128
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,  # Reduced epochs for large dataset
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        # Save final model
        model_path = f"trained_models/maximum_data_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        model.save(model_path)
        
        results = {
            'training_time': training_time,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'epochs_trained': len(history.history['loss']),
            'model_path': model_path,
            'parameters': model.count_params(),
            'batch_size': batch_size
        }
        
        training_results[model_name] = results
        
        print(f"      ✅ Completed in {training_time:.1f}s")
        print(f"      📊 Train Accuracy: {train_acc:.4f}")
        print(f"      📊 Val Accuracy: {val_acc:.4f}")
        print(f"      💾 Saved: {model_path}")
    
    return training_results

def test_signal_generation_advanced(models: Dict, X_val, y_val):
    """Test advanced signal generation"""
    print(f"\n🔮 ADVANCED SIGNAL GENERATION TEST...")
    print("=" * 60)
    
    # Ensemble prediction function
    def ensemble_predict(X_batch):
        predictions = []
        for model in models.values():
            pred = model.predict(X_batch, verbose=0)
            predictions.append(pred)
        
        # Average ensemble
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Confidence based on agreement
        std_dev = np.std(predictions, axis=0)
        confidence = 1 - (std_dev * 2)  # Higher agreement = higher confidence
        confidence = np.clip(confidence, 0, 1)
        
        return ensemble_pred, confidence
    
    # Test trên validation set (sample để tránh memory issues)
    test_size = min(1000, len(X_val))
    X_test = X_val[-test_size:]
    y_test = y_val[-test_size:]
    
    print(f"   🔄 Testing on {test_size:,} samples...")
    
    # Generate predictions
    predictions, confidences = ensemble_predict(X_test)
    
    # Convert to signals
    signals = (predictions > 0.5).astype(int).flatten()
    actual = y_test.astype(int)
    
    # Calculate metrics
    accuracy = np.mean(signals == actual)
    
    # Signal analysis
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == 0)
    
    # Confidence analysis
    high_conf = np.sum(confidences > 0.7)
    med_conf = np.sum((confidences >= 0.5) & (confidences <= 0.7))
    low_conf = np.sum(confidences < 0.5)
    
    print(f"   📊 SIGNAL RESULTS:")
    print(f"      Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"      Buy Signals: {buy_signals:,} ({buy_signals/len(signals)*100:.1f}%)")
    print(f"      Sell Signals: {sell_signals:,} ({sell_signals/len(signals)*100:.1f}%)")
    
    print(f"   📊 CONFIDENCE ANALYSIS:")
    print(f"      High Confidence (>70%): {high_conf:,} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"      Medium Confidence (50-70%): {med_conf:,} ({med_conf/len(confidences)*100:.1f}%)")
    print(f"      Low Confidence (<50%): {low_conf:,} ({low_conf/len(confidences)*100:.1f}%)")
    
    # Real-time test
    print(f"\n   🔄 Real-time Signal Test (Last 10 samples)...")
    
    recent_signals = []
    for i in range(-10, 0):
        sample = X_test[i:i+1]
        actual_val = y_test[i]
        
        pred, conf = ensemble_predict(sample)
        
        signal = "BUY" if pred[0] > 0.5 else "SELL"
        actual_move = "UP" if actual_val == 1 else "DOWN"
        correct = "✅" if (pred[0] > 0.5) == (actual_val == 1) else "❌"
        
        recent_signals.append({
            'signal': signal,
            'confidence': float(conf[0]),
            'prediction': float(pred[0]),
            'actual': actual_move,
            'correct': correct == '✅'
        })
        
        print(f"      Signal: {signal} | Conf: {conf[0]:.3f} | Actual: {actual_move} {correct}")
    
    real_time_accuracy = np.mean([s['correct'] for s in recent_signals])
    
    results = {
        'overall_accuracy': float(accuracy),
        'buy_signals_pct': float(buy_signals/len(signals)*100),
        'sell_signals_pct': float(sell_signals/len(signals)*100),
        'high_confidence_pct': float(high_conf/len(confidences)*100),
        'medium_confidence_pct': float(med_conf/len(confidences)*100),
        'low_confidence_pct': float(low_conf/len(confidences)*100),
        'real_time_accuracy': float(real_time_accuracy),
        'recent_signals': recent_signals,
        'test_samples': test_size
    }
    
    print(f"   🎯 Real-time Accuracy: {real_time_accuracy:.4f} ({real_time_accuracy*100:.2f}%)")
    
    return results

def generate_final_report(training_results: Dict, signal_results: Dict, data_info: Dict):
    """Generate comprehensive final report"""
    print(f"\n📋 MAXIMUM DATA TRAINING REPORT")
    print("=" * 70)
    
    # Best model
    best_model = max(training_results.keys(), key=lambda k: training_results[k]['val_accuracy'])
    best_accuracy = training_results[best_model]['val_accuracy']
    
    # Training summary
    total_time = sum(r['training_time'] for r in training_results.values())
    total_params = sum(r['parameters'] for r in training_results.values())
    
    print(f"🎯 TRAINING SUMMARY:")
    print(f"   Dataset Size: {data_info.get('total_records', 'N/A'):,} records")
    print(f"   Training Samples: {data_info.get('training_samples', 'N/A'):,}")
    print(f"   Validation Samples: {data_info.get('validation_samples', 'N/A'):,}")
    print(f"   Models Trained: {len(training_results)}")
    print(f"   Best Model: {best_model.upper()} ({best_accuracy:.4f})")
    print(f"   Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Total Parameters: {total_params:,}")
    
    print(f"\n📊 MODEL PERFORMANCE:")
    for model_name, results in training_results.items():
        print(f"   {model_name.upper()}:")
        print(f"      Val Accuracy: {results['val_accuracy']:.4f}")
        print(f"      Training Time: {results['training_time']:.1f}s")
        print(f"      Parameters: {results['parameters']:,}")
    
    print(f"\n🔮 SIGNAL GENERATION:")
    print(f"   Overall Accuracy: {signal_results['overall_accuracy']:.4f}")
    print(f"   Real-time Accuracy: {signal_results['real_time_accuracy']:.4f}")
    print(f"   High Confidence Signals: {signal_results['high_confidence_pct']:.1f}%")
    
    # Overall assessment
    overall_score = (
        best_accuracy + 
        signal_results['overall_accuracy'] + 
        signal_results['real_time_accuracy']
    ) / 3
    
    if overall_score >= 0.75:
        grade = "A - EXCELLENT"
        status = "🟢 PRODUCTION READY"
    elif overall_score >= 0.65:
        grade = "B - GOOD"
        status = "🟡 NEAR PRODUCTION"
    elif overall_score >= 0.55:
        grade = "C - SATISFACTORY"
        status = "🟠 NEEDS OPTIMIZATION"
    else:
        grade = "D - NEEDS WORK"
        status = "🔴 REQUIRES IMPROVEMENT"
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    print(f"   Overall Score: {overall_score:.4f}")
    print(f"   Grade: {grade}")
    print(f"   Status: {status}")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_info': data_info,
        'training_results': training_results,
        'signal_results': signal_results,
        'summary': {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'overall_score': overall_score,
            'grade': grade,
            'status': status
        }
    }
    
    filename = f"maximum_data_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved: {filename}")
    return report

def main():
    """Main execution"""
    print("🚀 MAXIMUM DATA TRAINING & SIGNAL GENERATION")
    print("=" * 70)
    print(f"🕒 Started: {datetime.now()}")
    print()
    
    # 1. Load maximum data
    df = load_maximum_data()
    if df is None:
        return
    
    # 2. Prepare large dataset (limit to manageable size)
    max_samples = 100000  # Use 100K samples để tránh memory issues
    X_train, X_val, y_train, y_val, scaler = prepare_large_dataset(df, max_samples=max_samples)
    
    data_info = {
        'total_records': len(df),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'max_samples_used': max_samples
    }
    
    # 3. Create optimized models
    models = create_optimized_gpu_models()
    
    # 4. Train on large dataset
    training_results = train_on_large_dataset(models, X_train, X_val, y_train, y_val)
    
    # 5. Test signal generation
    signal_results = test_signal_generation_advanced(models, X_val, y_val)
    
    # 6. Generate final report
    report = generate_final_report(training_results, signal_results, data_info)
    
    print(f"\n✅ MAXIMUM DATA TRAINING COMPLETED!")
    print(f"🕒 Finished: {datetime.now()}")
    print(f"📊 Overall Score: {report['summary']['overall_score']:.4f}")
    print(f"🏆 Final Grade: {report['summary']['grade']}")

if __name__ == "__main__":
    main() 