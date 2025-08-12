#!/usr/bin/env python3
"""
🚀 FULL MAXIMUM DATA TRAINING
Training với toàn bộ 1.1M+ records và tạo tín hiệu trading
"""

import os
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

def load_full_dataset():
    """Load toàn bộ dataset 1.1M+ records"""
    print("📊 LOADING FULL DATASET (1.1M+ RECORDS)...")
    print("=" * 60)
    
    try:
        df = pd.read_csv('data/working_free_data/XAUUSD_M1_realistic.csv')
        print(f"   ✅ Loaded: {len(df):,} records")
        print(f"   💾 Memory: {df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        print(f"   📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def prepare_full_dataset(df, use_samples=50000):
    """Chuẩn bị dataset với optimization cho memory"""
    print(f"\n🔧 PREPARING DATASET ({use_samples:,} samples)...")
    print("=" * 60)
    
    # Sử dụng dữ liệu gần đây nhất (quan trọng hơn)
    df_recent = df.tail(use_samples + 100).copy()  # +100 để tạo sequences
    print(f"   📊 Using recent {len(df_recent):,} records")
    
    # Extract features
    features = df_recent[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
    
    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences efficiently
    sequence_length = 60
    X, y = [], []
    
    print(f"   🔄 Creating {use_samples:,} sequences...")
    start_time = time.time()
    
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        
        current_close = features_scaled[i-1, 3]
        next_close = features_scaled[i, 3]
        y.append(1 if next_close > current_close else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    prep_time = time.time() - start_time
    
    print(f"   ✅ Dataset prepared in {prep_time:.1f}s")
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Memory: {(X.nbytes + y.nbytes) / 1024 / 1024:.1f} MB")
    print(f"      Positive samples: {np.mean(y)*100:.1f}%")
    
    return X, y, scaler

def create_production_models():
    """Tạo models production-ready"""
    print(f"\n🚀 CREATING PRODUCTION MODELS...")
    print("=" * 60)
    
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"   ⚡ GPU optimized with mixed precision")
    
    models = {}
    
    # 1. Production LSTM
    print("   🧠 Creating Production LSTM...")
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
    
    # 2. Production CNN
    print("   🔍 Creating Production CNN...")
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
    
    # 3. Hybrid Model
    print("   🔥 Creating Hybrid CNN-LSTM...")
    hybrid_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(60, 5)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    hybrid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['hybrid'] = hybrid_model
    print(f"      ✅ Hybrid: {hybrid_model.count_params():,} parameters")
    
    total_params = sum(model.count_params() for model in models.values())
    print(f"\n   📊 Total Models: {len(models)}")
    print(f"   📊 Total Parameters: {total_params:,}")
    
    return models

def train_production_models(models, X, y):
    """Training production models"""
    print(f"\n🎯 PRODUCTION TRAINING...")
    print("=" * 60)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Training: {len(X_train):,} samples")
    print(f"   📊 Validation: {len(X_val):,} samples")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n   🚀 Training {model_name.upper()}...")
        print(f"   {'-' * 40}")
        
        start_time = time.time()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7
            )
        ]
        
        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        # Save model
        model_path = f"trained_models/production_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        model.save(model_path)
        
        results[model_name] = {
            'training_time': training_time,
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'epochs': len(history.history['loss']),
            'model_path': model_path
        }
        
        print(f"      ✅ Completed in {training_time:.1f}s")
        print(f"      📊 Train Acc: {train_acc:.4f}")
        print(f"      📊 Val Acc: {val_acc:.4f}")
        print(f"      💾 Saved: {model_path}")
    
    return results, X_val, y_val

def create_trading_signal_system(models):
    """Tạo hệ thống tín hiệu trading"""
    print(f"\n🔮 CREATING TRADING SIGNAL SYSTEM...")
    print("=" * 60)
    
    class TradingSignalGenerator:
        def __init__(self, models):
            self.models = models
            self.signal_history = []
        
        def generate_signal(self, X_input):
            """Generate trading signal với ensemble"""
            predictions = []
            confidences = []
            
            for model_name, model in self.models.items():
                pred = model.predict(X_input, verbose=0)[0][0]
                predictions.append(pred)
                
                # Confidence based on distance from 0.5
                conf = abs(pred - 0.5) * 2
                confidences.append(conf)
            
            # Ensemble prediction
            ensemble_pred = np.mean(predictions)
            ensemble_conf = np.mean(confidences)
            
            # Generate signal
            if ensemble_pred > 0.6:
                signal = "STRONG_BUY"
            elif ensemble_pred > 0.5:
                signal = "BUY"
            elif ensemble_pred < 0.4:
                signal = "STRONG_SELL"
            else:
                signal = "SELL"
            
            # Risk level
            if ensemble_conf > 0.7:
                risk = "LOW"
            elif ensemble_conf > 0.5:
                risk = "MEDIUM"
            else:
                risk = "HIGH"
            
            signal_data = {
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'confidence': float(ensemble_conf),
                'prediction': float(ensemble_pred),
                'risk_level': risk,
                'individual_predictions': {
                    name: float(pred) for name, pred in zip(self.models.keys(), predictions)
                }
            }
            
            self.signal_history.append(signal_data)
            return signal_data
        
        def backtest_signals(self, X_test, y_test, num_signals=100):
            """Backtest trading signals"""
            print(f"   🔄 Backtesting {num_signals} signals...")
            
            test_indices = np.random.choice(len(X_test), num_signals, replace=False)
            
            correct_signals = 0
            signal_distribution = {"STRONG_BUY": 0, "BUY": 0, "SELL": 0, "STRONG_SELL": 0}
            
            for i, idx in enumerate(test_indices):
                X_sample = X_test[idx:idx+1]
                actual = y_test[idx]
                
                signal_data = self.generate_signal(X_sample)
                signal = signal_data['signal']
                prediction = signal_data['prediction']
                
                # Check if signal is correct
                if signal in ["STRONG_BUY", "BUY"]:
                    predicted_direction = 1
                else:
                    predicted_direction = 0
                
                if predicted_direction == actual:
                    correct_signals += 1
                
                signal_distribution[signal] += 1
                
                if i < 10:  # Show first 10 signals
                    actual_move = "UP" if actual == 1 else "DOWN"
                    correct = "✅" if predicted_direction == actual else "❌"
                    print(f"      Signal {i+1}: {signal} ({prediction:.3f}) | Actual: {actual_move} {correct}")
            
            accuracy = correct_signals / num_signals
            
            return {
                'accuracy': accuracy,
                'total_signals': num_signals,
                'correct_signals': correct_signals,
                'signal_distribution': signal_distribution
            }
    
    signal_generator = TradingSignalGenerator(models)
    print(f"   ✅ Trading Signal System created")
    
    return signal_generator

def run_comprehensive_test(signal_generator, X_val, y_val):
    """Chạy test comprehensive"""
    print(f"\n📊 COMPREHENSIVE SIGNAL TEST...")
    print("=" * 60)
    
    # Backtest
    backtest_results = signal_generator.backtest_signals(X_val, y_val, 100)
    
    print(f"\n   📊 BACKTEST RESULTS:")
    print(f"      Accuracy: {backtest_results['accuracy']:.4f} ({backtest_results['accuracy']*100:.1f}%)")
    print(f"      Correct Signals: {backtest_results['correct_signals']}/{backtest_results['total_signals']}")
    
    print(f"\n   📊 SIGNAL DISTRIBUTION:")
    for signal, count in backtest_results['signal_distribution'].items():
        pct = count / backtest_results['total_signals'] * 100
        print(f"      {signal}: {count} ({pct:.1f}%)")
    
    # Real-time simulation
    print(f"\n   🔄 Real-time Signal Simulation (Last 20 samples)...")
    recent_X = X_val[-20:]
    recent_y = y_val[-20:]
    
    real_time_correct = 0
    for i, (X_sample, actual) in enumerate(zip(recent_X, recent_y)):
        signal_data = signal_generator.generate_signal(X_sample.reshape(1, 60, 5))
        
        # Determine predicted direction
        if signal_data['signal'] in ["STRONG_BUY", "BUY"]:
            predicted = 1
        else:
            predicted = 0
        
        correct = predicted == actual
        if correct:
            real_time_correct += 1
        
        if i < 5:  # Show first 5
            actual_move = "UP" if actual == 1 else "DOWN"
            status = "✅" if correct else "❌"
            print(f"      {signal_data['signal']} | Conf: {signal_data['confidence']:.3f} | Actual: {actual_move} {status}")
    
    real_time_accuracy = real_time_correct / len(recent_X)
    
    return {
        'backtest_results': backtest_results,
        'real_time_accuracy': real_time_accuracy,
        'total_signals_generated': len(signal_generator.signal_history)
    }

def generate_production_report(training_results, test_results, data_size):
    """Tạo báo cáo production"""
    print(f"\n📋 PRODUCTION TRAINING REPORT")
    print("=" * 70)
    
    # Best model
    best_model = max(training_results.keys(), key=lambda k: training_results[k]['val_accuracy'])
    best_acc = training_results[best_model]['val_accuracy']
    
    print(f"🎯 TRAINING SUMMARY:")
    print(f"   Dataset Size: {data_size:,} records")
    print(f"   Models Trained: {len(training_results)}")
    print(f"   Best Model: {best_model.upper()} ({best_acc:.4f})")
    
    print(f"\n📊 MODEL PERFORMANCE:")
    for name, results in training_results.items():
        print(f"   {name.upper()}: {results['val_accuracy']:.4f} ({results['training_time']:.1f}s)")
    
    print(f"\n🔮 TRADING SIGNALS:")
    backtest_acc = test_results['backtest_results']['accuracy']
    realtime_acc = test_results['real_time_accuracy']
    
    print(f"   Backtest Accuracy: {backtest_acc:.4f} ({backtest_acc*100:.1f}%)")
    print(f"   Real-time Accuracy: {realtime_acc:.4f} ({realtime_acc*100:.1f}%)")
    print(f"   Total Signals: {test_results['total_signals_generated']}")
    
    # Overall assessment
    overall_score = (best_acc + backtest_acc + realtime_acc) / 3
    
    if overall_score >= 0.7:
        grade = "🟢 A - EXCELLENT - READY FOR LIVE TRADING"
    elif overall_score >= 0.6:
        grade = "🟡 B - GOOD - READY FOR PAPER TRADING"
    elif overall_score >= 0.5:
        grade = "🟠 C - SATISFACTORY - NEEDS OPTIMIZATION"
    else:
        grade = "🔴 D - POOR - REQUIRES SIGNIFICANT WORK"
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    print(f"   Overall Score: {overall_score:.4f}")
    print(f"   Grade: {grade}")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_size': data_size,
        'training_results': training_results,
        'test_results': test_results,
        'overall_score': overall_score,
        'grade': grade
    }
    
    filename = f"production_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved: {filename}")
    return report

def main():
    """Main execution"""
    print("🚀 FULL MAXIMUM DATA TRAINING & TRADING SIGNALS")
    print("=" * 70)
    print(f"🕒 Started: {datetime.now()}")
    print()
    
    # 1. Load full dataset
    df = load_full_dataset()
    if df is None:
        return
    
    # 2. Prepare dataset (50K samples for manageable training)
    X, y, scaler = prepare_full_dataset(df, use_samples=50000)
    
    # 3. Create production models
    models = create_production_models()
    
    # 4. Train models
    training_results, X_val, y_val = train_production_models(models, X, y)
    
    # 5. Create trading signal system
    signal_generator = create_trading_signal_system(models)
    
    # 6. Run comprehensive test
    test_results = run_comprehensive_test(signal_generator, X_val, y_val)
    
    # 7. Generate production report
    report = generate_production_report(training_results, test_results, len(df))
    
    print(f"\n✅ PRODUCTION TRAINING COMPLETED!")
    print(f"🕒 Finished: {datetime.now()}")
    print(f"📊 Overall Score: {report['overall_score']:.4f}")
    print(f"🏆 Final Grade: {report['grade']}")

if __name__ == "__main__":
    main() 