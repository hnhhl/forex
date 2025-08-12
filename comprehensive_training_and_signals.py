#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE TRAINING & SIGNAL GENERATION TEST
Training hệ thống AI3.0 trên dữ liệu tối đa và test khả năng đưa ra tín hiệu
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
    """Load toàn bộ dữ liệu available"""
    print("📊 LOADING MAXIMUM DATA...")
    print("=" * 50)
    
    all_data = []
    data_sources = [
        'data/working_free_data/XAUUSD_M1_realistic.csv',
        'data/maximum_mt5_v2/XAUUSDc_M1_20250618_115847.csv',
        'data/real_free_data/XAUUSD_M1_forexsb.csv'
    ]
    
    total_records = 0
    
    for source in data_sources:
        if os.path.exists(source):
            try:
                df = pd.read_csv(source)
                print(f"   ✅ {source}: {len(df):,} records")
                
                # Standardize columns
                if 'Time' in df.columns:
                    df['Datetime'] = df['Time']
                elif 'Date' in df.columns:
                    df['Datetime'] = df['Date']
                
                # Ensure required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    df = df[['Datetime'] + required_cols].copy()
                    all_data.append(df)
                    total_records += len(df)
                else:
                    print(f"   ⚠️  Missing required columns in {source}")
                    
            except Exception as e:
                print(f"   ❌ Error loading {source}: {e}")
        else:
            print(f"   ❌ File not found: {source}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates if Datetime column exists
        if 'Datetime' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['Datetime'])
            combined_df = combined_df.sort_values('Datetime')
        
        print(f"\n   📊 TOTAL COMBINED DATA:")
        print(f"      Records: {len(combined_df):,}")
        print(f"      Columns: {list(combined_df.columns)}")
        print(f"      Memory Usage: {combined_df.memory_usage().sum() / 1024 / 1024:.1f} MB")
        
        return combined_df
    else:
        print("   ❌ No data loaded successfully")
        return None

def prepare_training_data(df: pd.DataFrame, sequence_length: int = 60):
    """Chuẩn bị dữ liệu training"""
    print(f"\n🔧 PREPARING TRAINING DATA...")
    print("=" * 50)
    
    # Extract features
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
    print(f"   📊 Raw features shape: {features.shape}")
    
    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"   ✅ Data normalized")
    
    # Create sequences
    X, y = [], []
    
    print(f"   🔄 Creating sequences (length={sequence_length})...")
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
        
        # Target: next close price direction (1 if up, 0 if down)
        current_close = features_scaled[i-1, 3]  # Close price index
        next_close = features_scaled[i, 3]
        y.append(1 if next_close > current_close else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"   ✅ Sequences created:")
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Positive samples: {np.sum(y):.0f} ({np.mean(y)*100:.1f}%)")
    
    # Train/validation split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Data split:")
    print(f"      Training: {X_train.shape[0]:,} samples")
    print(f"      Validation: {X_val.shape[0]:,} samples")
    
    return X_train, X_val, y_train, y_val, scaler

def create_advanced_gpu_models():
    """Tạo các models GPU advanced"""
    print(f"\n🚀 CREATING ADVANCED GPU MODELS...")
    print("=" * 50)
    
    # Check GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"   🎯 GPU Available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"      - {gpu}")
        
        # Configure GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"   ⚡ Mixed precision enabled")
    else:
        print(f"   ⚠️  No GPU available, using CPU")
    
    models = {}
    
    # 1. Advanced LSTM Model
    print(f"   🧠 Creating Advanced LSTM Model...")
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 5)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
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
    print(f"      ✅ LSTM Model: {lstm_model.count_params():,} parameters")
    
    # 2. Advanced CNN Model
    print(f"   🔍 Creating Advanced CNN Model...")
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(60, 5)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
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
    print(f"      ✅ CNN Model: {cnn_model.count_params():,} parameters")
    
    # 3. Hybrid CNN-LSTM Model
    print(f"   🔥 Creating Hybrid CNN-LSTM Model...")
    hybrid_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(60, 5)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    hybrid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['hybrid'] = hybrid_model
    print(f"      ✅ Hybrid Model: {hybrid_model.count_params():,} parameters")
    
    total_params = sum(model.count_params() for model in models.values())
    print(f"\n   📊 Total Models: {len(models)}")
    print(f"   📊 Total Parameters: {total_params:,}")
    
    return models

def train_models_comprehensive(models: Dict, X_train, X_val, y_train, y_val):
    """Training comprehensive tất cả models"""
    print(f"\n🎯 COMPREHENSIVE TRAINING...")
    print("=" * 50)
    
    training_results = {}
    
    for model_name, model in models.items():
        print(f"\n   🚀 Training {model_name.upper()} Model...")
        print(f"   {'-' * 40}")
        
        start_time = time.time()
        
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
                min_lr=1e-7
            )
        ]
        
        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        # Save model
        model_path = f"trained_models/comprehensive_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        model.save(model_path)
        
        results = {
            'training_time': training_time,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'epochs_trained': len(history.history['loss']),
            'model_path': model_path,
            'parameters': model.count_params()
        }
        
        training_results[model_name] = results
        
        print(f"      ✅ Training completed in {training_time:.1f}s")
        print(f"      📊 Train Accuracy: {train_acc:.4f}")
        print(f"      📊 Val Accuracy: {val_acc:.4f}")
        print(f"      💾 Model saved: {model_path}")
    
    return training_results

def create_ensemble_predictor(models: Dict):
    """Tạo ensemble predictor"""
    print(f"\n🏆 CREATING ENSEMBLE PREDICTOR...")
    print("=" * 50)
    
    def ensemble_predict(X):
        """Ensemble prediction với voting"""
        predictions = []
        confidences = []
        
        for model_name, model in models.items():
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
            
            # Calculate confidence (distance from 0.5)
            confidence = np.abs(pred - 0.5) * 2
            confidences.append(confidence)
        
        # Weighted average based on confidence
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Normalize weights
        weights = confidences / np.sum(confidences, axis=0)
        
        # Weighted ensemble
        ensemble_pred = np.sum(predictions * weights, axis=0)
        ensemble_confidence = np.mean(confidences, axis=0)
        
        return ensemble_pred, ensemble_confidence
    
    print(f"   ✅ Ensemble predictor created with {len(models)} models")
    return ensemble_predict

def test_signal_generation(ensemble_predictor, X_val, y_val, scaler):
    """Test khả năng đưa ra tín hiệu trading"""
    print(f"\n🔮 TESTING SIGNAL GENERATION...")
    print("=" * 50)
    
    # Test trên validation data
    print(f"   🔄 Generating signals for {len(X_val):,} samples...")
    
    predictions, confidences = ensemble_predictor(X_val)
    
    # Convert to binary signals
    signals = (predictions > 0.5).astype(int).flatten()
    actual = y_val.astype(int)
    
    # Calculate metrics
    accuracy = np.mean(signals == actual)
    
    # Signal distribution
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == 0)
    
    # Confidence analysis
    high_confidence = np.sum(confidences > 0.7)
    medium_confidence = np.sum((confidences >= 0.5) & (confidences <= 0.7))
    low_confidence = np.sum(confidences < 0.5)
    
    print(f"   📊 SIGNAL GENERATION RESULTS:")
    print(f"      Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"      Buy Signals: {buy_signals:,} ({buy_signals/len(signals)*100:.1f}%)")
    print(f"      Sell Signals: {sell_signals:,} ({sell_signals/len(signals)*100:.1f}%)")
    
    print(f"\n   📊 CONFIDENCE DISTRIBUTION:")
    print(f"      High Confidence (>70%): {high_confidence:,} ({high_confidence/len(confidences)*100:.1f}%)")
    print(f"      Medium Confidence (50-70%): {medium_confidence:,} ({medium_confidence/len(confidences)*100:.1f}%)")
    print(f"      Low Confidence (<50%): {low_confidence:,} ({low_confidence/len(confidences)*100:.1f}%)")
    
    # Test real-time signal generation
    print(f"\n   🔄 Testing Real-time Signal Generation...")
    
    # Take last 10 samples for real-time test
    real_time_samples = X_val[-10:]
    real_time_actual = y_val[-10:]
    
    real_time_signals = []
    
    for i, sample in enumerate(real_time_samples):
        sample_input = sample.reshape(1, 60, 5)
        pred, conf = ensemble_predictor(sample_input)
        
        signal = "BUY" if pred[0] > 0.5 else "SELL"
        actual_move = "UP" if real_time_actual[i] == 1 else "DOWN"
        correct = "✅" if (pred[0] > 0.5) == (real_time_actual[i] == 1) else "❌"
        
        real_time_signals.append({
            'sample': i+1,
            'signal': signal,
            'confidence': float(conf[0]),
            'prediction': float(pred[0]),
            'actual': actual_move,
            'correct': correct
        })
        
        print(f"      Sample {i+1}: {signal} (conf: {conf[0]:.3f}) | Actual: {actual_move} {correct}")
    
    # Calculate real-time accuracy
    real_time_accuracy = np.mean([1 if s['correct'] == '✅' else 0 for s in real_time_signals])
    
    results = {
        'overall_accuracy': float(accuracy),
        'buy_signals_pct': float(buy_signals/len(signals)*100),
        'sell_signals_pct': float(sell_signals/len(signals)*100),
        'high_confidence_pct': float(high_confidence/len(confidences)*100),
        'medium_confidence_pct': float(medium_confidence/len(confidences)*100),
        'low_confidence_pct': float(low_confidence/len(confidences)*100),
        'real_time_accuracy': float(real_time_accuracy),
        'real_time_signals': real_time_signals
    }
    
    print(f"\n   🎯 Real-time Accuracy: {real_time_accuracy:.4f} ({real_time_accuracy*100:.2f}%)")
    
    return results

def generate_comprehensive_report(training_results: Dict, signal_results: Dict):
    """Tạo báo cáo comprehensive"""
    print(f"\n📋 COMPREHENSIVE TRAINING & SIGNALS REPORT")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_results': training_results,
        'signal_results': signal_results,
        'summary': {}
    }
    
    # Training summary
    best_model = max(training_results.keys(), key=lambda k: training_results[k]['val_accuracy'])
    best_accuracy = training_results[best_model]['val_accuracy']
    
    total_training_time = sum(r['training_time'] for r in training_results.values())
    total_parameters = sum(r['parameters'] for r in training_results.values())
    
    print(f"🎯 TRAINING SUMMARY:")
    print(f"   Models Trained: {len(training_results)}")
    print(f"   Best Model: {best_model.upper()} ({best_accuracy:.4f} accuracy)")
    print(f"   Total Training Time: {total_training_time:.1f}s")
    print(f"   Total Parameters: {total_parameters:,}")
    
    print(f"\n📊 MODEL PERFORMANCE:")
    for model_name, results in training_results.items():
        print(f"   {model_name.upper()}:")
        print(f"      Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"      Training Time: {results['training_time']:.1f}s")
        print(f"      Parameters: {results['parameters']:,}")
    
    print(f"\n🔮 SIGNAL GENERATION SUMMARY:")
    print(f"   Overall Accuracy: {signal_results['overall_accuracy']:.4f}")
    print(f"   Real-time Accuracy: {signal_results['real_time_accuracy']:.4f}")
    print(f"   High Confidence Signals: {signal_results['high_confidence_pct']:.1f}%")
    print(f"   Buy/Sell Distribution: {signal_results['buy_signals_pct']:.1f}% / {signal_results['sell_signals_pct']:.1f}%")
    
    # Overall assessment
    overall_score = (best_accuracy + signal_results['overall_accuracy'] + signal_results['real_time_accuracy']) / 3
    
    if overall_score >= 0.8:
        grade = "A - EXCELLENT"
        status = "🟢 PRODUCTION READY"
    elif overall_score >= 0.7:
        grade = "B - GOOD"
        status = "🟡 NEAR PRODUCTION READY"
    elif overall_score >= 0.6:
        grade = "C - SATISFACTORY"
        status = "🟠 NEEDS OPTIMIZATION"
    else:
        grade = "D - NEEDS IMPROVEMENT"
        status = "🔴 REQUIRES SIGNIFICANT WORK"
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   Overall Score: {overall_score:.4f}")
    print(f"   Grade: {grade}")
    print(f"   Status: {status}")
    
    report['summary'] = {
        'best_model': best_model,
        'best_accuracy': best_accuracy,
        'total_training_time': total_training_time,
        'total_parameters': total_parameters,
        'overall_score': overall_score,
        'grade': grade,
        'status': status
    }
    
    # Save report
    report_file = f"comprehensive_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved: {report_file}")
    
    return report

def main():
    """Main execution"""
    print("🚀 COMPREHENSIVE TRAINING & SIGNAL GENERATION")
    print("=" * 70)
    print(f"🕒 Started at: {datetime.now()}")
    print()
    
    # 1. Load maximum data
    df = load_maximum_data()
    if df is None:
        print("❌ Failed to load data")
        return
    
    # 2. Prepare training data
    X_train, X_val, y_train, y_val, scaler = prepare_training_data(df)
    
    # 3. Create advanced models
    models = create_advanced_gpu_models()
    
    # 4. Train models
    training_results = train_models_comprehensive(models, X_train, X_val, y_train, y_val)
    
    # 5. Create ensemble predictor
    ensemble_predictor = create_ensemble_predictor(models)
    
    # 6. Test signal generation
    signal_results = test_signal_generation(ensemble_predictor, X_val, y_val, scaler)
    
    # 7. Generate comprehensive report
    report = generate_comprehensive_report(training_results, signal_results)
    
    print(f"\n✅ COMPREHENSIVE TRAINING COMPLETED!")
    print(f"🕒 Finished at: {datetime.now()}")
    print(f"📊 Overall Score: {report['summary']['overall_score']:.4f}")
    print(f"🏆 Grade: {report['summary']['grade']}")

if __name__ == "__main__":
    main() 