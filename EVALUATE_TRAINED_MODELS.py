import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import os
from datetime import datetime

print("🔍 ĐÁNH GIÁ THỰC TẾ CÁC MODELS ĐÃ TRAIN")
print("="*60)

# Load test data
print("\n📊 Loading test data...")
try:
    test_data = pd.read_csv('training/xauusdc/data/unified_multi_timeframe_data.csv')
    print(f"✅ Loaded {len(test_data)} samples")
    
    # Prepare features and targets
    feature_cols = [col for col in test_data.columns if col not in ['timestamp', 'y_direction_2', 'y_direction_4', 'y_direction_8']]
    X = test_data[feature_cols].values
    
    targets = {
        'y_direction_2': test_data['y_direction_2'].values,
        'y_direction_4': test_data['y_direction_4'].values, 
        'y_direction_8': test_data['y_direction_8'].values
    }
    
    # Split test data (use last 20% as true test set)
    test_size = int(len(X) * 0.2)
    X_test = X[-test_size:]
    y_test = {target: values[-test_size:] for target, values in targets.items()}
    
    print(f"📈 Test set size: {test_size} samples")
    print(f"📊 Features: {len(feature_cols)}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Initialize results
results = {
    'evaluation_time': datetime.now().isoformat(),
    'test_samples': test_size,
    'models_evaluated': {},
    'summary': {}
}

print("\n🧠 EVALUATING NEURAL ENSEMBLE MODELS")
print("-" * 40)

# Evaluate Neural Models
for target in ['y_direction_2', 'y_direction_4', 'y_direction_8']:
    print(f"\n📊 Target: {target}")
    
    try:
        # Load scaler
        scaler_path = f'trained_models/scaler_{target}.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X_test_scaled = scaler.transform(X_test)
        
        # Load and evaluate LSTM model
        lstm_path = f'trained_models/neural_ensemble_{target}_lstm.h5'
        lstm_model = tf.keras.models.load_model(lstm_path)
        
        # Prepare LSTM input (reshape for sequence)
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)
        lstm_pred_binary = (lstm_pred > 0.5).astype(int).flatten()
        
        # Load and evaluate Dense model
        dense_path = f'trained_models/neural_ensemble_{target}_dense.h5'
        dense_model = tf.keras.models.load_model(dense_path)
        
        dense_pred = dense_model.predict(X_test_scaled, verbose=0)
        dense_pred_binary = (dense_pred > 0.5).astype(int).flatten()
        
        # Ensemble prediction (average)
        ensemble_pred = (lstm_pred + dense_pred) / 2
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int).flatten()
        
        # Calculate metrics
        y_true = y_test[target]
        
        lstm_acc = accuracy_score(y_true, lstm_pred_binary)
        dense_acc = accuracy_score(y_true, dense_pred_binary)
        ensemble_acc = accuracy_score(y_true, ensemble_pred_binary)
        
        lstm_f1 = f1_score(y_true, lstm_pred_binary)
        dense_f1 = f1_score(y_true, dense_pred_binary)
        ensemble_f1 = f1_score(y_true, ensemble_pred_binary)
        
        print(f"  🔸 LSTM Accuracy: {lstm_acc:.4f} | F1: {lstm_f1:.4f}")
        print(f"  🔸 Dense Accuracy: {dense_acc:.4f} | F1: {dense_f1:.4f}")
        print(f"  🔸 Ensemble Accuracy: {ensemble_acc:.4f} | F1: {ensemble_f1:.4f}")
        
        results['models_evaluated'][f'neural_{target}'] = {
            'lstm_accuracy': float(lstm_acc),
            'dense_accuracy': float(dense_acc),
            'ensemble_accuracy': float(ensemble_acc),
            'lstm_f1': float(lstm_f1),
            'dense_f1': float(dense_f1),
            'ensemble_f1': float(ensemble_f1)
        }
        
    except Exception as e:
        print(f"  ❌ Error evaluating neural models for {target}: {e}")

print("\n🌲 EVALUATING TRADITIONAL ML MODELS")
print("-" * 40)

# Evaluate Traditional Models
for target in ['y_direction_2', 'y_direction_4', 'y_direction_8']:
    print(f"\n📊 Target: {target}")
    
    try:
        # Load scaler
        scaler_path = f'trained_models/scaler_{target}.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X_test_scaled = scaler.transform(X_test)
        
        models = ['random_forest', 'gradient_boost', 'lightgbm']
        model_results = {}
        
        for model_name in models:
            model_path = f'trained_models/{model_name}_{target}.pkl'
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test[target], pred)
            f1 = f1_score(y_test[target], pred)
            
            print(f"  🔸 {model_name}: Accuracy: {acc:.4f} | F1: {f1:.4f}")
            model_results[model_name] = {'accuracy': float(acc), 'f1': float(f1)}
        
        results['models_evaluated'][f'traditional_{target}'] = model_results
        
    except Exception as e:
        print(f"  ❌ Error evaluating traditional models for {target}: {e}")

# Calculate summary statistics
print("\n📈 TỔNG KẾT ĐÁNH GIÁ")
print("=" * 60)

all_accuracies = []
best_models = {}

for model_group, models in results['models_evaluated'].items():
    if 'neural' in model_group:
        best_acc = max([models.get('lstm_accuracy', 0), models.get('dense_accuracy', 0), models.get('ensemble_accuracy', 0)])
        all_accuracies.append(best_acc)
        best_models[model_group] = f"Best Neural: {best_acc:.4f}"
    else:
        best_acc = max([model_data.get('accuracy', 0) for model_data in models.values()])
        all_accuracies.append(best_acc)
        best_models[model_group] = f"Best Traditional: {best_acc:.4f}"

if all_accuracies:
    avg_accuracy = np.mean(all_accuracies)
    max_accuracy = np.max(all_accuracies)
    min_accuracy = np.min(all_accuracies)
    
    print(f"🎯 Độ chính xác trung bình: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"🏆 Độ chính xác cao nhất: {max_accuracy:.4f} ({max_accuracy*100:.2f}%)")
    print(f"📉 Độ chính xác thấp nhất: {min_accuracy:.4f} ({min_accuracy*100:.2f}%)")
    
    results['summary'] = {
        'average_accuracy': float(avg_accuracy),
        'max_accuracy': float(max_accuracy),
        'min_accuracy': float(min_accuracy),
        'total_models_evaluated': len(all_accuracies)
    }

print("\n🎯 CHI TIẾT TỪNG MODEL:")
for model_group, result in best_models.items():
    print(f"  {model_group}: {result}")

# Save results
results_file = f'model_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Kết quả đã lưu: {results_file}")

# Performance Assessment
print("\n🏁 ĐÁNH GIÁ TỔNG QUAN:")
if all_accuracies:
    if avg_accuracy >= 0.70:
        print("✅ HỆ THỐNG HOẠT ĐỘNG TỐT - Độ chính xác trung bình >= 70%")
    elif avg_accuracy >= 0.60:
        print("⚠️ HỆ THỐNG HOẠT ĐỘNG KHẤP KHIU - Độ chính xác 60-70%")
    else:
        print("❌ HỆ THỐNG CẦN CẢI THIỆN - Độ chính xác < 60%")
        
    print(f"📊 Tổng cộng đã đánh giá {len(all_accuracies)} models")
    print(f"🎯 Mức độ thành công: {(avg_accuracy*100):.1f}%")
else:
    print("❌ KHÔNG THỂ ĐÁNH GIÁ - Không có models hoạt động") 