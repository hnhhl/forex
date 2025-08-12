import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from datetime import datetime

print("🎯 KIỂM TRA THỰC TẾ CHẤT LƯỢNG MODELS")
print("="*50)

# Load sample data from one timeframe to test
print("\n📊 Loading test data...")
try:
    # Load M15 data as sample
    with open('training/xauusdc/data/M15_data.pkl', 'rb') as f:
        m15_data = pickle.load(f)
    
    print(f"✅ Loaded M15 data: {len(m15_data)} samples")
    
    # Prepare test data (last 500 samples)
    test_size = min(500, len(m15_data))
    test_data = m15_data.tail(test_size).copy()
    
    # Get features (exclude target columns)
    feature_cols = [col for col in test_data.columns if not col.startswith('y_direction') and col != 'timestamp']
    X_test = test_data[feature_cols].values
    
    # Get targets
    targets = {}
    for target_col in ['y_direction_2', 'y_direction_4', 'y_direction_8']:
        if target_col in test_data.columns:
            targets[target_col] = test_data[target_col].values
    
    print(f"📈 Test samples: {test_size}")
    print(f"📊 Features: {len(feature_cols)}")
    print(f"🎯 Targets: {list(targets.keys())}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Test results
results = {
    'test_time': datetime.now().isoformat(),
    'test_samples': test_size,
    'models_tested': {},
    'summary': {}
}

print("\n🧠 TESTING NEURAL MODELS")
print("-" * 30)

neural_accuracies = []

for target in targets.keys():
    print(f"\n📊 Testing {target}...")
    
    try:
        # Load scaler
        scaler_path = f'trained_models/scaler_{target}.pkl'
        if not os.path.exists(scaler_path):
            print(f"  ❌ Scaler not found: {scaler_path}")
            continue
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X_test_scaled = scaler.transform(X_test)
        
        # Test LSTM model
        lstm_path = f'trained_models/neural_ensemble_{target}_lstm.h5'
        if os.path.exists(lstm_path):
            lstm_model = tf.keras.models.load_model(lstm_path)
            
            # Prepare LSTM input
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)
            lstm_pred_binary = (lstm_pred > 0.5).astype(int).flatten()
            
            lstm_acc = accuracy_score(targets[target], lstm_pred_binary)
            lstm_f1 = f1_score(targets[target], lstm_pred_binary)
            
            print(f"  🔸 LSTM: Accuracy={lstm_acc:.4f} ({lstm_acc*100:.1f}%) | F1={lstm_f1:.4f}")
            neural_accuracies.append(lstm_acc)
        
        # Test Dense model
        dense_path = f'trained_models/neural_ensemble_{target}_dense.h5'
        if os.path.exists(dense_path):
            dense_model = tf.keras.models.load_model(dense_path)
            
            dense_pred = dense_model.predict(X_test_scaled, verbose=0)
            dense_pred_binary = (dense_pred > 0.5).astype(int).flatten()
            
            dense_acc = accuracy_score(targets[target], dense_pred_binary)
            dense_f1 = f1_score(targets[target], dense_pred_binary)
            
            print(f"  🔸 Dense: Accuracy={dense_acc:.4f} ({dense_acc*100:.1f}%) | F1={dense_f1:.4f}")
            neural_accuracies.append(dense_acc)
        
        results['models_tested'][f'neural_{target}'] = {
            'lstm_accuracy': float(lstm_acc) if 'lstm_acc' in locals() else 0,
            'dense_accuracy': float(dense_acc) if 'dense_acc' in locals() else 0
        }
        
    except Exception as e:
        print(f"  ❌ Error testing {target}: {e}")

print("\n🌲 TESTING TRADITIONAL MODELS")
print("-" * 30)

traditional_accuracies = []

for target in targets.keys():
    print(f"\n📊 Testing {target}...")
    
    try:
        # Load scaler
        scaler_path = f'trained_models/scaler_{target}.pkl'
        if not os.path.exists(scaler_path):
            continue
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        X_test_scaled = scaler.transform(X_test)
        
        models = ['random_forest', 'gradient_boost', 'lightgbm']
        
        for model_name in models:
            model_path = f'trained_models/{model_name}_{target}.pkl'
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                pred = model.predict(X_test_scaled)
                acc = accuracy_score(targets[target], pred)
                f1 = f1_score(targets[target], pred)
                
                print(f"  🔸 {model_name}: Accuracy={acc:.4f} ({acc*100:.1f}%) | F1={f1:.4f}")
                traditional_accuracies.append(acc)
        
    except Exception as e:
        print(f"  ❌ Error testing traditional models for {target}: {e}")

# Calculate summary
print("\n📊 TỔNG KẾT THỰC TẾ")
print("=" * 50)

all_accuracies = neural_accuracies + traditional_accuracies

if all_accuracies:
    avg_acc = np.mean(all_accuracies)
    max_acc = np.max(all_accuracies)
    min_acc = np.min(all_accuracies)
    
    print(f"🎯 Độ chính xác trung bình: {avg_acc:.4f} ({avg_acc*100:.1f}%)")
    print(f"🏆 Độ chính xác cao nhất: {max_acc:.4f} ({max_acc*100:.1f}%)")
    print(f"📉 Độ chính xác thấp nhất: {min_acc:.4f} ({min_acc*100:.1f}%)")
    print(f"📊 Tổng số models test: {len(all_accuracies)}")
    
    # Performance assessment
    print("\n🏁 ĐÁNH GIÁ MỨC ĐỘ THÀNH CÔNG:")
    if avg_acc >= 0.70:
        success_level = "XUẤT SẮC"
        status = "✅"
    elif avg_acc >= 0.60:
        success_level = "KHÁ TỐT"
        status = "⚠️"
    elif avg_acc >= 0.55:
        success_level = "TRUNG BÌNH"
        status = "🔶"
    else:
        success_level = "CẦN CẢI THIỆN"
        status = "❌"
    
    print(f"{status} Mức độ: {success_level}")
    print(f"📈 Tỷ lệ thành công: {avg_acc*100:.1f}%")
    
    # Specific analysis
    if len(neural_accuracies) > 0:
        neural_avg = np.mean(neural_accuracies)
        print(f"🧠 Neural Models trung bình: {neural_avg:.4f} ({neural_avg*100:.1f}%)")
    
    if len(traditional_accuracies) > 0:
        trad_avg = np.mean(traditional_accuracies)
        print(f"🌲 Traditional Models trung bình: {trad_avg:.4f} ({trad_avg*100:.1f}%)")
    
    results['summary'] = {
        'average_accuracy': float(avg_acc),
        'max_accuracy': float(max_acc),
        'min_accuracy': float(min_acc),
        'success_level': success_level,
        'neural_models_count': len(neural_accuracies),
        'traditional_models_count': len(traditional_accuracies)
    }
    
    # Save results
    results_file = f'quick_model_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Kết quả chi tiết đã lưu: {results_file}")
    
else:
    print("❌ KHÔNG THỂ TEST MODEL NÀO - Có vấn đề nghiêm trọng!")

print("\n" + "="*50)
print("🎯 KẾT LUẬN: THÀNH CÔNG LÀ...")
if all_accuracies and avg_acc >= 0.55:
    print("✅ CÓ THÀNH CÔNG - Models hoạt động và cho kết quả khả quan")
    print(f"📊 {len(all_accuracies)} models hoạt động với độ chính xác {avg_acc*100:.1f}%")
else:
    print("❌ CHƯA THÀNH CÔNG HOÀN TOÀN - Cần cải thiện thêm")
print("="*50) 