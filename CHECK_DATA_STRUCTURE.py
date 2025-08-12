import pickle
import pandas as pd
import numpy as np

print("🔍 KIỂM TRA CẤU TRÚC DỮ LIỆU THỰC TẾ")
print("="*40)

# Check M15 data structure
print("\n📊 Checking M15 data structure...")
try:
    with open('training/xauusdc/data/M15_data.pkl', 'rb') as f:
        m15_data = pickle.load(f)
    
    print(f"✅ Type: {type(m15_data)}")
    
    if isinstance(m15_data, dict):
        print(f"📋 Dict keys: {list(m15_data.keys())}")
        for key, value in m15_data.items():
            print(f"  🔸 {key}: {type(value)} - {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
    
    elif isinstance(m15_data, pd.DataFrame):
        print(f"📊 DataFrame shape: {m15_data.shape}")
        print(f"📋 Columns: {list(m15_data.columns)}")
        print(f"📈 Sample data:\n{m15_data.head()}")
    
    elif isinstance(m15_data, np.ndarray):
        print(f"📊 Array shape: {m15_data.shape}")
    
    else:
        print(f"❓ Unknown type: {type(m15_data)}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Check if we have actual trained models
print(f"\n🧠 KIỂM TRA MODELS ĐÃ TRAIN")
print("-"*30)

import os
model_files = []
for file in os.listdir('trained_models'):
    if file.endswith(('.h5', '.pkl')):
        model_files.append(file)
        
print(f"📁 Found {len(model_files)} model files:")
for i, file in enumerate(model_files[:10]):  # Show first 10
    size = os.path.getsize(f'trained_models/{file}') / (1024*1024)  # MB
    print(f"  {i+1}. {file} ({size:.1f}MB)")

if len(model_files) > 10:
    print(f"  ... và {len(model_files)-10} files khác")

# Try to load one model to test
print(f"\n🎯 TEST LOAD MỘT MODEL")
print("-"*25)

try:
    # Try neural model
    import tensorflow as tf
    model_path = 'trained_models/neural_ensemble_y_direction_2_lstm.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Neural model loaded successfully!")
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
        print(f"🧠 Total params: {model.count_params():,}")
    
    # Try traditional model  
    rf_path = 'trained_models/random_forest_y_direction_2.pkl'
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        print(f"✅ Random Forest loaded successfully!")
        print(f"📊 Type: {type(rf_model)}")
        
except Exception as e:
    print(f"❌ Error loading models: {e}")

print(f"\n🏁 KẾT LUẬN THỰC TẾ:")
print("="*40)
print(f"📁 Có {len(model_files)} model files")
print(f"💾 Tổng dung lượng: {sum(os.path.getsize(f'trained_models/{f}') for f in model_files)/(1024*1024):.1f}MB")

if len(model_files) >= 15:
    print("✅ THÀNH CÔNG: Có đủ models đã được train!")
else:
    print("⚠️ CHƯA ĐỦ: Thiếu một số models") 