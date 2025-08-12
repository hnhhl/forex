import pandas as pd
import numpy as np
import os
from datetime import datetime

print("🧪 QUICK SYSTEM TEST - ULTIMATE SYSTEM V5.0")
print("="*50)
print("🔍 Kiểm tra nguyên nhân process bị treo...")

# Test 1: Import libraries
print("\n1️⃣ TESTING IMPORTS:")
try:
    import tensorflow as tf
    print("  ✅ TensorFlow:", tf.__version__)
except Exception as e:
    print(f"  ❌ TensorFlow error: {e}")

try:
    import sklearn
    print("  ✅ Scikit-learn:", sklearn.__version__)
except Exception as e:
    print(f"  ❌ Scikit-learn error: {e}")

try:
    import lightgbm as lgb
    print("  ✅ LightGBM:", lgb.__version__)
except Exception as e:
    print(f"  ❌ LightGBM error: {e}")

try:
    import xgboost as xgb
    print("  ✅ XGBoost:", xgb.__version__)
except Exception as e:
    print(f"  ❌ XGBoost error: {e}")

# Test 2: Data loading
print("\n2️⃣ TESTING DATA LOADING:")
data_files = [
    'training/xauusdc/data/M15_data.pkl',
    'training/xauusdc/data/M30_data.pkl',
    'training/xauusdc/data/H1_data.pkl'
]

data_found = False
for file_path in data_files:
    if os.path.exists(file_path):
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
                
            print(f"  ✅ Loaded {file_path}: {len(df)} rows, {len(df.columns)} cols")
            data_found = True
            break
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")

if not data_found:
    print("  ⚠️ No data files found - using synthetic data")

# Test 3: Simple model creation
print("\n3️⃣ TESTING MODEL CREATION:")
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([
        Dense(10, input_dim=5, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("  ✅ Simple neural network created successfully")
except Exception as e:
    print(f"  ❌ Neural network error: {e}")

# Test 4: Memory usage
print("\n4️⃣ TESTING MEMORY:")
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"  💾 Available RAM: {memory.available/1024**3:.1f}GB")
    print(f"  💾 Used RAM: {memory.percent:.1f}%")
    
    if memory.percent > 80:
        print("  ⚠️ High memory usage - may cause issues")
    else:
        print("  ✅ Memory usage normal")
except Exception as e:
    print(f"  ❌ Memory check error: {e}")

# Test 5: Synthetic training test
print("\n5️⃣ QUICK TRAINING TEST:")
try:
    # Create small synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X, y)
    accuracy = rf.score(X, y)
    
    print(f"  ✅ Quick RF training: {accuracy:.3f} accuracy")
    
    # Test neural network
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    nn = Sequential([
        Dense(5, input_dim=10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn.fit(X, y, epochs=5, verbose=0)
    
    print(f"  ✅ Quick NN training: {history.history['accuracy'][-1]:.3f} accuracy")
    
except Exception as e:
    print(f"  ❌ Training test error: {e}")

# Test 6: File system
print("\n6️⃣ TESTING FILE SYSTEM:")
try:
    test_file = f"test_file_{datetime.now().strftime('%H%M%S')}.tmp"
    with open(test_file, 'w') as f:
        f.write("test data")
    
    # Read back
    with open(test_file, 'r') as f:
        content = f.read()
    
    os.remove(test_file)
    print("  ✅ File I/O working normally")
    
except Exception as e:
    print(f"  ❌ File I/O error: {e}")

print(f"\n{'='*50}")
print("🎯 DIAGNOSIS SUMMARY:")

# Possible issues analysis
issues = []
if not data_found:
    issues.append("❌ Data loading issues")

print("📋 POSSIBLE CAUSES OF HANGING:")
print("  1. 🔄 Infinite loop in training process")
print("  2. 💾 Memory leak causing slowdown") 
print("  3. 🧠 Neural network training stuck")
print("  4. 📊 Cross-validation taking too long")
print("  5. 🔢 Large dataset causing timeout")

print("\n💡 RECOMMENDED SOLUTIONS:")
print("  1. 🚀 Run simplified version with smaller data")
print("  2. ⚡ Reduce epochs/iterations")
print("  3. 🎯 Skip complex components first")
print("  4. 📊 Add progress monitoring")

print(f"\n⏰ Test completed at: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*50}") 