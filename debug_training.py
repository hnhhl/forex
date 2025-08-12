#!/usr/bin/env python3
"""Debug script để tìm lỗi training"""

print("🔍 DEBUG TRAINING SCRIPT")
print("="*50)

try:
    print("1. Testing basic imports...")
    import os
    import sys
    print(f"   ✅ Python: {sys.version}")
    print(f"   ✅ Working dir: {os.getcwd()}")
    
    print("2. Testing pandas...")
    import pandas as pd
    print(f"   ✅ Pandas: {pd.__version__}")
    
    print("3. Testing numpy...")
    import numpy as np
    print(f"   ✅ NumPy: {np.__version__}")
    
    print("4. Testing PyTorch...")
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    print("5. Testing data loading...")
    data_path = "data/working_free_data/XAUUSD_M1_realistic.csv"
    if os.path.exists(data_path):
        print(f"   ✅ Data file exists: {data_path}")
        df = pd.read_csv(data_path, nrows=100)
        print(f"   ✅ Data loaded: {len(df)} rows")
        print(f"   ✅ Columns: {list(df.columns)}")
    else:
        print(f"   ❌ Data file missing: {data_path}")
    
    print("6. Testing main script import...")
    # Import main script to check for syntax errors
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", "ULTIMATE_REAL_DATA_TRAINING_171_MODELS.py")
    main_module = importlib.util.module_from_spec(spec)
    print("   ✅ Main script syntax OK")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("Main script should work. Running it now...")
    
    # Execute main script
    spec.loader.exec_module(main_module)
    
except Exception as e:
    print(f"\n❌ ERROR FOUND: {e}")
    import traceback
    print("\n🔍 FULL TRACEBACK:")
    traceback.print_exc()
    
    print(f"\n📋 ERROR TYPE: {type(e).__name__}")
    print(f"📋 ERROR MESSAGE: {str(e)}")

print("="*60)
print("DEBUGGING GPU TRAINING ISSUE")
print("="*60)

# Check CUDA
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    device = torch.device('cuda')
    print(f"Using device: {device}")
else:
    print("❌ CUDA NOT AVAILABLE - USING CPU")
    device = torch.device('cpu')

print("-"*60)

# Create simple model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Create model and move to device
model = TestModel().to(device)
print(f"Model device: {next(model.parameters()).device}")

# Create sample data
batch_size = 1000
features = 20
X = torch.randn(batch_size, features).to(device)
y = torch.randint(0, 2, (batch_size,)).float().to(device)

print(f"Input tensor device: {X.device}")
print(f"Target tensor device: {y.device}")

# Test training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("\nStarting test training...")
start_time = time.time()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

end_time = time.time()
print(f"\nTraining time: {end_time - start_time:.2f} seconds")

# Check GPU utilization
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.3f} GB")
    print(f"GPU Utilization: Check nvidia-smi now!")

print("="*60) 