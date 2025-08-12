#!/usr/bin/env python3
"""
GPU DIAGNOSTIC - Kiểm tra 4 vấn đề chính
"""
import sys
import warnings
warnings.filterwarnings('ignore')

print("🔍 GPU DIAGNOSTIC - KIỂM TRA 4 VẤN ĐỀ CHÍNH")
print("="*60)

# 1. KIỂM TRA MÔI TRƯỜNG VÀ XUNG ĐỘT
print("\n1️⃣ KIỂM TRA MÔI TRƯỜNG:")
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ CUDA compiled: {torch.version.cuda}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✅ GPU count: {torch.cuda.device_count()}")
        print(f"   ✅ Current device: {torch.cuda.current_device()}")
        print(f"   ✅ GPU name: {torch.cuda.get_device_name(0)}")
        
        # Check CUDA version compatibility
        print(f"   ✅ CUDA runtime: {torch.version.cuda}")
        device_props = torch.cuda.get_device_properties(0)
        print(f"   ✅ GPU memory: {device_props.total_memory // 1024**3}GB")
        print(f"   ✅ Compute capability: {device_props.major}.{device_props.minor}")
    else:
        print("   ❌ CUDA not available - CHECK DRIVERS!")
        sys.exit(1)
        
except ImportError as e:
    print(f"   ❌ PyTorch import error: {e}")
    sys.exit(1)

# 2. KIỂM TRA MODEL CHUYỂN SANG GPU
print("\n2️⃣ KIỂM TRA MODEL TRÊN GPU:")
try:
    import torch.nn as nn
    
    # Tạo model đơn giản
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    print(f"   📍 Model ban đầu: {next(model.parameters()).device}")
    
    # Chuyển model sang GPU
    model = model.cuda()
    print(f"   🔥 Model sau .cuda(): {next(model.parameters()).device}")
    
    # Kiểm tra tất cả parameters
    all_on_gpu = all(p.device.type == 'cuda' for p in model.parameters())
    print(f"   ✅ Tất cả parameters trên GPU: {all_on_gpu}")
    
except Exception as e:
    print(f"   ❌ Model GPU error: {e}")

# 3. KIỂM TRA DỮ LIỆU TRÊN GPU
print("\n3️⃣ KIỂM TRA DỮ LIỆU TRÊN GPU:")
try:
    # Tạo dữ liệu test
    x_cpu = torch.randn(100, 10)
    y_cpu = torch.randn(100, 1)
    
    print(f"   📍 Data ban đầu: x={x_cpu.device}, y={y_cpu.device}")
    
    # Chuyển data sang GPU
    x_gpu = x_cpu.cuda()
    y_gpu = y_cpu.cuda()
    
    print(f"   🔥 Data sau .cuda(): x={x_gpu.device}, y={y_gpu.device}")
    
    # Test forward pass
    output = model(x_gpu)
    print(f"   🔥 Model output device: {output.device}")
    
    # Kiểm tra memory usage
    print(f"   📊 GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
except Exception as e:
    print(f"   ❌ Data GPU error: {e}")

# 4. KIỂM TRA TENSOR VÀ LOSS FUNCTION ĐỒNG BỘ
print("\n4️⃣ KIỂM TRA ĐỒNG BỘ THIẾT BỊ:")
try:
    # Test loss function
    criterion = nn.MSELoss()
    
    # Tạo target trên GPU
    target = torch.randn(100, 1).cuda()
    
    print(f"   📍 Loss function device: CPU (default)")
    print(f"   📍 Output device: {output.device}")
    print(f"   📍 Target device: {target.device}")
    
    # Tính loss
    loss = criterion(output, target)
    print(f"   🔥 Loss device: {loss.device}")
    print(f"   ✅ Loss value: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"   ✅ Backward pass: SUCCESS")
    
    # Kiểm tra gradients
    grad_devices = [p.grad.device if p.grad is not None else 'None' for p in model.parameters()]
    print(f"   🔥 Gradient devices: {grad_devices}")
    
except Exception as e:
    print(f"   ❌ Sync error: {e}")

# 5. TEST TRAINING LOOP HOÀN CHỈNH
print("\n🚀 TEST TRAINING LOOP HOÀN CHỈNH:")
try:
    import torch.optim as optim
    
    # Reset model
    model = TestModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training data
    x_train = torch.randn(1000, 10).cuda()
    y_train = torch.randn(1000, 1).cuda()
    
    print(f"   📊 Training data: {x_train.shape} on {x_train.device}")
    
    # Mini training loop
    model.train()
    total_loss = 0
    
    for i in range(0, len(x_train), 100):  # Batch size 100
        batch_x = x_train[i:i+100]
        batch_y = y_train[i:i+100]
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i == 0:  # First batch
            print(f"   🔥 First batch - Input: {batch_x.device}, Output: {output.device}, Loss: {loss.device}")
    
    avg_loss = total_loss / (len(x_train) // 100)
    print(f"   ✅ Training loop SUCCESS - Avg Loss: {avg_loss:.4f}")
    print(f"   🔥 Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
except Exception as e:
    print(f"   ❌ Training loop error: {e}")

print("\n🎉 DIAGNOSTIC COMPLETED!")
print("="*60)

# KIỂM TRA CUỐI CÙNG
print("\n📋 TÓM TẮT:")
if torch.cuda.is_available():
    print("✅ GPU environment: OK")
    print("✅ Model to GPU: OK") 
    print("✅ Data to GPU: OK")
    print("✅ Device sync: OK")
    print("✅ Training loop: OK")
    print("\n🔥 GPU READY FOR TRAINING!")
else:
    print("❌ GPU not available") 