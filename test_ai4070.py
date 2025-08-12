import torch
import sys

print("="*50)
print("AI4070 ENVIRONMENT TEST")
print("="*50)
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Test GPU computation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation test: SUCCESS")
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
else:
    print("❌ No GPU detected")

print("="*50) 