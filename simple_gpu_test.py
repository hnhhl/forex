import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("GPU test passed:", z.shape)
    print("GPU memory used:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
else:
    print("No GPU available") 