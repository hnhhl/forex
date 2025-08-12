#!/usr/bin/env python3
import torch
import psutil
import platform
import os
import sys
from datetime import datetime

def check_system():
    print("="*60)
    print("🔍 KIỂM TRA HỆ THỐNG TRƯỚC KHI TRAINING")
    print("="*60)
    print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System Info
    print("💻 THÔNG TIN HỆ THỐNG:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.architecture()[0]}")
    print(f"   Processor: {platform.processor()}")
    print()
    
    # Python Info
    print("🐍 PYTHON ENVIRONMENT:")
    print(f"   Version: {sys.version.split()[0]}")
    print(f"   Executable: {sys.executable}")
    print(f"   PyTorch: {torch.__version__}")
    print()
    
    # Memory Info
    memory = psutil.virtual_memory()
    print("💾 BỘ NHỚ:")
    print(f"   Total RAM: {memory.total/1024**3:.1f} GB")
    print(f"   Available: {memory.available/1024**3:.1f} GB")
    print(f"   Used: {(memory.total-memory.available)/1024**3:.1f} GB")
    print(f"   Usage: {memory.percent:.1f}%")
    print()
    
    # CPU Info
    print("⚡ CPU:")
    print(f"   Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"   Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print()
    
    # Disk Info
    print("💽 Ổ CỨNG:")
    for partition in psutil.disk_partitions():
        try:
            disk_usage = psutil.disk_usage(partition.mountpoint)
            print(f"   {partition.device}: {disk_usage.total/1024**3:.1f} GB total, {disk_usage.free/1024**3:.1f} GB free")
        except:
            pass
    print()
    
    # GPU Info
    print("🚀 GPU INFORMATION:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA Available: True")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"   Memory: {props.total_memory/1024**3:.1f} GB")
        
        # GPU Memory Usage
        print(f"   Current GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        # GPU Test
        print("   🧪 GPU Performance Test:")
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            x = torch.randn(5000, 5000).cuda()
            y = torch.randn(5000, 5000).cuda()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            print(f"   Matrix multiplication (5000x5000): {elapsed_time:.2f} ms")
            print("   ✅ GPU Test: PASSED")
        except Exception as e:
            print(f"   ❌ GPU Test Failed: {e}")
    else:
        print("   ❌ CUDA Not Available")
    
    print()
    
    # Training Readiness
    print("🎯 ĐÁNH GIÁ SẴN SÀNG TRAINING:")
    
    # Check RAM
    if memory.available/1024**3 > 8:
        print("   ✅ RAM: Đủ cho training (>8GB available)")
    else:
        print("   ⚠️ RAM: Có thể thiếu cho training lớn (<8GB available)")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory/1024**3
        if gpu_memory > 8:
            print("   ✅ GPU Memory: Tuyệt vời cho training (>8GB)")
        elif gpu_memory > 4:
            print("   ✅ GPU Memory: Tốt cho training (>4GB)")
        else:
            print("   ⚠️ GPU Memory: Hạn chế cho training lớn (<4GB)")
    
    # Check Disk Space
    disk_usage = psutil.disk_usage('C:')
    if disk_usage.free/1024**3 > 50:
        print("   ✅ Disk Space: Đủ cho training (>50GB free)")
    else:
        print("   ⚠️ Disk Space: Có thể thiếu cho training lớn (<50GB free)")
    
    print()
    print("="*60)
    print("🚀 HỆ THỐNG SẴN SÀNG CHO TRAINING!")
    print("="*60)

if __name__ == "__main__":
    check_system() 