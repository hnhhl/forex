#!/usr/bin/env python3
"""
GPU Environment Check for 116+ Models Training
Kiểm tra GPU và thiết lập môi trường training
"""

import sys
import psutil
import platform
import subprocess
import json
from datetime import datetime

def check_system_info():
    """Kiểm tra thông tin hệ thống"""
    print("🖥️ SYSTEM INFORMATION")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.total / (1024**3):.1f} GB total, {disk.free / (1024**3):.1f} GB free")
    print()

def check_python_environment():
    """Kiểm tra Python environment"""
    print("🐍 PYTHON ENVIRONMENT")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()

def check_pytorch():
    """Kiểm tra PyTorch và CUDA"""
    print("🔥 PYTORCH & CUDA CHECK")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Check GPU memory usage
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"   Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
        else:
            print("❌ CUDA not available - CPU training only")
            
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
    print()

def check_tensorflow():
    """Kiểm tra TensorFlow"""
    print("🧠 TENSORFLOW CHECK")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPU devices detected: {len(gpus)}")
        
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
            
        # Check if TensorFlow can use GPU
        if len(gpus) > 0:
            print("✅ TensorFlow can use GPU")
            
            # Try to create a simple operation on GPU
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print("✅ GPU computation test passed")
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
        else:
            print("❌ No GPU devices available for TensorFlow")
            
    except ImportError:
        print("❌ TensorFlow not installed")
    except Exception as e:
        print(f"❌ TensorFlow check failed: {e}")
    print()

def check_other_libraries():
    """Kiểm tra các thư viện khác"""
    print("📚 OTHER LIBRARIES CHECK")
    print("=" * 50)
    
    libraries = [
        'numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost',
        'joblib', 'pickle', 'multiprocessing', 'concurrent.futures'
    ]
    
    for lib in libraries:
        try:
            if lib == 'multiprocessing':
                import multiprocessing
                print(f"✅ {lib}: Available (CPU cores: {multiprocessing.cpu_count()})")
            elif lib == 'concurrent.futures':
                import concurrent.futures
                print(f"✅ {lib}: Available")
            elif lib == 'pickle':
                import pickle
                print(f"✅ {lib}: Available")
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {lib}: {version}")
        except ImportError:
            print(f"❌ {lib}: Not installed")
        except Exception as e:
            print(f"⚠️ {lib}: Error - {e}")
    print()

def estimate_training_capacity():
    """Ước tính khả năng training"""
    print("🚀 TRAINING CAPACITY ESTIMATION")
    print("=" * 50)
    
    # System specs
    cpu_cores = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check GPU
    gpu_available = False
    gpu_memory_gb = 0
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except:
        pass
    
    print(f"CPU Cores: {cpu_cores}")
    print(f"RAM: {memory_gb:.1f} GB")
    print(f"GPU: {'Yes' if gpu_available else 'No'}")
    if gpu_available:
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    # Estimate concurrent models
    if gpu_available:
        # GPU can handle neural networks
        neural_concurrent = min(6, max(1, int(gpu_memory_gb / 2)))  # 2GB per neural model
        tree_concurrent = min(12, cpu_cores)  # Tree models on CPU
        linear_concurrent = min(15, cpu_cores)  # Linear models on CPU
        
        total_concurrent = neural_concurrent + tree_concurrent + linear_concurrent
        
        print(f"\n📊 ESTIMATED CONCURRENT TRAINING:")
        print(f"   Neural Networks (GPU): {neural_concurrent} models")
        print(f"   Tree Models (CPU): {tree_concurrent} models")
        print(f"   Linear Models (CPU): {linear_concurrent} models")
        print(f"   Total Concurrent: {total_concurrent} models")
        
        # Time estimation
        if total_concurrent >= 40:
            training_time = "60-90 minutes"
            performance = "EXCELLENT 🚀"
        elif total_concurrent >= 25:
            training_time = "90-120 minutes"
            performance = "GOOD 👍"
        elif total_concurrent >= 15:
            training_time = "2-3 hours"
            performance = "MODERATE ⚠️"
        else:
            training_time = "4+ hours"
            performance = "LIMITED ❌"
            
        print(f"   Estimated Training Time: {training_time}")
        print(f"   Performance Rating: {performance}")
        
    else:
        # CPU only
        cpu_concurrent = min(20, cpu_cores)
        print(f"\n📊 CPU-ONLY TRAINING:")
        print(f"   Concurrent Models: {cpu_concurrent}")
        print(f"   Estimated Time: 4-8 hours")
        print(f"   Performance: LIMITED (CPU Only) ❌")
    
    print()

def generate_recommendations():
    """Tạo khuyến nghị cải thiện"""
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = psutil.cpu_count(logical=True)
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = 0
    except:
        gpu_available = False
        gpu_memory_gb = 0
    
    recommendations = []
    
    # Memory recommendations
    if memory_gb < 16:
        recommendations.append("❗ RAM: Upgrade to at least 32GB for optimal performance")
    elif memory_gb < 32:
        recommendations.append("⚠️ RAM: Consider upgrading to 64GB for heavy workloads")
    else:
        recommendations.append("✅ RAM: Sufficient for 116+ models training")
    
    # GPU recommendations
    if not gpu_available:
        recommendations.append("❗ GPU: Install CUDA-compatible GPU (RTX 3070+ recommended)")
    elif gpu_memory_gb < 8:
        recommendations.append("⚠️ GPU: Consider upgrading to GPU with 12GB+ VRAM")
    elif gpu_memory_gb < 12:
        recommendations.append("👍 GPU: Good for training, RTX 4090 would be optimal")
    else:
        recommendations.append("✅ GPU: Excellent for 116+ models training")
    
    # CPU recommendations
    if cpu_cores < 8:
        recommendations.append("❗ CPU: Upgrade to at least 8-core processor")
    elif cpu_cores < 16:
        recommendations.append("⚠️ CPU: 16+ cores recommended for parallel training")
    else:
        recommendations.append("✅ CPU: Sufficient cores for parallel processing")
    
    # Software recommendations
    recommendations.extend([
        "🔧 Enable GPU memory growth in TensorFlow",
        "🔧 Use mixed precision training (FP16) to save memory",
        "🔧 Implement gradient checkpointing for large models",
        "🔧 Use model parallelism for very large ensembles",
        "🔧 Monitor GPU temperature during training",
        "🔧 Use SSD storage for faster data loading"
    ])
    
    for rec in recommendations:
        print(f"   {rec}")
    print()

def save_environment_report():
    """Lưu báo cáo môi trường"""
    print("💾 SAVING ENVIRONMENT REPORT")
    print("=" * 50)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_cores": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3)
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable
        }
    }
    
    # Add GPU info
    try:
        import torch
        if torch.cuda.is_available():
            report["gpu"] = {
                "available": True,
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "cuda_version": torch.version.cuda
            }
        else:
            report["gpu"] = {"available": False}
    except:
        report["gpu"] = {"available": False, "error": "PyTorch not available"}
    
    # Add TensorFlow info
    try:
        import tensorflow as tf
        report["tensorflow"] = {
            "version": tf.__version__,
            "gpu_devices": len(tf.config.list_physical_devices('GPU'))
        }
    except:
        report["tensorflow"] = {"error": "TensorFlow not available"}
    
    # Save report
    report_file = f"environment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved: {report_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
    print()

def main():
    """Main function"""
    print("🔍 GPU ENVIRONMENT CHECK FOR 116+ MODELS TRAINING")
    print("=" * 60)
    print(f"Check started at: {datetime.now()}")
    print()
    
    check_system_info()
    check_python_environment()
    check_pytorch()
    check_tensorflow()
    check_other_libraries()
    estimate_training_capacity()
    generate_recommendations()
    save_environment_report()
    
    print("🎉 ENVIRONMENT CHECK COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main() 