#!/usr/bin/env python3
"""
📊 SYSTEM STATUS REPORT - AI3.0 TRADING SYSTEM
Báo cáo toàn diện về tình trạng hệ thống hiện tại
"""

import os
import sys
import json
import subprocess
import platform
from datetime import datetime
import psutil
import pandas as pd

def get_system_info():
    """Thu thập thông tin hệ thống cơ bản"""
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Memory info
    memory = psutil.virtual_memory()
    info["memory"] = {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_percent": memory.percent
    }
    
    # Disk info
    disk = psutil.disk_usage('C:')
    info["disk"] = {
        "total_gb": round(disk.total / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "used_percent": round((disk.used / disk.total) * 100, 2)
    }
    
    # CPU info
    info["cpu"] = {
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "current_usage": psutil.cpu_percent(interval=1)
    }
    
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    return info

def check_gpu_status():
    """Kiểm tra trạng thái GPU"""
    print("\n🎮 GPU STATUS")
    print("=" * 50)
    
    gpu_info = {}
    
    try:
        # NVIDIA-SMI check
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(', ')
            gpu_info = {
                "name": gpu_data[0],
                "memory_total_mb": int(gpu_data[1]),
                "memory_used_mb": int(gpu_data[2]),
                "utilization_percent": int(gpu_data[3]),
                "temperature_c": int(gpu_data[4]) if gpu_data[4] != '[Not Supported]' else 'N/A',
                "memory_free_mb": int(gpu_data[1]) - int(gpu_data[2]),
                "status": "ACTIVE"
            }
            
            print(f"GPU Name: {gpu_info['name']}")
            print(f"Memory: {gpu_info['memory_used_mb']}/{gpu_info['memory_total_mb']} MB ({gpu_info['memory_free_mb']} MB free)")
            print(f"Utilization: {gpu_info['utilization_percent']}%")
            print(f"Temperature: {gpu_info['temperature_c']}°C")
            print("Status: ✅ AVAILABLE")
            
        else:
            gpu_info = {"status": "NOT_FOUND", "error": result.stderr}
            print("Status: ❌ NO GPU DETECTED")
            
    except Exception as e:
        gpu_info = {"status": "ERROR", "error": str(e)}
        print(f"Status: ❌ ERROR - {e}")
    
    return gpu_info

def check_python_packages():
    """Kiểm tra các Python packages quan trọng"""
    print("\n📦 PYTHON PACKAGES")
    print("=" * 50)
    
    important_packages = [
        'tensorflow', 'tensorflow-gpu', 'keras', 'numpy', 'pandas', 
        'scikit-learn', 'matplotlib', 'seaborn', 'MetaTrader5', 
        'psutil', 'requests', 'fastapi', 'uvicorn'
    ]
    
    package_status = {}
    
    for package in important_packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                version = next((line.split(': ')[1] for line in lines if line.startswith('Version:')), 'Unknown')
                package_status[package] = {"installed": True, "version": version}
                print(f"✅ {package}: {version}")
            else:
                package_status[package] = {"installed": False, "version": None}
                print(f"❌ {package}: NOT INSTALLED")
        except Exception as e:
            package_status[package] = {"installed": False, "error": str(e)}
            print(f"⚠️  {package}: ERROR - {e}")
    
    return package_status

def check_tensorflow_gpu():
    """Kiểm tra TensorFlow GPU compatibility"""
    print("\n🤖 TENSORFLOW GPU STATUS")
    print("=" * 50)
    
    tf_status = {}
    
    try:
        import tensorflow as tf
        tf_status["tensorflow_version"] = tf.__version__
        
        # GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        tf_status["gpu_devices"] = len(gpus)
        
        if gpus:
            tf_status["gpu_detected"] = True
            tf_status["gpu_names"] = [gpu.name for gpu in gpus]
            
            # Test GPU operation
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                tf_status["gpu_operations"] = "WORKING"
                print("✅ TensorFlow GPU: WORKING")
            except Exception as e:
                tf_status["gpu_operations"] = f"ERROR: {str(e)}"
                print(f"⚠️  TensorFlow GPU: ERROR - {e}")
        else:
            tf_status["gpu_detected"] = False
            print("❌ TensorFlow GPU: NO DEVICES")
        
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"GPU Devices: {len(gpus)}")
        
    except ImportError as e:
        tf_status = {"error": f"TensorFlow not available: {str(e)}"}
        print(f"❌ TensorFlow: NOT AVAILABLE - {e}")
    except Exception as e:
        tf_status = {"error": f"TensorFlow error: {str(e)}"}
        print(f"⚠️  TensorFlow: ERROR - {e}")
    
    return tf_status

def check_data_files():
    """Kiểm tra các file dữ liệu quan trọng"""
    print("\n📊 DATA FILES STATUS")
    print("=" * 50)
    
    important_files = [
        "data/working_free_data/XAUUSD_M1_realistic.csv",
        "data/working_free_data/XAUUSD_H1_realistic.csv", 
        "data/working_free_data/XAUUSD_D1_realistic.csv",
        "data/maximum_mt5_v2/XAUUSDc_M1_20250618_115847.csv",
        "data/maximum_mt5_v2/XAUUSDc_H1_20250618_115847.csv",
        "data/maximum_mt5_v2/XAUUSDc_D1_20250618_115847.csv"
    ]
    
    data_status = {}
    
    for file_path in important_files:
        if os.path.exists(file_path):
            try:
                # Get file info
                stat = os.stat(file_path)
                size_mb = round(stat.st_size / (1024*1024), 2)
                modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Try to read CSV
                df = pd.read_csv(file_path, nrows=5)
                rows_sample = len(df)
                
                # Get total rows (estimate)
                with open(file_path, 'r') as f:
                    total_rows = sum(1 for line in f) - 1  # -1 for header
                
                data_status[file_path] = {
                    "exists": True,
                    "size_mb": size_mb,
                    "modified": modified,
                    "total_rows": total_rows,
                    "columns": list(df.columns)
                }
                
                print(f"✅ {file_path}")
                print(f"   Size: {size_mb} MB, Rows: {total_rows:,}, Modified: {modified}")
                
            except Exception as e:
                data_status[file_path] = {"exists": True, "error": str(e)}
                print(f"⚠️  {file_path}: READ ERROR - {e}")
        else:
            data_status[file_path] = {"exists": False}
            print(f"❌ {file_path}: NOT FOUND")
    
    return data_status

def check_trained_models():
    """Kiểm tra các model đã training"""
    print("\n🧠 TRAINED MODELS STATUS")
    print("=" * 50)
    
    model_dirs = [
        "trained_models",
        "trained_models_optimized", 
        "trained_models_real_data",
        "trained_models_smart"
    ]
    
    models_status = {}
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.pkl', '.json'))]
            
            models_status[model_dir] = {
                "exists": True,
                "files": []
            }
            
            print(f"📁 {model_dir}/")
            
            for file in files:
                file_path = os.path.join(model_dir, file)
                stat = os.stat(file_path)
                size_mb = round(stat.st_size / (1024*1024), 2)
                modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                models_status[model_dir]["files"].append({
                    "name": file,
                    "size_mb": size_mb,
                    "modified": modified
                })
                
                print(f"  ✅ {file} ({size_mb} MB, {modified})")
            
            if not files:
                print(f"  ⚠️  No model files found")
        else:
            models_status[model_dir] = {"exists": False}
            print(f"❌ {model_dir}/: NOT FOUND")
    
    return models_status

def check_ai3_system():
    """Kiểm tra AI3.0 system files"""
    print("\n🤖 AI3.0 SYSTEM FILES")
    print("=" * 50)
    
    core_files = [
        "src/core/ultimate_xau_system.py",
        "src/core/ai/advanced_ai2_technologies.py",
        "src/core/ai/ai_phases/main.py",
        "src/core/advanced_ai_ensemble.py",
        "src/core/integration/master_system.py"
    ]
    
    system_status = {}
    
    for file_path in core_files:
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            size_kb = round(stat.st_size / 1024, 2)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            system_status[file_path] = {
                "exists": True,
                "size_kb": size_kb,
                "modified": modified
            }
            
            print(f"✅ {file_path} ({size_kb} KB, {modified})")
        else:
            system_status[file_path] = {"exists": False}
            print(f"❌ {file_path}: NOT FOUND")
    
    return system_status

def generate_report():
    """Tạo báo cáo tổng hợp"""
    print("📋 GENERATING COMPREHENSIVE REPORT...")
    print("=" * 70)
    
    report = {
        "report_timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "gpu_status": check_gpu_status(), 
        "python_packages": check_python_packages(),
        "tensorflow_gpu": check_tensorflow_gpu(),
        "data_files": check_data_files(),
        "trained_models": check_trained_models(),
        "ai3_system": check_ai3_system()
    }
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"system_status_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 REPORT SUMMARY")
    print("=" * 50)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Saved to: {report_file}")
    
    # Quick summary
    gpu_ok = report['gpu_status'].get('status') == 'ACTIVE'
    tf_gpu_ok = report['tensorflow_gpu'].get('gpu_operations') == 'WORKING'
    data_files_count = sum(1 for f in report['data_files'].values() if f.get('exists'))
    models_count = sum(len(m.get('files', [])) for m in report['trained_models'].values() if m.get('exists'))
    
    print(f"🎮 GPU: {'✅ ACTIVE' if gpu_ok else '❌ INACTIVE'}")
    print(f"🤖 TensorFlow GPU: {'✅ WORKING' if tf_gpu_ok else '❌ NOT WORKING'}")
    print(f"📊 Data Files: {data_files_count}/6 available")
    print(f"🧠 Trained Models: {models_count} files found")
    
    return report_file

if __name__ == "__main__":
    print("🔍 AI3.0 SYSTEM STATUS ANALYSIS")
    print("=" * 70)
    print(f"🕒 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        report_file = generate_report()
        print(f"\n✅ ANALYSIS COMPLETED!")
        print(f"📄 Full report: {report_file}")
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}") 