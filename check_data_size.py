import os
import pandas as pd
import psutil

def check_system_resources():
    """Check system resources for training"""
    print("🔍 SYSTEM RESOURCES CHECK")
    print("=" * 50)
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"💾 RAM Total: {memory.total / (1024**3):.2f} GB")
    print(f"💾 RAM Available: {memory.available / (1024**3):.2f} GB")
    print(f"💾 RAM Used: {memory.percent:.1f}%")
    
    # CPU info
    print(f"🖥️ CPU Cores: {psutil.cpu_count()}")
    print(f"🖥️ CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    return memory.available / (1024**3)

def analyze_data_size():
    """Analyze total data size"""
    print("\n📊 DATA SIZE ANALYSIS")
    print("=" * 50)
    
    total_size = 0
    file_count = 0
    data_info = {}
    
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
                
                # Get folder name
                folder = os.path.basename(root)
                if folder not in data_info:
                    data_info[folder] = {'size': 0, 'files': 0}
                data_info[folder]['size'] += size
                data_info[folder]['files'] += 1
    
    print(f"📁 Total CSV files: {file_count}")
    print(f"💾 Total data size: {total_size / (1024**2):.2f} MB")
    
    # Detail by folder
    print("\n📂 BY FOLDER:")
    for folder, info in data_info.items():
        print(f"  {folder}: {info['size'] / (1024**2):.2f} MB ({info['files']} files)")
    
    return total_size / (1024**2)

def estimate_memory_usage(data_size_mb):
    """Estimate memory usage for training"""
    print("\n🧮 MEMORY USAGE ESTIMATION")
    print("=" * 50)
    
    # Rough estimates
    raw_data_memory = data_size_mb * 8  # CSV to DataFrame expansion
    feature_memory = raw_data_memory * 3  # Feature engineering
    model_memory = 2048  # ~2GB for models and training
    buffer_memory = 1024  # 1GB safety buffer
    
    total_estimated = (raw_data_memory + feature_memory + model_memory + buffer_memory) / 1024
    
    print(f"📄 Raw data in memory: ~{raw_data_memory:.0f} MB")
    print(f"🔧 Feature engineering: ~{feature_memory:.0f} MB") 
    print(f"🤖 Models + training: ~{model_memory:.0f} MB")
    print(f"🛡️ Safety buffer: ~{buffer_memory:.0f} MB")
    print(f"📊 TOTAL ESTIMATED: ~{total_estimated:.2f} GB")
    
    return total_estimated

def check_training_feasibility():
    """Check if training all data is feasible"""
    print("\n🎯 TRAINING FEASIBILITY CHECK")
    print("=" * 50)
    
    available_ram = check_system_resources()
    data_size = analyze_data_size()
    estimated_usage = estimate_memory_usage(data_size)
    
    feasible = available_ram > estimated_usage
    
    print(f"\n🔍 ANALYSIS:")
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Estimated usage: {estimated_usage:.2f} GB")
    
    if feasible:
        print("✅ FEASIBLE: Can train all data at once")
        print("💡 Recommended: Full comprehensive training")
    else:
        print("❌ NOT FEASIBLE: Insufficient memory")
        print("💡 Recommended: Batch training or data sampling")
        
        # Calculate safe batch size
        safe_ratio = available_ram / estimated_usage * 0.7  # 70% safety margin
        safe_records = int(1500000 * safe_ratio)  # Estimate total records
        print(f"🔧 Safe batch size: ~{safe_records:,} records")
    
    return feasible, available_ram, estimated_usage

if __name__ == "__main__":
    print("🚀 AI3.0 COMPREHENSIVE DATA TRAINING ANALYSIS")
    print("=" * 60)
    
    feasible, ram, usage = check_training_feasibility()
    
    print(f"\n🎯 RECOMMENDATION:")
    if feasible:
        print("✅ Proceed with FULL DATA TRAINING")
        print("🚀 All timeframes + All data sources")
        print("⚡ Expected training time: 45-90 minutes")
    else:
        print("⚠️ Use SMART BATCH TRAINING")
        print("📊 Train by timeframes separately")
        print("⏱️ Expected training time: 2-3 hours total") 