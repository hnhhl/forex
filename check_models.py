import os
import json
from datetime import datetime

def check_training_system():
    """Kiểm tra hệ thống training hiện tại"""
    
    print("🔍 KIỂM TRA HỆ THỐNG TRAINING AI3.0")
    print("=" * 50)
    
    # Kiểm tra trained_models
    models_dir = "trained_models"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        total_files = len(files)
        
        # Phân loại theo extension
        pkl_files = [f for f in files if f.endswith('.pkl')]
        keras_files = [f for f in files if f.endswith('.keras')]
        h5_files = [f for f in files if f.endswith('.h5')]
        json_files = [f for f in files if f.endswith('.json')]
        
        print(f"📁 TRAINED MODELS DIRECTORY:")
        print(f"   • Total files: {total_files}")
        print(f"   • PKL files (Traditional ML): {len(pkl_files)}")
        print(f"   • Keras files (Neural Networks): {len(keras_files)}")
        print(f"   • H5 files (Legacy Neural): {len(h5_files)}")
        print(f"   • JSON files (Config): {len(json_files)}")
        print(f"   • Total Models: {len(pkl_files) + len(keras_files) + len(h5_files)}")
        
        # Kiểm tra unified models
        unified_dir = os.path.join(models_dir, "unified")
        if os.path.exists(unified_dir):
            unified_files = os.listdir(unified_dir)
            unified_models = [f for f in unified_files if f.endswith(('.pkl', '.keras', '.h5'))]
            print(f"   • Unified Models: {len(unified_models)}")
        
    else:
        print("❌ trained_models directory not found!")
    
    print()
    
    # Kiểm tra training results
    results_dirs = [
        "training_results",
        "smart_training_results", 
        "training_reports",
        "optimized_training_results",
        "real_training_results"
    ]
    
    print(f"📊 TRAINING RESULTS:")
    total_results = 0
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            result_files = [f for f in files if f.endswith('.json')]
            total_results += len(result_files)
            print(f"   • {results_dir}: {len(result_files)} results")
    
    print(f"   • Total Results: {total_results}")
    print()
    
    # Kiểm tra training systems
    print(f"🚀 TRAINING SYSTEMS:")
    training_systems = [
        "MASS_TRAINING_SYSTEM_AI30.py",
        "comprehensive_training_fixed.py", 
        "demo_mass_training.py",
        "mass_training_demo.py"
    ]
    
    available_systems = []
    for system in training_systems:
        if os.path.exists(system):
            available_systems.append(system)
            size = os.path.getsize(system) / 1024  # KB
            print(f"   ✅ {system} ({size:.1f} KB)")
        else:
            print(f"   ❌ {system} (not found)")
    
    print(f"   • Available Systems: {len(available_systems)}")
    print()
    
    # Kiểm tra data directories
    print(f"💾 DATA DIRECTORIES:")
    data_dirs = [
        "data",
        "learning_data",
        "training/xauusdc"
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            try:
                files = os.listdir(data_dir)
                csv_files = [f for f in files if f.endswith('.csv')]
                pkl_files = [f for f in files if f.endswith('.pkl')]
                print(f"   ✅ {data_dir}: {len(files)} files ({len(csv_files)} CSV, {len(pkl_files)} PKL)")
            except:
                print(f"   ⚠️ {data_dir}: Access denied")
        else:
            print(f"   ❌ {data_dir}: Not found")
    
    print()
    
    # Summary
    print(f"🎯 SUMMARY:")
    print(f"   • Total Models Available: {len(pkl_files) + len(keras_files) + len(h5_files) if 'pkl_files' in locals() else 'Unknown'}")
    print(f"   • Training Systems: {len(available_systems)}")
    print(f"   • Training Results: {total_results}")
    print(f"   • System Status: {'Ready for Training' if available_systems else 'Setup Required'}")
    
    return {
        'total_models': len(pkl_files) + len(keras_files) + len(h5_files) if 'pkl_files' in locals() else 0,
        'training_systems': len(available_systems),
        'training_results': total_results,
        'status': 'ready' if available_systems else 'setup_required'
    }

if __name__ == "__main__":
    result = check_training_system()
    
    # Save results
    with open(f'training_system_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(result, f, indent=2) 