"""
⚡ QUICK TRAINING STATUS CHECK
Kiểm tra nhanh tiến trình training Ultimate XAU System V5.0
"""

import os
import glob
import json
from datetime import datetime

def check_training_status():
    """Quick check training status"""
    print("⚡ ULTIMATE XAU SYSTEM V5.0 - TRAINING STATUS CHECK")
    print("="*70)
    
    # 1. Check training data
    print("📊 TRAINING DATA:")
    data_files = glob.glob("training/xauusdc/data/*.pkl")
    print(f"   • Data files: {len(data_files)} found")
    
    # 2. Check unified system files  
    print("\n🔗 UNIFIED SYSTEM:")
    unified_files = glob.glob("UNIFIED_*.py")
    print(f"   • Unified files: {len(unified_files)} found")
    
    # 3. Check training script
    print("\n🚀 TRAINING SCRIPT:")
    if os.path.exists("ULTIMATE_SYSTEM_TRAINING.py"):
        print("   ✅ Training script exists")
        stat = os.stat("ULTIMATE_SYSTEM_TRAINING.py")
        print(f"   • Size: {stat.st_size:,} bytes")
        print(f"   • Modified: {datetime.fromtimestamp(stat.st_mtime)}")
    
    # 4. Check trained models
    print("\n🧠 TRAINED MODELS:")
    if os.path.exists("trained_models"):
        model_files = glob.glob("trained_models/*")
        print(f"   • Model files: {len(model_files)} found")
        
        if model_files:
            h5_files = glob.glob("trained_models/*.h5")
            pkl_files = glob.glob("trained_models/*.pkl") 
            json_files = glob.glob("trained_models/*.json")
            
            print(f"   • Neural models (.h5): {len(h5_files)}")
            print(f"   • Traditional models (.pkl): {len(pkl_files)}")
            print(f"   • Config files (.json): {len(json_files)}")
    else:
        print("   ⚠️ No trained_models directory yet")
    
    # 5. Check training reports
    print("\n📋 TRAINING REPORTS:")
    report_files = glob.glob("ULTIMATE_SYSTEM_TRAINING_REPORT_*.json")
    
    if report_files:
        latest_report = max(report_files, key=os.path.getctime)
        print(f"   ✅ Latest report: {latest_report}")
        
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            print(f"   • Training start: {report_data.get('training_start_time', 'N/A')}")
            print(f"   • Training end: {report_data.get('training_end_time', 'N/A')}")
            print(f"   • Total time: {report_data.get('total_training_time', 'N/A')}")
            
            if 'unified_data_info' in report_data:
                data_info = report_data['unified_data_info']
                print(f"   • Samples: {data_info.get('total_samples', 0):,}")
                print(f"   • Features: {data_info.get('total_features', 0):,}")
                print(f"   • Timeframes: {data_info.get('timeframes_used', 0)}")
            
            if 'training_results' in report_data:
                print(f"\n🎯 MODEL TRAINING RESULTS:")
                for model_type, results in report_data['training_results'].items():
                    status = results.get('status', 'unknown')
                    status_icon = '✅' if status == 'completed' else '❌' if status == 'failed' else '⚠️'
                    
                    print(f"   {status_icon} {model_type.upper()}: {status}")
                    
                    if 'avg_accuracy' in results:
                        print(f"      - Accuracy: {results['avg_accuracy']:.3f}")
                    if 'avg_reward' in results:
                        print(f"      - Reward: {results['avg_reward']:.3f}")
                    if 'targets_trained' in results:
                        print(f"      - Targets: {results['targets_trained']}")
                        
        except Exception as e:
            print(f"   ❌ Error reading report: {e}")
    else:
        print("   ⚠️ No training reports found yet")
    
    # 6. Check running processes
    print("\n🔄 PROCESS STATUS:")
    import subprocess
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
        if 'python.exe' in result.stdout:
            python_processes = [line for line in result.stdout.split('\n') if 'python.exe' in line]
            print(f"   ✅ Python processes running: {len(python_processes)}")
            for proc in python_processes[:3]:  # Show first 3
                print(f"      {proc.strip()}")
        else:
            print("   ⚠️ No Python processes found")
    except Exception as e:
        print(f"   ❌ Cannot check processes: {e}")
    
    print("\n" + "="*70)
    print(f"⏰ Status checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_training_status() 