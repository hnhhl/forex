#!/usr/bin/env python3
"""
📊 MONITOR TRAINING PROGRESS
===========================
Script để theo dõi tiến trình training của hệ thống
"""

import os
import time
import json
import glob
from datetime import datetime

def monitor_training():
    """Monitor training progress"""
    print("📊 THEO DÕI TIẾN TRÌNH TRAINING")
    print("=" * 40)
    
    results_dir = "training_results_real_data"
    models_dir = "trained_models_real_data"
    
    start_time = datetime.now()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔥 ULTIMATE XAU SYSTEM V4.0 - TRAINING MONITOR")
        print("=" * 60)
        print(f"⏰ Monitoring since: {start_time}")
        print(f"🕐 Current time: {datetime.now()}")
        print()
        
        # Check results directory
        if os.path.exists(results_dir):
            result_files = glob.glob(f"{results_dir}/*.json")
            print(f"📄 Result files: {len(result_files)}")
            
            if result_files:
                latest_file = max(result_files, key=os.path.getctime)
                print(f"📋 Latest result: {os.path.basename(latest_file)}")
                
                try:
                    with open(latest_file, 'r') as f:
                        results = json.load(f)
                    
                    print(f"📊 Training phases: {len(results.get('training_phases', {}))}")
                    
                    for tf, phase in results.get('training_phases', {}).items():
                        models_count = len(phase.get('models_trained', []))
                        data_points = phase.get('data_points', 0)
                        print(f"   {tf}: {models_count} models, {data_points:,} data points")
                        
                        # Show best accuracy if available
                        eval_results = phase.get('evaluation_results', {})
                        if eval_results:
                            best_acc = max([r.get('accuracy', 0) for r in eval_results.values()])
                            print(f"      Best accuracy: {best_acc:.4f}")
                
                except Exception as e:
                    print(f"❌ Error reading results: {e}")
        else:
            print("⏳ Waiting for training to start...")
        
        # Check models directory
        if os.path.exists(models_dir):
            model_files = glob.glob(f"{models_dir}/*")
            print(f"🧠 Model files created: {len(model_files)}")
            
            # Count by type
            keras_files = glob.glob(f"{models_dir}/*.keras")
            pkl_files = glob.glob(f"{models_dir}/*.pkl")
            json_files = glob.glob(f"{models_dir}/*.json")
            
            print(f"   📊 Keras models: {len(keras_files)}")
            print(f"   📊 Pickle models: {len(pkl_files)}")
            print(f"   📊 Config files: {len(json_files)}")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop monitoring")
        
        try:
            time.sleep(10)  # Update every 10 seconds
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped!")
            break

if __name__ == "__main__":
    monitor_training() 