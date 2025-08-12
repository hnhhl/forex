#!/usr/bin/env python3
import os
import time
import json
from datetime import datetime
from pathlib import Path

def monitor():
    print("🚀 ULTIMATE XAU TRAINING MONITOR")
    print("="*50)
    
    start_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        runtime = current_time - start_time
        
        print(f"\n⏰ {current_time.strftime('%H:%M:%S')} | Runtime: {str(runtime).split('.')[0]}")
        
        # Check models
        models_dir = Path("trained_models")
        model_count = len(list(models_dir.glob("*"))) if models_dir.exists() else 0
        print(f"🤖 Models: {model_count}")
        
        # Check results
        results_dir = Path("training_results")
        if results_dir.exists():
            recent_files = [f for f in results_dir.glob("*.json") 
                          if time.time() - os.path.getctime(f) < 3600]
            if recent_files:
                latest = max(recent_files, key=os.path.getctime)
                print(f"📊 Latest: {latest.name}")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                print(f"🔥 GPU: {gpu_mem:.2f}GB")
        except:
            pass
        
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n⏹️ Monitor stopped") 