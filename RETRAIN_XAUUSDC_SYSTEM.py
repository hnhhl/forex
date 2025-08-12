#!/usr/bin/env python3
"""
Re-Training XAU/USDc System với Improvements
Ultimate XAU Super System V4.0
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_system():
    """Re-training hệ thống với data mới và improvements"""
    
    print("🔄 RE-TRAINING XAU/USDc SYSTEM")
    print("=" * 60)
    print("🎯 Mục tiêu:")
    print("  • Cập nhật với data mới nhất")
    print("  • Tối ưu hyperparameters")
    print("  • Cải thiện accuracy")
    print("  • Training M1 và M5 models")
    print("=" * 60)
    
    # Backup models cũ
    backup_dir = f"training/xauusdc/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Chạy training mới
    from XAUUSDC_TRAINING_SYSTEM_OPTIMIZED import XAUUSDcTrainingSystem
    
    system = XAUUSDcTrainingSystem()
    results = system.run_training()
    
    if results:
        print("\n✅ RE-TRAINING THÀNH CÔNG!")
        print("📊 Kết quả đã được cập nhật")
    else:
        print("\n❌ RE-TRAINING THẤT BẠI!")
        
    return results

if __name__ == "__main__":
    retrain_system() 