import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

print("🔍 PHÂN TÍCH TOÀN BỘ DỮ LIỆU MULTI-TIMEFRAME")
print("="*60)

# Danh sách các timeframes
timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
data_summary = {}

print("📊 BƯỚC 1: PHÂN TÍCH CẤU TRÚC DỮ LIỆU")
print("-"*40)

for tf in timeframes:
    print(f"\n🔸 Analyzing {tf} data...")
    
    try:
        # Load data
        data_file = f'training/xauusdc/data/{tf}_data.pkl'
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Analyze structure
        summary = {
            'file_size_mb': os.path.getsize(data_file) / (1024*1024),
            'data_type': type(data).__name__,
            'keys': list(data.keys()) if isinstance(data, dict) else 'Not a dict',
            'samples': len(data['X']) if isinstance(data, dict) and 'X' in data else 'Unknown',
            'features': data['X'].shape[1] if isinstance(data, dict) and 'X' in data else 'Unknown',
            'feature_names': data['feature_names'][:5] if isinstance(data, dict) and 'feature_names' in data else 'Unknown',
            'timestamp_range': None,
            'targets_available': []
        }
        
        # Analyze timestamps
        if isinstance(data, dict) and 'timestamps' in data:
            timestamps = data['timestamps']
            if len(timestamps) > 0:
                # Convert to datetime if needed
                if isinstance(timestamps[0], (int, float)):
                    start_time = pd.to_datetime(timestamps[0], unit='s')
                    end_time = pd.to_datetime(timestamps[-1], unit='s')
                else:
                    start_time = timestamps[0]
                    end_time = timestamps[-1]
                
                summary['timestamp_range'] = {
                    'start': str(start_time),
                    'end': str(end_time),
                    'duration_days': (pd.to_datetime(end_time) - pd.to_datetime(start_time)).days
                }
        
        # Analyze targets
        if isinstance(data, dict):
            for key in data.keys():
                if key.startswith('y_'):
                    target_info = {
                        'name': key,
                        'shape': data[key].shape,
                        'positive_rate': np.mean(data[key]) if data[key].dtype in [bool, int] else 'N/A'
                    }
                    summary['targets_available'].append(target_info)
        
        data_summary[tf] = summary
        
        print(f"   ✅ {tf}: {summary['samples']} samples, {summary['features']} features")
        print(f"      📁 Size: {summary['file_size_mb']:.1f}MB")
        print(f"      🎯 Targets: {len(summary['targets_available'])}")
        
    except Exception as e:
        print(f"   ❌ Error loading {tf}: {e}")
        data_summary[tf] = {'error': str(e)}

print(f"\n📋 BƯỚC 2: TỔNG HỢP THÔNG TIN")
print("-"*40)

# Calculate totals
total_samples = sum(s['samples'] for s in data_summary.values() if isinstance(s.get('samples'), int))
total_size = sum(s['file_size_mb'] for s in data_summary.values() if 'file_size_mb' in s)
common_features = None

# Find common features
for tf, summary in data_summary.items():
    if 'feature_names' in summary and summary['feature_names'] != 'Unknown':
        if common_features is None:
            common_features = summary['feature_names']
        
print(f"📊 Tổng samples: {total_samples:,}")
print(f"💾 Tổng dung lượng: {total_size:.1f}MB")
print(f"🧮 Features per timeframe: {data_summary['M15']['features'] if 'M15' in data_summary else 'Unknown'}")

print(f"\n🎯 BƯỚC 3: PHÂN TÍCH TARGETS")
print("-"*40)

# Analyze targets across timeframes
if 'M15' in data_summary and 'targets_available' in data_summary['M15']:
    sample_targets = data_summary['M15']['targets_available']
    print(f"📋 Available targets (from M15):")
    for target in sample_targets:
        print(f"   • {target['name']}: {target['shape']} - Positive rate: {target['positive_rate']:.3f}" if isinstance(target['positive_rate'], float) else f"   • {target['name']}: {target['shape']}")

print(f"\n🔧 BƯỚC 4: ĐÁNH GIÁ KHẢ NĂNG UNIFIED")
print("-"*40)

# Check compatibility for unified approach
compatibility_issues = []

# Check feature consistency
feature_counts = [s['features'] for s in data_summary.values() if isinstance(s.get('features'), int)]
if len(set(feature_counts)) > 1:
    compatibility_issues.append(f"Inconsistent feature counts: {set(feature_counts)}")
else:
    print("✅ All timeframes have consistent feature count")

# Check sample counts
sample_counts = [(tf, s['samples']) for tf, s in data_summary.items() if isinstance(s.get('samples'), int)]
print(f"📊 Sample counts by timeframe:")
for tf, count in sample_counts:
    print(f"   • {tf}: {count:,} samples")

# Check timestamp alignment potential
print(f"\n⏰ Timestamp ranges:")
for tf, summary in data_summary.items():
    if 'timestamp_range' in summary and summary['timestamp_range']:
        tr = summary['timestamp_range']
        print(f"   • {tf}: {tr['start']} to {tr['end']} ({tr['duration_days']} days)")

if compatibility_issues:
    print(f"\n⚠️ COMPATIBILITY ISSUES:")
    for issue in compatibility_issues:
        print(f"   • {issue}")
else:
    print(f"\n✅ NO MAJOR COMPATIBILITY ISSUES FOUND")

print(f"\n🎯 BƯỚC 5: UNIFIED APPROACH STRATEGY")
print("-"*40)

print("💡 Đề xuất kiến trúc Unified Multi-Timeframe:")
print("   1. 📊 Concat features: 67 features × 7 timeframes = 469 features")
print("   2. ⏰ Time alignment: Sử dụng common timestamps")
print("   3. 🎯 Single target: Chọn một target chung (e.g., y_direction_2)")
print("   4. 🧠 Unified model: Một model nhìn toàn bộ thị trường")

print(f"\n💾 Lưu kết quả phân tích...")
import json
with open('multi_timeframe_analysis_results.json', 'w') as f:
    # Convert numpy types to regular types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Clean data for JSON
    clean_summary = {}
    for tf, summary in data_summary.items():
        clean_summary[tf] = {}
        for key, value in summary.items():
            clean_summary[tf][key] = convert_numpy(value)
    
    json.dump(clean_summary, f, indent=2, default=str)

print("✅ Kết quả đã được lưu vào: multi_timeframe_analysis_results.json")
print("\n🏁 PHÂN TÍCH HOÀN TẤT!") 