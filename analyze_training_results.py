#!/usr/bin/env python3
"""
Phân tích kết quả training để tìm models bị thiếu
"""

import json
import os
from collections import defaultdict

def analyze_training_results():
    # Tìm file kết quả mới nhất
    result_files = [f for f in os.listdir('.') if f.startswith('group_training_results_') and f.endswith('.json')]
    if not result_files:
        print("❌ Không tìm thấy file kết quả training")
        return
    
    latest_file = max(result_files, key=lambda x: os.path.getmtime(x))
    print(f"📊 Phân tích file: {latest_file}")
    
    # Đọc kết quả
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n🎯 TỔNG QUAN:")
    print(f"   • Total models: {data.get('total_models', 0)}")
    print(f"   • Successful models: {data.get('successful_models', 0)}")
    print(f"   • Training time: {data.get('total_training_time', 0):.2f}s")
    
    # Phân tích theo groups
    models = data.get('models', {})
    groups = defaultdict(list)
    
    for model_id, model_data in models.items():
        if isinstance(model_data, dict):
            success = model_data.get('success', False)
            arch = model_data.get('architecture', 'unknown')
            
            # Phân loại theo architecture
            if 'dense' in model_id or 'dense' in arch:
                groups['Dense'].append((model_id, success))
            elif 'cnn' in model_id or 'cnn' in arch:
                groups['CNN'].append((model_id, success))
            elif 'rnn' in model_id or 'lstm' in arch or 'gru' in arch:
                groups['RNN'].append((model_id, success))
            elif 'transformer' in model_id or 'transformer' in arch:
                groups['Transformer'].append((model_id, success))
            elif 'traditional' in model_id or 'traditional' in arch:
                groups['Traditional'].append((model_id, success))
            elif 'hybrid' in model_id or 'hybrid' in arch:
                groups['Hybrid'].append((model_id, success))
            else:
                groups['Other'].append((model_id, success))
    
    print(f"\n📊 PHÂN TÍCH THEO GROUPS:")
    total_expected = 0
    total_actual = 0
    
    expected_counts = {
        'Dense': 100,  # 50 cơ bản + 50 advanced
        'CNN': 80,     # 40 cơ bản + 40 advanced  
        'RNN': 60,     # 30 cơ bản + 30 advanced
        'Transformer': 40,  # 20 cơ bản + 20 advanced
        'Traditional': 100, # 60 cơ bản + 40 advanced
        'Hybrid': 50,  # 50 mới
        'Other': 20    # Existing models
    }
    
    for group_name, expected in expected_counts.items():
        actual = len(groups[group_name])
        successful = sum(1 for _, success in groups[group_name] if success)
        
        total_expected += expected
        total_actual += actual
        
        status = "✅" if actual >= expected else "❌"
        print(f"   {status} {group_name}: {actual}/{expected} models, {successful} successful")
        
        if actual < expected:
            print(f"      ⚠️ THIẾU {expected - actual} models!")
    
    print(f"\n🎯 TỔNG KẾT:")
    print(f"   • Expected: {total_expected} models")
    print(f"   • Actual: {total_actual} models") 
    print(f"   • Missing: {total_expected - total_actual} models")
    
    # Tìm models failed
    failed_models = []
    for model_id, model_data in models.items():
        if isinstance(model_data, dict) and not model_data.get('success', False):
            error = model_data.get('error_message', 'Unknown error')
            failed_models.append((model_id, error))
    
    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        for model_id, error in failed_models[:10]:  # Show first 10
            print(f"   • {model_id}: {error}")
        if len(failed_models) > 10:
            print(f"   • ... và {len(failed_models) - 10} models khác")

if __name__ == "__main__":
    analyze_training_results() 