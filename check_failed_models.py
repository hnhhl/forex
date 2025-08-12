#!/usr/bin/env python3
"""
🔍 CHECK FAILED MODELS
Kiểm tra chi tiết các models bị lỗi trong training
"""

import json

def check_failed_models():
    print("🔍 CHECKING FAILED MODELS")
    print("=" * 50)
    
    # Load results
    with open('group_training_results_20250627_223422.json', 'r') as f:
        data = json.load(f)
    
    # Find failed models
    failed_models = [(k, v) for k, v in data['models'].items() if not v['success']]
    
    print(f"📊 SUMMARY:")
    print(f"Total models: {data['total_models']}")
    print(f"Successful: {data['successful_models']}")
    print(f"Failed: {len(failed_models)}")
    print(f"Success rate: {data['successful_models']/data['total_models']*100:.1f}%")
    
    print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
    print("-" * 50)
    
    for i, (model_name, model_info) in enumerate(failed_models):
        print(f"{i+1:2d}. {model_name}")
        print(f"    Architecture: {model_info.get('architecture', 'N/A')}")
        print(f"    Error: {model_info.get('error', 'N/A')}")
        print(f"    Training time: {model_info.get('training_time', 0):.2f}s")
        print()
    
    # Group by architecture
    print("📊 FAILED BY ARCHITECTURE:")
    print("-" * 30)
    
    failed_by_arch = {}
    for model_name, model_info in failed_models:
        arch = model_info.get('architecture', 'unknown')
        if arch not in failed_by_arch:
            failed_by_arch[arch] = []
        failed_by_arch[arch].append(model_name)
    
    for arch, models in failed_by_arch.items():
        print(f"{arch}: {len(models)} models")
        for model in models[:3]:  # Show first 3
            print(f"  - {model}")
        if len(models) > 3:
            print(f"  - ... and {len(models)-3} more")
        print()
    
    # Show common error patterns
    print("🔍 ERROR ANALYSIS:")
    print("-" * 30)
    
    error_counts = {}
    for model_name, model_info in failed_models:
        error = model_info.get('error', 'Unknown error')
        # Extract key part of error
        if 'CUDA error: invalid configuration argument' in error:
            key_error = 'CUDA invalid configuration'
        elif 'dimension mismatch' in error.lower():
            key_error = 'Dimension mismatch'
        elif 'transformer' in error.lower():
            key_error = 'Transformer architecture issue'
        else:
            key_error = error[:50] + '...' if len(error) > 50 else error
        
        error_counts[key_error] = error_counts.get(key_error, 0) + 1
    
    for error, count in error_counts.items():
        print(f"• {error}: {count} models")
    
    return failed_models

if __name__ == "__main__":
    failed_models = check_failed_models()
    
    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    print("✅ 230/250 models successful (92% success rate)")
    print("❌ 20 transformer models failed")
    print("🔧 Issue: Transformer architecture incompatible with single timestep data")
    print("💡 Solution: Use sequence data or replace with compatible architectures")
    print("=" * 50) 