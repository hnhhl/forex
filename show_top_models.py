#!/usr/bin/env python3
import json

# Load results
with open('group_training_results_20250627_223422.json', 'r') as f:
    data = json.load(f)

# Get successful models and sort by accuracy
models = [(k, v['validation_accuracy']) for k, v in data['models'].items() if v['success']]
models.sort(key=lambda x: x[1], reverse=True)

print("🏆 TOP 20 MODELS:")
print("="*50)
for i, (model_name, accuracy) in enumerate(models[:20]):
    print(f"{i+1:2d}. {model_name:25s} - {accuracy:.4f}")

print(f"\n📊 SUMMARY:")
print(f"Total models: {data['total_models']}")
print(f"Successful: {data['successful_models']}")
print(f"Success rate: {data['successful_models']/data['total_models']*100:.1f}%")
print(f"Best accuracy: {models[0][1]:.4f}")
print(f"Training time: {data['total_training_time']:.1f}s ({data['total_training_time']/60:.1f} minutes)") 