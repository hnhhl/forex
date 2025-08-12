import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig

def test_current_model_usage():
    """Test cách hệ thống hiện tại sử dụng models"""
    print("🔍 TESTING CURRENT MODEL USAGE")
    print("=" * 60)
    
    # Initialize system
    config = SystemConfig()
    config.enable_integrated_training = True
    system = UltimateXAUSystem(config)
    
    print(f"📊 SYSTEM STATUS:")
    print(f"  Model loaded: {system.model_loaded}")
    if system.ai_model:
        print(f"  Model type: {type(system.ai_model).__name__}")
        print(f"  Input shape: {system.ai_model.input_shape}")
        print(f"  Output shape: {system.ai_model.output_shape}")
    
    return system

def check_available_models():
    """Kiểm tra các models có sẵn"""
    print("\n📁 AVAILABLE MODELS:")
    print("=" * 60)
    
    unified_path = "trained_models/unified"
    if os.path.exists(unified_path):
        model_files = [f for f in os.listdir(unified_path) if f.endswith('.keras')]
        
        print(f"📂 Models in {unified_path}:")
        for i, model_file in enumerate(model_files, 1):
            file_path = os.path.join(unified_path, model_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {i}. {model_file} ({file_size:.1f} MB)")
        
        return model_files
    else:
        print("❌ No unified models directory found")
        return []

def test_model_selection_logic():
    """Test logic chọn model"""
    print("\n🎯 MODEL SELECTION LOGIC:")
    print("=" * 60)
    
    # Read training results to see performance
    results_path = "training_results/comprehensive_training_20250624_230438.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print("📊 MODEL PERFORMANCE:")
        models_perf = results.get('results', {})
        
        # Sort by accuracy
        sorted_models = sorted(models_perf.items(), 
                             key=lambda x: x[1].get('val_accuracy', 0), 
                             reverse=True)
        
        for i, (model_name, perf) in enumerate(sorted_models, 1):
            accuracy = perf.get('val_accuracy', 0) * 100
            loss = perf.get('val_loss', 0)
            print(f"  {i}. {model_name.upper():8s}: {accuracy:6.2f}% accuracy, {loss:.3f} loss")
        
        best_model = sorted_models[0][0] if sorted_models else None
        print(f"\n🏆 BEST MODEL: {best_model.upper() if best_model else 'None'}")
        
        return sorted_models
    else:
        print("❌ No training results found")
        return []

def test_signal_generation():
    """Test việc generate signal"""
    print("\n🎯 SIGNAL GENERATION TEST:")
    print("=" * 60)
    
    system = UltimateXAUSystem()
    
    print("🔄 Generating 3 test signals...")
    for i in range(3):
        signal = system.generate_signal()
        
        print(f"\n📊 Signal {i+1}:")
        print(f"  Action: {signal.get('action', 'N/A')}")
        print(f"  Confidence: {signal.get('confidence', 0)}%")
        print(f"  AI Model: {signal.get('ai_model', 'N/A')}")
        print(f"  Features Used: {signal.get('features_used', 0)}")
        print(f"  Prediction Value: {signal.get('prediction_value', 'N/A')}")

def explain_current_architecture():
    """Giải thích kiến trúc hiện tại"""
    print("\n🏗️ CURRENT ARCHITECTURE EXPLANATION:")
    print("=" * 60)
    
    print("📋 HIỆN TẠI HỆ THỐNG:")
    print("  ❌ KHÔNG dùng 4 models cùng lúc")
    print("  ✅ CHỈ dùng 1 model để giao dịch")
    print("  🎯 Load model đầu tiên tìm thấy")
    
    print("\n🔍 LOGIC HIỆN TẠI:")
    print("  1. Tìm models trong trained_models/unified/")
    print("  2. Load model đầu tiên (.keras file)")
    print("  3. Sử dụng model đó cho tất cả predictions")
    print("  4. Nếu không có model → dùng fallback logic")
    
    print("\n⚠️ VẤN ĐỀ:")
    print("  • Không tận dụng được 4 models đã train")
    print("  • Không có ensemble voting")
    print("  • Không có model selection thông minh")

def propose_ensemble_solution():
    """Đề xuất giải pháp ensemble"""
    print("\n💡 ENSEMBLE SOLUTION PROPOSAL:")
    print("=" * 60)
    
    print("🎯 ĐỀ XUẤT NÂNG CẤP:")
    print("  ✅ Load tất cả 4 models")
    print("  ✅ Ensemble voting cho final decision")
    print("  ✅ Weight models theo performance")
    print("  ✅ Confidence từ model agreement")
    
    print("\n📊 ENSEMBLE VOTING LOGIC:")
    print("  1. Dense Model (73.35%): Weight = 0.4")
    print("  2. CNN Model (51.51%):   Weight = 0.2") 
    print("  3. LSTM Model (50.50%):  Weight = 0.2")
    print("  4. Hybrid Model (50.50%): Weight = 0.2")
    
    print("\n🔄 PREDICTION PROCESS:")
    print("  1. Mỗi model predict riêng")
    print("  2. Weighted average theo performance")
    print("  3. Final decision = ensemble result")
    print("  4. Confidence = model agreement level")
    
    print("\n📈 EXPECTED IMPROVEMENT:")
    print("  • Accuracy: 73.35% → 75-80%")
    print("  • Stability: Cao hơn")
    print("  • Confidence: Chính xác hơn")

if __name__ == "__main__":
    print("🤖 AI3.0 MODEL USAGE ANALYSIS")
    print("Analyzing how the system uses trained models")
    print("=" * 60)
    
    try:
        # 1. Test current usage
        system = test_current_model_usage()
        
        # 2. Check available models
        models = check_available_models()
        
        # 3. Test model selection
        performance = test_model_selection_logic()
        
        # 4. Test signal generation
        test_signal_generation()
        
        # 5. Explain current architecture
        explain_current_architecture()
        
        # 6. Propose ensemble solution
        propose_ensemble_solution()
        
        print(f"\n✅ ANALYSIS COMPLETED")
        print(f"📝 CONCLUSION: Hiện tại chỉ dùng 1 model, cần nâng cấp ensemble")
        
    except Exception as e:
        print(f"❌ Error: {e}") 