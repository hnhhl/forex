#!/usr/bin/env python3
"""
Demo: Training Component hoạt động TRONG Main System
Chứng minh Training KHÔNG phải bản sao mà là COMPONENT
"""

import sys
sys.path.append('src')

from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
from datetime import datetime

def demo_unified_training():
    """Demo training component trong main system"""
    
    print("🎯 DEMO: TRAINING COMPONENT TRONG MAIN SYSTEM")
    print("=" * 60)
    
    # 1. TẠO DUY NHẤT 1 HỆ THỐNG
    print("\n1️⃣ Khởi tạo MAIN SYSTEM (duy nhất)...")
    config = SystemConfig()
    config.enable_integrated_training = True
    
    # ✅ CHỈ CÓ 1 HỆ THỐNG DUY NHẤT
    main_system = UltimateXAUSystem(config)
    
    print(f"   ✅ Main System initialized")
    print(f"   ✅ Training System: {'INTEGRATED' if main_system.training_system else 'DISABLED'}")
    print(f"   ✅ Feature Engine: {main_system.feature_engine.__class__.__name__}")
    
    # 2. KIỂM TRA TRAINING LÀ COMPONENT
    print("\n2️⃣ Kiểm tra Training là COMPONENT của Main System...")
    
    # Training system LÀ PHẦN CỦA main system
    training_component = main_system.training_system
    
    print(f"   ✅ Training component type: {type(training_component).__name__}")
    print(f"   ✅ Same feature engine: {training_component.feature_engine is main_system.feature_engine}")
    print(f"   ✅ Same model architecture: {training_component.model_architecture is main_system.model_architecture}")
    
    # 3. SỬ DỤNG TRAINING COMPONENT
    print("\n3️⃣ Sử dụng Training Component...")
    
    # Simulate data collection (happens automatically in production)
    print("   📊 Collecting training data...")
    for i in range(60):
        sample_data = {
            'open': 2000.0 + i * 0.1,
            'high': 2005.0 + i * 0.1,
            'low': 1995.0 + i * 0.1,
            'close': 2002.0 + i * 0.1,
            'volume': 1000.0
        }
        training_component.collect_training_data(sample_data)
    
    # Check training status
    status = main_system.get_training_status()
    print(f"   ✅ Data points collected: {status['training_data_points']}")
    print(f"   ✅ Should retrain: {status['should_retrain']}")
    
    # 4. CHỨNG MINH CÙNG 1 HỆ THỐNG
    print("\n4️⃣ Chứng minh CÙNG 1 HỆ THỐNG...")
    
    # Generate signal sử dụng CÙNG feature engine
    signal = main_system.generate_signal()
    print(f"   ✅ Signal generated: {signal['action']} ({signal['confidence']:.1f}%)")
    print(f"   ✅ Features used: {signal.get('features_used', 'N/A')}")
    print(f"   ✅ Feature engine: {signal.get('feature_engine', 'N/A')}")
    
    # Training sử dụng CÙNG feature engine
    features = main_system.get_market_features()
    print(f"   ✅ Market features shape: {features.shape}")
    print(f"   ✅ Same 19 features: {len(features) == 19}")
    
    # 5. SYSTEM STATUS
    print("\n5️⃣ System Status (Unified)...")
    system_status = main_system.get_system_status()
    
    print(f"   ✅ Unified Architecture: {system_status['unified_architecture']}")
    print(f"   ✅ Feature Engine: {system_status['feature_engine']}")
    print(f"   ✅ Model Architecture: {system_status['model_architecture']}")
    
    if 'training_system' in system_status:
        training_status = system_status['training_system']
        print(f"   ✅ Training Data Points: {training_status['training_data_points']}")
        print(f"   ✅ Training Models: {training_status['trained_models']}")
    
    # 6. KẾT LUẬN
    print("\n🎉 KẾT LUẬN:")
    print("   ✅ CHỈ CÓ 1 HỆ THỐNG DUY NHẤT: UltimateXAUSystem")
    print("   ✅ Training là COMPONENT của main system")
    print("   ✅ KHÔNG phải bản sao hay hệ thống riêng")
    print("   ✅ Cùng logic, cùng features, cùng architecture")
    
    return main_system

def demo_training_vs_production():
    """Demo sự khác biệt giữa training và production trong CÙNG hệ thống"""
    
    print("\n" + "="*60)
    print("🔍 DEMO: TRAINING vs PRODUCTION trong CÙNG HỆ THỐNG")
    print("="*60)
    
    # Khởi tạo hệ thống
    system = UltimateXAUSystem()
    
    print("\n📊 PRODUCTION MODE (Real-time prediction):")
    # Production: Generate signal
    signal = system.generate_signal()
    print(f"   🎯 Action: {signal['action']}")
    print(f"   📈 Confidence: {signal['confidence']:.1f}%")
    print(f"   🔧 Source: {signal.get('source', 'N/A')}")
    
    print("\n🧠 TRAINING MODE (Model improvement):")
    # Training: Same system, different function
    if system.training_system:
        training_status = system.get_training_status()
        print(f"   📊 Data Points: {training_status['training_data_points']}")
        print(f"   🔄 Should Retrain: {training_status['should_retrain']}")
        print(f"   🤖 Trained Models: {training_status['trained_models']}")
    
    print("\n✅ CÙNG HỆ THỐNG - KHÁC CHỨC NĂNG:")
    print("   • Production: Tạo trading signals")
    print("   • Training: Cải thiện models")
    print("   • Shared: Cùng features, cùng logic")

if __name__ == "__main__":
    print("🚀 Starting Unified System Demo...")
    
    # Demo 1: Training component trong main system
    main_system = demo_unified_training()
    
    # Demo 2: Training vs Production trong cùng hệ thống
    demo_training_vs_production()
    
    print("\n🎉 Demo completed! Training is COMPONENT, not COPY!") 