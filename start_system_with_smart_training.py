#!/usr/bin/env python3
"""
🚀 KHỞI ĐỘNG HỆ THỐNG CHÍNH VỚI SMART TRAINING
ULTIMATE XAU SUPER SYSTEM V4.0 + Smart Training Integration
"""

import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

def start_main_system_with_smart_training():
    """Khởi động hệ thống chính với Smart Training"""
    
    print("🚀 KHỞI ĐỘNG HỆ THỐNG CHÍNH VỚI SMART TRAINING")
    print("ULTIMATE XAU SUPER SYSTEM V4.0 + Smart Training Integration")
    print("=" * 80)
    
    try:
        # Import systems
        from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        from core.smart_training_integration import SmartTrainingSystem
        
        print("✅ Imported systems successfully")
        
        # Create system configuration
        config = SystemConfig()
        config.live_trading = False  # Safe mode
        config.paper_trading = True
        
        # Initialize main system
        print("\n🔧 INITIALIZING MAIN SYSTEM...")
        main_system = UltimateXAUSystem(config)
        
        # Initialize Smart Training System
        print("\n🧠 INITIALIZING SMART TRAINING SYSTEM...")
        smart_training = SmartTrainingSystem(config)
        
        if smart_training.initialize():
            print("✅ Smart Training System initialized successfully")
            
            # Register Smart Training with main system
            main_system.system_manager.register_system(smart_training)
            print("✅ Smart Training System registered with main system")
            
            # Get system status
            print("\n📊 SYSTEM STATUS:")
            main_status = main_system.get_system_status()
            smart_status = smart_training.get_smart_training_status()
            
            print(f"   Main System: {main_status['total_systems']} systems active")
            print(f"   Smart Training: {smart_status['current_phase_name']}")
            print(f"   Baseline Accuracy: {smart_status['baseline_accuracy']:.2%}")
            print(f"   Target Accuracy: {smart_status['target_accuracy']:.2%}")
            
            # Generate initial signal to test integration
            print("\n🎯 TESTING SYSTEM INTEGRATION:")
            signal = main_system.generate_signal()
            print(f"   Signal generated: {signal['action']} (Confidence: {signal['confidence']:.2%})")
            
            print("\n🎉 SYSTEM READY WITH SMART TRAINING!")
            print("\n📋 AVAILABLE COMMANDS:")
            print("   1. Start Smart Training: smart_training.execute_smart_training_pipeline()")
            print("   2. Check Status: smart_training.get_smart_training_status()")
            print("   3. Generate Signal: main_system.generate_signal()")
            
            return main_system, smart_training
            
        else:
            print("❌ Smart Training System initialization failed")
            return main_system, None
            
    except Exception as e:
        print(f"❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def demo_smart_training():
    """Demo Smart Training functionality"""
    
    print("\n🧠 DEMO SMART TRAINING FUNCTIONALITY")
    print("=" * 60)
    
    try:
        main_system, smart_training = start_main_system_with_smart_training()
        
        if smart_training:
            print("\n🚀 Starting Smart Training Pipeline Demo...")
            
            # Execute Smart Training
            training_request = {'action': 'start_smart_training'}
            results = smart_training.process(training_request)
            
            if results.get('pipeline_status') == 'completed':
                print("\n🎉 SMART TRAINING COMPLETED!")
                print("\n📊 RESULTS SUMMARY:")
                
                for phase in results['phases_completed']:
                    print(f"   ✅ {phase}")
                
                print("\n📈 IMPROVEMENTS:")
                for improvement, value in results['improvements'].items():
                    print(f"   • {improvement.replace('_', ' ').title()}: {value}")
                
                print("\n🏆 FINAL METRICS:")
                final_metrics = results['final_metrics']
                for metric, value in final_metrics.items():
                    print(f"   • {metric.replace('_', ' ').title()}: {value}")
                
                print("\n💾 Results saved to smart_training_results/")
                
            else:
                print(f"❌ Smart Training failed: {results.get('error', 'Unknown error')}")
                
        else:
            print("❌ Smart Training not available")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Start system with Smart Training
    main_system, smart_training = start_main_system_with_smart_training()
    
    if main_system and smart_training:
        # Run demo
        demo_smart_training()
    else:
        print("❌ System startup failed")
