#!/usr/bin/env python3
"""
🔍 AI3.0 SYSTEM COMPREHENSIVE TEST
Kiểm tra toàn diện hệ thống AI3.0 sau khi fix tất cả vấn đề
"""

import os
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from datetime import datetime
import json

def test_system_imports():
    """Test import các components chính"""
    print("📦 TESTING SYSTEM IMPORTS...")
    print("=" * 50)
    
    try:
        # Core system
        from core.ultimate_xau_system import UltimateXAUSystem
        from core.base_system import SystemConfig
        print("   ✅ Core System: UltimateXAUSystem")
        
        # GPU Neural System
        from core.gpu_neural_system import GPUNeuralNetworkSystem
        print("   ✅ GPU Neural System: GPUNeuralNetworkSystem")
        
        # AI2.0 Technologies
        from core.ultimate_xau_system import AI2AdvancedTechnologiesSystem
        print("   ✅ AI2.0 Technologies: AI2AdvancedTechnologiesSystem")
        
        # AI Phases
        from core.ai.ai_phases.main import AISystem
        print("   ✅ AI Phases: AISystem")
        
        # Master Integration
        from core.integration.master_system import MasterIntegrationSystem
        print("   ✅ Master Integration: MasterIntegrationSystem")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import Error: {e}")
        return False

def test_system_initialization():
    """Test khởi tạo hệ thống"""
    print("\n🚀 TESTING SYSTEM INITIALIZATION...")
    print("=" * 50)
    
    try:
        from core.ultimate_xau_system import UltimateXAUSystem
        from core.base_system import SystemConfig
        
        # Create config
        config = SystemConfig()
        print("   ✅ SystemConfig created")
        
        # Create system
        system = UltimateXAUSystem(config)
        print("   ✅ UltimateXAUSystem created")
        
        # Initialize system
        print("   🔄 Initializing system...")
        success = system.initialize()
        
        if success:
            print("   ✅ System initialization: SUCCESS")
            
            # Get system info
            active_systems = len([s for s in system.system_manager.systems.values() if s.is_active])
            total_systems = len(system.system_manager.systems)
            
            print(f"   📊 Active Systems: {active_systems}/{total_systems}")
            
            # List active systems
            print("   📋 Active Systems:")
            for name, sys in system.system_manager.systems.items():
                status = "✅ ACTIVE" if sys.is_active else "❌ INACTIVE"
                print(f"      - {name}: {status}")
            
            return system, True
        else:
            print("   ❌ System initialization: FAILED")
            return None, False
            
    except Exception as e:
        print(f"   ❌ Initialization Error: {e}")
        return None, False

def test_data_preparation():
    """Test chuẩn bị dữ liệu"""
    print("\n📊 TESTING DATA PREPARATION...")
    print("=" * 50)
    
    try:
        # Load sample data
        data_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
        
        if not os.path.exists(data_file):
            print(f"   ❌ Data file not found: {data_file}")
            return None
        
        # Load data
        df = pd.read_csv(data_file)
        print(f"   ✅ Data loaded: {len(df):,} records")
        
        # Use last 100 records for test
        df_test = df.tail(100).copy()
        print(f"   📊 Test data: {len(df_test)} records")
        
        # Prepare features
        features = df_test[['Open', 'High', 'Low', 'Close', 'Volume']].values
        print(f"   ✅ Features prepared: {features.shape}")
        
        # Create sequences
        sequence_length = 60
        if len(features) >= sequence_length + 1:
            X = []
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
            
            X = np.array(X, dtype=np.float32)
            print(f"   ✅ Sequences created: {X.shape}")
            
            return X
        else:
            print(f"   ⚠️  Insufficient data for sequences (need {sequence_length+1}, have {len(features)})")
            return None
            
    except Exception as e:
        print(f"   ❌ Data preparation error: {e}")
        return None

def test_signal_generation(system, data):
    """Test tạo trading signals"""
    print("\n🔮 TESTING SIGNAL GENERATION...")
    print("=" * 50)
    
    try:
        if system is None or data is None:
            print("   ❌ System or data not available")
            return None
        
        # Test signal generation
        print("   🔄 Generating trading signal...")
        
        # Get latest data point
        latest_data = data[-1:] if len(data) > 0 else None
        
        if latest_data is not None:
            # Call system signal generation
            signal_result = system.generate_signal(latest_data)
            
            print(f"   ✅ Signal generated successfully")
            print(f"   📊 Signal Result:")
            
            if isinstance(signal_result, dict):
                for key, value in signal_result.items():
                    if isinstance(value, (int, float)):
                        print(f"      - {key}: {value:.4f}")
                    else:
                        print(f"      - {key}: {value}")
            else:
                print(f"      - Signal: {signal_result}")
            
            return signal_result
        else:
            print("   ❌ No data available for signal generation")
            return None
            
    except Exception as e:
        print(f"   ❌ Signal generation error: {e}")
        return None

def test_gpu_integration():
    """Test GPU integration"""
    print("\n🚀 TESTING GPU INTEGRATION...")
    print("=" * 50)
    
    try:
        from core.gpu_neural_system import GPUNeuralNetworkSystem
        
        # Mock config
        config = type('Config', (), {})()
        
        # Create GPU system
        gpu_system = GPUNeuralNetworkSystem(config)
        print("   ✅ GPU Neural System created")
        
        # Check GPU availability
        if gpu_system.is_gpu_available:
            print("   ✅ GPU Available: YES")
            print(f"   📊 GPU Models: {len(gpu_system.models)} created")
        else:
            print("   ⚠️  GPU Available: NO")
        
        # Initialize
        if gpu_system.initialize():
            print("   ✅ GPU System initialized")
            
            # Test prediction with dummy data
            dummy_data = np.random.random((1, 60, 5)).astype(np.float32)
            
            ensemble_pred, confidence = gpu_system.get_ensemble_prediction(dummy_data)
            
            if ensemble_pred is not None:
                print(f"   ✅ GPU Prediction: {ensemble_pred:.4f}")
                print(f"   📊 Confidence: {confidence:.4f}")
            else:
                print("   ⚠️  GPU Prediction: Not available")
            
            # Cleanup
            gpu_system.cleanup()
            
            return True
        else:
            print("   ❌ GPU System initialization failed")
            return False
            
    except Exception as e:
        print(f"   ❌ GPU integration error: {e}")
        return False

def test_ai_phases():
    """Test AI Phases system"""
    print("\n🧠 TESTING AI PHASES SYSTEM...")
    print("=" * 50)
    
    try:
        from core.ai.ai_phases.main import AISystem
        
        # Create AI Phases system
        ai_phases = AISystem()
        print("   ✅ AI Phases System created")
        
        # Get system status
        status = ai_phases.get_system_status()
        
        print("   📊 AI Phases Status:")
        if 'system_state' in status:
            system_state = status['system_state']
            print(f"      - Initialized: {system_state.get('initialized', 'Unknown')}")
            print(f"      - Performance Boost: +{system_state.get('total_performance_boost', 0)}%")
            print(f"      - Active Phases: {len(system_state.get('active_phases', []))}")
        
        # Test evolution
        print("   🔄 Testing system evolution...")
        evolution_result = ai_phases.evolve_system(1)
        
        if 'error' not in evolution_result:
            print("   ✅ Evolution test: SUCCESS")
        else:
            print(f"   ⚠️  Evolution test: {evolution_result['error']}")
        
        # Cleanup
        ai_phases.shutdown()
        
        return True
        
    except Exception as e:
        print(f"   ❌ AI Phases error: {e}")
        return False

def generate_comprehensive_report():
    """Tạo báo cáo toàn diện"""
    print("\n📋 GENERATING COMPREHENSIVE REPORT...")
    print("=" * 70)
    
    # Run all tests
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': {}
    }
    
    # Test 1: Imports
    results['test_results']['imports'] = test_system_imports()
    
    # Test 2: System Initialization
    system, init_success = test_system_initialization()
    results['test_results']['initialization'] = init_success
    
    # Test 3: Data Preparation
    data = test_data_preparation()
    results['test_results']['data_preparation'] = data is not None
    
    # Test 4: Signal Generation
    signal = test_signal_generation(system, data)
    results['test_results']['signal_generation'] = signal is not None
    
    # Test 5: GPU Integration
    results['test_results']['gpu_integration'] = test_gpu_integration()
    
    # Test 6: AI Phases
    results['test_results']['ai_phases'] = test_ai_phases()
    
    # Calculate overall score
    passed_tests = sum(1 for result in results['test_results'].values() if result)
    total_tests = len(results['test_results'])
    success_rate = (passed_tests / total_tests) * 100
    
    results['summary'] = {
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'success_rate': success_rate,
        'overall_status': 'EXCELLENT' if success_rate >= 90 else 'GOOD' if success_rate >= 70 else 'NEEDS_IMPROVEMENT'
    }
    
    # Save report
    report_file = f"ai3_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n🎯 AI3.0 SYSTEM TEST SUMMARY:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Overall Status: {results['summary']['overall_status']}")
    print(f"   Report Saved: {report_file}")
    
    # Detailed results
    print(f"\n📊 DETAILED TEST RESULTS:")
    for test_name, result in results['test_results'].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.upper()}: {status}")
    
    return results

def main():
    """Main test execution"""
    print("🔍 AI3.0 COMPREHENSIVE SYSTEM TEST")
    print("=" * 70)
    print(f"🕒 Test Time: {datetime.now()}")
    print()
    
    # Run comprehensive tests
    results = generate_comprehensive_report()
    
    # Final assessment
    success_rate = results['summary']['success_rate']
    
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT! AI3.0 System is performing at {success_rate:.1f}%")
        print("🚀 System is ready for production trading!")
    elif success_rate >= 70:
        print(f"\n👍 GOOD! AI3.0 System is performing at {success_rate:.1f}%")
        print("⚡ System is functional with minor issues")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT! AI3.0 System at {success_rate:.1f}%")
        print("🔧 System needs attention before production use")
    
    print(f"\n🕒 Test Completed: {datetime.now()}")

if __name__ == "__main__":
    main() 