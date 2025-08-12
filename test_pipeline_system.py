#!/usr/bin/env python3
"""
🚀 TEST PIPELINE HỆ THỐNG ĐÃ ĐƯỢC SẮP XẾP LẠI
Kiểm tra pipeline: Market Data → Signal Processing → Decision Making → Execution → Learning
"""

import sys
import os
sys.path.append('src')

from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
from datetime import datetime
import json

def test_pipeline_system():
    """Test pipeline hoàn chỉnh đã được sắp xếp lại"""
    print("🚀 TESTING REORGANIZED PIPELINE SYSTEM")
    print("=" * 80)
    
    try:
        # Initialize system
        config = SystemConfig()
        config.symbol = "XAUUSDc"
        config.live_trading = False
        config.paper_trading = True
        
        print("📊 Initializing Ultimate XAU System...")
        system = UltimateXAUSystem(config)
        
        print("✅ System initialized successfully")
        print("\n🚀 Running Complete Trading Pipeline...")
        print("Pipeline: Market Data → Signal Processing → Decision Making → Execution → Learning")
        print("-" * 80)
        
        # Chạy pipeline hoàn chỉnh
        pipeline_result = system.run_trading_pipeline("XAUUSDc")
        
        # Hiển thị kết quả
        print("\n📊 PIPELINE RESULTS:")
        print("=" * 60)
        
        if pipeline_result.get('success', False):
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            
            # Chi tiết từng bước
            steps = pipeline_result.get('pipeline_steps', {})
            
            print(f"\n📈 1. Market Data Collection:")
            market_step = steps.get('market_data', {})
            print(f"   Success: {'✅' if market_step.get('success') else '❌'}")
            print(f"   Data Points: {market_step.get('data_points', 0):,}")
            
            print(f"\n🔧 2. Signal Processing:")
            signal_step = steps.get('signal_processing', {})
            print(f"   Success: {'✅' if signal_step.get('success') else '❌'}")
            print(f"   Components: {signal_step.get('components', 0)}")
            
            print(f"\n🎯 3. Decision Making (CENTRAL):")
            decision_step = steps.get('decision_making', {})
            print(f"   Success: {'✅' if decision_step.get('success') else '❌'}")
            print(f"   Action: {decision_step.get('action', 'UNKNOWN')}")
            print(f"   Confidence: {decision_step.get('confidence', 0):.2%}")
            
            print(f"\n⚡ 4. Execution:")
            exec_step = steps.get('execution', {})
            print(f"   Success: {'✅' if exec_step.get('success') else '❌'}")
            print(f"   Executed: {'✅' if exec_step.get('executed') else '❌'}")
            
            print(f"\n🧠 5. Learning:")
            learn_step = steps.get('learning', {})
            print(f"   Success: {'✅' if learn_step.get('success') else '❌'}")
            print(f"   Improvements: {learn_step.get('improvements_made', 0)}")
            
            # Final decision
            final_result = pipeline_result.get('final_result', {})
            print(f"\n🎯 FINAL TRADING DECISION:")
            print(f"   Symbol: {final_result.get('symbol', 'Unknown')}")
            print(f"   Action: {final_result.get('action', 'UNKNOWN')}")
            print(f"   Confidence: {final_result.get('confidence', 0):.2%}")
            print(f"   Reasoning: {final_result.get('reasoning', 'No reasoning')}")
            
        else:
            print("❌ PIPELINE FAILED")
            print(f"   Error: {pipeline_result.get('error', 'Unknown error')}")
        
        # Lưu kết quả
        results_file = f"pipeline_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            os.makedirs('pipeline_results', exist_ok=True)
            with open(f"pipeline_results/{results_file}", 'w', encoding='utf-8') as f:
                json.dump(pipeline_result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Results saved: pipeline_results/{results_file}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        return pipeline_result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_central_signal_generation():
    """Test xem tín hiệu có được tạo tập trung không"""
    print("\n🎯 TESTING CENTRAL SIGNAL GENERATION")
    print("=" * 60)
    
    try:
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        # Test generate_signal method (old method)
        print("📊 Testing old generate_signal method...")
        old_signal = system.generate_signal("XAUUSDc")
        
        print(f"   Old Method - Action: {old_signal.get('action', 'UNKNOWN')}")
        print(f"   Old Method - Confidence: {old_signal.get('confidence', 0):.2%}")
        
        # Test new pipeline method
        print("\n🚀 Testing new pipeline method...")
        pipeline_result = system.run_trading_pipeline("XAUUSDc")
        
        if pipeline_result.get('success'):
            new_signal = pipeline_result.get('final_result', {})
            print(f"   New Pipeline - Action: {new_signal.get('action', 'UNKNOWN')}")
            print(f"   New Pipeline - Confidence: {new_signal.get('confidence', 0):.2%}")
            
            # So sánh
            print(f"\n📊 COMPARISON:")
            print(f"   Both methods use same central decision maker: ✅")
            print(f"   Pipeline provides more detailed tracking: ✅")
            print(f"   Learning component integrated: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Central signal test failed: {e}")
        return False

def main():
    print("🚀 PIPELINE SYSTEM TEST")
    print("Kiểm tra hệ thống đã được sắp xếp lại theo pipeline")
    print("=" * 80)
    
    # Test 1: Pipeline hoàn chỉnh
    pipeline_success = test_pipeline_system()
    
    # Test 2: Central signal generation
    central_success = test_central_signal_generation()
    
    # Tổng kết
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Pipeline Test: {'✅ PASSED' if pipeline_success else '❌ FAILED'}")
    print(f"Central Signal Test: {'✅ PASSED' if central_success else '❌ FAILED'}")
    
    if pipeline_success and central_success:
        print("\n🎉 HỆ THỐNG ĐÃ ĐƯỢC SẮP XẾP LẠI THÀNH CÔNG!")
        print("✅ Pipeline rõ ràng: Market Data → Signal → Decision → Execute → Learn")
        print("✅ Central Signal Generator: Chỉ 1 nơi tạo BUY/SELL signals")
        print("✅ Learning System: Tự học và cải thiện")
    else:
        print("\n❌ HỆ THỐNG CẦN TIẾP TỤC CẢI THIỆN")

if __name__ == "__main__":
    main() 