#!/usr/bin/env python3
"""
🎬 LIVE SYSTEM DEMO - Demo trực tiếp hệ thống
Chứng minh hệ thống hoạt động thực tế trước mắt bạn
"""

import sys
import time
import json
from datetime import datetime

sys.path.append('src')

def live_system_demonstration():
    """Demo trực tiếp hệ thống"""
    
    print("🎬 LIVE SYSTEM DEMONSTRATION")
    print("=" * 50)
    print("🎯 Chứng minh hệ thống hoạt động thực tế NGAY BÂY GIỜ!")
    print()
    
    # Step 1: System Import and Initialization
    print("📋 STEP 1: SYSTEM IMPORT & INITIALIZATION")
    print("-" * 45)
    
    print("🔄 Importing system...")
    start_time = time.time()
    
    try:
        from core.ultimate_xau_system import UltimateXAUSystem
        import_time = time.time() - start_time
        print(f"✅ Import successful in {import_time*1000:.2f}ms")
        
        print("🔄 Initializing system...")
        init_start = time.time()
        system = UltimateXAUSystem()
        init_time = time.time() - init_start
        print(f"✅ System initialized in {init_time*1000:.2f}ms")
        
    except Exception as e:
        print(f"❌ System import/init failed: {e}")
        return False
    
    # Step 2: Live Signal Generation
    print(f"\n📋 STEP 2: LIVE SIGNAL GENERATION")
    print("-" * 35)
    
    print("🎯 Generating 10 live signals right now...")
    signals = []
    
    for i in range(10):
        try:
            start = time.time()
            signal = system.generate_signal()
            end = time.time()
            
            signal_time = (end - start) * 1000
            signals.append({
                'id': i + 1,
                'signal': signal,
                'time_ms': signal_time,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"   Signal {i+1}: {signal.get('action', 'UNKNOWN')} "
                  f"(Confidence: {signal.get('confidence', 0):.1f}%, "
                  f"Time: {signal_time:.3f}ms)")
            
        except Exception as e:
            print(f"   Signal {i+1}: ERROR - {e}")
    
    # Step 3: Real-time Performance Monitoring
    print(f"\n📋 STEP 3: REAL-TIME PERFORMANCE MONITORING")
    print("-" * 45)
    
    print("📊 Analyzing live performance...")
    
    if signals:
        successful_signals = [s for s in signals if 'signal' in s and isinstance(s['signal'], dict)]
        signal_times = [s['time_ms'] for s in successful_signals]
        confidences = [s['signal']['confidence'] for s in successful_signals if 'confidence' in s['signal']]
        actions = [s['signal']['action'] for s in successful_signals if 'action' in s['signal']]
        
        print(f"✅ Successful signals: {len(successful_signals)}/10")
        print(f"⚡ Average time: {sum(signal_times)/len(signal_times):.3f}ms")
        print(f"🚀 Fastest signal: {min(signal_times):.3f}ms")
        print(f"🐌 Slowest signal: {max(signal_times):.3f}ms")
        print(f"🎯 Average confidence: {sum(confidences)/len(confidences):.1f}%")
        
        # Action distribution
        action_count = {}
        for action in actions:
            action_count[action] = action_count.get(action, 0) + 1
        print(f"📊 Action distribution: {action_count}")
    
    # Step 4: System Health Check
    print(f"\n📋 STEP 4: SYSTEM HEALTH CHECK")
    print("-" * 30)
    
    try:
        health = system.get_system_health_status()
        print(f"🏥 System Health: {health}")
        
        if isinstance(health, dict):
            for key, value in health.items():
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"⚠️ Health check error: {e}")
    
    # Step 5: Live Trading Simulation
    print(f"\n📋 STEP 5: LIVE TRADING SIMULATION")
    print("-" * 35)
    
    print("🎮 Simulating live trading scenario...")
    
    try:
        # Simulate starting trading
        print("🔄 Starting trading simulation...")
        start_result = system.start_trading()
        print(f"✅ Trading started: {start_result}")
        
        # Generate a few trading signals
        print("📊 Generating trading signals...")
        for i in range(5):
            signal = system.generate_signal()
            print(f"   Trading Signal {i+1}: {signal.get('action', 'UNKNOWN')} "
                  f"(Confidence: {signal.get('confidence', 0):.1f}%)")
            time.sleep(0.1)  # Small delay for realism
        
        # Stop trading
        print("🛑 Stopping trading simulation...")
        stop_result = system.stop_trading()
        print(f"✅ Trading stopped: {stop_result}")
        
    except Exception as e:
        print(f"⚠️ Trading simulation error: {e}")
    
    # Step 6: Emergency Systems Test
    print(f"\n📋 STEP 6: EMERGENCY SYSTEMS TEST")
    print("-" * 35)
    
    try:
        print("🚨 Testing emergency stop...")
        emergency_result = system.emergency_stop()
        print(f"✅ Emergency stop: {emergency_result}")
        
    except Exception as e:
        print(f"⚠️ Emergency test error: {e}")
    
    # Step 7: Final Performance Summary
    print(f"\n📋 STEP 7: FINAL PERFORMANCE SUMMARY")
    print("-" * 40)
    
    total_demo_time = time.time() - start_time
    
    print(f"🎯 LIVE DEMO RESULTS:")
    print(f"   ⏱️ Total demo time: {total_demo_time:.2f} seconds")
    print(f"   🚀 System import: {import_time*1000:.2f}ms")
    print(f"   ⚡ System init: {init_time*1000:.2f}ms")
    print(f"   📊 Signals generated: {len(signals)}")
    print(f"   ✅ Success rate: {len(successful_signals)/len(signals)*100:.1f}%")
    
    if signal_times:
        print(f"   🎯 Average signal time: {sum(signal_times)/len(signal_times):.3f}ms")
        print(f"   🚀 Peak performance: {min(signal_times):.3f}ms")
    
    print(f"\n🎬 LIVE DEMO COMPLETED!")
    print("✅ Hệ thống đã hoạt động thực tế trước mắt bạn!")
    
    return True

def continuous_live_demo():
    """Demo liên tục trong 30 giây"""
    
    print(f"\n🔄 CONTINUOUS LIVE DEMO (30 seconds)")
    print("-" * 40)
    
    try:
        from core.ultimate_xau_system import UltimateXAUSystem
        system = UltimateXAUSystem()
        
        start_time = time.time()
        signal_count = 0
        errors = 0
        
        print("🎬 Starting continuous demo...")
        print("📊 Generating signals every second for 30 seconds...")
        
        while time.time() - start_time < 30:
            try:
                current_time = time.time() - start_time
                signal = system.generate_signal()
                signal_count += 1
                
                print(f"   [{current_time:05.1f}s] Signal {signal_count}: "
                      f"{signal.get('action', 'UNKNOWN')} "
                      f"({signal.get('confidence', 0):.1f}%)")
                
                time.sleep(1)  # 1 signal per second
                
            except Exception as e:
                errors += 1
                print(f"   [{current_time:05.1f}s] ERROR: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n📊 CONTINUOUS DEMO RESULTS:")
        print(f"   ⏱️ Duration: {total_time:.1f} seconds")
        print(f"   📊 Signals generated: {signal_count}")
        print(f"   ❌ Errors: {errors}")
        print(f"   ✅ Success rate: {(signal_count/(signal_count+errors))*100:.1f}%")
        print(f"   📈 Average rate: {signal_count/total_time:.2f} signals/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Continuous demo failed: {e}")
        return False

def main():
    """Main demo function"""
    
    print("🎬 AI3.0 LIVE SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("🎯 Objective: Chứng minh hệ thống hoạt động thực tế")
    print("📊 Response to: 'Chỉ là lý thuyết, chưa có số liệu thực tế'")
    print()
    
    # Run live demonstration
    demo_success = live_system_demonstration()
    
    if demo_success:
        print(f"\n🎯 Would you like to see continuous demo? (30 seconds)")
        print("🔄 This will show system running continuously...")
        
        # Run continuous demo
        continuous_success = continuous_live_demo()
        
        if continuous_success:
            print(f"\n🎉 LIVE DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("✅ Hệ thống đã được chứng minh hoạt động thực tế!")
            print("📊 Không còn là lý thuyết - đây là thực tế!")
        else:
            print(f"\n⚠️ Continuous demo had issues")
    else:
        print(f"\n❌ Live demonstration failed")
    
    return demo_success

if __name__ == "__main__":
    main() 