#!/usr/bin/env python3
"""
AI3.0 QUICK START DEMO
Demo nhanh hệ thống AI3.0 Unified Master System
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src path
sys.path.insert(0, 'src')

async def quick_demo():
    """Demo nhanh hệ thống"""
    print("🚀 AI3.0 UNIFIED MASTER SYSTEM - QUICK DEMO")
    print("=" * 60)
    
    try:
        # Import system
        from UNIFIED_AI3_MASTER_SYSTEM import create_development_system
        
        print("✅ Importing system components...")
        
        # Create system
        print("🔧 Creating development system...")
        system = create_development_system()
        
        # Show system status
        print("📊 System status:")
        status = system.get_system_status()
        print(f"   System: {status['system_info']['name']} v{status['system_info']['version']}")
        print(f"   Mode: {status['system_info']['mode']}")
        print(f"   Components: {sum(status['components'].values())}/{len(status['components'])} active")
        
        # List active components
        print("\n🔧 Active components:")
        for name, active in status['components'].items():
            status_icon = "✅" if active else "❌"
            print(f"   {status_icon} {name}")
        
        # Start system
        print("\n🚀 Starting system...")
        if await system.start_system():
            print("✅ System started successfully!")
            
            # Run demo for 30 seconds
            print("⏳ Running demo for 30 seconds...")
            start_time = datetime.now()
            
            for i in range(6):  # 6 iterations x 5 seconds = 30 seconds
                await asyncio.sleep(5)
                
                # Show status
                current_status = system.get_system_status()
                uptime = current_status['system_info']['uptime_seconds']
                signals = current_status['recent_signals']
                
                print(f"⏰ [{i+1}/6] Uptime: {uptime:.0f}s | Signals: {signals}")
            
            # Show final results
            print("\n📊 DEMO RESULTS:")
            final_status = system.get_system_status()
            print(f"   Total runtime: {final_status['system_info']['uptime_seconds']:.1f} seconds")
            print(f"   Signals generated: {final_status['recent_signals']}")
            
            performance = final_status.get('performance', {})
            if performance:
                print(f"   Performance metrics:")
                for key, value in performance.items():
                    print(f"     {key}: {value}")
            
            # Stop system
            print("\n🛑 Stopping system...")
            await system.stop_system()
            print("✅ System stopped successfully!")
            
        else:
            print("❌ Failed to start system")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("📖 For more options, run: python SYSTEM_LAUNCHER.py --help")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install numpy pandas tensorflow joblib")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("🔍 Run system validator: python SYSTEM_VALIDATOR.py")

def main():
    """Main function"""
    print("Starting AI3.0 Quick Demo...")
    asyncio.run(quick_demo())

if __name__ == "__main__":
    main() 