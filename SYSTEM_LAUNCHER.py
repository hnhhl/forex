#!/usr/bin/env python3
"""
AI3.0 SYSTEM LAUNCHER
Launcher cho hệ thống AI3.0 với các chế độ khác nhau
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any

from UNIFIED_AI3_MASTER_SYSTEM import (
    UnifiedAI3MasterSystem,
    UnifiedSystemConfig,
    SystemMode,
    create_development_system,
    create_simulation_system,
    create_live_trading_system
)

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ai3_system.log'),
            logging.StreamHandler()
        ]
    )

def create_custom_system(config_file: str) -> UnifiedAI3MasterSystem:
    """Create system from config file"""
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        config = UnifiedSystemConfig(**config_data)
        return UnifiedAI3MasterSystem(config)
    except Exception as e:
        logging.error(f"Failed to create custom system: {e}")
        return create_development_system()

async def run_system_demo(system: UnifiedAI3MasterSystem, duration: int = 60):
    """Run system demo for specified duration"""
    print(f"\n🎯 DEMO MODE - Running for {duration} seconds")
    print("=" * 50)
    
    # Start system
    if await system.start_system():
        print("✅ System started - Generating signals...")
        
        # Monitor for specified duration
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < duration:
            await asyncio.sleep(10)
            
            # Show periodic status
            status = system.get_system_status()
            uptime = status['system_info']['uptime_seconds']
            recent_signals = status['recent_signals']
            
            print(f"⏰ Uptime: {uptime:.0f}s | Recent signals: {recent_signals}")
        
        # Stop system
        await system.stop_system()
        
        # Show final results
        final_status = system.get_system_status()
        print(f"\n📊 DEMO COMPLETED:")
        print(f"   Duration: {duration}s")
        print(f"   Total signals: {final_status['recent_signals']}")
        
        performance = final_status.get('performance', {})
        if performance:
            print(f"   Performance metrics: {json.dumps(performance, indent=2)}")
    else:
        print("❌ Failed to start system")

async def run_interactive_mode(system: UnifiedAI3MasterSystem):
    """Run system in interactive mode"""
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Commands:")
    print("  start   - Start the system")
    print("  stop    - Stop the system") 
    print("  status  - Show system status")
    print("  signals - Show recent signals")
    print("  quit    - Exit interactive mode")
    print()
    
    while True:
        try:
            command = input("AI3.0> ").strip().lower()
            
            if command == "start":
                await system.start_system()
            
            elif command == "stop":
                await system.stop_system()
            
            elif command == "status":
                status = system.get_system_status()
                print(json.dumps(status, indent=2, default=str))
            
            elif command == "signals":
                signals = system.signals_history[-10:] if system.signals_history else []
                for i, signal in enumerate(signals, 1):
                    print(f"  {i}. {signal.timestamp.strftime('%H:%M:%S')} | {signal.action} | {signal.confidence:.3f} | {signal.source}")
            
            elif command == "quit":
                if system.is_active:
                    await system.stop_system()
                break
            
            elif command == "help":
                print("Available commands: start, stop, status, signals, quit")
            
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            if system.is_active:
                await system.stop_system()
            break
        except Exception as e:
            print(f"Error: {e}")

async def run_continuous_mode(system: UnifiedAI3MasterSystem):
    """Run system continuously"""
    print("\n♾️ CONTINUOUS MODE")
    print("=" * 50)
    print("System will run continuously. Press Ctrl+C to stop.")
    
    try:
        await system.start_system()
        
        # Run indefinitely until interrupted
        while system.is_active:
            await asyncio.sleep(60)  # Check every minute
            
            # Periodic status update
            status = system.get_system_status()
            uptime = status['system_info']['uptime_seconds']
            recent_signals = status['recent_signals']
            
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Uptime: {uptime:.0f}s | Signals: {recent_signals}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
        await system.stop_system()
        print("✅ System stopped")

async def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="AI3.0 System Launcher")
    parser.add_argument("--mode", choices=["development", "simulation", "live"], 
                       default="development", help="System mode")
    parser.add_argument("--config", type=str, help="Custom config file")
    parser.add_argument("--demo", type=int, help="Run demo for specified seconds")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create system
    if args.config:
        system = create_custom_system(args.config)
    elif args.mode == "development":
        system = create_development_system()
    elif args.mode == "simulation":
        system = create_simulation_system()
    elif args.mode == "live":
        system = create_live_trading_system()
    else:
        system = create_development_system()
    
    # Display system info
    print("🚀 AI3.0 UNIFIED MASTER SYSTEM LAUNCHER")
    print("=" * 60)
    
    status = system.get_system_status()
    print(f"System: {status['system_info']['name']} v{status['system_info']['version']}")
    print(f"Mode: {status['system_info']['mode']}")
    print(f"Components: {sum(status['components'].values())}/{len(status['components'])} active")
    
    # Run based on mode
    if args.demo:
        await run_system_demo(system, args.demo)
    elif args.interactive:
        await run_interactive_mode(system)
    elif args.continuous:
        await run_continuous_mode(system)
    else:
        # Default: run quick demo
        await run_system_demo(system, 30)
    
    print("\n✅ Launcher completed")

if __name__ == "__main__":
    asyncio.run(main()) 