#!/usr/bin/env python3
"""
🚀 ULTIMATE XAU SUPER SYSTEM - SIMPLE STARTER
Khởi động hệ thống ULTIMATE XAU SUPER SYSTEM - Hệ thống giao dịch XAU siêu việt
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Import UltimateXAUSystem từ ULTIMATE_XAU_SUPER_SYSTEM
from src.core.ultimate_xau_system import UltimateXAUSystem
from enhanced_auto_trading import EnhancedAutoTrading
import MetaTrader5 as mt5

class SimpleSystemStarter:
    def __init__(self):
        print("🚀 KHỞI ĐỘNG ULTIMATE XAU SUPER SYSTEM...")
        
        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return
        
        print("✅ MT5 connected successfully")
        
        # Initialize ULTIMATE XAU SUPER SYSTEM
        self.ultimate_system = UltimateXAUSystem()
        print("✅ ULTIMATE XAU SUPER SYSTEM loaded")
        
        # Initialize Enhanced Auto Trading
        self.auto_trading = EnhancedAutoTrading(self.ultimate_system, mt5)
        print("✅ Enhanced Auto Trading initialized")
        
    def start(self):
        """Khởi động hệ thống"""
        try:
            print("\n🎯 STARTING AUTO TRADING...")
            
            # Start auto trading
            self.auto_trading.start_auto_trading()
            self.auto_trading.enable_auto_trading()
            
            print("✅ System started successfully!")
            print("📊 Monitoring for signals from ULTIMATE XAU SUPER SYSTEM...")
            
            # Keep running
            while True:
                status = self.auto_trading.get_status()
                positions = self.auto_trading.get_positions_summary()
                
                print(f"\n📈 Status: Auto={status['auto_enabled']}, Mode={status['current_mode']}")
                print(f"💰 Positions: Active={positions['active_positions']}, Total Profit=${positions['total_profit']:.2f}")
                
                time.sleep(30)  # Update every 30 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping system...")
            self.auto_trading.stop_auto_trading()
            mt5.shutdown()
            print("✅ System stopped")

if __name__ == "__main__":
    starter = SimpleSystemStarter()
    starter.start() 