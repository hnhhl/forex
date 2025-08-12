#!/usr/bin/env python3
"""
Test Script cho Ultimate XAU System V4.0
Kiểm tra từng component và chức năng
"""

import sys
import os
import traceback
from datetime import datetime

def test_imports():
    """Test các import cần thiết"""
    print("🔍 TESTING IMPORTS...")
    
    try:
        import pandas as pd
        print("✅ pandas: OK")
    except Exception as e:
        print(f"❌ pandas: {e}")
    
    try:
        import numpy as np
        print("✅ numpy: OK")
    except Exception as e:
        print(f"❌ numpy: {e}")
    
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5: OK")
    except Exception as e:
        print(f"❌ MetaTrader5: {e}")
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        print("✅ UltimateXAUSystem: OK")
    except Exception as e:
        print(f"❌ UltimateXAUSystem: {e}")
        return False
    
    return True

def test_system_config():
    """Test SystemConfig"""
    print("\n🔧 TESTING SYSTEM CONFIG...")
    
    try:
        from ultimate_xau_system import SystemConfig
        config = SystemConfig()
        
        print(f"✅ Symbol: {config.symbol}")
        print(f"✅ Timeframe: {config.timeframe}")
        print(f"✅ Kelly Criterion: {config.enable_kelly_criterion}")
        print(f"✅ Multi-timeframe: {config.enable_multi_timeframe_training}")
        print(f"✅ Live Trading: {config.live_trading}")
        print(f"✅ Paper Trading: {config.paper_trading}")
        
        return True
    except Exception as e:
        print(f"❌ SystemConfig error: {e}")
        return False

def test_system_initialization():
    """Test khởi tạo hệ thống"""
    print("\n🚀 TESTING SYSTEM INITIALIZATION...")
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        print("Creating config...")
        config = SystemConfig()
        
        print("Initializing Ultimate XAU System...")
        system = UltimateXAUSystem(config)
        
        print(f"✅ System initialized successfully!")
        print(f"   Status: {system.system_state.get('status')}")
        print(f"   Version: {system.system_state.get('version')}")
        print(f"   Systems Active: {system.system_state.get('systems_active')}")
        print(f"   Systems Total: {system.system_state.get('systems_total')}")
        
        return system
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        traceback.print_exc()
        return None

def test_signal_generation(system):
    """Test tạo signal"""
    print("\n📊 TESTING SIGNAL GENERATION...")
    
    try:
        print("Generating test signal...")
        signal = system.generate_signal()
        
        print(f"✅ Signal generated successfully!")
        print(f"   Action: {signal.get('action')}")
        print(f"   Confidence: {signal.get('confidence')}")
        print(f"   Prediction: {signal.get('prediction')}")
        print(f"   Symbol: {signal.get('symbol')}")
        print(f"   Timestamp: {signal.get('timestamp')}")
        
        return True
    except Exception as e:
        print(f"❌ Signal generation error: {e}")
        traceback.print_exc()
        return False

def test_system_status(system):
    """Test system status"""
    print("\n📈 TESTING SYSTEM STATUS...")
    
    try:
        status = system.get_system_status()
        
        print(f"✅ System status retrieved!")
        print(f"   Overall Status: {status.get('system_info', {}).get('status')}")
        print(f"   Trading Active: {status.get('trading_status', {}).get('active')}")
        print(f"   Health: {status.get('system_health', {}).get('health_percentage')}%")
        
        return True
    except Exception as e:
        print(f"❌ System status error: {e}")
        return False

def test_subsystems(system):
    """Test các subsystem"""
    print("\n🔧 TESTING SUBSYSTEMS...")
    
    try:
        systems = system.system_manager.systems
        
        print(f"Total registered systems: {len(systems)}")
        
        for name, subsystem in systems.items():
            try:
                status = subsystem.get_status()
                active = "✅" if subsystem.is_active else "❌"
                print(f"   {active} {name}: {status.get('status', 'Unknown')}")
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
        
        return True
    except Exception as e:
        print(f"❌ Subsystems test error: {e}")
        return False

def test_data_sources():
    """Test data sources"""
    print("\n💾 TESTING DATA SOURCES...")
    
    # Test MT5 data files
    data_dirs = [
        "data/maximum_mt5_v2",
        "data/working_free_data", 
        "data/real_free_data"
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"✅ {data_dir}: {len(csv_files)} CSV files")
        else:
            print(f"❌ {data_dir}: Not found")

def identify_satellite_systems():
    """Xác định các hệ thống vệ tinh"""
    print("\n🛰️ IDENTIFYING SATELLITE SYSTEMS...")
    
    # Tìm các file có thể là hệ thống vệ tinh
    satellite_patterns = [
        "*training*.py",
        "*demo*.py", 
        "*test*.py",
        "*mode*.py",
        "*phase*.py",
        "*backup*.py"
    ]
    
    import glob
    
    satellite_systems = {}
    
    for pattern in satellite_patterns:
        files = glob.glob(pattern)
        if files:
            category = pattern.replace("*", "").replace(".py", "").upper()
            satellite_systems[category] = files
    
    for category, files in satellite_systems.items():
        print(f"\n📂 {category} SYSTEMS:")
        for file in files[:5]:  # Hiển thị tối đa 5 files
            print(f"   - {file}")
        if len(files) > 5:
            print(f"   ... và {len(files) - 5} files khác")

def main():
    """Main test function"""
    print("🔬 ULTIMATE XAU SYSTEM V4.0 - COMPONENT TESTING")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed - stopping tests")
        return
    
    # Test 2: Config
    if not test_system_config():
        print("❌ Config test failed")
        return
    
    # Test 3: System initialization
    system = test_system_initialization()
    if not system:
        print("❌ System initialization failed")
        return
    
    # Test 4: Signal generation
    test_signal_generation(system)
    
    # Test 5: System status
    test_system_status(system)
    
    # Test 6: Subsystems
    test_subsystems(system)
    
    # Test 7: Data sources
    test_data_sources()
    
    # Test 8: Satellite systems
    identify_satellite_systems()
    
    print("\n✅ TESTING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main() 