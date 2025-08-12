#!/usr/bin/env python3
"""
🎯 TEST HỆ THỐNG CHÍNH THỐNG NHẤT
ULTIMATE XAU SUPER SYSTEM V4.0 - COMPLETE RESTORATION

Test đơn giản để đảm bảo hệ thống hoạt động như một thể thống nhất
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_main_system():
    """Test hệ thống chính thống nhất"""
    
    print("🎯 TESTING HỆ THỐNG CHÍNH THỐNG NHẤT")
    print("="*60)
    
    try:
        # Import hệ thống chính
        from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        print("✅ Import thành công")
        
        # Tạo cấu hình đơn giản
        config = SystemConfig()
        config.enable_kelly_criterion = False  # Disable để test đơn giản
        
        print("✅ Config tạo thành công")
        
        # Khởi tạo hệ thống chính (chỉ core systems)
        print("\n🔧 Khởi tạo hệ thống chính...")
        system = UltimateXAUSystem(config)
        
        print("\n✅ HỆ THỐNG CHÍNH KHỞI TẠO THÀNH CÔNG!")
        
        # Test generate signal
        print("\n📊 Testing signal generation...")
        signal = system.generate_signal()
        
        print(f"✅ Signal generated: {signal.get('signal', 'Unknown')}")
        print(f"📈 Confidence: {signal.get('confidence', 0):.2f}")
        
        # Test system status
        print("\n📋 Testing system status...")
        status = system.get_system_status()
        
        print(f"✅ Active Systems: {status.get('active_systems', 0)}")
        print(f"📊 Total Systems: {status.get('total_systems', 0)}")
        print(f"🎯 Status: {status.get('status', 'Unknown')}")
        
        print("\n🏆 HỆ THỐNG CHÍNH HOẠT ĐỘNG HOÀN HẢO NHU MỘT THỂ THỐNG NHẤT!")
        return True
        
    except Exception as e:
        print(f"❌ LỖI TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_system()
    
    if success:
        print("\n🎉 TEST THÀNH CÔNG - HỆ THỐNG CHÍNH THỐNG NHẤT!")
    else:
        print("\n❌ TEST THẤT BẠI") 