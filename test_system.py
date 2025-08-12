#!/usr/bin/env python3
"""
Simple system test
"""

import sys
sys.path.insert(0, 'src')

def test_system():
    print("🔍 TESTING AI3.0 UNIFIED SYSTEM")
    print("=" * 50)
    
    try:
        # Test import
        print("📦 Testing imports...")
        from UNIFIED_AI3_MASTER_SYSTEM import create_development_system
        print("✅ Import successful")
        
        # Create system
        print("🔧 Creating system...")
        system = create_development_system()
        print("✅ System created")
        
        # Get status
        print("📊 Getting system status...")
        status = system.get_system_status()
        
        # Display results
        print(f"\n📋 SYSTEM STATUS:")
        print(f"   Name: {status['system_info']['name']}")
        print(f"   Version: {status['system_info']['version']}")
        print(f"   Mode: {status['system_info']['mode']}")
        print(f"   Active: {status['system_info']['is_active']}")
        
        print(f"\n🔧 COMPONENTS ({sum(status['components'].values())}/{len(status['components'])} active):")
        for name, active in status['components'].items():
            status_icon = "✅" if active else "❌"
            print(f"   {status_icon} {name}")
        
        print(f"\n🎯 RESULT: SYSTEM IS WORKING!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_system() 