"""
Quick test để kiểm tra signal generation sau khi fix components
"""

import sys
sys.path.append('src')

try:
    from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig
    
    print("🧪 Quick test signal generation...")
    
    # Tạo system với config cơ bản
    config = SystemConfig(
        symbol="XAUUSD",
        mt5_login=0,
        mt5_password="", 
        mt5_server=""
    )
    
    system = UltimateXAUSystem(config)
    
    # Generate signal
    signal = system.generate_signal()
    
    print(f"📊 Signal generated:")
    print(f"   Action: {signal.get('action', 'UNKNOWN')}")
    print(f"   Confidence: {signal.get('confidence', 0.0):.3f}")
    print(f"   Prediction: {signal.get('prediction', 0.0):.3f}")
    print(f"   Components: {len(signal.get('signal_components', {}))}")
    
    # Kiểm tra từng component
    components = signal.get('signal_components', {})
    print(f"\n🔍 Component predictions:")
    
    for name, data in components.items():
        if isinstance(data, dict):
            pred = data.get('prediction', 'N/A')
            conf = data.get('confidence', 'N/A')
            print(f"   {name}: pred={pred}, conf={conf}")
        else:
            print(f"   {name}: {type(data)} (needs fixing)")
    
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 