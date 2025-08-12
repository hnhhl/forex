
import sys
sys.path.append('src')
from core.ultimate_xau_system import UltimateXAUSystem
import pandas as pd
import numpy as np

def validate_system():
    print("🔍 VALIDATING REPAIRED SYSTEM...")
    
    try:
        # Initialize system
        system = UltimateXAUSystem()
        print("✅ System initialization: SUCCESS")
        
        # Check component activation
        active_components = 0
        for component_name in ['data_quality_monitor', 'latency_optimizer', 
                              'mt5_connection_manager', 'neural_network_system',
                              'ai_phase_system', 'ai2_advanced_technologies',
                              'advanced_ai_ensemble', 'realtime_mt5_data']:
            if hasattr(system, component_name):
                active_components += 1
        
        print(f"✅ Active components: {active_components}/8")
        
        # Test signal generation
        test_data = pd.DataFrame({
            'open': [2000.0, 2001.0, 2002.0],
            'high': [2005.0, 2006.0, 2007.0], 
            'low': [1995.0, 1996.0, 1997.0],
            'close': [2003.0, 2004.0, 2005.0],
            'volume': [1000, 1100, 1200]
        })
        
        signal = system.generate_signal(test_data)
        print(f"✅ Signal generation: SUCCESS")
        print(f"   Signal: {signal.get('signal', 'N/A')}")
        print(f"   Confidence: {signal.get('confidence', 'N/A')}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_system()
