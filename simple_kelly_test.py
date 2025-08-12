#!/usr/bin/env python3
"""Simple Kelly System Test"""

from src.core.kelly_system import KellyCriterionSystem
from src.core.ultimate_xau_system import SystemConfig
import pandas as pd

def main():
    print("🧪 SIMPLE KELLY SYSTEM TEST")
    
    # Create proper config object
    config = SystemConfig()
    
    # Create Kelly system
    kelly = KellyCriterionSystem(config)
    kelly.initialize()
    
    # Create test data
    data = pd.DataFrame({
        'close': [2050, 2051, 2049, 2052, 2048],
        'volume': [1000, 1100, 900, 1200, 1050]
    })
    
    # Process data
    result = kelly.process(data)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        kelly_calc = result.get('kelly_calculation', {})
        print("✅ Kelly System Test Successful!")
        print(f"Kelly Fraction: {kelly_calc.get('safe_kelly_fraction', 0):.3f}")
        print(f"Position Size: ${kelly_calc.get('position_size_usd', 0):,.2f}")
        print(f"Recommendation: {kelly_calc.get('recommendation', 'N/A')}")

if __name__ == "__main__":
    main() 