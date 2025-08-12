import sys
import os
sys.path.append("src")

try:
    from core.specialists.rsi_specialist import create_rsi_specialist
    from core.specialists.macd_specialist import create_macd_specialist
    from core.specialists.fibonacci_specialist import create_fibonacci_specialist
    from core.specialists.chart_pattern_specialist import create_chart_pattern_specialist
    from core.specialists.var_risk_specialist import create_var_risk_specialist
    from core.specialists.democratic_voting_engine import create_democratic_voting_engine
    
    print("🎯 TESTING CORE 5 SPECIALISTS")
    print("✅ All imports successful")
    
    # Create specialists
    specialists = [
        create_rsi_specialist(),
        create_macd_specialist(),
        create_fibonacci_specialist(),
        create_chart_pattern_specialist(),
        create_var_risk_specialist()
    ]
    
    print(f"✅ Created {len(specialists)} specialists")
    
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    data = pd.DataFrame({
        "open": [2000, 2010, 2005],
        "high": [2015, 2020, 2010],
        "low": [1995, 2000, 1995],
        "close": [2010, 2005, 2008],
        "volume": [1000, 1500, 1200]
    })
    
    print("✅ Testing individual specialists...")
    for i, spec in enumerate(specialists, 1):
        try:
            vote = spec.analyze(data, 2008.0)
            print(f"   {i}. {spec.name}: {vote.vote} (confidence: {vote.confidence:.2f})")
        except Exception as e:
            print(f"   {i}. {spec.name}: ERROR - {e}")
    
    # Test democratic voting
    voting_engine = create_democratic_voting_engine()
    result = voting_engine.conduct_vote(specialists, data, 2008.0)
    
    print(f"\\n🗳️ Democratic Result: {result.vote} (confidence: {result.confidence:.2f})")
    print("🎉 CORE SYSTEM WORKING!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
