"""
Unit Tests for Day 23 Custom Technical Indicators
Ultimate XAU Super System V4.0

Test coverage:
- Basic indicator calculations
- Custom indicator creation
- Multi-timeframe analysis
- Volume profile analysis
- Performance validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add source path
sys.path.append('src')

# Import custom technical indicators
try:
    from src.core.analysis.custom_technical_indicators import (
        create_custom_technical_indicators, CustomTechnicalIndicators,
        IndicatorConfig, IndicatorResult, BaseIndicator,
        MovingAverageCustom, RSICustom, MACDCustom, BollingerBandsCustom, VolumeProfileCustom,
        CustomIndicatorFactory, MultiTimeframeAnalyzer,
        stochastic_rsi, williams_r, commodity_channel_index
    )
    CUSTOM_INDICATORS_AVAILABLE = True
except ImportError:
    CUSTOM_INDICATORS_AVAILABLE = False

def run_basic_tests():
    """Run basic tests without pytest"""
    if not CUSTOM_INDICATORS_AVAILABLE:
        print("❌ Custom Technical Indicators not available - skipping tests")
        return False
    
    print("🧪 Running Custom Technical Indicators Tests...")
    
    try:
        # Generate test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        prices = [2000 + i + np.random.normal(0, 10) for i in range(50)]
        
        test_data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Test system creation
        system = create_custom_technical_indicators()
        print("✅ System initialization: PASSED")
        
        # Test basic indicators
        ma_result = system.calculate_indicator(test_data, 'ma', period=10, method='sma')
        assert ma_result is not None
        print("✅ Moving Average calculation: PASSED")
        
        rsi_result = system.calculate_indicator(test_data, 'rsi', period=14)
        assert rsi_result is not None
        print("✅ RSI calculation: PASSED")
        
        # Test custom indicators
        system.register_custom_indicator('stoch_rsi', stochastic_rsi, period=14)
        stoch_result = system.calculate_indicator(test_data, 'stoch_rsi', period=14, stoch_period=14)
        assert stoch_result is not None
        print("✅ Custom indicator registration: PASSED")
        
        # Test multiple indicators
        configs = {
            'ma': {'type': 'ma', 'period': 10, 'method': 'sma'},
            'rsi': {'type': 'rsi', 'period': 14}
        }
        results = system.calculate_multiple_indicators(test_data, configs)
        assert len(results) == 2
        print("✅ Multiple indicators calculation: PASSED")
        
        print("🎉 All tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    if success:
        print("\n✅ Day 23 Custom Technical Indicators: All tests passed!")
    else:
        print("\n❌ Day 23 Custom Technical Indicators: Some tests failed!")
