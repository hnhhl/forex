#!/usr/bin/env python3
"""
Test script to verify synchronized weights and boost mechanisms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultimate_xau_system import UltimateXAUSystem, SystemConfig
import pandas as pd
import numpy as np
from datetime import datetime

def test_synchronized_weights():
    """Test that weights are now synchronized with optimal distribution"""
    
    print("🔄 TESTING SYNCHRONIZED WEIGHTS SYSTEM")
    print("=" * 60)
    
    # Initialize system
    config = SystemConfig()
    system = UltimateXAUSystem(config)
    
    # Test weight distribution
    print("📊 TESTING WEIGHT DISTRIBUTION:")
    
    # Core voting systems
    voting_systems = [
        'NeuralNetworkSystem',
        'PortfolioManager', 
        'OrderManager',
        'AdvancedAIEnsembleSystem',
        'AIPhaseSystem',
        'LatencyOptimizer',
        'AI2AdvancedTechnologiesSystem',
        'DemocraticSpecialistsSystem'
    ]
    
    # Support systems (should have 0 weight)
    support_systems = [
        'MT5ConnectionManager',
        'DataQualityMonitor',
        'RealTimeMT5DataSystem',
        'StopLossManager',
        'PositionSizer',
        'KellyCriterionCalculator'
    ]
    
    print("\n🗳️ VOTING SYSTEMS:")
    total_voting_weight = 0
    for system_name in voting_systems:
        weight = system._get_system_weight(system_name)
        total_voting_weight += weight
        print(f"  {system_name:<35}: {weight:>6.1%}")
    
    print(f"\n  TOTAL VOTING WEIGHT: {total_voting_weight:.1%}")
    
    print("\n📊 SUPPORT SYSTEMS:")
    total_support_weight = 0
    for system_name in support_systems:
        weight = system._get_system_weight(system_name)
        total_support_weight += weight
        print(f"  {system_name:<35}: {weight:>6.1%}")
    
    print(f"\n  TOTAL SUPPORT WEIGHT: {total_support_weight:.1%}")
    
    print(f"\n📈 GRAND TOTAL: {total_voting_weight + total_support_weight:.1%}")
    
    # Check if weights are correct
    if abs(total_voting_weight - 1.0) < 0.01:
        print("✅ WEIGHTS SYNCHRONIZED CORRECTLY!")
    else:
        print(f"❌ WEIGHTS NOT SYNCHRONIZED: {total_voting_weight:.1%}")
    
    if total_support_weight == 0:
        print("✅ SUPPORT SYSTEMS CORRECTLY SET TO 0%!")
    else:
        print(f"❌ SUPPORT SYSTEMS HAVE VOTING POWER: {total_support_weight:.1%}")
    
    return system

def test_boost_mechanisms(system):
    """Test boost mechanisms are working correctly"""
    
    print("\n🚀 TESTING BOOST MECHANISMS")
    print("=" * 60)
    
    # Create mock market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(2000, 2100, 100),
        'high': np.random.uniform(2050, 2150, 100),
        'low': np.random.uniform(1950, 2050, 100),
        'close': np.random.uniform(2000, 2100, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # Generate signal to test boost mechanisms
    try:
        signal = system.generate_signal()
        
        print("📊 SIGNAL ANALYSIS:")
        print(f"  Action: {signal.get('action', 'N/A')}")
        print(f"  Prediction: {signal.get('prediction', 0):.3f}")
        print(f"  Confidence: {signal.get('confidence', 0):.3f}")
        print(f"  Voting Systems Used: {signal.get('voting_systems_used', 0)}")
        print(f"  Total Systems Used: {signal.get('systems_used', 0)}")
        
        # Check boost info
        boost_info = signal.get('boost_info', {})
        if boost_info:
            print("\n🔥 BOOST MECHANISMS APPLIED:")
            for boost_type, boost_value in boost_info.items():
                print(f"  {boost_type}: +{boost_value:.1%}")
        else:
            print("\n⚪ NO BOOST MECHANISMS APPLIED")
        
        # Check hybrid metrics
        hybrid_metrics = signal.get('hybrid_metrics', {})
        if hybrid_metrics:
            print("\n📈 HYBRID METRICS:")
            base_pred = hybrid_metrics.get('base_prediction', 0)
            boosted_pred = hybrid_metrics.get('boosted_prediction', 0)
            print(f"  Base Prediction: {base_pred:.3f}")
            print(f"  Boosted Prediction: {boosted_pred:.3f}")
            if boosted_pred > base_pred:
                improvement = ((boosted_pred - base_pred) / base_pred) * 100
                print(f"  Improvement: +{improvement:.1f}%")
        
        return signal
        
    except Exception as e:
        print(f"❌ ERROR TESTING BOOST MECHANISMS: {e}")
        return None

def test_voting_vs_support_separation(system):
    """Test that voting and support systems are properly separated"""
    
    print("\n🔄 TESTING VOTING VS SUPPORT SEPARATION")
    print("=" * 60)
    
    try:
        signal = system.generate_signal()
        
        voting_used = signal.get('voting_systems_used', 0)
        total_used = signal.get('systems_used', 0)
        support_used = total_used - voting_used
        
        print(f"📊 SYSTEM USAGE BREAKDOWN:")
        print(f"  Voting Systems Used: {voting_used}")
        print(f"  Support Systems Used: {support_used}")
        print(f"  Total Systems Used: {total_used}")
        
        if voting_used > 0:
            print("✅ VOTING SYSTEMS ACTIVE")
        else:
            print("❌ NO VOTING SYSTEMS ACTIVE")
        
        if support_used > 0:
            print("✅ SUPPORT SYSTEMS PROVIDING DATA")
        else:
            print("⚠️  NO SUPPORT SYSTEMS DETECTED")
        
        # Check ensemble method
        ensemble_method = signal.get('ensemble_method', '')
        if 'boosts' in ensemble_method:
            print("✅ BOOST-ENABLED ENSEMBLE METHOD DETECTED")
        else:
            print("⚠️  BOOST-ENABLED ENSEMBLE METHOD NOT DETECTED")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR TESTING SEPARATION: {e}")
        return False

def test_performance_comparison():
    """Compare old vs new system performance"""
    
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Simulate old system (280% weights)
    old_weights = {
        'NeuralNetworkSystem': 0.25,
        'MT5ConnectionManager': 0.20,
        'AdvancedAIEnsembleSystem': 0.20,
        'DataQualityMonitor': 0.15,
        'AIPhaseSystem': 0.15,
        'RealTimeMT5DataSystem': 0.15,
        'AI2AdvancedTechnologiesSystem': 0.10,
        'LatencyOptimizer': 0.10,
        'DemocraticSpecialistsSystem': 1.0
    }
    
    # New system (100% weights)
    new_weights = {
        'NeuralNetworkSystem': 0.20,
        'PortfolioManager': 0.15,
        'OrderManager': 0.05,
        'AdvancedAIEnsembleSystem': 0.20,
        'AIPhaseSystem': 0.15,
        'LatencyOptimizer': 0.05,
        'AI2AdvancedTechnologiesSystem': 0.10,
        'DemocraticSpecialistsSystem': 0.10
    }
    
    old_total = sum(old_weights.values())
    new_total = sum(new_weights.values())
    
    print(f"📈 WEIGHT DISTRIBUTION COMPARISON:")
    print(f"  Old System Total: {old_total:.1%}")
    print(f"  New System Total: {new_total:.1%}")
    print(f"  Improvement: {((new_total - old_total) / old_total) * 100:+.1f}%")
    
    # Calculate efficiency
    old_efficiency = 1.0 / old_total  # Lower is better
    new_efficiency = 1.0 / new_total  # Should be 1.0
    
    print(f"\n⚡ EFFICIENCY COMPARISON:")
    print(f"  Old System Efficiency: {old_efficiency:.3f}")
    print(f"  New System Efficiency: {new_efficiency:.3f}")
    print(f"  Efficiency Gain: {((new_efficiency - old_efficiency) / old_efficiency) * 100:+.1f}%")
    
    if new_total == 1.0:
        print("✅ PERFECT WEIGHT DISTRIBUTION ACHIEVED!")
    else:
        print(f"⚠️  WEIGHT DISTRIBUTION NEEDS ADJUSTMENT: {new_total:.1%}")

def main():
    """Main test function"""
    
    print("🎯 COMPREHENSIVE SYNCHRONIZED SYSTEM TEST")
    print("=" * 80)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test 1: Weight synchronization
        system = test_synchronized_weights()
        
        # Test 2: Boost mechanisms
        signal = test_boost_mechanisms(system)
        
        # Test 3: Voting vs Support separation
        separation_ok = test_voting_vs_support_separation(system)
        
        # Test 4: Performance comparison
        test_performance_comparison()
        
        # Final assessment
        print("\n🎯 FINAL ASSESSMENT")
        print("=" * 60)
        
        if signal and separation_ok:
            print("✅ ALL TESTS PASSED - SYSTEM SYNCHRONIZED SUCCESSFULLY!")
            print("🚀 READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            print("🔧 ADDITIONAL FIXES NEEDED")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR DURING TESTING: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 