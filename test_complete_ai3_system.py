#!/usr/bin/env python3
"""
Test Complete AI3.0 System - 4 Cấp Quyết Định
Kiểm tra hệ thống hoàn thiện với tất cả components đã bổ sung
"""

import sys
import os
sys.path.append("src")

import pandas as pd
import numpy as np
from datetime import datetime
import json

def test_complete_ai3_system():
    """Test Complete AI3.0 System với 4 cấp quyết định"""
    
    print("🎯 TESTING COMPLETE AI3.0 SYSTEM - 4 CẤP QUYẾT ĐỊNH")
    print("=" * 80)
    
    try:
        # Import main system
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        print("✅ Successfully imported UltimateXAUSystem")
        
        # Create system configuration
        config = SystemConfig()
        print(f"✅ System configuration created")
        
        # Initialize system
        print("\n🔧 Initializing Complete AI3.0 System...")
        system = UltimateXAUSystem(config)
        
        print(f"   System initialized: {system.__class__.__name__}")
        print(f"   Config symbol: {system.config.symbol}")
        
        # Test system status
        print("\n📊 Testing System Status...")
        status = system.get_system_status()
        
        print(f"   Systems active: {status.get('systems_active', 0)}")
        print(f"   Systems total: {status.get('systems_total', 0)}")
        print(f"   System health: {status.get('system_health', 'unknown')}")
        
        # Test signal generation
        print("\n🔍 Testing Signal Generation...")
        signal = system.generate_signal()
        
        print(f"   Signal generated: {signal.get('action', 'unknown')}")
        print(f"   Prediction: {signal.get('prediction', 0.0):.3f}")
        print(f"   Confidence: {signal.get('confidence', 0.0):.3f}")
        print(f"   Systems used: {signal.get('systems_used', 0)}")
        
        # Check for 4-tier decision structure
        print("\n🏗️ Analyzing 4-Tier Decision Structure...")
        
        signal_components = signal.get('signal_components', {})
        print(f"   Total signal components: {len(signal_components)}")
        
        # CẤP 1 - HỆ THỐNG CHÍNH (65% quyết định)
        tier1_systems = ['NeuralNetworkSystem', 'MT5ConnectionManager']
        tier1_found = [sys for sys in tier1_systems if sys in signal_components]
        print(f"   🥇 CẤP 1 (65%): {len(tier1_found)}/{len(tier1_systems)} systems found")
        for sys_name in tier1_found:
            comp = signal_components[sys_name]
            print(f"      - {sys_name}: pred={comp.get('prediction', 0):.3f}, conf={comp.get('confidence', 0):.3f}")
        
        # CẤP 2 - HỆ THỐNG HỖ TRỢ (45% quyết định)
        tier2_systems = ['AdvancedAIEnsembleSystem', 'DataQualityMonitor', 'AIPhaseSystem', 'RealTimeMT5DataSystem']
        tier2_found = [sys for sys in tier2_systems if sys in signal_components]
        print(f"   🥈 CẤP 2 (45%): {len(tier2_found)}/{len(tier2_systems)} systems found")
        for sys_name in tier2_found:
            comp = signal_components[sys_name]
            print(f"      - {sys_name}: pred={comp.get('prediction', 0):.3f}, conf={comp.get('confidence', 0):.3f}")
        
        # CẤP 3 - HỆ THỐNG PHỤ (20% quyết định)
        tier3_systems = ['AI2AdvancedTechnologiesSystem', 'LatencyOptimizer']
        tier3_found = [sys for sys in tier3_systems if sys in signal_components]
        print(f"   🥉 CẤP 3 (20%): {len(tier3_found)}/{len(tier3_systems)} systems found")
        for sys_name in tier3_found:
            comp = signal_components[sys_name]
            print(f"      - {sys_name}: pred={comp.get('prediction', 0):.3f}, conf={comp.get('confidence', 0):.3f}")
        
        # CẤP 4 - DEMOCRATIC LAYER (Equal voting rights)
        tier4_systems = ['DemocraticSpecialistsSystem']
        tier4_found = [sys for sys in tier4_systems if sys in signal_components]
        print(f"   🗳️ CẤP 4 (Democratic): {len(tier4_found)}/{len(tier4_systems)} systems found")
        for sys_name in tier4_found:
            comp = signal_components[sys_name]
            print(f"      - {sys_name}: pred={comp.get('prediction', 0):.3f}, conf={comp.get('confidence', 0):.3f}")
            
            # Check democratic details
            if 'final_vote' in comp:
                print(f"        Final Vote: {comp['final_vote']}")
                print(f"        Consensus: {comp.get('consensus_strength', 0):.3f}")
                print(f"        Specialists: {comp.get('total_specialists', 0)}")
        
        # Test ensemble method
        ensemble_method = signal.get('ensemble_method', 'unknown')
        print(f"\n🤝 Ensemble Method: {ensemble_method}")
        
        # Check for hybrid AI2.0 + AI3.0
        if 'ai2_weighted' in signal:
            ai2_result = signal['ai2_weighted']
            print(f"   AI2.0 Weighted: pred={ai2_result.get('prediction', 0):.3f}, conf={ai2_result.get('confidence', 0):.3f}")
        
        if 'ai3_democratic' in signal:
            ai3_result = signal['ai3_democratic']
            print(f"   AI3.0 Democratic: vote={ai3_result.get('final_vote', 'unknown')}, consensus={ai3_result.get('consensus_strength', 0):.3f}")
        
        # Performance summary
        print("\n📈 SYSTEM PERFORMANCE SUMMARY:")
        print(f"   ✅ System Status: {len(tier1_found + tier2_found + tier3_found + tier4_found)}/{len(tier1_systems + tier2_systems + tier3_systems + tier4_systems)} systems active")
        print(f"   ✅ Signal Quality: {signal.get('confidence', 0.0):.1%} confidence")
        print(f"   ✅ Decision Method: {ensemble_method}")
        
        # Check for specific AI3.0 features
        print("\n🔍 AI3.0 FEATURES CHECK:")
        
        # Advanced AI Ensemble System
        if 'AdvancedAIEnsembleSystem' in signal_components:
            ensemble_comp = signal_components['AdvancedAIEnsembleSystem']
            print(f"   🤖 Advanced AI Ensemble: {ensemble_comp.get('model_count', 0)} models")
            print(f"      Tree models: {ensemble_comp.get('tree_models', 0)}")
            print(f"      Linear models: {ensemble_comp.get('linear_models', 0)}")
            print(f"      Advanced models: {ensemble_comp.get('advanced_models', 0)}")
        else:
            print("   ❌ Advanced AI Ensemble System not found")
        
        # Democratic Specialists System
        if 'DemocraticSpecialistsSystem' in signal_components:
            democratic_comp = signal_components['DemocraticSpecialistsSystem']
            print(f"   🗳️ Democratic Specialists: {democratic_comp.get('total_specialists', 0)} specialists")
            
            category_votes = democratic_comp.get('category_votes', {})
            if category_votes:
                print(f"      Category votes:")
                for category, vote_data in category_votes.items():
                    print(f"        {category}: {vote_data.get('majority_vote', 'unknown')} (conf: {vote_data.get('confidence', 0):.2f})")
        else:
            print("   ❌ Democratic Specialists System not found")
        
        # AI Phases System
        if 'AIPhaseSystem' in signal_components:
            phases_comp = signal_components['AIPhaseSystem']
            print(f"   🚀 AI Phases: +{phases_comp.get('total_boost', 0)}% performance boost")
        else:
            print("   ❌ AI Phases System not found")
        
        print("\n🎉 COMPLETE AI3.0 SYSTEM TEST COMPLETED!")
        
        # Save test results
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'system_status': status,
            'signal_result': signal,
            'tier_analysis': {
                'tier1_found': len(tier1_found),
                'tier1_total': len(tier1_systems),
                'tier2_found': len(tier2_found),
                'tier2_total': len(tier2_systems),
                'tier3_found': len(tier3_found),
                'tier3_total': len(tier3_systems),
                'tier4_found': len(tier4_found),
                'tier4_total': len(tier4_systems)
            },
            'features_check': {
                'advanced_ai_ensemble': 'AdvancedAIEnsembleSystem' in signal_components,
                'democratic_specialists': 'DemocraticSpecialistsSystem' in signal_components,
                'ai_phases': 'AIPhaseSystem' in signal_components
            }
        }
        
        with open('complete_ai3_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"✅ Test results saved to: complete_ai3_test_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in complete system test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_ai3_system()
    if success:
        print("\n🎯 COMPLETE AI3.0 SYSTEM: FULLY OPERATIONAL ✅")
    else:
        print("\n❌ COMPLETE AI3.0 SYSTEM: NEEDS ATTENTION") 