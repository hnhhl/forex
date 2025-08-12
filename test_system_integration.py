#!/usr/bin/env python3
"""
Test System Integration - Kiểm tra tích hợp hệ thống
"""

import sys
sys.path.append('src/core')

def test_system_integration():
    """Test xem các hệ thống có tích hợp với nhau không"""
    print("🎯 TESTING SYSTEM INTEGRATION...")
    print("=" * 50)
    
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        
        # Initialize system
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        
        # Test system status
        status = system.get_system_status()
        print(f"✅ System Status: {status['system_state']['status']}")
        print(f"📊 Total Systems: {status['system_state']['systems_total']}")
        print(f"🔥 Active Systems: {status['system_state']['systems_active']}")
        
        # Test signal generation
        signal = system.generate_signal()
        print(f"🎯 Signal Generated: {signal['action']} (Confidence: {signal['confidence']:.2%})")
        print(f"🔗 Systems Used: {signal.get('systems_used', 0)}")
        print(f"⚙️ Ensemble Method: {signal.get('ensemble_method', 'unknown')}")
        
        # Check integration details
        if 'voting_results' in signal:
            voting = signal['voting_results']
            print(f"🗳️ Voting Results: BUY={voting['buy_votes']}, SELL={voting['sell_votes']}, HOLD={voting['hold_votes']}")
        
        if 'hybrid_metrics' in signal:
            metrics = signal['hybrid_metrics']
            print(f"📊 Hybrid Consensus: {metrics['hybrid_consensus']:.2%}")
            print(f"📈 Signal Strength: {metrics['signal_strength']:.3f}")
        
        # Test system integration
        systems_used = signal.get('systems_used', 0)
        if systems_used > 0:
            print(f"✅ SYSTEMS ARE INTEGRATED! {systems_used} systems working together")
            
            # Test data flow
            print("\n🔄 TESTING DATA FLOW...")
            system_manager = system.system_manager
            active_systems = [name for name, sys in system_manager.systems.items() if sys.is_active]
            print(f"📡 Active Systems: {', '.join(active_systems)}")
            
            return True
        else:
            print("❌ Systems appear to be working in isolation")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_modules():
    """Test các analysis modules"""
    print(f"\n📚 TEST 4: ANALYSIS MODULES")
    print("-" * 30)
    
    modules_to_test = [
        ('Technical Analysis', 'src.core.analysis.technical_analysis'),
        ('Pattern Recognition', 'src.core.analysis.advanced_pattern_recognition'),
        ('Risk Management', 'src.core.analysis.advanced_risk_management'),
        ('Market Regime Detection', 'src.core.analysis.market_regime_detection')
    ]
    
    available_modules = []
    
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"✅ {name} module available")
            available_modules.append(name)
        except Exception as e:
            print(f"❌ {name} module failed: {e}")
    
    print(f"\n📊 Available analysis modules: {len(available_modules)}/4")
    return available_modules

def test_voting_systems():
    """Test voting systems"""
    print(f"\n🗳️ TEST 5: VOTING SYSTEMS")
    print("-" * 30)
    
    try:
        from src.core.integration.ai_master_integration import DecisionStrategy
        strategies = list(DecisionStrategy)
        print(f"✅ Available decision strategies: {len(strategies)}")
        for strategy in strategies:
            print(f"   • {strategy.value}")
        return True
    except Exception as e:
        print(f"❌ Voting systems test failed: {e}")
        return False

def generate_integration_report():
    """Tạo báo cáo tích hợp"""
    print(f"\n📋 INTEGRATION REPORT")
    print("=" * 60)
    
    # Run all tests
    system_test = test_system_integration()
    analysis_modules = test_analysis_modules()
    voting_test = test_voting_systems()
    
    # Summary
    print(f"\n🎯 TỔNG KẾT:")
    print("-" * 20)
    
    integration_score = 0
    
    if system_test:
        print("✅ Core system: INTEGRATED & WORKING")
        integration_score += 40
    else:
        print("❌ Core system: NOT WORKING")
    
    module_score = len(analysis_modules) * 10
    print(f"📚 Analysis modules: {len(analysis_modules)}/4 available ({module_score}%)")
    integration_score += module_score
    
    if voting_test:
        print("✅ Voting systems: AVAILABLE")
        integration_score += 20
    else:
        print("❌ Voting systems: NOT AVAILABLE")
    
    print(f"\n🏆 INTEGRATION SCORE: {integration_score}/100")
    
    if integration_score >= 80:
        status = "🟢 EXCELLENT - Hệ thống tích hợp tốt"
    elif integration_score >= 60:
        status = "🟡 GOOD - Hệ thống tích hợp khá tốt"
    elif integration_score >= 40:
        status = "🟠 FAIR - Hệ thống tích hợp cơ bản"
    else:
        status = "🔴 POOR - Hệ thống chưa tích hợp tốt"
    
    print(f"📊 STATUS: {status}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if integration_score < 100:
        if not system_test:
            print("   • Fix core system integration issues")
        if len(analysis_modules) < 4:
            print("   • Complete analysis modules integration")
        if not voting_test:
            print("   • Fix voting systems")
        if integration_score >= 60:
            print("   • Ready for Multi-Perspective Ensemble upgrade!")
    else:
        print("   • System fully integrated - ready for any upgrade!")

if __name__ == "__main__":
    success = test_system_integration()
    if success:
        print("\n🎉 SYSTEM INTEGRATION TEST PASSED!")
    else:
        print("\n❌ SYSTEM INTEGRATION TEST FAILED!") 