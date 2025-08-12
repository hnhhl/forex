#!/usr/bin/env python3
"""
SCRIPT KIỂM TRA HỆ THỐNG AI3.0 - VALIDATION TEST
Kiểm tra tất cả các lỗi đã được sửa
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_imports():
    """Test 1: Kiểm tra import system"""
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        logger.info("✅ Test 1 PASSED: System imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: Import error - {e}")
        return False

def test_system_initialization():
    """Test 2: Kiểm tra khởi tạo hệ thống"""
    try:
        from ultimate_xau_system import UltimateXAUSystem, SystemConfig
        config = SystemConfig()
        system = UltimateXAUSystem(config)
        logger.info("✅ Test 2 PASSED: System initialization successful")
        return True, system
    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: Initialization error - {e}")
        return False, None

def test_voting_weights(system):
    """Test 3: Kiểm tra trọng số bỏ phiếu"""
    try:
        # Test all registered systems
        registered_systems = [
            'DataQualityMonitor', 'LatencyOptimizer', 'MT5ConnectionManager',
            'NeuralNetworkSystem', 'AIPhaseSystem', 'AI2AdvancedTechnologiesSystem',
            'AdvancedAIEnsembleSystem', 'RealTimeMT5DataSystem', 'DemocraticSpecialistsSystem',
            'PortfolioManager', 'OrderManager', 'StopLossManager', 
            'PositionSizer', 'KellyCriterionCalculator'
        ]
        
        total_weight = 0
        voting_systems = []
        support_systems = []
        
        for system_name in registered_systems:
            weight = system._get_system_weight(system_name)
            if weight > 0:
                voting_systems.append((system_name, weight))
                total_weight += weight
            else:
                support_systems.append(system_name)
        
        logger.info(f"📊 VOTING SYSTEMS ({len(voting_systems)}):")
        for name, weight in voting_systems:
            logger.info(f"   {name}: {weight:.1%}")
        
        logger.info(f"🔧 SUPPORT SYSTEMS ({len(support_systems)}):")
        for name in support_systems:
            logger.info(f"   {name}: 0% (support only)")
        
        logger.info(f"🎯 TOTAL VOTING WEIGHT: {total_weight:.1%}")
        
        if abs(total_weight - 1.0) < 0.001:
            logger.info("✅ Test 3 PASSED: Voting weights sum to 100%")
            return True
        else:
            logger.error(f"❌ Test 3 FAILED: Voting weights sum to {total_weight:.1%}, should be 100%")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: Voting weight test error - {e}")
        return False

def test_component_wrapper_fix(system):
    """Test 4: Kiểm tra Component Wrapper Fix"""
    try:
        # Test different result formats
        test_cases = [
            # NeuralNetworkSystem format
            ({'ensemble_prediction': {'prediction': 0.65, 'confidence': 0.8}}, 'NeuralNetworkSystem'),
            
            # AIPhaseSystem format
            ({'phase_results': {'phase1': {'prediction': 0.7}, 'phase2': 0.6}, 'prediction': 0.65, 'confidence': 0.75}, 'AIPhaseSystem'),
            
            # AdvancedAIEnsembleSystem format
            ({'tree_predictions': {'rf': {'prediction': 0.6}}, 'linear_predictions': {'lr': {'prediction': 0.7}}}, 'AdvancedAIEnsembleSystem'),
            
            # DemocraticSpecialistsSystem format
            ({'voting_results': {'buy_votes': 12, 'sell_votes': 6, 'hold_votes': 2}}, 'DemocraticSpecialistsSystem'),
            
            # PortfolioManager format
            ({'portfolio_allocation': {'confidence': 0.8}}, 'PortfolioManager'),
            
            # OrderManager format
            ({'execution_quality': 0.9}, 'OrderManager'),
            
            # StopLossManager format
            ({'risk_metrics': {'risk_score': 0.3}}, 'StopLossManager'),
            
            # PositionSizer format
            ({'kelly_fraction': 0.15}, 'PositionSizer'),
            
            # KellyCriterionCalculator format
            ({'optimal_fraction': 0.12}, 'KellyCriterionCalculator'),
        ]
        
        all_passed = True
        for result, system_name in test_cases:
            try:
                prediction, confidence = system._convert_to_prediction(result, system_name)
                if 0 <= prediction <= 1 and 0 <= confidence <= 1:
                    logger.info(f"   ✅ {system_name}: pred={prediction:.3f}, conf={confidence:.3f}")
                else:
                    logger.error(f"   ❌ {system_name}: Invalid range - pred={prediction:.3f}, conf={confidence:.3f}")
                    all_passed = False
            except Exception as e:
                logger.error(f"   ❌ {system_name}: Conversion error - {e}")
                all_passed = False
        
        if all_passed:
            logger.info("✅ Test 4 PASSED: Component Wrapper Fix working correctly")
            return True
        else:
            logger.error("❌ Test 4 FAILED: Component Wrapper Fix has issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: Component wrapper test error - {e}")
        return False

def test_signal_generation(system):
    """Test 5: Kiểm tra tạo signal"""
    try:
        # Create fake market data for testing
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        fake_data = pd.DataFrame({
            'time': dates,
            'open': np.random.uniform(2000, 2100, 100),
            'high': np.random.uniform(2050, 2150, 100),
            'low': np.random.uniform(1950, 2050, 100),
            'close': np.random.uniform(2000, 2100, 100),
            'tick_volume': np.random.randint(100, 1000, 100)
        })
        
        # Test signal generation (should not crash)
        signal = system.generate_signal("XAUUSD")
        
        required_fields = ['symbol', 'action', 'strength', 'prediction', 'confidence', 'timestamp']
        missing_fields = [field for field in required_fields if field not in signal]
        
        if not missing_fields:
            logger.info(f"✅ Test 5 PASSED: Signal generated successfully")
            logger.info(f"   Signal: {signal['action']} {signal['strength']} (pred={signal['prediction']:.3f}, conf={signal['confidence']:.3f})")
            return True
        else:
            logger.error(f"❌ Test 5 FAILED: Missing fields in signal: {missing_fields}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 5 FAILED: Signal generation error - {e}")
        return False

def test_system_status(system):
    """Test 6: Kiểm tra trạng thái hệ thống"""
    try:
        status = system.get_system_status()
        
        if isinstance(status, dict) and 'total_systems' in status:
            logger.info(f"✅ Test 6 PASSED: System status retrieved successfully")
            logger.info(f"   Total systems: {status.get('total_systems', 'N/A')}")
            logger.info(f"   Active systems: {status.get('active_systems', 'N/A')}")
            return True
        else:
            logger.error("❌ Test 6 FAILED: Invalid system status format")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test 6 FAILED: System status error - {e}")
        return False

def run_comprehensive_validation():
    """Chạy tất cả các test validation"""
    logger.info("🚀 BẮT ĐẦU KIỂM TRA TOÀN DIỆN HỆ THỐNG AI3.0")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Imports
    test_results.append(test_system_imports())
    
    # Test 2: Initialization
    init_success, system = test_system_initialization()
    test_results.append(init_success)
    
    if not init_success:
        logger.error("❌ Cannot continue tests - system initialization failed")
        return False
    
    # Test 3: Voting weights
    test_results.append(test_voting_weights(system))
    
    # Test 4: Component wrapper fix
    test_results.append(test_component_wrapper_fix(system))
    
    # Test 5: Signal generation
    test_results.append(test_signal_generation(system))
    
    # Test 6: System status
    test_results.append(test_system_status(system))
    
    # Final results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    logger.info("=" * 60)
    logger.info(f"📊 KẾT QUẢ KIỂM TRA: {passed_tests}/{total_tests} TESTS PASSED")
    
    if passed_tests == total_tests:
        logger.info("🎉 TẤT CẢ TESTS ĐÃ PASS - HỆ THỐNG HOẠT ĐỘNG TỐTI!")
        return True
    else:
        logger.error(f"⚠️ {total_tests - passed_tests} TESTS FAILED - CẦN SỬA LỖI THÊM")
        return False

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1) 