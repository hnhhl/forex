#!/usr/bin/env python3
"""
Test Script for Unified AI3.0 Ultimate XAU System
Validates integration of Training System into Main System with Unified Logic
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any

# Add src to path
sys.path.append('src')

# Import unified components
from core.shared.unified_feature_engine import UnifiedFeatureEngine
from core.shared.unified_model_architecture import UnifiedModelArchitecture
from core.shared.unified_prediction_logic import UnifiedPredictionLogic

# Import main system
from core.ultimate_xau_system import UltimateXAUSystem, SystemConfig, IntegratedTrainingSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedSystemTester:
    """Test suite for unified system integration"""
    
    def __init__(self):
        self.test_results = {}
        self.logger = logging.getLogger(f"{__name__}.UnifiedSystemTester")
        
    def run_all_tests(self) -> Dict:
        """Run all integration tests"""
        print("\n🧪 UNIFIED SYSTEM INTEGRATION TESTS")
        print("=" * 50)
        
        tests = [
            ("Feature Engine Test", self.test_unified_feature_engine),
            ("Model Architecture Test", self.test_unified_model_architecture),
            ("Prediction Logic Test", self.test_unified_prediction_logic),
            ("Training System Test", self.test_integrated_training_system),
            ("Main System Test", self.test_main_system_integration),
            ("End-to-End Test", self.test_end_to_end_workflow)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 {test_name}...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
                print(f"   {status}")
                if not result.get('success', False):
                    print(f"   Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                self.test_results[test_name] = {'success': False, 'error': str(e)}
                print(f"   ❌ FAILED - Exception: {e}")
        
        # Summary
        self._print_test_summary()
        return self.test_results
    
    def test_unified_feature_engine(self) -> Dict:
        """Test unified feature engine"""
        try:
            engine = UnifiedFeatureEngine()
            
            # Test feature info
            feature_info = engine.get_feature_info()
            assert feature_info['feature_count'] == 19, f"Expected 19 features, got {feature_info['feature_count']}"
            
            # Test with sample data
            sample_data = self._create_sample_data()
            features_df = engine.prepare_features_from_dataframe(sample_data)
            
            # Validate output
            assert len(features_df.columns) == 19, f"Expected 19 features, got {len(features_df.columns)}"
            assert len(features_df) > 0, "Features dataframe is empty"
            
            # Test feature validation
            features_array = features_df.iloc[-1].values
            assert engine.validate_features(features_array), "Feature validation failed"
            
            return {
                'success': True,
                'features_created': len(features_df.columns),
                'sample_features': features_array.tolist()[:5],  # First 5 features
                'feature_names': engine.get_feature_names()[:5]
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_unified_model_architecture(self) -> Dict:
        """Test unified model architecture"""
        try:
            architecture = UnifiedModelArchitecture()
            
            # Test model creation
            models_created = {}
            
            for model_type in ['dense', 'lstm', 'cnn', 'hybrid']:
                try:
                    model = architecture.create_model(model_type)
                    
                    # Validate model
                    assert model is not None, f"{model_type} model is None"
                    
                    # Check input shape compatibility
                    input_shape = model.input_shape
                    if model_type == 'dense':
                        assert input_shape[-1] == 19, f"Dense model expects 19 features, got {input_shape[-1]}"
                    else:
                        assert input_shape[-1] == 19, f"{model_type} model expects 19 features, got {input_shape[-1]}"
                    
                    models_created[model_type] = {
                        'success': True,
                        'input_shape': input_shape,
                        'output_shape': model.output_shape,
                        'parameters': model.count_params()
                    }
                    
                except Exception as e:
                    models_created[model_type] = {'success': False, 'error': str(e)}
            
            # Check if at least one model was created successfully
            successful_models = [k for k, v in models_created.items() if v.get('success', False)]
            
            return {
                'success': len(successful_models) > 0,
                'models_tested': len(models_created),
                'successful_models': len(successful_models),
                'models_created': models_created
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_unified_prediction_logic(self) -> Dict:
        """Test unified prediction logic"""
        try:
            prediction_logic = UnifiedPredictionLogic()
            
            # Create a simple test model
            architecture = UnifiedModelArchitecture()
            test_model = architecture.create_model('dense')
            
            # Create test features
            test_features = np.random.rand(19).astype(np.float32)
            
            # Test prediction processing
            prediction_result = prediction_logic.process_model_prediction(
                test_model, test_features, 'dense'
            )
            
            # Validate prediction result
            assert prediction_result.get('processing_status') == 'success', "Prediction processing failed"
            assert 'prediction_value' in prediction_result, "Missing prediction_value"
            assert 'confidence' in prediction_result, "Missing confidence"
            assert 'signal_type' in prediction_result, "Missing signal_type"
            
            # Test trading signal creation
            trading_signal = prediction_logic.create_trading_signal(
                prediction_result, current_price=2000.0, symbol="XAUUSD"
            )
            
            # Validate trading signal
            assert 'action' in trading_signal, "Missing action in trading signal"
            assert 'confidence' in trading_signal, "Missing confidence in trading signal"
            assert 'stop_loss' in trading_signal, "Missing stop_loss in trading signal"
            assert 'take_profit' in trading_signal, "Missing take_profit in trading signal"
            
            return {
                'success': True,
                'prediction_result': {
                    'prediction_value': prediction_result['prediction_value'],
                    'confidence': prediction_result['confidence'],
                    'signal_type': prediction_result['signal_type']
                },
                'trading_signal': {
                    'action': trading_signal['action'],
                    'confidence': trading_signal['confidence'],
                    'price': trading_signal['price']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_integrated_training_system(self) -> Dict:
        """Test integrated training system"""
        try:
            config = SystemConfig()
            config.min_training_data_points = 100  # Lower for testing
            
            training_system = IntegratedTrainingSystem(config)
            
            # Test data collection
            sample_market_data = {
                'open': 2000.0,
                'high': 2010.0,
                'low': 1990.0,
                'close': 2005.0,
                'volume': 1000.0
            }
            
            # Collect multiple data points
            for i in range(150):  # Collect enough data for training
                market_data = sample_market_data.copy()
                market_data['close'] = 2000.0 + np.random.uniform(-20, 20)
                training_system.collect_training_data(market_data)
            
            # Test training status
            status = training_system.get_training_status()
            assert status['training_data_points'] >= 100, f"Insufficient training data: {status['training_data_points']}"
            
            # Test should_retrain
            should_retrain = training_system.should_retrain()
            assert isinstance(should_retrain, bool), "should_retrain should return boolean"
            
            return {
                'success': True,
                'training_data_points': status['training_data_points'],
                'should_retrain': should_retrain,
                'status': status
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_main_system_integration(self) -> Dict:
        """Test main system with integrated training"""
        try:
            # Create system with training enabled
            config = SystemConfig()
            config.enable_integrated_training = True
            
            system = UltimateXAUSystem(config)
            
            # Test system initialization
            assert system.feature_engine is not None, "Feature engine not initialized"
            assert system.model_architecture is not None, "Model architecture not initialized"
            assert system.training_system is not None, "Training system not initialized"
            
            # Test feature generation
            features = system.get_market_features()
            assert len(features) == 19, f"Expected 19 features, got {len(features)}"
            
            # Test signal generation
            signal = system.generate_signal()
            assert 'action' in signal, "Missing action in signal"
            assert 'confidence' in signal, "Missing confidence in signal"
            assert 'features_used' in signal, "Missing features_used in signal"
            assert signal['features_used'] == 19, f"Expected 19 features used, got {signal['features_used']}"
            
            # Test system status
            status = system.get_system_status()
            assert status['unified_architecture'] == True, "Unified architecture not enabled"
            assert status['feature_engine'] == 'unified_19_features', "Wrong feature engine"
            
            return {
                'success': True,
                'features_shape': features.shape,
                'signal': {
                    'action': signal['action'],
                    'confidence': signal['confidence'],
                    'features_used': signal['features_used']
                },
                'system_status': {
                    'unified_architecture': status['unified_architecture'],
                    'feature_engine': status['feature_engine']
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_end_to_end_workflow(self) -> Dict:
        """Test complete end-to-end workflow"""
        try:
            # 1. Initialize system
            config = SystemConfig()
            config.enable_integrated_training = True
            config.min_training_data_points = 50  # Lower for testing
            
            system = UltimateXAUSystem(config)
            
            # 2. Simulate market data collection
            for i in range(60):
                # This happens automatically in get_market_features if training is enabled
                features = system.get_market_features()
            
            # 3. Check training data collection
            training_status = system.get_training_status()
            data_points = training_status.get('training_data_points', 0)
            
            # 4. Generate signals
            signals = []
            for i in range(5):
                signal = system.generate_signal()
                signals.append(signal)
            
            # 5. Validate workflow
            assert len(signals) == 5, f"Expected 5 signals, got {len(signals)}"
            assert all('features_used' in s for s in signals), "Not all signals have features_used"
            assert all(s['features_used'] == 19 for s in signals), "Not all signals use 19 features"
            
            # 6. Test system health
            system_status = system.get_system_status()
            
            return {
                'success': True,
                'workflow_steps': {
                    'system_initialized': True,
                    'data_collected': data_points,
                    'signals_generated': len(signals),
                    'unified_features': all(s['features_used'] == 19 for s in signals)
                },
                'final_status': system_status
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing"""
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Generate realistic price data
        base_price = 2000.0
        price_changes = np.random.normal(0, 5, 100)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] + change
            prices.append(max(1800, min(2200, new_price)))  # Keep within reasonable range
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price + np.random.uniform(0, 10)
            low = price - np.random.uniform(0, 10)
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(500, 2000)
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df
    
    def _print_test_summary(self):
        """Print test summary"""
        print("\n📊 TEST SUMMARY")
        print("=" * 30)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if not result.get('success', False):
                    print(f"   • {test_name}: {result.get('error', 'Unknown error')}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"unified_system_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\n💾 Test results saved: {results_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save results: {e}")


def main():
    """Main test function"""
    print("🚀 Starting Unified System Integration Tests...")
    
    tester = UnifiedSystemTester()
    results = tester.run_all_tests()
    
    # Overall result
    total_success = all(result.get('success', False) for result in results.values())
    
    if total_success:
        print("\n🎉 ALL TESTS PASSED! Unified system integration successful!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the results above.")
        return 1


if __name__ == "__main__":
    exit(main()) 