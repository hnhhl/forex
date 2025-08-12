"""
Test suite for Day 22 Advanced Pattern Recognition
Ultimate XAU Super System V4.0

Comprehensive testing of:
- Advanced pattern detection algorithms
- Machine learning pattern classification
- Real-time alert system
- Performance tracking
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add source path
sys.path.append('src')

try:
    from src.core.analysis.advanced_pattern_recognition import (
        PatternConfig, AdvancedPattern, PatternAlert,
        AdvancedPatternDetector, MachineLearningPatternClassifier,
        RealTimePatternAlerter, AdvancedPatternRecognition,
        create_advanced_pattern_recognition
    )
    ADVANCED_PATTERN_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERN_AVAILABLE = False


class TestPatternConfig(unittest.TestCase):
    """Test PatternConfig class"""
    
    def test_pattern_config_initialization(self):
        """Test PatternConfig initialization with default values"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        config = PatternConfig()
        
        self.assertEqual(config.min_pattern_length, 20)
        self.assertEqual(config.max_pattern_length, 100)
        self.assertEqual(config.pattern_similarity_threshold, 0.85)
        self.assertTrue(config.use_ml_classification)
        self.assertTrue(config.enable_performance_tracking)
        self.assertTrue(config.enable_real_time_alerts)
        self.assertEqual(config.alert_confidence_threshold, 0.7)
    
    def test_pattern_config_custom_values(self):
        """Test PatternConfig with custom values"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        config = PatternConfig(
            min_pattern_length=30,
            max_pattern_length=150,
            pattern_similarity_threshold=0.9,
            use_ml_classification=False,
            enable_performance_tracking=False,
            enable_real_time_alerts=False,
            alert_confidence_threshold=0.8
        )
        
        self.assertEqual(config.min_pattern_length, 30)
        self.assertEqual(config.max_pattern_length, 150)
        self.assertEqual(config.pattern_similarity_threshold, 0.9)
        self.assertFalse(config.use_ml_classification)
        self.assertFalse(config.enable_performance_tracking)
        self.assertFalse(config.enable_real_time_alerts)
        self.assertEqual(config.alert_confidence_threshold, 0.8)


class TestAdvancedPattern(unittest.TestCase):
    """Test AdvancedPattern class"""
    
    def test_advanced_pattern_creation(self):
        """Test AdvancedPattern creation"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        pattern = AdvancedPattern(
            pattern_id="test_pattern_1",
            pattern_name="Test Pattern",
            pattern_type="BULLISH",
            confidence=0.85,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            pattern_data=np.array([100, 101, 102, 103]),
            ml_features={"test_feature": 0.5},
            classification_score=0.8,
            performance_metrics={"success_rate": 0.75},
            target_price=105.0,
            stop_loss=98.0
        )
        
        self.assertEqual(pattern.pattern_id, "test_pattern_1")
        self.assertEqual(pattern.pattern_name, "Test Pattern")
        self.assertEqual(pattern.pattern_type, "BULLISH")
        self.assertEqual(pattern.confidence, 0.85)
        self.assertEqual(pattern.target_price, 105.0)
        self.assertEqual(pattern.stop_loss, 98.0)
        self.assertIn("test_feature", pattern.ml_features)


class TestPatternAlert(unittest.TestCase):
    """Test PatternAlert class"""
    
    def test_pattern_alert_creation(self):
        """Test PatternAlert creation"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create a mock pattern first
        pattern = AdvancedPattern(
            pattern_id="test_pattern",
            pattern_name="Test Pattern",
            pattern_type="BULLISH",
            confidence=0.8,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            pattern_data=np.array([100, 101, 102]),
            ml_features={},
            classification_score=0.7,
            performance_metrics={}
        )
        
        alert = PatternAlert(
            alert_id="test_alert_1",
            pattern=pattern,
            alert_time=datetime.now(),
            alert_type="FORMATION",
            price_at_alert=102.5,
            recommended_action="BUY",
            urgency_level="HIGH"
        )
        
        self.assertEqual(alert.alert_id, "test_alert_1")
        self.assertEqual(alert.alert_type, "FORMATION")
        self.assertEqual(alert.price_at_alert, 102.5)
        self.assertEqual(alert.recommended_action, "BUY")
        self.assertEqual(alert.urgency_level, "HIGH")
        self.assertEqual(alert.pattern.pattern_name, "Test Pattern")


class TestAdvancedPatternDetector(unittest.TestCase):
    """Test AdvancedPatternDetector class"""
    
    def setUp(self):
        """Set up test data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.config = PatternConfig()
        self.detector = AdvancedPatternDetector(self.config)
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        np.random.seed(42)
        
        prices = []
        base_price = 2000.0
        for i in range(100):
            if i < 30:
                change = np.random.normal(0.001, 0.005)
            elif i < 60:
                # Create triangle pattern
                change = np.random.normal(0.002, 0.003) if i % 2 == 0 else np.random.normal(-0.001, 0.003)
            else:
                change = np.random.normal(0, 0.008)
            
            new_price = base_price * (1 + change) if i == 0 else prices[-1] * (1 + change)
            prices.append(new_price)
        
        self.sample_data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.assertIsInstance(self.detector.config, PatternConfig)
        self.assertEqual(len(self.detector.pattern_history), 0)
    
    def test_detect_triangular_patterns(self):
        """Test triangular pattern detection"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        patterns = self.detector.detect_triangular_patterns(self.sample_data)
        
        # Should be able to detect patterns or return empty list
        self.assertIsInstance(patterns, list)
        
        # If patterns found, verify structure
        for pattern in patterns:
            self.assertIsInstance(pattern, AdvancedPattern)
            self.assertIn(pattern.pattern_type, ['BULLISH', 'BEARISH', 'NEUTRAL'])
            self.assertGreaterEqual(pattern.confidence, 0)
            self.assertLessEqual(pattern.confidence, 1)
    
    def test_detect_flag_pennant_patterns(self):
        """Test flag and pennant pattern detection"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        patterns = self.detector.detect_flag_pennant_patterns(self.sample_data)
        
        self.assertIsInstance(patterns, list)
        
        for pattern in patterns:
            self.assertIsInstance(pattern, AdvancedPattern)
            self.assertEqual(pattern.pattern_name, "Flag Pattern")
            self.assertIn(pattern.pattern_type, ['BULLISH', 'BEARISH'])
    
    def test_detect_harmonic_patterns(self):
        """Test harmonic pattern detection"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        patterns = self.detector.detect_harmonic_patterns(self.sample_data)
        
        self.assertIsInstance(patterns, list)
        
        for pattern in patterns:
            self.assertIsInstance(pattern, AdvancedPattern)
            self.assertIn("Pattern", pattern.pattern_name)
            self.assertIn(pattern.pattern_type, ['BULLISH', 'BEARISH'])
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create very small dataset
        small_data = self.sample_data.iloc[:5]
        
        triangular_patterns = self.detector.detect_triangular_patterns(small_data)
        flag_patterns = self.detector.detect_flag_pennant_patterns(small_data)
        harmonic_patterns = self.detector.detect_harmonic_patterns(small_data)
        
        # Should return empty lists for insufficient data
        self.assertEqual(len(triangular_patterns), 0)
        self.assertEqual(len(flag_patterns), 0)
        self.assertEqual(len(harmonic_patterns), 0)


class TestMachineLearningPatternClassifier(unittest.TestCase):
    """Test MachineLearningPatternClassifier class"""
    
    def setUp(self):
        """Set up test data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.classifier = MachineLearningPatternClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.assertFalse(self.classifier.is_trained)
        self.assertEqual(len(self.classifier.pattern_clusters), 0)
    
    def test_extract_features(self):
        """Test feature extraction"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Test with normal data
        pattern_data = np.array([100, 101, 102, 103, 104, 105])
        features = self.classifier.extract_features(pattern_data)
        
        self.assertEqual(len(features), 20)  # Should return 20 features
        self.assertIsInstance(features, np.ndarray)
        
        # Test with insufficient data
        small_data = np.array([100, 101])
        features_small = self.classifier.extract_features(small_data)
        
        self.assertEqual(len(features_small), 20)  # Should still return 20 features (zeros)
    
    def test_train_classifier_insufficient_data(self):
        """Test training with insufficient data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create few patterns (less than 10)
        patterns = []
        for i in range(5):
            pattern = AdvancedPattern(
                pattern_id=f"pattern_{i}",
                pattern_name="Test Pattern",
                pattern_type="BULLISH",
                confidence=0.8,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                pattern_data=np.random.rand(20),
                ml_features={},
                classification_score=0.7,
                performance_metrics={}
            )
            patterns.append(pattern)
        
        result = self.classifier.train_classifier(patterns)
        self.assertFalse(result)  # Should fail with insufficient data
        self.assertFalse(self.classifier.is_trained)
    
    def test_classify_pattern_untrained(self):
        """Test pattern classification when untrained"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        pattern_data = np.array([100, 101, 102, 103, 104])
        pattern_type, confidence = self.classifier.classify_pattern(pattern_data)
        
        self.assertEqual(pattern_type, "UNKNOWN")
        self.assertEqual(confidence, 0.5)


class TestRealTimePatternAlerter(unittest.TestCase):
    """Test RealTimePatternAlerter class"""
    
    def setUp(self):
        """Set up test data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.config = PatternConfig()
        self.alerter = RealTimePatternAlerter(self.config)
    
    def test_alerter_initialization(self):
        """Test alerter initialization"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.assertEqual(len(self.alerter.active_alerts), 0)
        self.assertEqual(len(self.alerter.alert_history), 0)
    
    def test_check_for_alerts(self):
        """Test alert checking"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create high-confidence pattern
        pattern = AdvancedPattern(
            pattern_id="test_pattern",
            pattern_name="Test Pattern",
            pattern_type="BULLISH",
            confidence=0.8,  # Above threshold
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            pattern_data=np.array([100, 101, 102]),
            ml_features={},
            classification_score=0.7,
            performance_metrics={},
            key_levels=[100, 105]
        )
        
        patterns = [pattern]
        current_price = 103.0
        
        alerts = self.alerter.check_for_alerts(patterns, current_price)
        
        self.assertIsInstance(alerts, list)
        if alerts:  # If alerts generated
            for alert in alerts:
                self.assertIsInstance(alert, PatternAlert)
                self.assertEqual(alert.price_at_alert, current_price)
    
    def test_no_alerts_low_confidence(self):
        """Test no alerts generated for low confidence patterns"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create low-confidence pattern
        pattern = AdvancedPattern(
            pattern_id="test_pattern",
            pattern_name="Test Pattern",
            pattern_type="BULLISH",
            confidence=0.5,  # Below threshold
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            pattern_data=np.array([100, 101, 102]),
            ml_features={},
            classification_score=0.5,
            performance_metrics={}
        )
        
        patterns = [pattern]
        current_price = 103.0
        
        alerts = self.alerter.check_for_alerts(patterns, current_price)
        
        # Should not generate alerts for low confidence
        self.assertEqual(len(alerts), 0)


class TestAdvancedPatternRecognition(unittest.TestCase):
    """Test main AdvancedPatternRecognition class"""
    
    def setUp(self):
        """Set up test data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.config = PatternConfig()
        self.recognizer = AdvancedPatternRecognition(self.config)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        np.random.seed(42)
        
        prices = [2000.0]
        for i in range(49):
            change = np.random.normal(0.001, 0.008)
            prices.append(prices[-1] * (1 + change))
        
        self.sample_data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
    
    def test_recognizer_initialization(self):
        """Test recognizer initialization"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        self.assertIsInstance(self.recognizer.detector, AdvancedPatternDetector)
        self.assertIsInstance(self.recognizer.ml_classifier, MachineLearningPatternClassifier)
        self.assertIsInstance(self.recognizer.alerter, RealTimePatternAlerter)
        self.assertEqual(len(self.recognizer.pattern_performance), 0)
    
    def test_analyze_patterns_basic(self):
        """Test basic pattern analysis"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        results = self.recognizer.analyze_patterns(self.sample_data, current_price=2010.0)
        
        self.assertIsInstance(results, dict)
        self.assertIn('timestamp', results)
        self.assertIn('data_points', results)
        self.assertIn('patterns', results)
        self.assertIn('ml_classifications', results)
        self.assertIn('alerts', results)
        self.assertIn('performance_summary', results)
        self.assertIn('recommendations', results)
        
        self.assertEqual(results['data_points'], len(self.sample_data))
        self.assertIsInstance(results['patterns'], list)
        self.assertIsInstance(results['ml_classifications'], list)
        self.assertIsInstance(results['alerts'], list)
        self.assertIsInstance(results['recommendations'], list)
    
    def test_analyze_patterns_insufficient_data(self):
        """Test pattern analysis with insufficient data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create very small dataset
        small_data = self.sample_data.iloc[:10]
        
        with self.assertRaises(ValueError):
            self.recognizer.analyze_patterns(small_data)
    
    def test_analyze_patterns_empty_data(self):
        """Test pattern analysis with empty data"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.recognizer.analyze_patterns(empty_data)
    
    def test_train_ml_classifier(self):
        """Test ML classifier training"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create training patterns
        patterns = []
        for i in range(15):  # Sufficient for training
            pattern = AdvancedPattern(
                pattern_id=f"pattern_{i}",
                pattern_name="Training Pattern",
                pattern_type="BULLISH" if i % 2 == 0 else "BEARISH",
                confidence=0.8,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                pattern_data=np.random.rand(30),
                ml_features={},
                classification_score=0.7,
                performance_metrics={}
            )
            patterns.append(pattern)
        
        result = self.recognizer.train_ml_classifier(patterns)
        self.assertTrue(result)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    def test_create_advanced_pattern_recognition_default(self):
        """Test factory function with default config"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        recognizer = create_advanced_pattern_recognition()
        
        self.assertIsInstance(recognizer, AdvancedPatternRecognition)
        self.assertEqual(recognizer.config.min_pattern_length, 20)
        self.assertEqual(recognizer.config.max_pattern_length, 100)
    
    def test_create_advanced_pattern_recognition_custom(self):
        """Test factory function with custom config"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        custom_config = {
            'min_pattern_length': 30,
            'max_pattern_length': 150,
            'pattern_similarity_threshold': 0.9,
            'use_ml_classification': False
        }
        
        recognizer = create_advanced_pattern_recognition(custom_config)
        
        self.assertIsInstance(recognizer, AdvancedPatternRecognition)
        self.assertEqual(recognizer.config.min_pattern_length, 30)
        self.assertEqual(recognizer.config.max_pattern_length, 150)
        self.assertEqual(recognizer.config.pattern_similarity_threshold, 0.9)
        self.assertFalse(recognizer.config.use_ml_classification)


class TestPerformanceAndBenchmarks(unittest.TestCase):
    """Test performance and benchmarks"""
    
    def test_analysis_performance(self):
        """Test analysis performance benchmarks"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        # Create larger dataset for performance testing
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        np.random.seed(42)
        
        prices = [2000.0]
        for i in range(199):
            change = np.random.normal(0.001, 0.008)
            prices.append(prices[-1] * (1 + change))
        
        large_data = pd.DataFrame({
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        recognizer = create_advanced_pattern_recognition()
        
        import time
        start_time = time.time()
        results = recognizer.analyze_patterns(large_data, current_price=prices[-1])
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = len(large_data) / duration
        
        # Performance benchmarks
        self.assertLess(duration, 10.0)  # Should complete in under 10 seconds
        self.assertGreater(throughput, 10)  # Should process at least 10 data points per second
        
        # Results validation
        self.assertIsInstance(results, dict)
        self.assertEqual(results['data_points'], len(large_data))
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        if not ADVANCED_PATTERN_AVAILABLE:
            self.skipTest("Advanced Pattern Recognition not available")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and analyze multiple datasets
        for i in range(5):
            dates = pd.date_range('2024-01-01', periods=100, freq='h')
            prices = [2000.0 + np.random.rand() * 100 for _ in range(100)]
            
            data = pd.DataFrame({
                'open': [p * 0.999 for p in prices],
                'high': [p * 1.005 for p in prices],
                'low': [p * 0.995 for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            recognizer = create_advanced_pattern_recognition()
            recognizer.analyze_patterns(data, current_price=prices[-1])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase excessively (under 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Day 22 Advanced Pattern Recognition Tests...")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPatternConfig,
        TestAdvancedPattern,
        TestPatternAlert,
        TestAdvancedPatternDetector,
        TestMachineLearningPatternClassifier,
        TestRealTimePatternAlerter,
        TestAdvancedPatternRecognition,
        TestFactoryFunction,
        TestPerformanceAndBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"📊 TEST SUMMARY:")
    print(f"   Total Tests: {result.testsRun}")
    print(f"   Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   Failed: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, trace in result.failures:
            print(f"   {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, trace in result.errors:
            print(f"   {test}: {trace.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n🏆 OVERALL RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    return result


if __name__ == "__main__":
    # Check if advanced pattern recognition is available
    if not ADVANCED_PATTERN_AVAILABLE:
        print("❌ Advanced Pattern Recognition module not available")
        print("   Please ensure src/core/analysis/advanced_pattern_recognition.py exists")
        sys.exit(1)
    
    # Run comprehensive tests
    result = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if len(result.failures) == 0 and len(result.errors) == 0 else 1)