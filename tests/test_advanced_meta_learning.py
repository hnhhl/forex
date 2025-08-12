"""
Test suite for Advanced Meta-Learning System
Ultimate XAU Super System V4.0 - Phase 2 Week 4

Tests for MAML, Transfer Learning, and Continual Learning components
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import the system components
from src.core.ai.advanced_meta_learning import (
    AdvancedMetaLearningSystem,
    MetaLearningConfig,
    MetaLearningResult,
    MAMLLearner,
    TransferLearner,
    ContinualLearner,
    create_meta_learning_system
)

class TestMetaLearningConfig(unittest.TestCase):
    """Test MetaLearningConfig dataclass"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = MetaLearningConfig()
        
        self.assertEqual(config.maml_inner_lr, 0.01)
        self.assertEqual(config.maml_outer_lr, 0.001)
        self.assertEqual(config.input_features, 95)
        self.assertEqual(config.sequence_length, 50)
        self.assertEqual(config.output_units, 3)
        self.assertIsInstance(config.transfer_source_domains, list)
        self.assertEqual(len(config.transfer_source_domains), 4)
    
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        config = MetaLearningConfig(
            maml_inner_lr=0.02,
            continual_memory_size=2000,
            hidden_units=256
        )
        
        self.assertEqual(config.maml_inner_lr, 0.02)
        self.assertEqual(config.continual_memory_size, 2000)
        self.assertEqual(config.hidden_units, 256)

class TestMetaLearningResult(unittest.TestCase):
    """Test MetaLearningResult dataclass"""
    
    def test_result_creation(self):
        """Test result creation and serialization"""
        prediction = np.array([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]])
        
        result = MetaLearningResult(
            prediction=prediction,
            confidence=0.8,
            adaptation_score=0.7,
            transfer_effectiveness=0.6,
            continual_retention=0.9,
            meta_gradient_norm=0.1,
            timestamp=datetime.now()
        )
        
        self.assertIsInstance(result.prediction, np.ndarray)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.adaptation_score, 0.7)
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn('prediction', result_dict)
        self.assertIn('confidence', result_dict)
        self.assertIsInstance(result_dict['prediction'], list)

class TestMAMLLearner(unittest.TestCase):
    """Test MAML (Model-Agnostic Meta-Learning) implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MetaLearningConfig(
            input_features=10,
            sequence_length=5,
            hidden_units=32
        )
        self.maml_learner = MAMLLearner(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.support_x = np.random.randn(20, 5, 10)
        self.support_y = np.eye(3)[np.random.randint(0, 3, 20)]
        self.query_x = np.random.randn(10, 5, 10)
        self.query_y = np.eye(3)[np.random.randint(0, 3, 10)]
    
    def test_maml_initialization(self):
        """Test MAML learner initialization"""
        self.assertIsNotNone(self.maml_learner.meta_model)
        self.assertFalse(self.maml_learner.is_trained)
        self.assertIsNone(self.maml_learner.adapted_model)
        self.assertEqual(len(self.maml_learner.training_history), 0)
    
    def test_meta_model_architecture(self):
        """Test meta-model architecture"""
        model = self.maml_learner.meta_model
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 5, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 3))
        
        # Check that model can make predictions
        test_input = np.random.randn(1, 5, 10)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 3))
    
    def test_adaptation(self):
        """Test MAML adaptation to new task"""
        # First need to set is_trained to True for adaptation
        self.maml_learner.is_trained = True
        
        # Test adaptation
        self.maml_learner.adapt((self.support_x, self.support_y))
        
        # Check that adapted model is created
        self.assertIsNotNone(self.maml_learner.adapted_model)
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training"""
        with self.assertRaises(ValueError):
            self.maml_learner.predict(self.query_x)
    
    def test_prediction_with_meta_model(self):
        """Test prediction using meta-model"""
        # Set as trained
        self.maml_learner.is_trained = True
        
        result = self.maml_learner.predict(self.query_x)
        
        self.assertIsInstance(result, MetaLearningResult)
        self.assertEqual(result.prediction.shape, (10, 3))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.adaptation_score, 0.5)  # No adaptation performed

class TestTransferLearner(unittest.TestCase):
    """Test Transfer Learning implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MetaLearningConfig(
            input_features=10,
            sequence_length=5,
            hidden_units=32
        )
        self.transfer_learner = TransferLearner(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.source_x = np.random.randn(100, 5, 10)
        self.source_y = np.eye(3)[np.random.randint(0, 3, 100)]
        self.target_x = np.random.randn(50, 5, 10)
        self.target_y = np.eye(3)[np.random.randint(0, 3, 50)]
    
    def test_transfer_learner_initialization(self):
        """Test transfer learner initialization"""
        self.assertEqual(len(self.transfer_learner.source_models), 0)
        self.assertIsNone(self.transfer_learner.target_model)
        self.assertFalse(self.transfer_learner.is_trained)
        self.assertEqual(len(self.transfer_learner.transfer_effectiveness_history), 0)
    
    def test_source_domain_training(self):
        """Test training on source domain"""
        result = self.transfer_learner.train_source_domain(
            'EURUSD', 
            (self.source_x, self.source_y)
        )
        
        self.assertIn('EURUSD', self.transfer_learner.source_models)
        self.assertIsInstance(result, dict)
        self.assertIn('domain', result)
        self.assertIn('final_accuracy', result)
        self.assertIn('val_accuracy', result)
        self.assertEqual(result['domain'], 'EURUSD')
    
    def test_transfer_to_target(self):
        """Test transfer to target domain"""
        # First train source domain
        self.transfer_learner.train_source_domain(
            'EURUSD', 
            (self.source_x, self.source_y)
        )
        
        # Transfer to target
        result = self.transfer_learner.transfer_to_target(
            (self.target_x, self.target_y),
            'EURUSD'
        )
        
        self.assertTrue(self.transfer_learner.is_trained)
        self.assertIsNotNone(self.transfer_learner.target_model)
        self.assertIsInstance(result, dict)
        self.assertIn('transfer_effectiveness', result)
        self.assertIn('final_accuracy', result)
        self.assertEqual(len(self.transfer_learner.transfer_effectiveness_history), 1)
    
    def test_transfer_without_source(self):
        """Test transfer fails without source domain"""
        with self.assertRaises(ValueError):
            self.transfer_learner.transfer_to_target(
                (self.target_x, self.target_y),
                'NONEXISTENT'
            )
    
    def test_prediction_after_transfer(self):
        """Test prediction after transfer learning"""
        # Train source and transfer
        self.transfer_learner.train_source_domain(
            'EURUSD', 
            (self.source_x, self.source_y)
        )
        self.transfer_learner.transfer_to_target(
            (self.target_x, self.target_y),
            'EURUSD'
        )
        
        # Test prediction
        test_data = np.random.randn(5, 5, 10)
        result = self.transfer_learner.predict(test_data)
        
        self.assertIsInstance(result, MetaLearningResult)
        self.assertEqual(result.prediction.shape, (5, 3))
        self.assertEqual(result.adaptation_score, 0.7)
        self.assertGreaterEqual(result.transfer_effectiveness, 0.0)

class TestContinualLearner(unittest.TestCase):
    """Test Continual Learning implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MetaLearningConfig(
            input_features=10,
            sequence_length=5,
            hidden_units=32,
            continual_memory_size=100
        )
        self.continual_learner = ContinualLearner(self.config)
        
        # Generate test data for multiple tasks
        np.random.seed(42)
        self.task1_x = np.random.randn(50, 5, 10)
        self.task1_y = np.eye(3)[np.random.randint(0, 3, 50)]
        self.task2_x = np.random.randn(50, 5, 10)
        self.task2_y = np.eye(3)[np.random.randint(0, 3, 50)]
    
    def test_continual_learner_initialization(self):
        """Test continual learner initialization"""
        self.assertIsNotNone(self.continual_learner.model)
        self.assertEqual(len(self.continual_learner.memory_buffer), 0)
        self.assertEqual(len(self.continual_learner.task_history), 0)
        self.assertEqual(len(self.continual_learner.retention_scores), 0)
        self.assertFalse(self.continual_learner.is_trained)
    
    def test_learn_single_task(self):
        """Test learning a single task"""
        result = self.continual_learner.learn_task(
            (self.task1_x, self.task1_y),
            'task1'
        )
        
        self.assertTrue(self.continual_learner.is_trained)
        self.assertGreater(len(self.continual_learner.memory_buffer), 0)
        self.assertEqual(len(self.continual_learner.task_history), 1)
        self.assertEqual(len(self.continual_learner.retention_scores), 1)
        
        self.assertIsInstance(result, dict)
        self.assertIn('task_id', result)
        self.assertIn('retention_score', result)
        self.assertIn('final_accuracy', result)
        self.assertEqual(result['task_id'], 'task1')
    
    def test_learn_multiple_tasks(self):
        """Test learning multiple sequential tasks"""
        # Learn first task
        result1 = self.continual_learner.learn_task(
            (self.task1_x, self.task1_y),
            'task1'
        )
        
        # Learn second task
        result2 = self.continual_learner.learn_task(
            (self.task2_x, self.task2_y),
            'task2'
        )
        
        self.assertEqual(len(self.continual_learner.task_history), 2)
        self.assertEqual(len(self.continual_learner.retention_scores), 2)
        self.assertGreater(len(self.continual_learner.memory_buffer), 0)
        
        # Check that memory buffer doesn't exceed limit
        self.assertLessEqual(
            len(self.continual_learner.memory_buffer),
            self.config.continual_memory_size
        )
    
    def test_memory_buffer_management(self):
        """Test memory buffer size management"""
        # Set small memory size for testing
        self.continual_learner.config.continual_memory_size = 20
        
        # Learn multiple tasks to exceed memory limit
        for i in range(5):
            task_x = np.random.randn(30, 5, 10)
            task_y = np.eye(3)[np.random.randint(0, 3, 30)]
            self.continual_learner.learn_task((task_x, task_y), f'task_{i}')
        
        # Check memory buffer size is within limit
        self.assertLessEqual(
            len(self.continual_learner.memory_buffer),
            20
        )
    
    def test_prediction_after_continual_learning(self):
        """Test prediction after continual learning"""
        # Learn a task
        self.continual_learner.learn_task(
            (self.task1_x, self.task1_y),
            'task1'
        )
        
        # Test prediction
        test_data = np.random.randn(5, 5, 10)
        result = self.continual_learner.predict(test_data)
        
        self.assertIsInstance(result, MetaLearningResult)
        self.assertEqual(result.prediction.shape, (5, 3))
        self.assertEqual(result.adaptation_score, 0.6)
        self.assertGreaterEqual(result.continual_retention, 0.0)

class TestAdvancedMetaLearningSystem(unittest.TestCase):
    """Test the integrated Advanced Meta-Learning System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MetaLearningConfig(
            input_features=10,
            sequence_length=5,
            hidden_units=32
        )
        self.system = AdvancedMetaLearningSystem(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.test_data = np.random.randn(10, 5, 10)
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsInstance(self.system.maml_learner, MAMLLearner)
        self.assertIsInstance(self.system.transfer_learner, TransferLearner)
        self.assertIsInstance(self.system.continual_learner, ContinualLearner)
        
        self.assertIn('initialized', self.system.system_state)
        self.assertTrue(self.system.system_state['initialized'])
        self.assertEqual(self.system.system_state['performance_boost'], 3.5)
    
    def test_ensemble_prediction_no_trained_learners(self):
        """Test ensemble prediction fails with no trained learners"""
        with self.assertRaises(ValueError):
            self.system.ensemble_predict(self.test_data)
    
    def test_ensemble_prediction_with_trained_learners(self):
        """Test ensemble prediction with trained learners"""
        # Train continual learner
        task_x = np.random.randn(50, 5, 10)
        task_y = np.eye(3)[np.random.randint(0, 3, 50)]
        self.system.continual_learner.learn_task((task_x, task_y), 'test_task')
        
        # Test ensemble prediction
        result = self.system.ensemble_predict(self.test_data)
        
        self.assertIsInstance(result, MetaLearningResult)
        self.assertEqual(result.prediction.shape, (10, 3))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_custom_ensemble_weights(self):
        """Test ensemble prediction with custom weights"""
        # Train continual learner
        task_x = np.random.randn(50, 5, 10)
        task_y = np.eye(3)[np.random.randint(0, 3, 50)]
        self.system.continual_learner.learn_task((task_x, task_y), 'test_task')
        
        # Custom weights
        weights = {'maml': 0.0, 'transfer': 0.0, 'continual': 1.0}
        result = self.system.ensemble_predict(self.test_data, weights)
        
        self.assertIsInstance(result, MetaLearningResult)
        # Should only use continual learner
        self.assertEqual(result.adaptation_score, 0.6)
    
    def test_get_system_status(self):
        """Test system status retrieval"""
        status = self.system.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('system_state', status)
        self.assertIn('learner_status', status)
        self.assertIn('performance_history', status)
        self.assertIn('timestamp', status)
        
        # Check learner status structure
        learner_status = status['learner_status']
        self.assertIn('maml', learner_status)
        self.assertIn('transfer', learner_status)
        self.assertIn('continual', learner_status)
    
    def test_export_system_data(self):
        """Test system data export"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.system.export_system_data(temp_path)
            
            self.assertTrue(result['success'])
            self.assertEqual(result['filepath'], temp_path)
            
            # Check file exists and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('config', data)
            self.assertIn('system_status', data)
            self.assertIn('export_timestamp', data)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestFactoryFunction(unittest.TestCase):
    """Test factory function for system creation"""
    
    def test_create_system_default_config(self):
        """Test creating system with default configuration"""
        system = create_meta_learning_system()
        
        self.assertIsInstance(system, AdvancedMetaLearningSystem)
        self.assertEqual(system.config.maml_inner_lr, 0.01)
        self.assertEqual(system.config.input_features, 95)
    
    def test_create_system_custom_config(self):
        """Test creating system with custom configuration"""
        config = {
            'maml_inner_lr': 0.02,
            'hidden_units': 256,
            'continual_memory_size': 2000
        }
        
        system = create_meta_learning_system(config)
        
        self.assertIsInstance(system, AdvancedMetaLearningSystem)
        self.assertEqual(system.config.maml_inner_lr, 0.02)
        self.assertEqual(system.config.hidden_units, 256)
        self.assertEqual(system.config.continual_memory_size, 2000)

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MetaLearningConfig(
            input_features=10,
            sequence_length=5,
            hidden_units=32,
            continual_memory_size=50
        )
        self.system = AdvancedMetaLearningSystem(self.config)
    
    def test_transfer_learning_workflow(self):
        """Test complete transfer learning workflow"""
        # Generate source and target data
        np.random.seed(42)
        source_x = np.random.randn(100, 5, 10)
        source_y = np.eye(3)[np.random.randint(0, 3, 100)]
        target_x = np.random.randn(50, 5, 10)
        target_y = np.eye(3)[np.random.randint(0, 3, 50)]
        
        # Train source domain
        source_result = self.system.transfer_learner.train_source_domain(
            'EURUSD', (source_x, source_y)
        )
        self.assertIn('final_accuracy', source_result)
        
        # Transfer to target
        transfer_result = self.system.transfer_learner.transfer_to_target(
            (target_x, target_y), 'EURUSD'
        )
        self.assertIn('transfer_effectiveness', transfer_result)
        
        # Make prediction
        test_data = np.random.randn(5, 5, 10)
        prediction = self.system.transfer_learner.predict(test_data)
        self.assertEqual(prediction.prediction.shape, (5, 3))
    
    def test_continual_learning_workflow(self):
        """Test complete continual learning workflow"""
        # Generate sequential tasks
        np.random.seed(42)
        tasks = []
        for i in range(3):
            task_x = np.random.randn(50, 5, 10)
            task_y = np.eye(3)[np.random.randint(0, 3, 50)]
            tasks.append((task_x, task_y, f'task_{i}'))
        
        # Learn tasks sequentially
        results = []
        for task_x, task_y, task_name in tasks:
            result = self.system.continual_learner.learn_task(
                (task_x, task_y), task_name
            )
            results.append(result)
        
        # Check retention scores
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('retention_score', result)
            self.assertGreaterEqual(result['retention_score'], 0.0)
            self.assertLessEqual(result['retention_score'], 1.0)
        
        # Make prediction
        test_data = np.random.randn(5, 5, 10)
        prediction = self.system.continual_learner.predict(test_data)
        self.assertEqual(prediction.prediction.shape, (5, 3))
    
    def test_mixed_learning_scenario(self):
        """Test scenario with both transfer and continual learning"""
        np.random.seed(42)
        
        # Transfer learning setup
        source_x = np.random.randn(80, 5, 10)
        source_y = np.eye(3)[np.random.randint(0, 3, 80)]
        target_x = np.random.randn(40, 5, 10)
        target_y = np.eye(3)[np.random.randint(0, 3, 40)]
        
        # Continual learning setup
        task_x = np.random.randn(60, 5, 10)
        task_y = np.eye(3)[np.random.randint(0, 3, 60)]
        
        # Execute transfer learning
        self.system.transfer_learner.train_source_domain(
            'EURUSD', (source_x, source_y)
        )
        self.system.transfer_learner.transfer_to_target(
            (target_x, target_y), 'EURUSD'
        )
        
        # Execute continual learning
        self.system.continual_learner.learn_task(
            (task_x, task_y), 'mixed_task'
        )
        
        # Test ensemble prediction
        test_data = np.random.randn(5, 5, 10)
        ensemble_result = self.system.ensemble_predict(test_data)
        
        self.assertEqual(ensemble_result.prediction.shape, (5, 3))
        # Transfer effectiveness can be 0.0 in some cases, so we check >= 0.0
        self.assertGreaterEqual(ensemble_result.transfer_effectiveness, 0.0)
        self.assertGreater(ensemble_result.continual_retention, 0.0)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMetaLearningConfig,
        TestMetaLearningResult,
        TestMAMLLearner,
        TestTransferLearner,
        TestContinualLearner,
        TestAdvancedMetaLearningSystem,
        TestFactoryFunction,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"\n🚀 Advanced Meta-Learning System testing completed!")