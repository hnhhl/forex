"""
Test Suite for AI Master Integration System
Ultimate XAU Super System V4.0 - Day 18 Implementation

Comprehensive testing of:
- AI Master Integration System
- Multi-AI ensemble decision making
- Performance tracking and optimization
- Decision strategies
"""

import unittest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the system to test
from src.core.integration.ai_master_integration import (
    AIMasterIntegrationSystem, AISystemConfig, AIMarketData, AIPrediction,
    EnsembleDecision, DecisionStrategy, AISystemType, AIPerformanceTracker,
    create_ai_master_system
)


class TestAISystemConfig(unittest.TestCase):
    """Test AISystemConfig dataclass"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = AISystemConfig()
        
        self.assertTrue(config.enable_neural_ensemble)
        self.assertTrue(config.enable_reinforcement_learning)
        self.assertTrue(config.enable_meta_learning)
        self.assertEqual(config.decision_strategy, DecisionStrategy.ADAPTIVE_ENSEMBLE)
        self.assertEqual(config.min_confidence_threshold, 0.6)
        self.assertEqual(config.max_position_size, 0.25)
        self.assertEqual(config.sequence_length, 50)
        self.assertEqual(config.input_features, 95)
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = AISystemConfig(
            enable_neural_ensemble=False,
            decision_strategy=DecisionStrategy.MAJORITY_VOTING,
            min_confidence_threshold=0.8,
            max_position_size=0.3,
            sequence_length=30,
            input_features=100
        )
        
        self.assertFalse(config.enable_neural_ensemble)
        self.assertEqual(config.decision_strategy, DecisionStrategy.MAJORITY_VOTING)
        self.assertEqual(config.min_confidence_threshold, 0.8)
        self.assertEqual(config.max_position_size, 0.3)
        self.assertEqual(config.sequence_length, 30)
        self.assertEqual(config.input_features, 100)


class TestAIMarketData(unittest.TestCase):
    """Test AIMarketData structure"""
    
    def setUp(self):
        """Set up test data"""
        self.market_data = AIMarketData(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            price=2000.0,
            high=2010.0,
            low=1990.0,
            volume=5000.0,
            sma_20=1995.0,
            sma_50=1985.0,
            rsi=65.0,
            macd=2.5,
            volatility=0.15,
            momentum=0.05
        )
    
    def test_market_data_creation(self):
        """Test market data creation"""
        self.assertEqual(self.market_data.symbol, "XAUUSD")
        self.assertEqual(self.market_data.price, 2000.0)
        self.assertEqual(self.market_data.volume, 5000.0)
        self.assertEqual(self.market_data.rsi, 65.0)
    
    def test_to_feature_vector(self):
        """Test feature vector conversion"""
        features = self.market_data.to_feature_vector()
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 95)  # Expected feature size
        self.assertEqual(features[0], 2000.0)  # Price should be first feature


class TestAIPrediction(unittest.TestCase):
    """Test AIPrediction structure"""
    
    def setUp(self):
        """Set up test prediction"""
        self.prediction = AIPrediction(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            source=AISystemType.NEURAL_ENSEMBLE,
            action="BUY",
            confidence=0.85,
            probability_distribution=np.array([0.1, 0.1, 0.8]),
            recommended_position_size=0.2,
            risk_score=0.3,
            expected_return=0.05,
            uncertainty=0.15
        )
    
    def test_prediction_creation(self):
        """Test prediction creation"""
        self.assertEqual(self.prediction.action, "BUY")
        self.assertEqual(self.prediction.confidence, 0.85)
        self.assertEqual(self.prediction.source, AISystemType.NEURAL_ENSEMBLE)
        self.assertEqual(len(self.prediction.probability_distribution), 3)
    
    def test_prediction_validation(self):
        """Test prediction data validation"""
        self.assertGreaterEqual(self.prediction.confidence, 0.0)
        self.assertLessEqual(self.prediction.confidence, 1.0)
        self.assertIn(self.prediction.action, ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(self.prediction.recommended_position_size, 0.0)


class TestAIPerformanceTracker(unittest.TestCase):
    """Test AIPerformanceTracker"""
    
    def setUp(self):
        """Set up performance tracker"""
        self.tracker = AIPerformanceTracker(window_size=50)
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.window_size, 50)
        self.assertEqual(len(self.tracker.predictions_history), 0)
        self.assertIn(AISystemType.NEURAL_ENSEMBLE, self.tracker.system_weights)
    
    def test_add_prediction(self):
        """Test adding predictions to tracker"""
        prediction = AIPrediction(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            source=AISystemType.NEURAL_ENSEMBLE,
            action="BUY",
            confidence=0.8,
            probability_distribution=np.array([0.1, 0.1, 0.8]),
            recommended_position_size=0.2,
            risk_score=0.3,
            expected_return=0.05,
            uncertainty=0.2
        )
        
        self.tracker.add_prediction(prediction, 0.02)
        self.assertEqual(len(self.tracker.predictions_history), 1)
        self.assertEqual(self.tracker.predictions_history[0]['actual_outcome'], 0.02)
    
    def test_calculate_system_performance(self):
        """Test system performance calculation"""
        # Add multiple predictions
        for i in range(20):
            prediction = AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.NEURAL_ENSEMBLE,
                action="BUY" if i % 2 == 0 else "SELL",
                confidence=0.7 + i * 0.01,
                probability_distribution=np.array([0.1, 0.1, 0.8]),
                recommended_position_size=0.2,
                risk_score=0.3,
                expected_return=0.05,
                uncertainty=0.2
            )
            outcome = 0.01 if i % 2 == 0 else -0.01  # Match BUY/SELL with positive/negative
            self.tracker.add_prediction(prediction, outcome)
        
        performance = self.tracker.calculate_system_performance()
        
        self.assertIn(AISystemType.NEURAL_ENSEMBLE, performance)
        self.assertIn('accuracy', performance[AISystemType.NEURAL_ENSEMBLE])
        self.assertIn('avg_confidence', performance[AISystemType.NEURAL_ENSEMBLE])
    
    def test_update_system_weights(self):
        """Test system weight updates"""
        initial_weights = self.tracker.system_weights.copy()
        
        # Add some performance data
        for i in range(15):
            prediction = AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.META_LEARNING,
                action="BUY",
                confidence=0.9,  # High confidence for meta-learning
                probability_distribution=np.array([0.1, 0.1, 0.8]),
                recommended_position_size=0.2,
                risk_score=0.2,
                expected_return=0.08,
                uncertainty=0.1
            )
            self.tracker.add_prediction(prediction, 0.05)  # Good outcomes
        
        new_weights = self.tracker.update_system_weights()
        
        self.assertIsInstance(new_weights, dict)
        self.assertAlmostEqual(sum(new_weights.values()), 1.0, places=2)


class TestAIMasterIntegrationSystem(unittest.TestCase):
    """Test AI Master Integration System"""
    
    def setUp(self):
        """Set up test system"""
        self.config = AISystemConfig(
            enable_neural_ensemble=True,
            enable_reinforcement_learning=True,
            enable_meta_learning=True,
            decision_strategy=DecisionStrategy.ADAPTIVE_ENSEMBLE,
            sequence_length=10,  # Smaller for testing
            input_features=95
        )
        self.system = AIMasterIntegrationSystem(self.config)
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsInstance(self.system.config, AISystemConfig)
        self.assertIsInstance(self.system.performance_tracker, AIPerformanceTracker)
        self.assertEqual(len(self.system.market_data_buffer), 0)
        self.assertEqual(len(self.system.decision_history), 0)
    
    def test_prepare_sequence_data_insufficient_data(self):
        """Test sequence preparation with insufficient data"""
        # Add less data than sequence_length
        for i in range(5):
            market_data = AIMarketData(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                price=2000.0 + i,
                high=2010.0,
                low=1990.0,
                volume=5000.0
            )
            self.system.market_data_buffer.append(market_data)
        
        sequence = self.system._prepare_sequence_data()
        self.assertIsNone(sequence)
    
    def test_prepare_sequence_data_sufficient_data(self):
        """Test sequence preparation with sufficient data"""
        # Add enough data
        for i in range(15):
            market_data = AIMarketData(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                price=2000.0 + i,
                high=2010.0,
                low=1990.0,
                volume=5000.0
            )
            self.system.market_data_buffer.append(market_data)
        
        sequence = self.system._prepare_sequence_data()
        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.shape, (1, self.config.sequence_length, self.config.input_features))
    
    def test_convert_to_action(self):
        """Test action conversion"""
        # Test different prediction arrays
        prediction_buy = np.array([0.1, 0.1, 0.8])
        prediction_sell = np.array([0.1, 0.8, 0.1])
        prediction_hold = np.array([0.8, 0.1, 0.1])
        
        self.assertEqual(self.system._convert_to_action(prediction_buy), "BUY")
        self.assertEqual(self.system._convert_to_action(prediction_sell), "SELL")
        self.assertEqual(self.system._convert_to_action(prediction_hold), "HOLD")
    
    def test_calculate_consensus(self):
        """Test consensus calculation"""
        predictions = [
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.NEURAL_ENSEMBLE,
                action="BUY",
                confidence=0.8,
                probability_distribution=np.array([0.1, 0.1, 0.8]),
                recommended_position_size=0.2,
                risk_score=0.3,
                expected_return=0.05,
                uncertainty=0.2
            ),
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.REINFORCEMENT_LEARNING,
                action="BUY",
                confidence=0.7,
                probability_distribution=np.array([0.1, 0.1, 0.8]),
                recommended_position_size=0.15,
                risk_score=0.4,
                expected_return=0.03,
                uncertainty=0.3
            ),
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.META_LEARNING,
                action="SELL",
                confidence=0.6,
                probability_distribution=np.array([0.1, 0.8, 0.1]),
                recommended_position_size=0.1,
                risk_score=0.5,
                expected_return=0.02,
                uncertainty=0.4
            )
        ]
        
        consensus = self.system._calculate_consensus(predictions)
        self.assertAlmostEqual(consensus, 2/3, places=2)  # 2 out of 3 agree on BUY
    
    def test_create_default_decision(self):
        """Test default decision creation"""
        market_data = AIMarketData(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            price=2000.0,
            high=2010.0,
            low=1990.0,
            volume=5000.0
        )
        
        decision = self.system._create_default_decision(market_data)
        
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.confidence, 0.5)
        self.assertEqual(decision.position_size, 0.0)
        self.assertEqual(len(decision.individual_predictions), 0)
    
    @patch('src.core.integration.ai_master_integration.NEURAL_ENSEMBLE_AVAILABLE', True)
    @patch('src.core.integration.ai_master_integration.RL_AVAILABLE', True)
    @patch('src.core.integration.ai_master_integration.META_LEARNING_AVAILABLE', True)
    def test_process_market_data_with_mocked_systems(self):
        """Test market data processing with mocked AI systems"""
        # Mock AI systems
        self.system.neural_ensemble = Mock()
        self.system.rl_agent = Mock()
        self.system.rl_environment = Mock()
        self.system.meta_learning_system = Mock()
        
        # Mock neural ensemble prediction
        mock_neural_result = Mock()
        mock_neural_result.prediction = np.array([0.1, 0.1, 0.8])
        mock_neural_result.confidence = 0.85
        mock_neural_result.consensus_score = 0.9
        self.system.neural_ensemble.predict_ensemble.return_value = mock_neural_result
        
        # Mock RL prediction
        self.system.rl_agent.select_action.return_value = 2  # BUY action
        
        # Mock meta-learning prediction
        mock_meta_result = Mock()
        mock_meta_result.prediction = np.array([0.1, 0.1, 0.8])
        mock_meta_result.confidence = 0.8
        mock_meta_result.adaptation_score = 0.7
        mock_meta_result.transfer_effectiveness = 0.6
        mock_meta_result.continual_retention = 0.8
        mock_meta_result.meta_gradient_norm = 0.5
        self.system.meta_learning_system.ensemble_predict.return_value = mock_meta_result
        
        # Generate sufficient market data
        for i in range(15):
            market_data = AIMarketData(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol="XAUUSD",
                price=2000.0 + i,
                high=2010.0,
                low=1990.0,
                volume=5000.0
            )
            self.system.market_data_buffer.append(market_data)
        
        # Process new data point
        new_data = AIMarketData(
            timestamp=datetime.now() + timedelta(minutes=15),
            symbol="XAUUSD",
            price=2015.0,
            high=2025.0,
            low=2005.0,
            volume=5500.0
        )
        
        decision = self.system.process_market_data(new_data)
        
        self.assertIsNotNone(decision)
        self.assertIsInstance(decision, EnsembleDecision)
        self.assertIn(decision.action, ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertEqual(len(decision.individual_predictions), 3)  # All three systems
    
    def test_update_performance(self):
        """Test performance update"""
        # Create a mock decision
        predictions = [
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.NEURAL_ENSEMBLE,
                action="BUY",
                confidence=0.8,
                probability_distribution=np.array([0.1, 0.1, 0.8]),
                recommended_position_size=0.2,
                risk_score=0.3,
                expected_return=0.05,
                uncertainty=0.2
            )
        ]
        
        decision = EnsembleDecision(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            action="BUY",
            confidence=0.8,
            consensus_score=1.0,
            position_size=0.2,
            risk_score=0.3,
            expected_return=0.05,
            neural_ensemble_weight=1.0,
            rl_weight=0.0,
            meta_learning_weight=0.0,
            total_processing_time=0.1,
            individual_predictions=predictions,
            decision_strategy="adaptive_ensemble"
        )
        
        initial_history_length = len(self.system.performance_tracker.predictions_history)
        self.system.update_performance(decision, 0.03)
        
        # Check that prediction was added to tracker
        self.assertEqual(
            len(self.system.performance_tracker.predictions_history),
            initial_history_length + 1
        )
    
    def test_get_system_status(self):
        """Test system status retrieval"""
        status = self.system.get_system_status()
        
        self.assertIn('timestamp', status)
        self.assertIn('systems_active', status)
        self.assertIn('system_weights', status)
        self.assertIn('performance_metrics', status)
        self.assertIn('config', status)
        
        # Check systems_active structure
        self.assertIn('neural_ensemble', status['systems_active'])
        self.assertIn('reinforcement_learning', status['systems_active'])
        self.assertIn('meta_learning', status['systems_active'])
    
    def test_export_system_data(self):
        """Test system data export"""
        export_result = self.system.export_system_data()
        
        self.assertIn('success', export_result)
        if export_result['success']:
            self.assertIn('filepath', export_result)
        else:
            self.assertIn('error', export_result)


class TestDecisionStrategies(unittest.TestCase):
    """Test different decision strategies"""
    
    def setUp(self):
        """Set up test data"""
        self.system = AIMasterIntegrationSystem(AISystemConfig())
        self.market_data = AIMarketData(
            timestamp=datetime.now(),
            symbol="XAUUSD",
            price=2000.0,
            high=2010.0,
            low=1990.0,
            volume=5000.0
        )
        
        self.predictions = [
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.NEURAL_ENSEMBLE,
                action="BUY",
                confidence=0.8,
                probability_distribution=np.array([0.1, 0.1, 0.8]),
                recommended_position_size=0.2,
                risk_score=0.3,
                expected_return=0.05,
                uncertainty=0.2
            ),
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.REINFORCEMENT_LEARNING,
                action="BUY",
                confidence=0.7,
                probability_distribution=np.array([0.1, 0.2, 0.7]),
                recommended_position_size=0.15,
                risk_score=0.4,
                expected_return=0.03,
                uncertainty=0.3
            ),
            AIPrediction(
                timestamp=datetime.now(),
                symbol="XAUUSD",
                source=AISystemType.META_LEARNING,
                action="SELL",
                confidence=0.6,
                probability_distribution=np.array([0.1, 0.7, 0.2]),
                recommended_position_size=0.1,
                risk_score=0.5,
                expected_return=0.02,
                uncertainty=0.4
            )
        ]
    
    def test_majority_voting_decision(self):
        """Test majority voting strategy"""
        decision = self.system._majority_voting_decision(self.predictions, self.market_data)
        
        self.assertEqual(decision.action, "BUY")  # 2 out of 3 predictions are BUY
        self.assertAlmostEqual(decision.consensus_score, 2/3, places=2)
        self.assertIsInstance(decision, EnsembleDecision)
    
    def test_confidence_weighted_decision(self):
        """Test confidence weighted strategy"""
        decision = self.system._confidence_weighted_decision(self.predictions, self.market_data)
        
        self.assertIsInstance(decision, EnsembleDecision)
        self.assertIn(decision.action, ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
    
    def test_adaptive_ensemble_decision(self):
        """Test adaptive ensemble strategy"""
        weights = {
            AISystemType.NEURAL_ENSEMBLE: 0.4,
            AISystemType.REINFORCEMENT_LEARNING: 0.3,
            AISystemType.META_LEARNING: 0.3
        }
        
        decision = self.system._adaptive_ensemble_decision(self.predictions, weights, self.market_data)
        
        self.assertIsInstance(decision, EnsembleDecision)
        self.assertIn(decision.action, ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        
        # Check that system weights are used
        self.assertGreaterEqual(decision.neural_ensemble_weight, 0.0)
        self.assertGreaterEqual(decision.rl_weight, 0.0)
        self.assertGreaterEqual(decision.meta_learning_weight, 0.0)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function"""
    
    def test_create_ai_master_system_default(self):
        """Test creating system with default config"""
        system = create_ai_master_system()
        
        self.assertIsInstance(system, AIMasterIntegrationSystem)
        self.assertTrue(system.config.enable_neural_ensemble)
        self.assertTrue(system.config.enable_reinforcement_learning)
        self.assertTrue(system.config.enable_meta_learning)
    
    def test_create_ai_master_system_custom_config(self):
        """Test creating system with custom config"""
        config = {
            'enable_neural_ensemble': False,
            'decision_strategy': 'majority_voting',
            'min_confidence_threshold': 0.8,
            'max_position_size': 0.3
        }
        
        system = create_ai_master_system(config)
        
        self.assertIsInstance(system, AIMasterIntegrationSystem)
        self.assertFalse(system.config.enable_neural_ensemble)
        self.assertEqual(system.config.decision_strategy, DecisionStrategy.MAJORITY_VOTING)
        self.assertEqual(system.config.min_confidence_threshold, 0.8)
        self.assertEqual(system.config.max_position_size, 0.3)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_processing(self):
        """Test end-to-end market data processing"""
        system = create_ai_master_system({
            'sequence_length': 5,  # Small for testing
            'enable_neural_ensemble': False,  # Disable to avoid import issues
            'enable_reinforcement_learning': False,
            'enable_meta_learning': False
        })
        
        # Generate test data
        market_data_list = []
        for i in range(10):
            data = AIMarketData(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol="XAUUSD",
                price=2000.0 + i,
                high=2010.0 + i,
                low=1990.0 + i,
                volume=5000.0 + i * 100
            )
            market_data_list.append(data)
        
        decisions = []
        for data in market_data_list:
            decision = system.process_market_data(data)
            if decision:
                decisions.append(decision)
        
        # With all AI systems disabled, should get default decisions
        for decision in decisions:
            self.assertEqual(decision.action, "HOLD")
            self.assertEqual(decision.confidence, 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)