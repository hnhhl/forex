"""
Test Suite for Neural Network Ensemble System
Tests the AI neural network ensemble for Phase 2
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.ai.neural_ensemble import (
    NeuralEnsemble, NetworkConfig, NetworkType, PredictionType,
    LSTMNetwork, GRUNetwork, CNNNetwork, DenseNetwork,
    create_default_ensemble_configs, PredictionResult, EnsembleResult
)


class TestNeuralEnsemble:
    """Test Neural Network Ensemble System"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        
        # Create synthetic price data with trend and noise
        trend = np.linspace(2000, 2100, 1000)
        noise = np.random.normal(0, 10, 1000)
        prices = trend + noise
        
        # Create additional features
        data = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.uniform(100, 1000, 1000),
            'high': prices + np.random.uniform(0, 5, 1000),
            'low': prices - np.random.uniform(0, 5, 1000),
            'rsi': np.random.uniform(20, 80, 1000),
            'macd': np.random.normal(0, 2, 1000),
            'bb_upper': prices + np.random.uniform(10, 20, 1000),
            'bb_lower': prices - np.random.uniform(10, 20, 1000),
            'volatility': np.random.uniform(0.01, 0.05, 1000),
            'returns': np.random.normal(0, 0.02, 1000)
        })
        
        return data
    
    @pytest.fixture
    def simple_config(self):
        """Create simple network configuration for testing"""
        return NetworkConfig(
            network_type=NetworkType.LSTM,
            prediction_type=PredictionType.PRICE,
            sequence_length=10,  # Short for fast testing
            features=5,
            hidden_units=[16, 8],  # Small for fast testing
            epochs=2,  # Very few epochs for testing
            batch_size=16
        )
    
    @pytest.fixture
    def ensemble_configs(self):
        """Create ensemble configurations for testing"""
        return [
            NetworkConfig(
                network_type=NetworkType.LSTM,
                prediction_type=PredictionType.PRICE,
                sequence_length=10,
                features=5,
                hidden_units=[16, 8],
                epochs=2,
                batch_size=16,
                weight=1.0
            ),
            NetworkConfig(
                network_type=NetworkType.GRU,
                prediction_type=PredictionType.TREND,
                sequence_length=10,
                features=5,
                hidden_units=[16, 8],
                epochs=2,
                batch_size=16,
                weight=0.8
            )
        ]
    
    def test_network_config_creation(self):
        """Test NetworkConfig creation"""
        config = NetworkConfig(
            network_type=NetworkType.LSTM,
            prediction_type=PredictionType.PRICE
        )
        
        assert config.network_type == NetworkType.LSTM
        assert config.prediction_type == PredictionType.PRICE
        assert config.sequence_length == 60  # Default value
        assert config.features == 10  # Default value
        assert isinstance(config.hidden_units, list)
    
    def test_lstm_network_creation(self, simple_config):
        """Test LSTM network creation"""
        network = LSTMNetwork(simple_config)
        
        assert network.config == simple_config
        assert network.model is None
        assert not network.is_trained
        assert network.scaler is not None
    
    def test_lstm_model_building(self, simple_config):
        """Test LSTM model building"""
        network = LSTMNetwork(simple_config)
        model = network.build_model()
        
        assert model is not None
        assert len(model.layers) > 0
        
        # Check input shape
        expected_input_shape = (simple_config.sequence_length, simple_config.features)
        assert model.input_shape[1:] == expected_input_shape
    
    def test_gru_network_creation(self, simple_config):
        """Test GRU network creation"""
        config = simple_config
        config.network_type = NetworkType.GRU
        
        network = GRUNetwork(config)
        model = network.build_model()
        
        assert model is not None
        assert len(model.layers) > 0
    
    def test_cnn_network_creation(self, simple_config):
        """Test CNN network creation"""
        config = simple_config
        config.network_type = NetworkType.CNN
        
        network = CNNNetwork(config)
        model = network.build_model()
        
        assert model is not None
        assert len(model.layers) > 0
    
    def test_dense_network_creation(self, simple_config):
        """Test Dense network creation"""
        config = simple_config
        config.network_type = NetworkType.DENSE
        
        network = DenseNetwork(config)
        model = network.build_model()
        
        assert model is not None
        assert len(model.layers) > 0
    
    def test_data_preparation(self, simple_config, sample_data):
        """Test data preparation for training"""
        network = LSTMNetwork(simple_config)
        
        # Use subset of data for testing
        test_data = sample_data.iloc[:100].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        X, y = network.prepare_data(test_data, 'price')
        
        assert X.shape[0] == len(test_data) - simple_config.sequence_length
        assert X.shape[1] == simple_config.sequence_length
        assert X.shape[2] == len(feature_columns)
        assert len(y) == len(test_data) - simple_config.sequence_length
    
    def test_sequence_creation(self, simple_config, sample_data):
        """Test sequence creation"""
        network = LSTMNetwork(simple_config)
        
        # Use subset of data
        test_data = sample_data.iloc[:50].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        X, y = network._create_sequences(test_data, 'price')
        
        expected_samples = len(test_data) - simple_config.sequence_length
        assert X.shape == (expected_samples, simple_config.sequence_length, len(feature_columns))
        assert y.shape == (expected_samples,)
    
    @pytest.mark.slow
    def test_network_training(self, simple_config, sample_data):
        """Test individual network training"""
        network = LSTMNetwork(simple_config)
        
        # Prepare small dataset
        test_data = sample_data.iloc[:100].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        X, y = network.prepare_data(test_data, 'price')
        
        # Train network
        result = network.train(X, y)
        
        assert network.is_trained
        assert network.model is not None
        assert 'training_time' in result
        assert 'final_loss' in result
        assert result['training_time'] > 0
    
    @pytest.mark.slow
    def test_network_prediction(self, simple_config, sample_data):
        """Test individual network prediction"""
        network = LSTMNetwork(simple_config)
        
        # Prepare and train
        test_data = sample_data.iloc[:100].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        X, y = network.prepare_data(test_data, 'price')
        network.train(X, y)
        
        # Make prediction
        test_input = X[:1]  # Single sample
        result = network.predict(test_input)
        
        assert isinstance(result, PredictionResult)
        assert result.network_type == NetworkType.LSTM
        assert result.prediction_type == PredictionType.PRICE
        assert result.prediction is not None
        assert 0 <= result.confidence <= 1
        assert result.processing_time > 0
    
    def test_ensemble_creation(self, ensemble_configs):
        """Test ensemble creation"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        assert len(ensemble.networks) == 2
        assert not ensemble.is_trained
        assert ensemble.ensemble_weights is None
    
    def test_ensemble_initialization(self, ensemble_configs):
        """Test ensemble network initialization"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        # Check that networks are properly initialized
        assert 'lstm_price' in ensemble.networks
        assert 'gru_trend' in ensemble.networks
        
        assert isinstance(ensemble.networks['lstm_price'], LSTMNetwork)
        assert isinstance(ensemble.networks['gru_trend'], GRUNetwork)
    
    @pytest.mark.slow
    def test_ensemble_training(self, ensemble_configs, sample_data):
        """Test ensemble training"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        # Prepare small dataset
        test_data = sample_data.iloc[:100].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        # Train ensemble
        result = ensemble.train_ensemble(test_data, 'price')
        
        assert ensemble.is_trained
        assert 'total_training_time' in result
        assert 'individual_results' in result
        assert 'ensemble_weights' in result
        assert len(result['individual_results']) == 2
        assert ensemble.ensemble_weights is not None
    
    @pytest.mark.slow
    def test_ensemble_prediction(self, ensemble_configs, sample_data):
        """Test ensemble prediction"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        # Prepare and train
        test_data = sample_data.iloc[:100].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        # Get training data
        network = ensemble.networks['lstm_price']
        X, y = network.prepare_data(test_data, 'price')
        
        # Train ensemble
        ensemble.train_ensemble(test_data, 'price')
        
        # Make prediction
        test_input = X[:1]  # Single sample
        result = ensemble.predict(test_input)
        
        assert isinstance(result, EnsembleResult)
        assert result.final_prediction is not None
        assert len(result.individual_predictions) > 0
        assert 0 <= result.ensemble_confidence <= 1
        assert 0 <= result.consensus_score <= 1
        assert result.processing_time > 0
    
    def test_ensemble_weights_calculation(self, ensemble_configs):
        """Test ensemble weights calculation"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        # Mock training results
        training_results = {
            'lstm_price': {'final_val_loss': 0.1},
            'gru_trend': {'final_val_loss': 0.2}
        }
        
        ensemble._calculate_ensemble_weights(training_results)
        
        assert ensemble.ensemble_weights is not None
        assert len(ensemble.ensemble_weights) == 2
        
        # Network with lower loss should have higher weight
        assert ensemble.ensemble_weights['lstm_price'] > ensemble.ensemble_weights['gru_trend']
        
        # Weights should sum to 1
        total_weight = sum(ensemble.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_confidence_calculation(self, simple_config):
        """Test confidence calculation"""
        network = LSTMNetwork(simple_config)
        
        # Test with different prediction scenarios
        prediction1 = np.array([[1.0]])  # Single prediction
        confidence1 = network._calculate_confidence(prediction1)
        assert 0 <= confidence1 <= 1
        
        prediction2 = np.array([[1.0], [1.1], [0.9]])  # Multiple similar predictions
        confidence2 = network._calculate_confidence(prediction2)
        assert 0 <= confidence2 <= 1
    
    def test_consensus_score_calculation(self, ensemble_configs):
        """Test consensus score calculation"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        # Create mock predictions
        pred1 = PredictionResult(
            network_type=NetworkType.LSTM,
            prediction_type=PredictionType.PRICE,
            prediction=np.array([[1.0]]),
            confidence=0.8,
            timestamp=datetime.now(),
            features_used=[],
            model_version="test",
            processing_time=0.1
        )
        
        pred2 = PredictionResult(
            network_type=NetworkType.GRU,
            prediction_type=PredictionType.TREND,
            prediction=np.array([[1.1]]),
            confidence=0.7,
            timestamp=datetime.now(),
            features_used=[],
            model_version="test",
            processing_time=0.1
        )
        
        predictions = [pred1, pred2]
        consensus = ensemble._calculate_consensus_score(predictions)
        
        assert 0 <= consensus <= 1
    
    def test_ensemble_summary(self, ensemble_configs):
        """Test ensemble summary"""
        ensemble = NeuralEnsemble(ensemble_configs)
        summary = ensemble.get_ensemble_summary()
        
        assert 'networks_count' in summary
        assert 'is_trained' in summary
        assert 'network_types' in summary
        assert 'prediction_types' in summary
        
        assert summary['networks_count'] == 2
        assert not summary['is_trained']
        assert 'lstm' in summary['network_types']
        assert 'gru' in summary['network_types']
    
    def test_default_ensemble_configs(self):
        """Test default ensemble configuration creation"""
        configs = create_default_ensemble_configs()
        
        assert len(configs) == 4  # LSTM, GRU, CNN, Dense
        assert all(isinstance(config, NetworkConfig) for config in configs)
        
        # Check that different network types are included
        network_types = [config.network_type for config in configs]
        assert NetworkType.LSTM in network_types
        assert NetworkType.GRU in network_types
        assert NetworkType.CNN in network_types
        assert NetworkType.DENSE in network_types
    
    def test_error_handling_untrained_prediction(self, simple_config):
        """Test error handling for prediction without training"""
        network = LSTMNetwork(simple_config)
        
        # Try to predict without training
        test_input = np.random.random((1, 10, 5))
        
        with pytest.raises(ValueError, match="Model must be trained"):
            network.predict(test_input)
    
    def test_error_handling_ensemble_untrained_prediction(self, ensemble_configs):
        """Test error handling for ensemble prediction without training"""
        ensemble = NeuralEnsemble(ensemble_configs)
        
        test_input = np.random.random((1, 10, 5))
        
        with pytest.raises(ValueError, match="Ensemble must be trained"):
            ensemble.predict(test_input)
    
    @pytest.mark.slow
    def test_model_save_load(self, simple_config, sample_data):
        """Test model saving and loading"""
        network = LSTMNetwork(simple_config)
        
        # Train network
        test_data = sample_data.iloc[:50].copy()
        feature_columns = ['volume', 'high', 'low', 'rsi', 'macd']
        test_data = test_data[feature_columns + ['price']]
        
        X, y = network.prepare_data(test_data, 'price')
        network.train(X, y)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model")
            network.save_model(model_path)
            
            # Create new network and load
            new_network = LSTMNetwork(simple_config)
            new_network.load_model(f"{model_path}_{network.model_version}")
            
            assert new_network.is_trained
            assert new_network.model is not None
    
    def test_prediction_result_structure(self):
        """Test PredictionResult structure"""
        result = PredictionResult(
            network_type=NetworkType.LSTM,
            prediction_type=PredictionType.PRICE,
            prediction=np.array([[1.0]]),
            confidence=0.8,
            timestamp=datetime.now(),
            features_used=['feature1', 'feature2'],
            model_version="v1.0",
            processing_time=0.1
        )
        
        assert result.network_type == NetworkType.LSTM
        assert result.prediction_type == PredictionType.PRICE
        assert result.prediction.shape == (1, 1)
        assert result.confidence == 0.8
        assert isinstance(result.timestamp, datetime)
        assert len(result.features_used) == 2
        assert result.processing_time == 0.1
    
    def test_ensemble_result_structure(self):
        """Test EnsembleResult structure"""
        individual_pred = PredictionResult(
            network_type=NetworkType.LSTM,
            prediction_type=PredictionType.PRICE,
            prediction=np.array([[1.0]]),
            confidence=0.8,
            timestamp=datetime.now(),
            features_used=[],
            model_version="v1.0",
            processing_time=0.1
        )
        
        result = EnsembleResult(
            final_prediction=np.array([[1.0]]),
            individual_predictions=[individual_pred],
            ensemble_confidence=0.85,
            consensus_score=0.9,
            timestamp=datetime.now(),
            processing_time=0.2,
            metadata={'test': 'value'}
        )
        
        assert result.final_prediction.shape == (1, 1)
        assert len(result.individual_predictions) == 1
        assert result.ensemble_confidence == 0.85
        assert result.consensus_score == 0.9
        assert isinstance(result.timestamp, datetime)
        assert result.processing_time == 0.2
        assert 'test' in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 