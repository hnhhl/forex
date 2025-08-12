"""
Neural Network Ensemble System
Advanced AI component for Ultimate XAU Super System V4.0 Phase 2

This module implements a sophisticated ensemble of neural networks for:
- Price prediction and trend analysis
- Risk assessment and volatility forecasting
- Pattern recognition and signal generation
- Multi-timeframe analysis and decision making
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging level
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """Types of neural networks in ensemble"""
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    DENSE = "dense"
    HYBRID = "hybrid"


class PredictionType(Enum):
    """Types of predictions"""
    PRICE = "price"
    TREND = "trend"
    VOLATILITY = "volatility"
    SIGNAL = "signal"
    RISK = "risk"


@dataclass
class NetworkConfig:
    """Configuration for individual neural network"""
    network_type: NetworkType
    prediction_type: PredictionType
    sequence_length: int = 60
    features: int = 10
    hidden_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    weight: float = 1.0  # Weight in ensemble


@dataclass
class PredictionResult:
    """Result from neural network prediction"""
    network_type: NetworkType
    prediction_type: PredictionType
    prediction: np.ndarray
    confidence: float
    timestamp: datetime
    features_used: List[str]
    model_version: str
    processing_time: float


@dataclass
class EnsembleResult:
    """Result from ensemble prediction"""
    final_prediction: np.ndarray
    individual_predictions: List[PredictionResult]
    ensemble_confidence: float
    consensus_score: float
    timestamp: datetime
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNeuralNetwork:
    """Base class for neural networks"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = None
        self.model_version = f"{config.network_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def build_model(self) -> keras.Model:
        """Build neural network model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        try:
            # Create sequences
            X, y = self._create_sequences(data, target_column)
            
            # Update config features to match actual data
            self.config.features = X.shape[-1]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _create_sequences(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        try:
            # Select feature columns (exclude target)
            feature_columns = [col for col in data.columns if col != target_column]
            
            X, y = [], []
            for i in range(self.config.sequence_length, len(data)):
                # Features sequence
                X.append(data[feature_columns].iloc[i-self.config.sequence_length:i].values)
                # Target value
                y.append(data[target_column].iloc[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the neural network"""
        try:
            logger.info(f"Training {self.config.network_type.value} model...")
            
            # Update config to match actual data dimensions
            self.config.features = X.shape[-1]
            
            # Build model if not exists
            if self.model is None:
                self.model = self.build_model()
            
            # Compile model
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            start_time = time.time()
            self.training_history = self.model.fit(
                X, y,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks_list,
                verbose=0
            )
            
            training_time = time.time() - start_time
            self.is_trained = True
            
            # Calculate training metrics
            train_pred = self.model.predict(X, verbose=0)
            train_mse = mean_squared_error(y, train_pred)
            train_mae = mean_absolute_error(y, train_pred)
            train_r2 = r2_score(y, train_pred)
            
            logger.info(f"{self.config.network_type.value} training completed in {training_time:.2f}s")
            
            return {
                'training_time': training_time,
                'final_loss': self.training_history.history['loss'][-1],
                'final_val_loss': self.training_history.history['val_loss'][-1],
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'epochs_trained': len(self.training_history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Error training {self.config.network_type.value}: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make prediction"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            start_time = time.time()
            
            # Scale input data
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Make prediction
            prediction = self.model.predict(X_scaled, verbose=0)
            
            # Calculate confidence (simplified - based on prediction variance)
            confidence = self._calculate_confidence(prediction)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                network_type=self.config.network_type,
                prediction_type=self.config.prediction_type,
                prediction=prediction,
                confidence=confidence,
                timestamp=datetime.now(),
                features_used=[f"feature_{i}" for i in range(X.shape[-1])],
                model_version=self.model_version,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        """Calculate prediction confidence"""
        try:
            # Simple confidence based on prediction consistency
            if len(prediction) > 1:
                variance = np.var(prediction)
                confidence = max(0.1, min(0.99, 1.0 - variance))
            else:
                confidence = 0.8  # Default confidence for single prediction
            
            return float(confidence)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            if self.model is not None:
                self.model.save(f"{filepath}_{self.model_version}.keras")
                joblib.dump(self.scaler, f"{filepath}_{self.model_version}_scaler.pkl")
                logger.info(f"Model saved: {filepath}_{self.model_version}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(f"{filepath}.keras")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.is_trained = True
            logger.info(f"Model loaded: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")


class LSTMNetwork(BaseNeuralNetwork):
    """LSTM Neural Network for time series prediction"""
    
    def build_model(self) -> keras.Model:
        """Build LSTM model"""
        model = models.Sequential([
            layers.LSTM(
                self.config.hidden_units[0],
                return_sequences=True,
                input_shape=(self.config.sequence_length, self.config.features)
            ),
            layers.Dropout(self.config.dropout_rate),
            
            layers.LSTM(
                self.config.hidden_units[1],
                return_sequences=len(self.config.hidden_units) > 2
            ),
            layers.Dropout(self.config.dropout_rate),
        ])
        
        # Add additional LSTM layers if specified
        for i in range(2, len(self.config.hidden_units)):
            model.add(layers.LSTM(
                self.config.hidden_units[i],
                return_sequences=i < len(self.config.hidden_units) - 1
            ))
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        return model


class GRUNetwork(BaseNeuralNetwork):
    """GRU Neural Network for time series prediction"""
    
    def build_model(self) -> keras.Model:
        """Build GRU model"""
        model = models.Sequential([
            layers.GRU(
                self.config.hidden_units[0],
                return_sequences=True,
                input_shape=(self.config.sequence_length, self.config.features)
            ),
            layers.Dropout(self.config.dropout_rate),
            
            layers.GRU(
                self.config.hidden_units[1],
                return_sequences=len(self.config.hidden_units) > 2
            ),
            layers.Dropout(self.config.dropout_rate),
        ])
        
        # Add additional GRU layers if specified
        for i in range(2, len(self.config.hidden_units)):
            model.add(layers.GRU(
                self.config.hidden_units[i],
                return_sequences=i < len(self.config.hidden_units) - 1
            ))
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        return model


class CNNNetwork(BaseNeuralNetwork):
    """CNN Neural Network for pattern recognition"""
    
    def build_model(self) -> keras.Model:
        """Build CNN model"""
        model = models.Sequential([
            layers.Conv1D(
                filters=self.config.hidden_units[0],
                kernel_size=3,
                activation='relu',
                input_shape=(self.config.sequence_length, self.config.features)
            ),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(self.config.dropout_rate),
            
            layers.Conv1D(
                filters=self.config.hidden_units[1],
                kernel_size=3,
                activation='relu'
            ),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(self.config.dropout_rate),
            
            layers.Flatten(),
            layers.Dense(self.config.hidden_units[-1], activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(1)
        ])
        
        return model


class DenseNetwork(BaseNeuralNetwork):
    """Dense Neural Network for feature-based prediction"""
    
    def build_model(self) -> keras.Model:
        """Build Dense model"""
        model = models.Sequential([
            layers.Flatten(input_shape=(self.config.sequence_length, self.config.features)),
        ])
        
        # Add dense layers
        for units in self.config.hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.config.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        return model


class NeuralEnsemble:
    """Ensemble of Neural Networks for comprehensive prediction"""
    
    def __init__(self, configs: List[NetworkConfig]):
        self.configs = configs
        self.networks: Dict[str, BaseNeuralNetwork] = {}
        self.is_trained = False
        self.ensemble_weights = None
        self.training_metrics = {}
        
        # Initialize networks
        self._initialize_networks()
        
        logger.info(f"Neural Ensemble initialized with {len(self.networks)} networks")
    
    def _initialize_networks(self):
        """Initialize individual neural networks"""
        network_classes = {
            NetworkType.LSTM: LSTMNetwork,
            NetworkType.GRU: GRUNetwork,
            NetworkType.CNN: CNNNetwork,
            NetworkType.DENSE: DenseNetwork
        }
        
        for config in self.configs:
            if config.network_type in network_classes:
                network_id = f"{config.network_type.value}_{config.prediction_type.value}"
                self.networks[network_id] = network_classes[config.network_type](config)
            else:
                logger.warning(f"Network type {config.network_type} not implemented yet")
    
    def train_ensemble(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train all networks in ensemble"""
        try:
            logger.info("Starting ensemble training...")
            start_time = time.time()
            
            training_results = {}
            
            for network_id, network in self.networks.items():
                logger.info(f"Training network: {network_id}")
                
                # Prepare data for this network
                X, y = network.prepare_data(data, target_column)
                
                # Train network
                result = network.train(X, y)
                training_results[network_id] = result
                
                logger.info(f"Network {network_id} trained - Loss: {result['final_loss']:.6f}")
            
            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights(training_results)
            
            total_time = time.time() - start_time
            self.is_trained = True
            self.training_metrics = training_results
            
            logger.info(f"Ensemble training completed in {total_time:.2f}s")
            
            return {
                'total_training_time': total_time,
                'individual_results': training_results,
                'ensemble_weights': self.ensemble_weights,
                'networks_trained': len(self.networks)
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def _calculate_ensemble_weights(self, training_results: Dict[str, Any]):
        """Calculate weights for ensemble based on training performance"""
        try:
            weights = {}
            
            # Calculate weights based on validation loss (inverse relationship)
            val_losses = {
                network_id: results['final_val_loss']
                for network_id, results in training_results.items()
            }
            
            # Inverse of validation loss (lower loss = higher weight)
            inv_losses = {
                network_id: 1.0 / (loss + 1e-8)
                for network_id, loss in val_losses.items()
            }
            
            # Normalize weights
            total_weight = sum(inv_losses.values())
            weights = {
                network_id: weight / total_weight
                for network_id, weight in inv_losses.items()
            }
            
            # Apply config weights
            for network_id, network in self.networks.items():
                if network_id in weights:
                    weights[network_id] *= network.config.weight
            
            # Renormalize
            total_weight = sum(weights.values())
            self.ensemble_weights = {
                network_id: weight / total_weight
                for network_id, weight in weights.items()
            }
            
            logger.info("Ensemble weights calculated:")
            for network_id, weight in self.ensemble_weights.items():
                logger.info(f"  {network_id}: {weight:.4f}")
                
        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {e}")
            # Default to equal weights
            self.ensemble_weights = {
                network_id: 1.0 / len(self.networks)
                for network_id in self.networks.keys()
            }
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Make ensemble prediction"""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before prediction")
            
            start_time = time.time()
            individual_predictions = []
            weighted_predictions = []
            
            # Get predictions from all networks
            for network_id, network in self.networks.items():
                try:
                    prediction_result = network.predict(X)
                    individual_predictions.append(prediction_result)
                    
                    # Apply ensemble weight
                    weight = self.ensemble_weights.get(network_id, 0.0)
                    weighted_pred = prediction_result.prediction * weight
                    weighted_predictions.append(weighted_pred)
                    
                except Exception as e:
                    logger.warning(f"Error getting prediction from {network_id}: {e}")
            
            if not weighted_predictions:
                raise ValueError("No valid predictions obtained from ensemble")
            
            # Combine weighted predictions
            final_prediction = np.sum(weighted_predictions, axis=0)
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(individual_predictions)
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(individual_predictions)
            
            processing_time = time.time() - start_time
            
            return EnsembleResult(
                final_prediction=final_prediction,
                individual_predictions=individual_predictions,
                ensemble_confidence=ensemble_confidence,
                consensus_score=consensus_score,
                timestamp=datetime.now(),
                processing_time=processing_time,
                metadata={
                    'ensemble_weights': self.ensemble_weights,
                    'networks_used': len(individual_predictions)
                }
            )
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            raise
    
    def _calculate_ensemble_confidence(self, predictions: List[PredictionResult]) -> float:
        """Calculate overall ensemble confidence"""
        try:
            if not predictions:
                return 0.0
            
            # Weighted average of individual confidences
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for pred in predictions:
                network_id = f"{pred.network_type.value}_{pred.prediction_type.value}"
                weight = self.ensemble_weights.get(network_id, 0.0)
                weighted_confidence += pred.confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_confidence / total_weight
            else:
                return np.mean([pred.confidence for pred in predictions])
                
        except Exception as e:
            logger.warning(f"Error calculating ensemble confidence: {e}")
            return 0.5
    
    def _calculate_consensus_score(self, predictions: List[PredictionResult]) -> float:
        """Calculate consensus score among predictions"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            # Calculate standard deviation of predictions
            pred_values = [pred.prediction.flatten()[0] if pred.prediction.size > 0 else 0.0 
                          for pred in predictions]
            
            if len(pred_values) > 1:
                std_dev = np.std(pred_values)
                mean_val = np.mean(np.abs(pred_values))
                
                # Consensus score: higher when predictions agree (lower std dev)
                if mean_val > 0:
                    consensus = max(0.0, 1.0 - (std_dev / mean_val))
                else:
                    consensus = 1.0 if std_dev < 1e-6 else 0.0
            else:
                consensus = 1.0
            
            return float(consensus)
            
        except Exception as e:
            logger.warning(f"Error calculating consensus score: {e}")
            return 0.5
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble status"""
        return {
            'networks_count': len(self.networks),
            'is_trained': self.is_trained,
            'network_types': [config.network_type.value for config in self.configs],
            'prediction_types': [config.prediction_type.value for config in self.configs],
            'ensemble_weights': self.ensemble_weights,
            'training_metrics': self.training_metrics
        }
    
    def save_ensemble(self, base_path: str):
        """Save entire ensemble"""
        try:
            for network_id, network in self.networks.items():
                network.save_model(f"{base_path}_{network_id}")
            
            # Save ensemble metadata
            metadata = {
                'ensemble_weights': self.ensemble_weights,
                'training_metrics': self.training_metrics,
                'configs': [
                    {
                        'network_type': config.network_type.value,
                        'prediction_type': config.prediction_type.value,
                        'sequence_length': config.sequence_length,
                        'features': config.features,
                        'weight': config.weight
                    }
                    for config in self.configs
                ]
            }
            
            import json
            with open(f"{base_path}_ensemble_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Ensemble saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
    
    def load_ensemble(self, base_path: str):
        """Load entire ensemble"""
        try:
            # Load ensemble metadata
            import json
            with open(f"{base_path}_ensemble_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.ensemble_weights = metadata['ensemble_weights']
            self.training_metrics = metadata['training_metrics']
            
            # Load individual networks
            for network_id, network in self.networks.items():
                network.load_model(f"{base_path}_{network_id}")
            
            self.is_trained = True
            logger.info(f"Ensemble loaded from {base_path}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")


# Factory function for creating ensemble configurations
def create_default_ensemble_configs() -> List[NetworkConfig]:
    """Create default ensemble configuration"""
    configs = [
        # LSTM for price prediction
        NetworkConfig(
            network_type=NetworkType.LSTM,
            prediction_type=PredictionType.PRICE,
            sequence_length=60,
            features=10,
            hidden_units=[128, 64, 32],
            weight=1.5  # Higher weight for price prediction
        ),
        
        # GRU for trend prediction
        NetworkConfig(
            network_type=NetworkType.GRU,
            prediction_type=PredictionType.TREND,
            sequence_length=30,
            features=10,
            hidden_units=[64, 32],
            weight=1.2
        ),
        
        # CNN for pattern recognition
        NetworkConfig(
            network_type=NetworkType.CNN,
            prediction_type=PredictionType.SIGNAL,
            sequence_length=40,
            features=10,
            hidden_units=[64, 32, 16],
            weight=1.0
        ),
        
        # Dense for volatility prediction
        NetworkConfig(
            network_type=NetworkType.DENSE,
            prediction_type=PredictionType.VOLATILITY,
            sequence_length=20,
            features=10,
            hidden_units=[128, 64, 32],
            weight=0.8
        )
    ]
    
    return configs


if __name__ == "__main__":
    # Example usage
    print("🧠 Neural Network Ensemble System")
    print("Phase 2 - AI Systems Expansion")
    
    # Create ensemble
    configs = create_default_ensemble_configs()
    ensemble = NeuralEnsemble(configs)
    
    print(f"✅ Ensemble created with {len(ensemble.networks)} networks")
    print("Ready for training and prediction!") 