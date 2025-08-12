"""
Unified Model Architecture for AI3.0 Ultimate XAU System
SINGLE SOURCE OF TRUTH for model architectures
Used by BOTH Training and Production systems
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class UnifiedModelArchitecture:
    """
    Unified Model Architecture Manager
    Standard architectures for both Training and Production
    """
    
    # STANDARD MODEL CONFIGURATIONS
    MODEL_CONFIGS = {
        'lstm': {
            'name': 'LSTM',
            'description': 'Long Short-Term Memory Network',
            'input_shape': (60, 19),  # 60 timesteps, 19 features
            'output_shape': 1,
            'architecture': 'sequence'
        },
        'cnn': {
            'name': 'CNN',
            'description': 'Convolutional Neural Network',
            'input_shape': (60, 19),
            'output_shape': 1,
            'architecture': 'convolutional'
        },
        'dense': {
            'name': 'Dense',
            'description': 'Deep Feed-Forward Network',
            'input_shape': (19,),  # Flattened features
            'output_shape': 1,
            'architecture': 'feedforward'
        },
        'hybrid': {
            'name': 'Hybrid',
            'description': 'CNN + LSTM Hybrid Network',
            'input_shape': (60, 19),
            'output_shape': 1,
            'architecture': 'hybrid'
        }
    }
    
    def __init__(self):
        logger.info("UnifiedModelArchitecture initialized with standard architectures")
    
    def create_lstm_model(self, input_shape: Tuple = None, 
                         sequence_length: int = 60, 
                         features: int = 19) -> keras.Model:
        """
        Create standard LSTM model
        
        Args:
            input_shape: Input shape tuple (sequence_length, features)
            sequence_length: Length of sequence
            features: Number of features
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        if input_shape is None:
            input_shape = (sequence_length, features)
            
        try:
            model = keras.Sequential([
                # Input layer
                layers.Input(shape=input_shape),
                
                # LSTM layers
                layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                
                # Dense layers
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.2),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"LSTM model created with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            raise
    
    def create_cnn_model(self, input_shape: Tuple = None,
                        sequence_length: int = 60,
                        features: int = 19) -> keras.Model:
        """
        Create standard CNN model
        
        Args:
            input_shape: Input shape tuple (sequence_length, features)
            sequence_length: Length of sequence
            features: Number of features
            
        Returns:
            keras.Model: Compiled CNN model
        """
        if input_shape is None:
            input_shape = (sequence_length, features)
            
        try:
            model = keras.Sequential([
                # Input layer
                layers.Input(shape=input_shape),
                
                # Convolutional layers
                layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2),
                
                layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.2),
                
                layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.GlobalMaxPooling1D(),
                
                # Dense layers
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"CNN model created with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating CNN model: {e}")
            raise
    
    def create_dense_model(self, input_shape: Tuple = None,
                          features: int = 19) -> keras.Model:
        """
        Create standard Dense model
        
        Args:
            input_shape: Input shape tuple (features,)
            features: Number of features
            
        Returns:
            keras.Model: Compiled Dense model
        """
        if input_shape is None:
            input_shape = (features,)
            
        try:
            model = keras.Sequential([
                # Input layer
                layers.Input(shape=input_shape),
                
                # Dense layers
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Dense model created with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating Dense model: {e}")
            raise
    
    def create_hybrid_model(self, input_shape: Tuple = None,
                           sequence_length: int = 60,
                           features: int = 19) -> keras.Model:
        """
        Create hybrid CNN+LSTM model
        
        Args:
            input_shape: Input shape tuple (sequence_length, features)
            sequence_length: Length of sequence
            features: Number of features
            
        Returns:
            keras.Model: Compiled Hybrid model
        """
        if input_shape is None:
            input_shape = (sequence_length, features)
            
        try:
            # Input
            inputs = layers.Input(shape=input_shape)
            
            # CNN branch
            cnn_branch = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
            cnn_branch = layers.BatchNormalization()(cnn_branch)
            cnn_branch = layers.MaxPooling1D(pool_size=2)(cnn_branch)
            cnn_branch = layers.Dropout(0.2)(cnn_branch)
            
            # LSTM branch
            lstm_branch = layers.LSTM(64, return_sequences=True, dropout=0.2)(cnn_branch)
            lstm_branch = layers.LSTM(32, return_sequences=False, dropout=0.2)(lstm_branch)
            
            # Dense layers
            dense = layers.Dense(64, activation='relu')(lstm_branch)
            dense = layers.Dropout(0.3)(dense)
            dense = layers.Dense(32, activation='relu')(dense)
            dense = layers.Dropout(0.2)(dense)
            
            # Output
            outputs = layers.Dense(1, activation='sigmoid')(dense)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Hybrid model created with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating Hybrid model: {e}")
            raise
    
    def create_model(self, model_type: str, **kwargs) -> keras.Model:
        """
        Create model by type
        
        Args:
            model_type: Type of model ('lstm', 'cnn', 'dense', 'hybrid')
            **kwargs: Additional arguments for model creation
            
        Returns:
            keras.Model: Compiled model
        """
        model_type = model_type.lower()
        
        if model_type == 'lstm':
            return self.create_lstm_model(**kwargs)
        elif model_type == 'cnn':
            return self.create_cnn_model(**kwargs)
        elif model_type == 'dense':
            return self.create_dense_model(**kwargs)
        elif model_type == 'hybrid':
            return self.create_hybrid_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        model_type = model_type.lower()
        if model_type in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_type].copy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        return list(self.MODEL_CONFIGS.keys())
    
    def get_standard_training_callbacks(self, model_name: str = "model") -> List:
        """
        Get standard training callbacks
        
        Args:
            model_name: Name for model checkpoint
            
        Returns:
            List: List of Keras callbacks
        """
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f'trained_models/unified_{model_name.lower()}.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def get_standard_training_params(self) -> Dict[str, Any]:
        """Get standard training parameters"""
        return {
            'epochs': 200,
            'batch_size': 64,
            'validation_split': 0.2,
            'shuffle': True,
            'verbose': 1
        }
    
    def create_ensemble_models(self, sequence_length: int = 60, 
                             features: int = 19) -> Dict[str, keras.Model]:
        """
        Create ensemble of all model types
        
        Args:
            sequence_length: Length of sequence for sequence models
            features: Number of features
            
        Returns:
            Dict[str, keras.Model]: Dictionary of models
        """
        models = {}
        
        try:
            # Create sequence models (LSTM, CNN, Hybrid)
            models['lstm'] = self.create_lstm_model(sequence_length=sequence_length, features=features)
            models['cnn'] = self.create_cnn_model(sequence_length=sequence_length, features=features)
            models['hybrid'] = self.create_hybrid_model(sequence_length=sequence_length, features=features)
            
            # Create dense model (flattened features)
            models['dense'] = self.create_dense_model(features=features)
            
            logger.info(f"Created ensemble of {len(models)} models: {list(models.keys())}")
            return models
            
        except Exception as e:
            logger.error(f"Error creating ensemble models: {e}")
            raise
    
    def validate_model_compatibility(self, model: keras.Model, 
                                   expected_features: int = 19) -> bool:
        """
        Validate that model is compatible with unified feature standard
        
        Args:
            model: Keras model to validate
            expected_features: Expected number of features
            
        Returns:
            bool: True if compatible
        """
        try:
            input_shape = model.input_shape
            
            # For sequence models, check last dimension
            if len(input_shape) == 3:  # (batch, sequence, features)
                return input_shape[-1] == expected_features
            elif len(input_shape) == 2:  # (batch, features)
                return input_shape[-1] == expected_features
            else:
                logger.warning(f"Unexpected input shape: {input_shape}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating model compatibility: {e}")
            return False 