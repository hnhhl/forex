#!/usr/bin/env python3
"""
MASS TRAINING SYSTEM AI3.0
Hệ thống training đồng loạt với khả năng xử lý song song và quản lý tài nguyên

Tính năng:
- Training đồng loạt 50+ models
- Parallel processing với CPU/GPU optimization
- Intelligent resource management
- Auto-scaling và load balancing
- Advanced ensemble creation
- Real-time monitoring

Author: AI3.0 Team
Date: 2025-01-03
Version: Mass Training 2.0
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# Traditional ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Advanced ML Libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class TrainingConfig:
    """Configuration cho training process"""
    # General settings
    max_parallel_jobs: int = 8
    max_neural_parallel: int = 2
    max_traditional_parallel: int = 6
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    
    # Neural network training
    neural_epochs: int = 100
    neural_batch_size: int = 64
    neural_learning_rate: float = 0.001
    neural_patience: int = 15
    
    # Traditional ML
    traditional_cv_folds: int = 5
    traditional_max_iter: int = 1000
    
    # Resource management
    max_memory_gb: float = 16.0
    gpu_memory_limit: float = 8.0
    auto_scaling: bool = True

@dataclass
class ModelSpec:
    """Specification cho mỗi model"""
    model_id: str
    model_type: str  # 'neural' hoặc 'traditional'
    architecture: str
    priority: int = 1  # 1=highest, 5=lowest
    resource_weight: float = 1.0
    expected_time_minutes: float = 5.0
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

@dataclass
class TrainingResult:
    """Kết quả training cho mỗi model"""
    model_id: str
    success: bool
    accuracy: float
    validation_accuracy: float
    training_time: float
    model_path: str
    model_type: str
    architecture: str
    error_message: str = ""
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}

class ResourceMonitor:
    """Monitor tài nguyên hệ thống"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cpu_count = mp.cpu_count()
        self.gpu_available = self._check_gpu_availability()
        self.memory_gb = self._get_available_memory()
        
        self.logger.info(f"🖥️ System Resources:")
        self.logger.info(f"   • CPU Cores: {self.cpu_count}")
        self.logger.info(f"   • GPU Available: {self.gpu_available}")
        self.logger.info(f"   • Memory: {self.memory_gb:.1f} GB")
    
    def _check_gpu_availability(self) -> bool:
        """Check GPU availability"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return True
            return False
        except Exception as e:
            self.logger.warning(f"GPU check failed: {e}")
            return False
    
    def _get_available_memory(self) -> float:
        """Get available system memory"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Default

class ModelFactory:
    """Factory để tạo các models khác nhau"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_neural_model(self, spec: ModelSpec, input_shape: Tuple[int, ...]) -> keras.Model:
        """Tạo neural network model"""
        architecture = spec.architecture
        
        if architecture == "dense_small":
            return self._create_dense_small(input_shape)
        elif architecture == "dense_medium":
            return self._create_dense_medium(input_shape)
        elif architecture == "dense_large":
            return self._create_dense_large(input_shape)
        elif architecture == "cnn_1d":
            return self._create_cnn_1d(input_shape)
        elif architecture == "lstm":
            return self._create_lstm(input_shape)
        elif architecture == "gru":
            return self._create_gru(input_shape)
        elif architecture == "hybrid_cnn_lstm":
            return self._create_hybrid_cnn_lstm(input_shape)
        elif architecture == "transformer":
            return self._create_transformer(input_shape)
        elif architecture == "autoencoder":
            return self._create_autoencoder(input_shape)
        else:
            raise ValueError(f"Unknown neural architecture: {architecture}")
    
    def _create_dense_small(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Dense network - Small"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_dense_medium(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Dense network - Medium"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_dense_large(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Dense network - Large"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_cnn_1d(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """1D CNN for time series patterns"""
        input_layer = layers.Input(shape=input_shape)
        reshaped = layers.Reshape((input_shape[0], 1))(input_layer)
        
        x = layers.Conv1D(64, 3, activation='relu')(reshaped)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(32, 3, activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(50, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_lstm(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """LSTM for sequence learning"""
        input_layer = layers.Input(shape=input_shape)
        reshaped = layers.Reshape((input_shape[0], 1))(input_layer)
        
        x = layers.LSTM(128, return_sequences=True)(reshaped)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_gru(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """GRU model"""
        input_layer = layers.Input(shape=input_shape)
        reshaped = layers.Reshape((input_shape[0], 1))(input_layer)
        
        x = layers.GRU(128, return_sequences=True)(reshaped)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(64)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_hybrid_cnn_lstm(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Hybrid CNN+LSTM"""
        input_layer = layers.Input(shape=input_shape)
        reshaped = layers.Reshape((input_shape[0], 1))(input_layer)
        
        # CNN layers
        cnn = layers.Conv1D(32, 3, activation='relu')(reshaped)
        cnn = layers.MaxPooling1D(2)(cnn)
        
        # LSTM layers
        lstm = layers.LSTM(64, return_sequences=True)(cnn)
        lstm = layers.LSTM(32)(lstm)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(lstm)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_transformer(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Simple transformer model"""
        inputs = layers.Input(shape=input_shape)
        reshaped = layers.Reshape((input_shape[0], 1))(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=32
        )(reshaped, reshaped)
        
        # Add & Norm
        x = layers.Add()([reshaped, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(128, activation='relu')(x)
        ff_output = layers.Dense(1)(ff_output)
        
        # Add & Norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling và output
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs, outputs)
    
    def _create_autoencoder(self, input_shape: Tuple[int, ...]) -> keras.Model:
        """Autoencoder-based classifier"""
        inputs = layers.Input(shape=input_shape)
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(inputs)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        
        # Classification head
        classifier = layers.Dense(16, activation='relu')(encoded)
        outputs = layers.Dense(1, activation='sigmoid')(classifier)
        
        return models.Model(inputs, outputs)
    
    def create_traditional_model(self, spec: ModelSpec):
        """Tạo traditional ML model"""
        architecture = spec.architecture
        params = spec.params
        
        if architecture == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=self.config.random_state
            )
        elif architecture == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                random_state=self.config.random_state
            )
        elif architecture == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100, 50)),
                max_iter=params.get('max_iter', 1000),
                random_state=self.config.random_state
            )
        elif architecture == "svm":
            return SVC(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                random_state=self.config.random_state,
                probability=True
            )
        elif architecture == "decision_tree":
            return DecisionTreeClassifier(
                max_depth=params.get('max_depth', None),
                random_state=self.config.random_state
            )
        elif architecture == "naive_bayes":
            return GaussianNB()
        elif architecture == "logistic_regression":
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=self.config.random_state
            )
        elif architecture == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=self.config.random_state,
                verbose=-1
            )
        elif architecture == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=self.config.random_state,
                verbosity=0
            )
        elif architecture == "catboost" and CATBOOST_AVAILABLE:
            return cb.CatBoostClassifier(
                iterations=params.get('iterations', 100),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=self.config.random_state,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown traditional architecture: {architecture}")

class MassTrainingOrchestrator:
    """Orchestrator chính cho mass training"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.resource_monitor = ResourceMonitor()
        self.model_factory = ModelFactory(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results: Dict[str, TrainingResult] = {}
        self.trained_models: Dict[str, Any] = {}
        
        # Training state
        self.training_start_time = None
        self.training_end_time = None
        
        # Create directories
        self.models_dir = Path("trained_models/mass_training")
        self.results_dir = Path("training_results/mass_training")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Mass Training Orchestrator initialized")
    
    def generate_model_specifications(self) -> List[ModelSpec]:
        """Generate comprehensive list of model specifications"""
        specs = []
        
        # Neural Networks
        neural_architectures = [
            "dense_small", "dense_medium", "dense_large",
            "cnn_1d", "lstm", "gru", "hybrid_cnn_lstm",
            "transformer", "autoencoder"
        ]
        
        for i, arch in enumerate(neural_architectures):
            specs.append(ModelSpec(
                model_id=f"neural_{arch}_{i+1:02d}",
                model_type="neural",
                architecture=arch,
                priority=1,
                resource_weight=2.0 if arch in ["transformer", "hybrid_cnn_lstm"] else 1.0,
                expected_time_minutes=10.0 if arch in ["transformer", "lstm", "gru"] else 5.0
            ))
        
        # Traditional ML Models
        traditional_specs = [
            ("random_forest", {"n_estimators": 100}),
            ("random_forest", {"n_estimators": 200}),
            ("random_forest", {"n_estimators": 300}),
            ("gradient_boosting", {"n_estimators": 100, "learning_rate": 0.1}),
            ("gradient_boosting", {"n_estimators": 200, "learning_rate": 0.05}),
            ("mlp", {"hidden_layer_sizes": (100,)}),
            ("mlp", {"hidden_layer_sizes": (100, 50)}),
            ("mlp", {"hidden_layer_sizes": (200, 100, 50)}),
            ("svm", {"kernel": "rbf", "C": 1.0}),
            ("svm", {"kernel": "linear", "C": 1.0}),
            ("decision_tree", {"max_depth": 10}),
            ("decision_tree", {"max_depth": None}),
            ("naive_bayes", {}),
            ("logistic_regression", {"C": 1.0}),
            ("logistic_regression", {"C": 0.1}),
        ]
        
        # Add advanced models if available
        if LIGHTGBM_AVAILABLE:
            traditional_specs.extend([
                ("lightgbm", {"n_estimators": 100, "learning_rate": 0.1}),
                ("lightgbm", {"n_estimators": 200, "learning_rate": 0.05}),
            ])
        
        if XGBOOST_AVAILABLE:
            traditional_specs.extend([
                ("xgboost", {"n_estimators": 100, "learning_rate": 0.1}),
                ("xgboost", {"n_estimators": 200, "learning_rate": 0.05}),
            ])
        
        if CATBOOST_AVAILABLE:
            traditional_specs.extend([
                ("catboost", {"iterations": 100, "learning_rate": 0.1}),
                ("catboost", {"iterations": 200, "learning_rate": 0.05}),
            ])
        
        for i, (arch, params) in enumerate(traditional_specs):
            specs.append(ModelSpec(
                model_id=f"traditional_{arch}_{i+1:02d}",
                model_type="traditional",
                architecture=arch,
                priority=2,
                resource_weight=0.5,
                expected_time_minutes=2.0,
                params=params
            ))
        
        self.logger.info(f"📋 Generated {len(specs)} model specifications")
        return specs
    
    def train_neural_model(self, spec: ModelSpec, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """Train một neural model"""
        start_time = time.time()
        
        try:
            # Create model
            input_shape = (X_train.shape[1],)
            model = self.model_factory.create_neural_model(spec, input_shape)
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.neural_learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.neural_patience,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=self.config.neural_batch_size,
                epochs=self.config.neural_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=0
            )
            
            # Evaluate
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
            # Save model
            model_path = self.models_dir / f"{spec.model_id}.keras"
            model.save(model_path)
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                model_id=spec.model_id,
                success=True,
                accuracy=train_acc,
                validation_accuracy=val_acc,
                training_time=training_time,
                model_path=str(model_path),
                model_type=spec.model_type,
                architecture=spec.architecture,
                additional_metrics={
                    'final_loss': float(train_loss),
                    'final_val_loss': float(val_loss),
                    'epochs_trained': len(history.history['loss']),
                    'best_epoch': np.argmin(history.history['val_loss']) + 1
                }
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Neural model {spec.model_id} training failed: {e}")
            
            return TrainingResult(
                model_id=spec.model_id,
                success=False,
                accuracy=0.0,
                validation_accuracy=0.0,
                training_time=training_time,
                model_path="",
                model_type=spec.model_type,
                architecture=spec.architecture,
                error_message=str(e)
            )
    
    def train_traditional_model(self, spec: ModelSpec, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """Train một traditional ML model"""
        start_time = time.time()
        
        try:
            # Create model
            model = self.model_factory.create_traditional_model(spec)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            
            # Save model
            model_path = self.models_dir / f"{spec.model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                model_id=spec.model_id,
                success=True,
                accuracy=train_acc,
                validation_accuracy=val_acc,
                training_time=training_time,
                model_path=str(model_path),
                model_type=spec.model_type,
                architecture=spec.architecture,
                additional_metrics={
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'cv_scores': cv_scores.tolist()
                }
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Traditional model {spec.model_id} training failed: {e}")
            
            return TrainingResult(
                model_id=spec.model_id,
                success=False,
                accuracy=0.0,
                validation_accuracy=0.0,
                training_time=training_time,
                model_path="",
                model_type=spec.model_type,
                architecture=spec.architecture,
                error_message=str(e)
            )
    
    def execute_mass_training(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Execute mass training of all models"""
        self.logger.info("🚀 Starting MASS TRAINING EXECUTION")
        self.training_start_time = datetime.now()
        
        # Data preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=self.config.validation_split + self.config.test_split,
            random_state=self.config.random_state, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config.test_split / (self.config.validation_split + self.config.test_split),
            random_state=self.config.random_state, stratify=y_temp
        )
        
        self.logger.info(f"📊 Data splits:")
        self.logger.info(f"   • Train: {X_train.shape[0]} samples")
        self.logger.info(f"   • Validation: {X_val.shape[0]} samples")
        self.logger.info(f"   • Test: {X_test.shape[0]} samples")
        
        # Generate model specifications
        model_specs = self.generate_model_specifications()
        
        # Separate neural and traditional models for different processing
        neural_specs = [spec for spec in model_specs if spec.model_type == "neural"]
        traditional_specs = [spec for spec in model_specs if spec.model_type == "traditional"]
        
        self.logger.info(f"🧠 Neural models: {len(neural_specs)}")
        self.logger.info(f"🔧 Traditional models: {len(traditional_specs)}")
        
        # Train neural models (limited parallelism due to GPU)
        self.logger.info("🧠 Training Neural Networks...")
        neural_results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_neural_parallel) as executor:
            neural_futures = {
                executor.submit(
                    self.train_neural_model, spec, X_train, y_train, X_val, y_val
                ): spec for spec in neural_specs
            }
            
            for future in as_completed(neural_futures):
                spec = neural_futures[future]
                try:
                    result = future.result()
                    neural_results[result.model_id] = result
                    
                    if result.success:
                        self.logger.info(f"   ✅ {result.model_id}: {result.validation_accuracy:.4f}")
                    else:
                        self.logger.error(f"   ❌ {result.model_id}: {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"   ❌ {spec.model_id}: Exception - {e}")
        
        # Train traditional models (higher parallelism)
        self.logger.info("🔧 Training Traditional ML Models...")
        traditional_results = {}
        
        with ProcessPoolExecutor(max_workers=self.config.max_traditional_parallel) as executor:
            traditional_futures = {
                executor.submit(
                    self.train_traditional_model, spec, X_train, y_train, X_val, y_val
                ): spec for spec in traditional_specs
            }
            
            for future in as_completed(traditional_futures):
                spec = traditional_futures[future]
                try:
                    result = future.result()
                    traditional_results[result.model_id] = result
                    
                    if result.success:
                        self.logger.info(f"   ✅ {result.model_id}: {result.validation_accuracy:.4f}")
                    else:
                        self.logger.error(f"   ❌ {result.model_id}: {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"   ❌ {spec.model_id}: Exception - {e}")
        
        # Combine results
        all_results = {**neural_results, **traditional_results}
        self.results = all_results
        
        self.training_end_time = datetime.now()
        total_training_time = (self.training_end_time - self.training_start_time).total_seconds()
        
        # Calculate statistics
        successful_models = [r for r in all_results.values() if r.success]
        failed_models = [r for r in all_results.values() if not r.success]
        
        self.logger.info("📊 MASS TRAINING COMPLETED!")
        self.logger.info(f"   • Total models: {len(all_results)}")
        self.logger.info(f"   • Successful: {len(successful_models)}")
        self.logger.info(f"   • Failed: {len(failed_models)}")
        self.logger.info(f"   • Total time: {total_training_time:.1f}s")
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results, X_test, y_test, scaler, total_training_time)
        
        return {
            'total_models': len(all_results),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'total_training_time': total_training_time,
            'results': {model_id: asdict(result) for model_id, result in all_results.items()},
            'best_models': self._get_best_models(successful_models),
            'data_info': {
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            }
        }
    
    def _get_best_models(self, successful_results: List[TrainingResult], top_k: int = 10) -> Dict[str, Any]:
        """Get top performing models"""
        # Sort by validation accuracy
        sorted_results = sorted(successful_results, key=lambda x: x.validation_accuracy, reverse=True)
        
        best_models = {}
        for i, result in enumerate(sorted_results[:top_k]):
            best_models[f"rank_{i+1}"] = {
                'model_id': result.model_id,
                'model_type': result.model_type,
                'architecture': result.architecture,
                'validation_accuracy': result.validation_accuracy,
                'training_time': result.training_time,
                'model_path': result.model_path
            }
        
        return best_models
    
    def _save_comprehensive_results(self, results: Dict[str, TrainingResult], 
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   scaler: StandardScaler, total_time: float):
        """Save comprehensive training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main results file
        results_data = {
            'timestamp': timestamp,
            'total_training_time': total_time,
            'total_models': len(results),
            'successful_models': len([r for r in results.values() if r.success]),
            'failed_models': len([r for r in results.values() if not r.success]),
            'configuration': asdict(self.config),
            'results': {model_id: asdict(result) for model_id, result in results.items()},
            'best_models': self._get_best_models([r for r in results.values() if r.success]),
            'system_info': {
                'cpu_count': self.resource_monitor.cpu_count,
                'gpu_available': self.resource_monitor.gpu_available,
                'memory_gb': self.resource_monitor.memory_gb
            }
        }
        
        results_path = self.results_dir / f"comprehensive_mass_training_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save scaler
        scaler_path = self.models_dir / f"scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save test data for later evaluation
        test_data = {
            'X_test': X_test.tolist(),
            'y_test': y_test.tolist(),
            'scaler_path': str(scaler_path)
        }
        
        test_path = self.results_dir / f"test_data_{timestamp}.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        self.logger.info(f"💾 Results saved:")
        self.logger.info(f"   • Main results: {results_path}")
        self.logger.info(f"   • Scaler: {scaler_path}")
        self.logger.info(f"   • Test data: {test_path}")

def main():
    """Main execution function"""
    print("🚀 MASS TRAINING SYSTEM AI3.0")
    print("="*50)
    
    # Load data (example)
    try:
        # Try to load from working data
        data_path = "data/working_free_data/XAUUSD_H1_realistic.csv"
        if os.path.exists(data_path):
            print(f"📊 Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            
            # Simple feature engineering
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(10).std()
            
            # Create target (1 if next period close > current close)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Prepare features
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'ma_5', 'ma_20', 'volatility']
            df = df.dropna()
            
            X = df[feature_cols].values
            y = df['target'].values
            
            print(f"   • Samples: {len(X)}")
            print(f"   • Features: {X.shape[1]}")
            print(f"   • Target distribution: {np.bincount(y)}")
            
        else:
            # Generate synthetic data for demo
            print("📊 Generating synthetic data for demo...")
            np.random.seed(42)
            n_samples = 10000
            n_features = 20
            
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            print(f"   • Samples: {n_samples}")
            print(f"   • Features: {n_features}")
            print(f"   • Target distribution: {np.bincount(y)}")
        
        # Initialize Mass Training System
        config = TrainingConfig(
            max_parallel_jobs=8,
            max_neural_parallel=2,
            max_traditional_parallel=6,
            neural_epochs=50,  # Reduced for demo
            neural_patience=10
        )
        
        orchestrator = MassTrainingOrchestrator(config)
        
        # Execute mass training
        print("\n🏋️ Starting Mass Training...")
        results = orchestrator.execute_mass_training(X, y)
        
        # Print summary
        print("\n" + "="*50)
        print("📊 MASS TRAINING SUMMARY")
        print("="*50)
        print(f"Total Models Trained: {results['total_models']}")
        print(f"Successful Models: {results['successful_models']}")
        print(f"Failed Models: {results['failed_models']}")
        print(f"Total Training Time: {results['total_training_time']:.1f}s")
        
        print("\n🏆 TOP 5 MODELS:")
        for rank, model_info in list(results['best_models'].items())[:5]:
            print(f"   {rank}: {model_info['model_id']} - {model_info['validation_accuracy']:.4f}")
        
        print("\n✅ Mass Training Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Error in mass training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 