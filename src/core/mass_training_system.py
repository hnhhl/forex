#!/usr/bin/env python3
"""
Mass Training System - Training 45 Models Simultaneously
Hệ thống training đồng loạt 45 models cho Enhanced Ensemble Parliament

Author: AI Assistant  
Date: 2025-01-03
Version: Mass Training 1.0
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# Traditional ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Try additional libraries
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

@dataclass
class TrainingJob:
    """Training job configuration"""
    model_id: str
    model_type: str
    architecture: str
    priority: int
    expected_time: float
    resource_requirements: Dict[str, Any]

@dataclass
class TrainingResult:
    """Training result"""
    model_id: str
    success: bool
    accuracy: float
    training_time: float
    model_path: str
    error_message: str = ""

class ResourceManager:
    """Quản lý tài nguyên cho mass training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # System resources
        self.cpu_count = mp.cpu_count()
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        self.memory_gb = self._get_available_memory()
        
        # Resource allocation
        self.max_parallel_neural = 2 if self.gpu_available else 1
        self.max_parallel_ml = min(4, self.cpu_count // 2)
        self.max_total_parallel = min(6, self.cpu_count)
        
        self.logger.info(f"🖥️ Resource Manager initialized:")
        self.logger.info(f"   • CPU Cores: {self.cpu_count}")
        self.logger.info(f"   • GPU Available: {self.gpu_available}")
        self.logger.info(f"   • Memory: {self.memory_gb:.1f} GB")
        self.logger.info(f"   • Max Parallel Neural: {self.max_parallel_neural}")
        self.logger.info(f"   • Max Parallel ML: {self.max_parallel_ml}")
    
    def _get_available_memory(self) -> float:
        """Estimate available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Default assumption
    
    def can_run_job(self, job: TrainingJob, current_jobs: int) -> bool:
        """Check if job can run given current resource usage"""
        if current_jobs >= self.max_total_parallel:
            return False
        
        if job.model_type == "neural" and current_jobs >= self.max_parallel_neural:
            return False
        
        if job.model_type == "traditional" and current_jobs >= self.max_parallel_ml:
            return False
        
        return True

class ModelFactory:
    """Factory để tạo các loại models khác nhau"""
    
    def __init__(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.logger = logging.getLogger(__name__)
    
    def create_neural_model(self, architecture: str) -> keras.Model:
        """Tạo neural network model"""
        if architecture == "dense":
            return self._create_dense_model()
        elif architecture == "cnn":
            return self._create_cnn_model()
        elif architecture == "lstm":
            return self._create_lstm_model()
        elif architecture == "hybrid":
            return self._create_hybrid_model()
        elif architecture == "gru":
            return self._create_gru_model()
        elif architecture == "transformer":
            return self._create_transformer_model()
        else:
            raise ValueError(f"Unknown neural architecture: {architecture}")
    
    def _create_dense_model(self) -> keras.Model:
        """Dense model for unified features"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=self.input_shape),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _create_cnn_model(self) -> keras.Model:
        """CNN model for pattern recognition"""
        # Reshape for CNN if needed
        input_layer = layers.Input(shape=self.input_shape)
        reshaped = layers.Reshape((self.input_shape[0], 1))(input_layer)
        
        x = layers.Conv1D(64, 3, activation='relu')(reshaped)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(32, 3, activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(50, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_lstm_model(self) -> keras.Model:
        """LSTM model for sequence learning"""
        input_layer = layers.Input(shape=self.input_shape)
        reshaped = layers.Reshape((self.input_shape[0], 1))(input_layer)
        
        x = layers.LSTM(128, return_sequences=True)(reshaped)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_hybrid_model(self) -> keras.Model:
        """Hybrid CNN+LSTM model"""
        input_layer = layers.Input(shape=self.input_shape)
        reshaped = layers.Reshape((self.input_shape[0], 1))(input_layer)
        
        # CNN branch
        cnn_branch = layers.Conv1D(32, 3, activation='relu')(reshaped)
        cnn_branch = layers.MaxPooling1D(2)(cnn_branch)
        
        # LSTM branch
        lstm_branch = layers.LSTM(64, return_sequences=True)(cnn_branch)
        lstm_branch = layers.LSTM(32)(lstm_branch)
        
        # Combine
        x = layers.Dense(64, activation='relu')(lstm_branch)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_gru_model(self) -> keras.Model:
        """GRU model"""
        input_layer = layers.Input(shape=self.input_shape)
        reshaped = layers.Reshape((self.input_shape[0], 1))(input_layer)
        
        x = layers.GRU(128, return_sequences=True)(reshaped)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(64)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def _create_transformer_model(self) -> keras.Model:
        """Simplified Transformer model"""
        input_layer = layers.Input(shape=self.input_shape)
        reshaped = layers.Reshape((self.input_shape[0], 1))(input_layer)
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(reshaped, reshaped)
        attention = layers.Add()([reshaped, attention])
        attention = layers.LayerNormalization()(attention)
        
        # Feed forward
        ff = layers.Dense(64, activation='relu')(attention)
        ff = layers.Dense(1)(ff)
        ff = layers.Add()([attention, ff])
        ff = layers.LayerNormalization()(ff)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(ff)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=input_layer, outputs=output)
    
    def create_traditional_model(self, architecture: str, **params):
        """Tạo traditional ML model"""
        if architecture == "random_forest":
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        elif architecture == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=42
            )
        elif architecture == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=42,
                eval_metric='logloss'
            )
        elif architecture == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 6),
                random_state=42,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown traditional architecture: {architecture}")

class MassTrainingSystem:
    """
    Mass Training System - Training 45 Models Simultaneously
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.resource_manager = ResourceManager()
        self.model_factory = None  # Will be initialized with data
        
        # Training configuration
        self.training_jobs: List[TrainingJob] = []
        self.training_results: List[TrainingResult] = []
        
        # Data
        self.training_data = None
        self.validation_data = None
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.logger.info("🏭 Mass Training System initialized")
    
    def prepare_training_jobs(self, input_shape: Tuple[int, ...]) -> List[TrainingJob]:
        """Chuẩn bị danh sách training jobs cho 45 models"""
        jobs = []
        
        # Initialize model factory
        self.model_factory = ModelFactory(input_shape)
        
        # 1. UNIFIED MODELS (4 models) - Highest Priority
        unified_architectures = ["dense", "cnn", "lstm", "hybrid"]
        for i, arch in enumerate(unified_architectures):
            jobs.append(TrainingJob(
                model_id=f"unified_{arch}",
                model_type="neural",
                architecture=arch,
                priority=1,  # Highest priority
                expected_time=300.0,  # 5 minutes
                resource_requirements={"gpu": True, "memory_gb": 2.0}
            ))
        
        # 2. NEURAL VARIANTS (20 models) - High Priority
        neural_variants = [
            ("dense_v2", "dense"), ("dense_v3", "dense"), ("dense_v4", "dense"),
            ("cnn_v2", "cnn"), ("cnn_v3", "cnn"), ("cnn_v4", "cnn"),
            ("lstm_v2", "lstm"), ("lstm_v3", "lstm"), ("lstm_v4", "lstm"),
            ("hybrid_v2", "hybrid"), ("hybrid_v3", "hybrid"), ("hybrid_v4", "hybrid"),
            ("gru_v1", "gru"), ("gru_v2", "gru"), ("gru_v3", "gru"),
            ("transformer_v1", "transformer"), ("transformer_v2", "transformer"),
            ("transformer_v3", "transformer"), ("transformer_v4", "transformer"),
            ("transformer_v5", "transformer")
        ]
        
        for model_id, arch in neural_variants:
            jobs.append(TrainingJob(
                model_id=model_id,
                model_type="neural",
                architecture=arch,
                priority=2,
                expected_time=240.0,  # 4 minutes
                resource_requirements={"gpu": False, "memory_gb": 1.5}
            ))
        
        # 3. TRADITIONAL ML MODELS (21 models) - Medium Priority
        traditional_configs = []
        
        # Random Forest variants
        for i, params in enumerate([
            {"n_estimators": 50, "max_depth": 8},
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 150, "max_depth": 12},
            {"n_estimators": 200, "max_depth": 15},
            {"n_estimators": 100, "max_depth": 8}
        ]):
            traditional_configs.append((f"random_forest_v{i+1}", "random_forest", params))
        
        # Gradient Boosting variants
        for i, params in enumerate([
            {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4},
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 8},
            {"n_estimators": 100, "learning_rate": 0.2, "max_depth": 4},
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6}
        ]):
            traditional_configs.append((f"gradient_boosting_v{i+1}", "gradient_boosting", params))
        
        # XGBoost variants (if available)
        if XGBOOST_AVAILABLE:
            for i, params in enumerate([
                {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4},
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
                {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 8},
                {"n_estimators": 100, "learning_rate": 0.2, "max_depth": 4},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6}
            ]):
                traditional_configs.append((f"xgboost_v{i+1}", "xgboost", params))
        
        # LightGBM variants (if available)
        if LIGHTGBM_AVAILABLE:
            for i, params in enumerate([
                {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4},
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
                {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 8},
                {"n_estimators": 100, "learning_rate": 0.2, "max_depth": 4},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
                {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 8}
            ]):
                traditional_configs.append((f"lightgbm_v{i+1}", "lightgbm", params))
        
        # Add traditional jobs
        for model_id, arch, params in traditional_configs:
            jobs.append(TrainingJob(
                model_id=model_id,
                model_type="traditional",
                architecture=arch,
                priority=3,
                expected_time=120.0,  # 2 minutes
                resource_requirements={"gpu": False, "memory_gb": 1.0}
            ))
        
        # Sort by priority
        jobs.sort(key=lambda x: x.priority)
        
        self.training_jobs = jobs
        self.logger.info(f"📋 Prepared {len(jobs)} training jobs:")
        self.logger.info(f"   • Neural models: {len([j for j in jobs if j.model_type == 'neural'])}")
        self.logger.info(f"   • Traditional models: {len([j for j in jobs if j.model_type == 'traditional'])}")
        
        return jobs
    
    def train_single_model(self, job: TrainingJob, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """Train một model duy nhất"""
        start_time = time.time()
        
        try:
            model_path = os.path.join(self.models_dir, f"mass_{job.model_id}.keras")
            
            if job.model_type == "neural":
                # Train neural network
                model = self.model_factory.create_neural_model(job.architecture)
                
                # Compile model
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with callbacks
                callbacks_list = [
                    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks_list,
                    verbose=0
                )
                
                # Evaluate
                _, accuracy = model.evaluate(X_val, y_val, verbose=0)
                
                # Save model
                model.save(model_path)
                
            else:
                # Train traditional ML
                # Extract architecture and params
                arch_parts = job.architecture.split('_')
                base_arch = '_'.join(arch_parts[:-1]) if len(arch_parts) > 1 else job.architecture
                
                # Default params (will be improved with proper param extraction)
                params = {"n_estimators": 100, "max_depth": 10}
                
                model = self.model_factory.create_traditional_model(base_arch, **params)
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                # Save model
                model_path = model_path.replace('.keras', '.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                model_id=job.model_id,
                success=True,
                accuracy=accuracy,
                training_time=training_time,
                model_path=model_path
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            return TrainingResult(
                model_id=job.model_id,
                success=False,
                accuracy=0.0,
                training_time=training_time,
                model_path="",
                error_message=str(e)
            )
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        validation_split: float = 0.2) -> Dict[str, Any]:
        """Training tất cả 45 models với parallel processing"""
        
        # Prepare data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Prepare training jobs
        input_shape = (X_train.shape[1],)
        jobs = self.prepare_training_jobs(input_shape)
        
        self.logger.info(f"🚀 Starting mass training of {len(jobs)} models...")
        self.logger.info(f"   • Training data: {X_train.shape}")
        self.logger.info(f"   • Validation data: {X_val.shape}")
        
        # Parallel training with resource management
        results = []
        completed_jobs = 0
        
        with ThreadPoolExecutor(max_workers=self.resource_manager.max_total_parallel) as executor:
            # Submit jobs based on resource availability
            future_to_job = {}
            
            for job in jobs:
                if self.resource_manager.can_run_job(job, len(future_to_job)):
                    future = executor.submit(
                        self.train_single_model, job, X_train, y_train, X_val, y_val
                    )
                    future_to_job[future] = job
                    self.logger.info(f"   🔄 Started training: {job.model_id}")
            
            # Process completed jobs and submit new ones
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                result = future.result()
                results.append(result)
                completed_jobs += 1
                
                if result.success:
                    self.logger.info(f"   ✅ {job.model_id}: {result.accuracy:.3f} accuracy ({result.training_time:.1f}s)")
                else:
                    self.logger.warning(f"   ❌ {job.model_id}: {result.error_message}")
                
                # Submit next job if available
                remaining_jobs = [j for j in jobs if j not in [future_to_job[f] for f in future_to_job]]
                if remaining_jobs:
                    next_job = remaining_jobs[0]
                    if self.resource_manager.can_run_job(next_job, len(future_to_job) - 1):
                        future = executor.submit(
                            self.train_single_model, next_job, X_train, y_train, X_val, y_val
                        )
                        future_to_job[future] = next_job
                        self.logger.info(f"   🔄 Started training: {next_job.model_id}")
        
        # Analyze results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            avg_accuracy = np.mean([r.accuracy for r in successful_results])
            best_model = max(successful_results, key=lambda x: x.accuracy)
            total_training_time = sum([r.training_time for r in successful_results])
        else:
            avg_accuracy = 0.0
            best_model = None
            total_training_time = 0.0
        
        training_summary = {
            'total_models': len(jobs),
            'successful_models': len(successful_results),
            'failed_models': len(failed_results),
            'success_rate': len(successful_results) / len(jobs),
            'average_accuracy': avg_accuracy,
            'best_model': {
                'model_id': best_model.model_id if best_model else None,
                'accuracy': best_model.accuracy if best_model else 0.0,
                'path': best_model.model_path if best_model else ""
            },
            'total_training_time': total_training_time,
            'results': results
        }
        
        self.training_results = results
        
        self.logger.info(f"🎯 MASS TRAINING COMPLETED:")
        self.logger.info(f"   • Total models: {training_summary['total_models']}")
        self.logger.info(f"   • Successful: {training_summary['successful_models']}")
        self.logger.info(f"   • Failed: {training_summary['failed_models']}")
        self.logger.info(f"   • Success rate: {training_summary['success_rate']:.1%}")
        self.logger.info(f"   • Average accuracy: {training_summary['average_accuracy']:.3f}")
        self.logger.info(f"   • Best model: {training_summary['best_model']['model_id']} ({training_summary['best_model']['accuracy']:.3f})")
        self.logger.info(f"   • Total training time: {training_summary['total_training_time']:.1f}s")
        
        return training_summary

if __name__ == "__main__":
    # Demo Mass Training System
    logging.basicConfig(level=logging.INFO)
    
    print("🏭 MASS TRAINING SYSTEM DEMO")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.random((1000, 19))  # 19 unified features
    y = (X.mean(axis=1) > 0.5).astype(int)  # Simple binary target
    
    # Initialize mass training system
    mass_trainer = MassTrainingSystem()
    
    # Train all models
    results = mass_trainer.train_all_models(X, y)
    
    print(f"\n🎉 Mass training completed!")
    print(f"   Success rate: {results['success_rate']:.1%}")
    print(f"   Best accuracy: {results['best_model']['accuracy']:.3f}") 