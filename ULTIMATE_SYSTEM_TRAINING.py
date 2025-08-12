"""
🚀 ULTIMATE XAU SYSTEM V5.0 - COMPLETE TRAINING
Training hệ thống hoàn chỉnh với dữ liệu unified multi-timeframe

TRAINING SCOPE:
- Unified Multi-Timeframe Data (62,727 samples)
- Neural Ensemble Training
- DQN Agent Training  
- Meta Learning Training
- Complete AI Coordination
- Production Model Export
"""

import asyncio
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import logging
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow
tf.get_logger().setLevel('ERROR')
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
except Exception:
    pass  # No GPU or configuration failed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateXAUSystemTraining:
    """
    🎯 ULTIMATE XAU SYSTEM V5.0 - COMPLETE TRAINING
    Training toàn bộ hệ thống AI với dữ liệu unified
    """
    
    def __init__(self):
        self.version = "5.0"
        self.training_start_time = datetime.now()
        
        # Data containers
        self.unified_data = None
        self.training_data = {}
        self.models = {}
        self.scalers = {}
        self.training_results = {}
        
        # Training configuration
        self.timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        self.model_types = [
            'neural_ensemble', 'dqn_agent', 'meta_learning',
            'random_forest', 'gradient_boost', 'xgboost', 'lightgbm'
        ]
        
        logger.info("🚀 Ultimate XAU System V5.0 Training initialized")
    
    async def execute_complete_training(self):
        """Execute complete system training"""
        try:
            logger.info("🔄 Starting complete system training...")
            
            # Phase 1: Load và prepare unified data
            await self._load_unified_data()
            
            # Phase 2: Prepare training datasets
            await self._prepare_training_datasets()
            
            # Phase 3: Train Neural Ensemble
            await self._train_neural_ensemble()
            
            # Phase 4: Train DQN Agent
            await self._train_dqn_agent()
            
            # Phase 5: Train Meta Learning System
            await self._train_meta_learning()
            
            # Phase 6: Train Traditional ML Models
            await self._train_traditional_models()
            
            # Phase 7: Create AI Coordination
            await self._create_ai_coordination()
            
            # Phase 8: Validate và export models
            await self._validate_and_export_models()
            
            # Phase 9: Generate training report
            await self._generate_training_report()
            
            logger.info("✅ Complete system training finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    async def _load_unified_data(self):
        """Load unified multi-timeframe data"""
        logger.info("📊 Loading unified multi-timeframe data...")
        
        try:
            # Load từ UNIFIED_DATA_SOLUTION results
            from UNIFIED_DATA_SOLUTION import UnifiedMultiTimeframeDataSolution
            
            solution = UnifiedMultiTimeframeDataSolution()
            self.unified_data = solution.build_unified_dataset()
            
            if self.unified_data:
                logger.info(f"✅ Unified data loaded: {self.unified_data['X'].shape}")
                logger.info(f"   • Features: {len(self.unified_data['feature_names'])}")
                logger.info(f"   • Targets: {len(self.unified_data['targets'])}")
                logger.info(f"   • Timeframes: {len(self.unified_data['original_timeframes'])}")
            else:
                # Fallback: Load individual timeframe data
                await self._load_individual_timeframe_data()
                
        except Exception as e:
            logger.warning(f"⚠️ Unified data loading failed: {e}")
            # Fallback to individual timeframe data
            await self._load_individual_timeframe_data()
    
    async def _load_individual_timeframe_data(self):
        """Fallback: Load individual timeframe data"""
        logger.info("📊 Loading individual timeframe data as fallback...")
        
        all_data = {}
        total_samples = 0
        
        for tf in self.timeframes:
            try:
                data_file = f'training/xauusdc/data/{tf}_data.pkl'
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and 'X' in data:
                    all_data[tf] = data
                    total_samples += data['X'].shape[0]
                    logger.info(f"   • {tf}: {data['X'].shape[0]:,} samples")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to load {tf}: {e}")
        
        # Create unified dataset from individual data
        if all_data:
            self.unified_data = await self._create_unified_from_individual(all_data)
            logger.info(f"✅ Fallback unified data created: {total_samples:,} total samples")
        else:
            raise Exception("No training data available")
    
    async def _create_unified_from_individual(self, all_data):
        """Create unified dataset from individual timeframe data"""
        logger.info("🔗 Creating unified dataset from individual timeframes...")
        
        # Use M15 as base (best performing)
        base_tf = 'M15'
        if base_tf not in all_data:
            base_tf = list(all_data.keys())[0]
        
        base_data = all_data[base_tf]
        
        # Combine features from all timeframes
        combined_features = []
        combined_feature_names = []
        
        # Add base timeframe features
        combined_features.append(base_data['X'])
        base_feature_names = [f"{base_tf}_{name}" for name in base_data['feature_names']]
        combined_feature_names.extend(base_feature_names)
        
        # Add other timeframes (simplified alignment)
        for tf, data in all_data.items():
            if tf != base_tf:
                tf_features = data['X']
                
                # Simple alignment: truncate or pad to match base length
                base_length = base_data['X'].shape[0]
                if tf_features.shape[0] > base_length:
                    tf_features = tf_features[:base_length]
                elif tf_features.shape[0] < base_length:
                    # Pad with last values
                    padding_needed = base_length - tf_features.shape[0]
                    last_values = np.repeat(tf_features[-1:], padding_needed, axis=0)
                    tf_features = np.vstack([tf_features, last_values])
                
                combined_features.append(tf_features)
                tf_feature_names = [f"{tf}_{name}" for name in data['feature_names']]
                combined_feature_names.extend(tf_feature_names)
        
        # Combine all features
        unified_X = np.hstack(combined_features)
        
        # Use base timeframe targets
        unified_targets = base_data.get('targets', {})
        if not unified_targets:
            # Create targets from available data
            for target_name, target_values in base_data.items():
                if 'y_' in target_name:
                    unified_targets[target_name] = target_values
        
        return {
            'X': unified_X,
            'feature_names': combined_feature_names,
            'targets': unified_targets,
            'timestamps': base_data.get('timestamps', []),
            'original_timeframes': list(all_data.keys())
        }
    
    async def _prepare_training_datasets(self):
        """Prepare training datasets for different models"""
        logger.info("🔧 Preparing training datasets...")
        
        if not self.unified_data:
            raise Exception("No unified data available")
        
        X = self.unified_data['X']
        targets = self.unified_data['targets']
        
        # Remove NaN values
        valid_indices = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_indices]
        
        logger.info(f"   • Original samples: {X.shape[0]:,}")
        logger.info(f"   • Clean samples: {X_clean.shape[0]:,}")
        logger.info(f"   • Features: {X_clean.shape[1]}")
        
        # Prepare datasets for each target
        for target_name, target_values in targets.items():
            if len(target_values) == len(valid_indices):
                y_clean = target_values[valid_indices]
                
                # Remove NaN targets
                target_valid = ~np.isnan(y_clean)
                X_target = X_clean[target_valid]
                y_target = y_clean[target_valid]
                
                if len(X_target) > 1000:  # Minimum samples required
                    # Check class distribution
                    unique_classes, class_counts = np.unique(y_target, return_counts=True)
                    
                    # Skip targets with insufficient class diversity
                    if len(unique_classes) < 2 or min(class_counts) < 10:
                        logger.warning(f"   ⚠️ {target_name}: Insufficient class diversity - {dict(zip(unique_classes, class_counts))}")
                        continue
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X_target)
                    
                    # Split data with stratification if possible
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y_target, test_size=0.2, random_state=42,
                            stratify=y_target
                        )
                    except ValueError:
                        # Fallback without stratification
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y_target, test_size=0.2, random_state=42
                        )
                    
                    self.training_data[target_name] = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'scaler': scaler,
                        'samples': len(X_target),
                        'class_distribution': dict(zip(unique_classes, class_counts))
                    }
                    
                    logger.info(f"   ✅ {target_name}: {len(X_target):,} samples, classes: {dict(zip(unique_classes, class_counts))}")
        
        logger.info(f"✅ Prepared {len(self.training_data)} training datasets")
    
    async def _train_neural_ensemble(self):
        """Train Neural Ensemble models"""
        logger.info("🧠 Training Neural Ensemble...")
        
        try:
            from tensorflow.keras import models, layers, optimizers, callbacks
            
            ensemble_results = {}
            
            for target_name, data in self.training_data.items():
                logger.info(f"   🔄 Training Neural Ensemble for {target_name}...")
                
                X_train, X_test = data['X_train'], data['X_test']
                y_train, y_test = data['y_train'], data['y_test']
                
                # Create ensemble of different architectures
                models_ensemble = {}
                
                # LSTM Model
                lstm_model = models.Sequential([
                    layers.Reshape((1, X_train.shape[1]), input_shape=(X_train.shape[1],)),
                    layers.LSTM(128, return_sequences=True),
                    layers.Dropout(0.3),
                    layers.LSTM(64),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                lstm_model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train LSTM
                start_time = time.time()
                history = lstm_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=[
                        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                    ],
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Evaluate LSTM
                y_pred = (lstm_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
                lstm_accuracy = accuracy_score(y_test, y_pred)
                
                models_ensemble['lstm'] = {
                    'model': lstm_model,
                    'accuracy': lstm_accuracy,
                    'training_time': training_time
                }
                
                # Dense Neural Network
                dense_model = models.Sequential([
                    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                    layers.Dropout(0.3),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                dense_model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train Dense
                start_time = time.time()
                dense_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=[
                        callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                    ],
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Evaluate Dense
                y_pred = (dense_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
                dense_accuracy = accuracy_score(y_test, y_pred)
                
                models_ensemble['dense'] = {
                    'model': dense_model,
                    'accuracy': dense_accuracy,
                    'training_time': training_time
                }
                
                # Ensemble prediction
                lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
                dense_pred = dense_model.predict(X_test, verbose=0).flatten()
                
                # Weighted ensemble (based on individual accuracy)
                total_acc = lstm_accuracy + dense_accuracy
                lstm_weight = lstm_accuracy / total_acc
                dense_weight = dense_accuracy / total_acc
                
                ensemble_pred = (lstm_pred * lstm_weight + dense_pred * dense_weight > 0.5).astype(int)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                
                ensemble_results[target_name] = {
                    'individual_models': models_ensemble,
                    'ensemble_accuracy': ensemble_accuracy,
                    'lstm_accuracy': lstm_accuracy,
                    'dense_accuracy': dense_accuracy,
                    'weights': {'lstm': lstm_weight, 'dense': dense_weight}
                }
                
                logger.info(f"   ✅ {target_name}: Ensemble {ensemble_accuracy:.3f}, LSTM {lstm_accuracy:.3f}, Dense {dense_accuracy:.3f}")
            
            self.models['neural_ensemble'] = ensemble_results
            self.training_results['neural_ensemble'] = {
                'status': 'completed',
                'targets_trained': len(ensemble_results),
                'avg_accuracy': np.mean([r['ensemble_accuracy'] for r in ensemble_results.values()]),
                'training_time': datetime.now()
            }
            
            logger.info(f"✅ Neural Ensemble training completed")
            
        except Exception as e:
            logger.error(f"❌ Neural Ensemble training failed: {e}")
            self.training_results['neural_ensemble'] = {'status': 'failed', 'error': str(e)}
    
    async def _train_dqn_agent(self):
        """Train DQN Agent (simplified version)"""
        logger.info("🤖 Training DQN Agent...")
        
        try:
            # Simplified DQN training for demonstration
            dqn_results = {}
            
            for target_name, data in list(self.training_data.items())[:1]:  # Train on first target
                logger.info(f"   🔄 Training DQN Agent for {target_name}...")
                
                X_train, X_test = data['X_train'], data['X_test']
                y_train, y_test = data['y_train'], data['y_test']
                
                # Create simple Q-Network
                q_network = models.Sequential([
                    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(3)  # 3 actions: BUY, HOLD, SELL
                ])
                
                q_network.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='mse'
                )
                
                # Simplified training (convert classification to Q-learning)
                start_time = time.time()
                
                # Create Q-targets (simplified)
                actions = np.where(y_train > 0.5, 0, 2)  # 0=BUY, 2=SELL
                q_targets = np.zeros((len(X_train), 3))
                q_targets[np.arange(len(X_train)), actions] = 1.0
                
                # Train Q-Network
                history = q_network.fit(
                    X_train, q_targets,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Evaluate
                q_values = q_network.predict(X_test, verbose=0)
                predicted_actions = np.argmax(q_values, axis=1)
                
                # Convert back to binary for evaluation
                y_pred = (predicted_actions == 0).astype(int)  # BUY predictions
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate average reward (simplified)
                rewards = np.where(
                    (predicted_actions == 0) & (y_test == 1), 1.0,  # Correct BUY
                    np.where((predicted_actions == 2) & (y_test == 0), 1.0,  # Correct SELL
                             -0.1)  # Wrong prediction penalty
                )
                avg_reward = np.mean(rewards)
                
                dqn_results[target_name] = {
                    'model': q_network,
                    'accuracy': accuracy,
                    'avg_reward': avg_reward,
                    'training_time': training_time,
                    'final_loss': history.history['loss'][-1]
                }
                
                logger.info(f"   ✅ {target_name}: Accuracy {accuracy:.3f}, Avg Reward {avg_reward:.3f}")
            
            self.models['dqn_agent'] = dqn_results
            self.training_results['dqn_agent'] = {
                'status': 'completed',
                'targets_trained': len(dqn_results),
                'avg_accuracy': np.mean([r['accuracy'] for r in dqn_results.values()]),
                'avg_reward': np.mean([r['avg_reward'] for r in dqn_results.values()]),
                'training_time': datetime.now()
            }
            
            logger.info(f"✅ DQN Agent training completed")
            
        except Exception as e:
            logger.error(f"❌ DQN Agent training failed: {e}")
            self.training_results['dqn_agent'] = {'status': 'failed', 'error': str(e)}
    
    async def _train_meta_learning(self):
        """Train Meta Learning System"""
        logger.info("🧪 Training Meta Learning System...")
        
        try:
            # Meta learning: Learn from multiple tasks (targets)
            meta_results = {}
            
            if len(self.training_data) >= 2:
                # Train a meta-model that can adapt to different targets
                all_X_train = []
                all_y_train = []
                all_task_ids = []
                
                for task_id, (target_name, data) in enumerate(self.training_data.items()):
                    X_train, y_train = data['X_train'], data['y_train']
                    
                    # Add task embedding
                    task_embedding = np.full((len(X_train), 1), task_id)
                    X_with_task = np.hstack([X_train, task_embedding])
                    
                    all_X_train.append(X_with_task)
                    all_y_train.append(y_train)
                    all_task_ids.append(np.full(len(y_train), task_id))
                
                # Combine all tasks
                X_meta = np.vstack(all_X_train)
                y_meta = np.hstack(all_y_train)
                task_ids = np.hstack(all_task_ids)
                
                # Create meta-model
                meta_model = models.Sequential([
                    layers.Dense(512, activation='relu', input_shape=(X_meta.shape[1],)),
                    layers.Dropout(0.4),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                meta_model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.0005),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train meta-model
                start_time = time.time()
                history = meta_model.fit(
                    X_meta, y_meta,
                    epochs=40,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[
                        callbacks.EarlyStopping(patience=15, restore_best_weights=True)
                    ],
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Evaluate on each task
                task_accuracies = []
                for task_id, (target_name, data) in enumerate(self.training_data.items()):
                    X_test, y_test = data['X_test'], data['y_test']
                    
                    # Add task embedding for testing
                    task_embedding = np.full((len(X_test), 1), task_id)
                    X_test_with_task = np.hstack([X_test, task_embedding])
                    
                    # Predict
                    y_pred = (meta_model.predict(X_test_with_task, verbose=0) > 0.5).astype(int).flatten()
                    accuracy = accuracy_score(y_test, y_pred)
                    task_accuracies.append(accuracy)
                    
                    logger.info(f"   ✅ Task {target_name}: {accuracy:.3f}")
                
                meta_results = {
                    'model': meta_model,
                    'task_accuracies': task_accuracies,
                    'avg_accuracy': np.mean(task_accuracies),
                    'training_time': training_time,
                    'num_tasks': len(self.training_data)
                }
                
                self.models['meta_learning'] = meta_results
                self.training_results['meta_learning'] = {
                    'status': 'completed',
                    'avg_accuracy': np.mean(task_accuracies),
                    'num_tasks': len(self.training_data),
                    'training_time': datetime.now()
                }
                
                logger.info(f"✅ Meta Learning training completed: {np.mean(task_accuracies):.3f} avg accuracy")
            else:
                logger.warning("⚠️ Not enough tasks for meta learning")
                self.training_results['meta_learning'] = {'status': 'skipped', 'reason': 'insufficient_tasks'}
                
        except Exception as e:
            logger.error(f"❌ Meta Learning training failed: {e}")
            self.training_results['meta_learning'] = {'status': 'failed', 'error': str(e)}
    
    async def _train_traditional_models(self):
        """Train traditional ML models"""
        logger.info("📊 Training traditional ML models...")
        
        traditional_results = {}
        
        for target_name, data in self.training_data.items():
            logger.info(f"   🔄 Training traditional models for {target_name}...")
            
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
            
            models_results = {}
            
            # Random Forest
            try:
                start_time = time.time()
                rf_model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                
                y_pred = rf_model.predict(X_test)
                rf_accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time
                
                models_results['random_forest'] = {
                    'model': rf_model,
                    'accuracy': rf_accuracy,
                    'training_time': training_time
                }
                
            except Exception as e:
                logger.warning(f"   ⚠️ Random Forest failed: {e}")
            
            # Gradient Boosting
            try:
                start_time = time.time()
                gb_model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, random_state=42
                )
                gb_model.fit(X_train, y_train)
                
                y_pred = gb_model.predict(X_test)
                gb_accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time
                
                models_results['gradient_boost'] = {
                    'model': gb_model,
                    'accuracy': gb_accuracy,
                    'training_time': training_time
                }
                
            except Exception as e:
                logger.warning(f"   ⚠️ Gradient Boosting failed: {e}")
            
            # XGBoost
            try:
                start_time = time.time()
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'
                )
                xgb_model.fit(X_train, y_train)
                
                y_pred = xgb_model.predict(X_test)
                xgb_accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time
                
                models_results['xgboost'] = {
                    'model': xgb_model,
                    'accuracy': xgb_accuracy,
                    'training_time': training_time
                }
                
            except Exception as e:
                logger.warning(f"   ⚠️ XGBoost failed: {e}")
            
            # LightGBM
            try:
                start_time = time.time()
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, random_state=42, verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                
                y_pred = lgb_model.predict(X_test)
                lgb_accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time
                
                models_results['lightgbm'] = {
                    'model': lgb_model,
                    'accuracy': lgb_accuracy,
                    'training_time': training_time
                }
                
            except Exception as e:
                logger.warning(f"   ⚠️ LightGBM failed: {e}")
            
            traditional_results[target_name] = models_results
            
            # Log results
            best_model = max(models_results.items(), key=lambda x: x[1]['accuracy'])
            logger.info(f"   ✅ Best: {best_model[0]} ({best_model[1]['accuracy']:.3f})")
        
        self.models['traditional'] = traditional_results
        self.training_results['traditional'] = {
            'status': 'completed',
            'targets_trained': len(traditional_results),
            'models_per_target': len(list(traditional_results.values())[0]) if traditional_results else 0,
            'training_time': datetime.now()
        }
        
        logger.info(f"✅ Traditional models training completed")
    
    async def _create_ai_coordination(self):
        """Create AI coordination system"""
        logger.info("🤝 Creating AI coordination system...")
        
        try:
            coordination_config = {
                'model_weights': {
                    'neural_ensemble': 0.35,
                    'dqn_agent': 0.25,
                    'meta_learning': 0.20,
                    'traditional': 0.20
                },
                'confidence_threshold': 0.7,
                'consensus_required': True,
                'voting_strategy': 'weighted_average'
            }
            
            # Calculate model performance for weight adjustment
            model_performances = {}
            
            if 'neural_ensemble' in self.training_results:
                model_performances['neural_ensemble'] = self.training_results['neural_ensemble'].get('avg_accuracy', 0.5)
            
            if 'dqn_agent' in self.training_results:
                model_performances['dqn_agent'] = self.training_results['dqn_agent'].get('avg_accuracy', 0.5)
            
            if 'meta_learning' in self.training_results:
                model_performances['meta_learning'] = self.training_results['meta_learning'].get('avg_accuracy', 0.5)
            
            # Adjust weights based on performance
            if model_performances:
                total_performance = sum(model_performances.values())
                for model_name, performance in model_performances.items():
                    coordination_config['model_weights'][model_name] = performance / total_performance
            
            self.models['ai_coordination'] = coordination_config
            self.training_results['ai_coordination'] = {
                'status': 'completed',
                'model_weights': coordination_config['model_weights'],
                'performance_based_weights': True,
                'creation_time': datetime.now()
            }
            
            logger.info(f"✅ AI coordination created with performance-based weights")
            
        except Exception as e:
            logger.error(f"❌ AI coordination creation failed: {e}")
            self.training_results['ai_coordination'] = {'status': 'failed', 'error': str(e)}
    
    async def _validate_and_export_models(self):
        """Validate and export trained models"""
        logger.info("💾 Validating and exporting models...")
        
        try:
            export_results = {}
            
            # Create models directory
            import os
            os.makedirs('trained_models', exist_ok=True)
            
            # Export neural ensemble
            if 'neural_ensemble' in self.models:
                for target_name, ensemble_data in self.models['neural_ensemble'].items():
                    for model_name, model_info in ensemble_data['individual_models'].items():
                        model_path = f"trained_models/neural_ensemble_{target_name}_{model_name}.h5"
                        model_info['model'].save(model_path)
                        export_results[f"neural_ensemble_{target_name}_{model_name}"] = model_path
            
            # Export DQN agent
            if 'dqn_agent' in self.models:
                for target_name, dqn_data in self.models['dqn_agent'].items():
                    model_path = f"trained_models/dqn_agent_{target_name}.h5"
                    dqn_data['model'].save(model_path)
                    export_results[f"dqn_agent_{target_name}"] = model_path
            
            # Export meta learning
            if 'meta_learning' in self.models:
                model_path = f"trained_models/meta_learning_model.h5"
                self.models['meta_learning']['model'].save(model_path)
                export_results['meta_learning'] = model_path
            
            # Export traditional models
            if 'traditional' in self.models:
                for target_name, models_dict in self.models['traditional'].items():
                    for model_name, model_info in models_dict.items():
                        model_path = f"trained_models/{model_name}_{target_name}.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump(model_info['model'], f)
                        export_results[f"{model_name}_{target_name}"] = model_path
            
            # Export scalers
            for target_name, data in self.training_data.items():
                scaler_path = f"trained_models/scaler_{target_name}.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(data['scaler'], f)
                export_results[f"scaler_{target_name}"] = scaler_path
            
            # Export coordination config
            coordination_path = "trained_models/ai_coordination_config.json"
            with open(coordination_path, 'w') as f:
                json.dump(self.models.get('ai_coordination', {}), f, indent=2, default=str)
            export_results['ai_coordination'] = coordination_path
            
            self.training_results['export'] = {
                'status': 'completed',
                'models_exported': len(export_results),
                'export_paths': export_results,
                'export_time': datetime.now()
            }
            
            logger.info(f"✅ Exported {len(export_results)} models and components")
            
        except Exception as e:
            logger.error(f"❌ Model export failed: {e}")
            self.training_results['export'] = {'status': 'failed', 'error': str(e)}
    
    async def _generate_training_report(self):
        """Generate comprehensive training report"""
        logger.info("📋 Generating training report...")
        
        total_training_time = datetime.now() - self.training_start_time
        
        report = {
            'system_version': self.version,
            'training_start_time': self.training_start_time.isoformat(),
            'training_end_time': datetime.now().isoformat(),
            'total_training_time': str(total_training_time),
            'unified_data_info': {
                'total_samples': self.unified_data['X'].shape[0] if self.unified_data else 0,
                'total_features': self.unified_data['X'].shape[1] if self.unified_data else 0,
                'timeframes_used': len(self.unified_data.get('original_timeframes', [])) if self.unified_data else 0
            },
            'training_datasets': {
                target: {
                    'samples': data['samples'],
                    'train_samples': len(data['X_train']),
                    'test_samples': len(data['X_test'])
                } for target, data in self.training_data.items()
            },
            'training_results': self.training_results,
            'model_summary': {}
        }
        
        # Add model performance summary
        if 'neural_ensemble' in self.training_results:
            report['model_summary']['neural_ensemble'] = {
                'status': self.training_results['neural_ensemble']['status'],
                'avg_accuracy': self.training_results['neural_ensemble'].get('avg_accuracy', 0.0)
            }
        
        if 'dqn_agent' in self.training_results:
            report['model_summary']['dqn_agent'] = {
                'status': self.training_results['dqn_agent']['status'],
                'avg_accuracy': self.training_results['dqn_agent'].get('avg_accuracy', 0.0),
                'avg_reward': self.training_results['dqn_agent'].get('avg_reward', 0.0)
            }
        
        if 'meta_learning' in self.training_results:
            report['model_summary']['meta_learning'] = {
                'status': self.training_results['meta_learning']['status'],
                'avg_accuracy': self.training_results['meta_learning'].get('avg_accuracy', 0.0)
            }
        
        # Save report
        report_path = f"ULTIMATE_SYSTEM_TRAINING_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Training report saved: {report_path}")
        return report

async def main():
    """Main training execution"""
    print("🚀 ULTIMATE XAU SYSTEM V5.0 - COMPLETE TRAINING")
    print("="*80)
    
    # Initialize training system
    training_system = UltimateXAUSystemTraining()
    
    try:
        # Execute complete training
        await training_system.execute_complete_training()
        
        # Display summary
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Training Results Summary:")
        
        for model_type, results in training_system.training_results.items():
            status_icon = '✅' if results['status'] == 'completed' else '❌' if results['status'] == 'failed' else '⚠️'
            print(f"• {status_icon} {model_type.upper()}: {results['status']}")
            
            if 'avg_accuracy' in results:
                print(f"  - Average Accuracy: {results['avg_accuracy']:.3f}")
            if 'avg_reward' in results:
                print(f"  - Average Reward: {results['avg_reward']:.3f}")
        
        print(f"\n🏆 ULTIMATE XAU SYSTEM V5.0 TRAINING COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        logger.error(f"Training execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())