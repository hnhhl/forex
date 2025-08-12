#!/usr/bin/env python3
"""
AI3.0 Ultimate XAU Trading System - UNIFIED ARCHITECTURE VERSION
Integrated Training System with Unified Logic
"""

import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
# REPLACED: import MetaTrader5 as mt5
try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False
    class _DummyMT5:
        TIMEFRAME_M1 = 'M1'
    mt5 = _DummyMT5()
import time
import joblib
from tensorflow import keras

# Import Unified Logic
sys.path.append('src')
from core.shared.unified_feature_engine import UnifiedFeatureEngine
from core.shared.unified_model_architecture import UnifiedModelArchitecture
from core.enhanced_ensemble_manager import EnhancedEnsembleManager

# System Configuration
SYSTEM_VERSION = "4.0.0"
SYSTEM_NAME = "ULTIMATE_XAU_SUPER_SYSTEM"
DEFAULT_SYMBOL = "XAUUSDc"
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_M1

class SystemConfig:
    """System configuration class"""
    def __init__(self):
        self.symbol = DEFAULT_SYMBOL
        self.timeframe = DEFAULT_TIMEFRAME
        self.live_trading = False
        self.paper_trading = True
        self.max_positions = 5
        self.risk_per_trade = 0.02
        self.max_daily_trades = 50
        self.use_mt5 = MT5_AVAILABLE
        self.monitoring_frequency = 60
        self.base_lot_size = 0.01
        self.max_lot_size = 1.0
        self.stop_loss_pips = 50
        self.take_profit_pips = 100
        self.enable_kelly_criterion = True
        self.trailing_stop = False
        self.auto_rebalancing = True
        self.continuous_learning = True
        self.close_positions_on_stop = False
        
        # Training System Configuration
        self.enable_integrated_training = True
        self.auto_retrain = True
        self.retrain_frequency_hours = 24
        self.min_training_data_points = 1000


class IntegratedTrainingSystem:
    """Integrated Training System - Part of Main System"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.IntegratedTrainingSystem")
        
        # Unified Logic Components
        self.feature_engine = UnifiedFeatureEngine()
        self.model_architecture = UnifiedModelArchitecture()
        
        # Training State
        self.is_training = False
        self.last_training_time = None
        self.training_data_buffer = []
        self.trained_models = {}
        self.model_performance = {}
        
        # Paths
        self.models_path = "trained_models/unified"
        self.training_data_path = "data/training_buffer"
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.training_data_path, exist_ok=True)
        
        self.logger.info("IntegratedTrainingSystem initialized with Unified Logic")
    
    def should_retrain(self) -> bool:
        """Check if system should retrain models"""
        if not self.config.auto_retrain:
            return False
            
        if self.last_training_time is None:
            return True
            
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= self.config.retrain_frequency_hours
    
    def collect_training_data(self, market_data: Dict) -> bool:
        """Collect market data for training"""
        try:
            # Add timestamp
            market_data['timestamp'] = datetime.now().isoformat()
            
            # Add to buffer
            self.training_data_buffer.append(market_data)
            
            # Keep buffer size manageable
            max_buffer_size = 10000
            if len(self.training_data_buffer) > max_buffer_size:
                self.training_data_buffer = self.training_data_buffer[-max_buffer_size:]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            return False
    
    def prepare_training_data(self) -> Optional[tuple]:
        """Prepare training data using unified feature engine"""
        try:
            if len(self.training_data_buffer) < self.config.min_training_data_points:
                self.logger.warning(f"Insufficient training data: {len(self.training_data_buffer)} < {self.config.min_training_data_points}")
                return None
            
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.training_data_buffer)
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns. Available: {list(df.columns)}")
                return None
            
            # Use unified feature engine to create 19 features
            features_df = self.feature_engine.prepare_features_from_dataframe(df)
            
            # Create target (next period direction)
            target = (df['close'].shift(-1) > df['close']).astype(int)
            target = target[:-1]  # Remove last NaN
            features_df = features_df[:-1]  # Align with target
            
            # Create sequences for LSTM/CNN models
            sequence_length = 60
            X_sequences, y_sequences = self._create_sequences(features_df.values, target.values, sequence_length)
            
            # Flattened features for Dense model
            X_flat = features_df.values
            y_flat = target.values
            
            self.logger.info(f"Training data prepared: {len(X_sequences)} sequences, {len(X_flat)} flat samples")
            return (X_sequences, y_sequences, X_flat, y_flat)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray, sequence_length: int) -> tuple:
        """Create sequences for time series models"""
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def train_models(self) -> Dict[str, Any]:
        """Train all models using unified architecture"""
        if self.is_training:
            return {"status": "already_training"}
        
        try:
            self.is_training = True
            self.logger.info("Starting integrated training with unified architecture...")
            
            # Prepare training data
            training_data = self.prepare_training_data()
            if training_data is None:
                return {"status": "insufficient_data"}
            
            X_sequences, y_sequences, X_flat, y_flat = training_data
            
            # Split data
            split_idx = int(len(X_sequences) * 0.8)
            X_seq_train, X_seq_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_seq_train, y_seq_val = y_sequences[:split_idx], y_sequences[split_idx:]
            
            split_idx_flat = int(len(X_flat) * 0.8)
            X_flat_train, X_flat_val = X_flat[:split_idx_flat], X_flat[split_idx_flat:]
            y_flat_train, y_flat_val = y_flat[:split_idx_flat], y_flat[split_idx_flat:]
            
            # Train models using unified architecture
            results = {}
            
            # Train sequence models (LSTM, CNN, Hybrid)
            for model_type in ['lstm', 'cnn', 'hybrid']:
                try:
                    self.logger.info(f"Training {model_type.upper()} model...")
                    
                    model = self.model_architecture.create_model(model_type)
                    callbacks = self.model_architecture.get_standard_training_callbacks(model_type)
                    params = self.model_architecture.get_standard_training_params()
                    
                    history = model.fit(
                        X_seq_train, y_seq_train,
                        validation_data=(X_seq_val, y_seq_val),
                        callbacks=callbacks,
                        **params
                    )
                    
                    # Evaluate
                    val_loss, val_acc = model.evaluate(X_seq_val, y_seq_val, verbose=0)
                    
                    # Save model
                    model_path = f"{self.models_path}/{model_type}_unified.keras"
                    model.save(model_path)
                    
                    self.trained_models[model_type] = model
                    self.model_performance[model_type] = {
                        'val_accuracy': float(val_acc),
                        'val_loss': float(val_loss),
                        'epochs_trained': len(history.history['loss']),
                        'model_path': model_path
                    }
                    
                    results[model_type] = self.model_performance[model_type]
                    self.logger.info(f"{model_type.upper()} training completed - Accuracy: {val_acc:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type}: {e}")
                    results[model_type] = {"error": str(e)}
            
            # Train Dense model
            try:
                self.logger.info("Training DENSE model...")
                
                model = self.model_architecture.create_model('dense')
                callbacks = self.model_architecture.get_standard_training_callbacks('dense')
                params = self.model_architecture.get_standard_training_params()
                
                history = model.fit(
                    X_flat_train, y_flat_train,
                    validation_data=(X_flat_val, y_flat_val),
                    callbacks=callbacks,
                    **params
                )
                
                # Evaluate
                val_loss, val_acc = model.evaluate(X_flat_val, y_flat_val, verbose=0)
                
                # Save model
                model_path = f"{self.models_path}/dense_unified.keras"
                model.save(model_path)
                
                self.trained_models['dense'] = model
                self.model_performance['dense'] = {
                    'val_accuracy': float(val_acc),
                    'val_loss': float(val_loss),
                    'epochs_trained': len(history.history['loss']),
                    'model_path': model_path
                }
                
                results['dense'] = self.model_performance['dense']
                self.logger.info(f"DENSE training completed - Accuracy: {val_acc:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training dense model: {e}")
                results['dense'] = {"error": str(e)}
            
            # Update training time
            self.last_training_time = datetime.now()
            
            # Save training results
            self._save_training_results(results)
            
            return {
                "status": "completed",
                "models_trained": len([r for r in results.values() if "error" not in r]),
                "results": results,
                "training_time": self.last_training_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}
            
        finally:
            self.is_training = False
    
    def _save_training_results(self, results: Dict):
        """Save training results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{self.training_data_path}/training_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Training results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")
    
    def get_best_model(self) -> Optional[keras.Model]:
        """Get best performing model"""
        if not self.model_performance:
            return None
        
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda k: self.model_performance[k].get('val_accuracy', 0))
        
        return self.trained_models.get(best_model_name)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_data_points': len(self.training_data_buffer),
            'trained_models': list(self.trained_models.keys()),
            'model_performance': self.model_performance,
            'should_retrain': self.should_retrain()
        }


class UltimateXAUSystem:
    """Ultimate XAU Trading System - Main Class with INTEGRATED TRAINING"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.is_trading_active = False
        self.error_count = 0
        self.daily_trade_count = 0
        self.last_signal_time = None
        self.start_time = None
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 🚀 UNIFIED LOGIC COMPONENTS
        self.feature_engine = UnifiedFeatureEngine()
        self.model_architecture = UnifiedModelArchitecture()
        
        # 🧠 INTEGRATED TRAINING SYSTEM
        if self.config.enable_integrated_training:
            self.training_system = IntegratedTrainingSystem(self.config)
            self.logger.info("✅ Integrated Training System enabled")
        else:
            self.training_system = None
            self.logger.info("⚠️ Integrated Training System disabled")
        
        # 🤖 ENSEMBLE AI SYSTEM (Updated to use all 4 models)
        self.ensemble_manager = None
        self.ensemble_loaded = False
        
        # Load ensemble models
        self._load_ensemble_models()
        
        self.logger.info("✅ UltimateXAUSystem initialized with Unified Architecture")
    
    def _load_ensemble_models(self):
        """Load enhanced ensemble models (45 models instead of 4)"""
        try:
            self.logger.info("🏛️ Loading Enhanced Ensemble Parliament...")
            
            # Initialize Enhanced Ensemble Manager
            self.ensemble_manager = EnhancedEnsembleManager()
            
            # Register and load priority models
            if self.ensemble_manager.register_models():
                # Load more models to avoid early saturation by FAILED high-priority entries
                if self.ensemble_manager.load_priority_models(max_models=40):  # Load top 40 models
                    self.ensemble_loaded = True
                    self.models_count = self.ensemble_manager.active_models
                    
                    # Get parliament status
                    status = self.ensemble_manager.get_parliament_status()
                    self.logger.info(f"✅ Enhanced Parliament loaded:")
                    self.logger.info(f"   • Total registered: {status['total_registered']} models")
                    self.logger.info(f"   • Active models: {status['active_models']} models")
                    self.logger.info(f"   • Parliament efficiency: {status['parliament_efficiency']:.1%}")
                    
                    # Show top performers
                    top_performers = self.ensemble_manager.get_top_performers(5)
                    self.logger.info("🏆 Top 5 Parliament Members:")
                    for i, performer in enumerate(top_performers, 1):
                        self.logger.info(f"   {i}. {performer['name']}: {performer['expected_accuracy']:.1f}%")
                    
                    # Fallback retry: if too few models loaded, try expanding further
                    if self.models_count < 6:
                        self.logger.info("🔁 Low active model count detected, retrying with broader cap (80)...")
                        self.ensemble_manager.load_priority_models(max_models=80)
                        self.models_count = self.ensemble_manager.active_models
                        self.logger.info(f"   • Active models after retry: {self.models_count}")
                    
                    return True
                else:
                    self.logger.error("Failed to load priority models")
                    return False
            else:
                self.logger.error("Failed to register models")
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced ensemble loading failed: {e}")
            return False
    
    def get_market_features(self) -> np.ndarray:
        """Get real market features using UNIFIED FEATURE ENGINE (19 features)"""
        try:
            # Try to get real market data
            data_path = "data/working_free_data/XAUUSD_H1_realistic.csv"
            
            if os.path.exists(data_path):
                # Load real market data
                df = pd.read_csv(data_path)
                
                # Use unified feature engine to create 19 features
                features_df = self.feature_engine.prepare_features_from_dataframe(df.tail(100))  # Use last 100 rows
                
                # Get latest features
                latest_features = features_df.iloc[-1].values
                
                # Collect data for training if enabled
                if self.training_system:
                    market_data = {
                        'open': df['open'].iloc[-1] if 'open' in df.columns else 2000.0,
                        'high': df['high'].iloc[-1] if 'high' in df.columns else 2010.0,
                        'low': df['low'].iloc[-1] if 'low' in df.columns else 1990.0,
                        'close': df['close'].iloc[-1] if 'close' in df.columns else 2000.0,
                        'volume': df.get('volume', pd.Series([1000.0])).iloc[-1]
                    }
                    self.training_system.collect_training_data(market_data)
                
                return latest_features
            else:
                # Fallback: use default features from unified engine
                self.logger.warning("Market data file not found, using default features")
                return self.feature_engine._get_default_features()
                
        except Exception as e:
            self.logger.error(f"Error getting market features: {e}")
            # Emergency fallback
            return self.feature_engine._get_default_features()
    
    def _prepare_features_for_prediction(self, data: Dict) -> Optional[np.ndarray]:
        """Prepare features for enhanced ensemble prediction"""
        try:
            # If data is provided, use it
            if data and isinstance(data, dict):
                # Convert dict to DataFrame for feature extraction
                if 'ohlc' in data:
                    df = pd.DataFrame(data['ohlc'])
                else:
                    # Create DataFrame from dict
                    df = pd.DataFrame([data])
                
                # Normalize column names
                df.columns = df.columns.str.lower()
                
                # Extract features
                features_df = self.feature_engine.prepare_features_from_dataframe(df)
                if features_df is not None and len(features_df) > 0:
                    return features_df.iloc[-1].values
            
            # Fallback to market features
            return self.get_market_features()
            
        except Exception as e:
            self.logger.error(f"Error preparing features for prediction: {e}")
            return self.get_market_features()
    
    def generate_signal(self, data: Dict = None) -> Dict:
        """Generate trading signal using enhanced ensemble"""
        try:
            if not self.ensemble_loaded:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Enhanced ensemble not loaded'
                }
            
            # Prepare features
            features = self._prepare_features_for_prediction(data)
            if features is None:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Feature preparation failed'
                }
            
            # Get enhanced ensemble prediction
            result = self.ensemble_manager.predict_ensemble(features)
            
            # Enhanced logging
            self.logger.info(f"🏛️ ENHANCED PARLIAMENT DECISION:")
            self.logger.info(f"   • Ensemble Prediction: {result['ensemble_prediction']:.4f}")
            self.logger.info(f"   • Signal: {result['signal']}")
            self.logger.info(f"   • Confidence: {result['confidence']:.3f}")
            self.logger.info(f"   • Agreement: {result['agreement']:.3f}")
            self.logger.info(f"   • Active Voters: {result['active_models']}/{self.models_count}")
            
            # Log individual model votes (top 3)
            individual_preds = result['individual_predictions']
            sorted_preds = sorted(individual_preds.items(), key=lambda x: x[1]['weight'], reverse=True)
            self.logger.info("🗳️ Top 3 Model Votes:")
            for i, (model_id, pred_info) in enumerate(sorted_preds[:3], 1):
                self.logger.info(f"   {i}. {pred_info['model_name']}: {pred_info['prediction']:.4f} (Weight: {pred_info['weight']:.3f})")
            
            return {
                'signal': result['signal'],
                'confidence': result['confidence'],
                'ensemble_prediction': result['ensemble_prediction'],
                'agreement': result['agreement'],
                'active_models': result['active_models'],
                'individual_predictions': result['individual_predictions'],
                'reason': f'Enhanced ensemble decision with {result["active_models"]} models'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced signal generation failed: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Enhanced ensemble error: {str(e)}'
            }
    
    def start_training(self) -> Dict[str, Any]:
        """Start integrated training"""
        if not self.training_system:
            return {"status": "training_disabled"}
        
        return self.training_system.train_models()
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training system status"""
        if not self.training_system:
            return {"status": "training_disabled"}
        
        return self.training_system.get_training_status()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system_version': SYSTEM_VERSION,
            'ensemble_architecture': True,
            'feature_engine': 'unified_19_features',
            'model_architecture': 'unified_standard',
            'is_trading_active': self.is_trading_active,
            'ensemble_loaded': self.ensemble_loaded,
            'ai_model': 'ensemble_system' if self.ensemble_loaded else 'fallback',
            'models_count': len(self.ensemble_manager.models) if self.ensemble_manager else 0,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'error_count': self.error_count,
            'daily_trade_count': self.daily_trade_count
        }
        
        # Add training status if enabled
        if self.training_system:
            status['training_system'] = self.get_training_status()
        
        return status

class SystemManager:
    """System manager for handling multiple components"""
    
    def __init__(self):
        self.systems = {}
    
    def initialize_all_systems(self):
        """Initialize all system components"""
        try:
            self.systems['main'] = UltimateXAUSystem()
            return True
        except Exception as e:
            print(f"Error initializing systems: {e}")
            return False
    
    def stop_all_systems(self):
        """Stop all running systems"""
        for name, system in self.systems.items():
            try:
                system.stop_trading()
            except Exception as e:
                print(f"Error stopping {name}: {e}")

def main():
    """Main function"""
    try:
        print("🚀 Starting AI3.0 Ultimate XAU System with REAL AI")
        system = UltimateXAUSystem()
        
        # Test AI signal generation
        print("\n🧪 Testing AI signal generation...")
        for i in range(3):
            signal = system.generate_signal()
            print(f"Signal {i+1}: {signal}")
            time.sleep(1)
        
        print("\n✅ AI3.0 System ready for operation!")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")

if __name__ == "__main__":
    main()
