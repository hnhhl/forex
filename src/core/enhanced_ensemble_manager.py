#!/usr/bin/env python3
"""
Enhanced Ensemble Model Manager - Mở rộng Quốc hội AI
Tích hợp tất cả models đã có sẵn trong hệ thống vào một ensemble lớn

Author: AI Assistant
Date: 2025-01-03
Version: Enhanced 2.0
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Try importing additional libraries
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
class ModelInfo:
    """Thông tin về từng model trong ensemble"""
    name: str
    model_type: str
    file_path: str
    weight: float
    expected_accuracy: float
    status: str = "AVAILABLE"
    model_object: Any = None

class EnhancedEnsembleManager:
    """
    Enhanced Ensemble Manager - Quốc hội AI mở rộng
    Tích hợp tất cả models có sẵn trong hệ thống
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
        
        # Model registry - Danh sách tất cả đại biểu
        self.model_registry: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.ensemble_accuracy = 0.0
        self.total_models = 0
        self.active_models = 0
        
        self.logger.info("🏛️ Enhanced Ensemble Manager initialized")
    
    def discover_available_models(self) -> Dict[str, List[str]]:
        """
        Khám phá tất cả models có sẵn trong hệ thống
        """
        available_models = {
            "neural_keras": [],
            "neural_h5": [],
            "traditional_pkl": [],
            "unified_models": [],
            "specialized_models": []
        }
        
        if not os.path.exists(self.models_dir):
            self.logger.warning(f"Models directory not found: {self.models_dir}")
            return available_models
        
        # Scan main models directory
        for file in os.listdir(self.models_dir):
            file_path = os.path.join(self.models_dir, file)
            
            if file.endswith('.keras'):
                available_models["neural_keras"].append(file_path)
            elif file.endswith('.h5'):
                available_models["neural_h5"].append(file_path)
            elif file.endswith('.pkl'):
                available_models["traditional_pkl"].append(file_path)
        
        # Scan unified models
        unified_dir = os.path.join(self.models_dir, "unified")
        if os.path.exists(unified_dir):
            for file in os.listdir(unified_dir):
                if file.endswith('.keras'):
                    file_path = os.path.join(unified_dir, file)
                    available_models["unified_models"].append(file_path)
        
        # Log discoveries
        self.logger.info("🔍 MODEL DISCOVERY RESULTS:")
        for category, files in available_models.items():
            self.logger.info(f"  {category}: {len(files)} models")
        
        return available_models
    
    def register_models(self) -> bool:
        """
        Đăng ký tất cả models vào model registry
        """
        try:
            available = self.discover_available_models()
            
            # 1. UNIFIED MODELS (Current Parliament - Highest Priority)
            for model_path in available["unified_models"]:
                model_name = os.path.basename(model_path).replace('.keras', '')
                
                # Assign weights based on known performance
                if 'dense' in model_name.lower():
                    weight, accuracy = 0.25, 73.4  # Thủ tướng
                elif 'cnn' in model_name.lower():
                    weight, accuracy = 0.15, 51.5  # Bộ trưởng Nhận dạng
                elif 'lstm' in model_name.lower():
                    weight, accuracy = 0.15, 50.5  # Bộ trưởng Dự báo
                elif 'hybrid' in model_name.lower():
                    weight, accuracy = 0.15, 50.5  # Bộ trưởng Tổng hợp
                else:
                    weight, accuracy = 0.10, 60.0  # Default
                
                self.model_registry[f"unified_{model_name}"] = ModelInfo(
                    name=f"Unified {model_name.title()}",
                    model_type="neural_keras",
                    file_path=model_path,
                    weight=weight,
                    expected_accuracy=accuracy,
                    status="HIGH_PRIORITY"
                )
            
            # 2. TRADITIONAL ML MODELS (New Parliament Members)
            ml_models_found = 0
            for model_path in available["traditional_pkl"]:
                model_name = os.path.basename(model_path).replace('.pkl', '')
                
                if 'random_forest' in model_name.lower():
                    weight, accuracy = 0.08, 68.0
                    friendly_name = "Random Forest Specialist"
                elif 'lightgbm' in model_name.lower():
                    weight, accuracy = 0.08, 70.0
                    friendly_name = "LightGBM Expert"
                elif 'gradient_boost' in model_name.lower():
                    weight, accuracy = 0.06, 65.0
                    friendly_name = "Gradient Boosting Advisor"
                elif 'xgboost' in model_name.lower():
                    weight, accuracy = 0.08, 67.0
                    friendly_name = "XGBoost Specialist"
                else:
                    continue  # Skip unknown models
                
                self.model_registry[f"ml_{model_name}"] = ModelInfo(
                    name=friendly_name,
                    model_type="traditional_pkl",
                    file_path=model_path,
                    weight=weight,
                    expected_accuracy=accuracy,
                    status="READY"
                )
                ml_models_found += 1
            
            # 3. SPECIALIZED NEURAL MODELS (Specialized Parliament)
            for model_path in available["neural_keras"]:
                model_name = os.path.basename(model_path).replace('.keras', '')
                
                if 'm1_' in model_name.lower():
                    weight, accuracy = 0.05, 75.0
                    friendly_name = "M1 Time Specialist"
                elif 'gpu_' in model_name.lower():
                    weight, accuracy = 0.04, 72.0
                    friendly_name = "GPU Accelerated Model"
                elif 'production_' in model_name.lower():
                    weight, accuracy = 0.06, 70.0
                    friendly_name = "Production Optimized"
                elif 'comprehensive_' in model_name.lower():
                    weight, accuracy = 0.05, 68.0
                    friendly_name = "Comprehensive Model"
                else:
                    weight, accuracy = 0.03, 65.0  # Lower priority for unknown
                    friendly_name = f"Neural {model_name.title()}"
                
                self.model_registry[f"neural_{model_name}"] = ModelInfo(
                    name=friendly_name,
                    model_type="neural_keras",
                    file_path=model_path,
                    weight=weight,
                    expected_accuracy=accuracy,
                    status="AVAILABLE"
                )
            
            self.total_models = len(self.model_registry)
            self.logger.info(f"✅ Registered {self.total_models} models in parliament")
            self.logger.info(f"   • Unified Models: {len(available['unified_models'])}")
            self.logger.info(f"   • Traditional ML: {ml_models_found}")
            self.logger.info(f"   • Specialized Neural: {len(available['neural_keras'])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            return False
    
    def load_priority_models(self, max_models: int = 10) -> bool:
        """
        Load các models có priority cao nhất
        """
        try:
            if not self.model_registry:
                self.register_models()
            
            # Sort models by weight (priority)
            sorted_models = sorted(
                self.model_registry.items(),
                key=lambda x: x[1].weight,
                reverse=True
            )
            
            loaded_count = 0
            for model_id, model_info in sorted_models[:max_models]:
                try:
                    if model_info.model_type == "neural_keras":
                        model = tf.keras.models.load_model(model_info.file_path)
                        self.loaded_models[model_id] = model
                        self.model_weights[model_id] = model_info.weight
                        
                    elif model_info.model_type == "traditional_pkl":
                        with open(model_info.file_path, 'rb') as f:
                            model = pickle.load(f)
                        self.loaded_models[model_id] = model
                        self.model_weights[model_id] = model_info.weight
                    
                    model_info.status = "LOADED"
                    loaded_count += 1
                    self.logger.info(f"   ✅ Loaded: {model_info.name} (Weight: {model_info.weight:.3f})")
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to load {model_info.name}: {e}")
                    model_info.status = "FAILED"
            
            self.active_models = loaded_count
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model_id in self.model_weights:
                    self.model_weights[model_id] /= total_weight
            
            self.logger.info(f"🏛️ ENHANCED PARLIAMENT LOADED:")
            self.logger.info(f"   • Active Models: {self.active_models}/{self.total_models}")
            self.logger.info(f"   • Total Weight: {sum(self.model_weights.values()):.3f}")
            
            return loaded_count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Priority model loading failed: {e}")
            return False
    
    def predict_ensemble(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Ensemble prediction từ tất cả active models
        """
        if not self.loaded_models:
            raise ValueError("No models loaded for prediction")
        
        predictions = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Get predictions from each model
        for model_id, model in self.loaded_models.items():
            try:
                weight = self.model_weights[model_id]
                
                if 'neural' in model_id or 'unified' in model_id:
                    # Neural network prediction
                    if len(features.shape) == 1:
                        features_reshaped = features.reshape(1, -1)
                    else:
                        features_reshaped = features
                    
                    pred = model.predict(features_reshaped, verbose=0)[0][0]
                    
                else:
                    # Traditional ML prediction
                    if len(features.shape) > 1:
                        features_flat = features.flatten().reshape(1, -1)
                    else:
                        features_flat = features.reshape(1, -1)
                    
                    # Handle different prediction types
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(features_flat)[0]
                        pred = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    else:
                        pred = model.predict(features_flat)[0]
                
                predictions[model_id] = {
                    'prediction': float(pred),
                    'weight': weight,
                    'model_name': self.model_registry[model_id].name
                }
                
                weighted_sum += pred * weight
                total_weight += weight
                
            except Exception as e:
                self.logger.warning(f"Prediction failed for {model_id}: {e}")
                continue
        
        # Calculate ensemble prediction
        if total_weight > 0:
            ensemble_prediction = weighted_sum / total_weight
        else:
            ensemble_prediction = 0.5  # Default neutral
        
        # Calculate agreement (consensus measure)
        pred_values = [p['prediction'] for p in predictions.values()]
        if len(pred_values) > 1:
            agreement = 1.0 - np.std(pred_values)  # Higher std = lower agreement
            agreement = max(0.0, min(1.0, agreement))  # Clamp to [0,1]
        else:
            agreement = 1.0
        
        # Generate trading signal
        if ensemble_prediction > 0.6 and agreement > 0.7:
            signal = "BUY"
            confidence = (ensemble_prediction - 0.5) * 2 * agreement
        elif ensemble_prediction < 0.4 and agreement > 0.7:
            signal = "SELL" 
            confidence = (0.5 - ensemble_prediction) * 2 * agreement
        else:
            signal = "HOLD"
            confidence = agreement * 0.5  # Conservative confidence for HOLD
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'signal': signal,
            'confidence': min(confidence, 1.0),
            'agreement': agreement,
            'individual_predictions': predictions,
            'active_models': len(predictions),
            'total_weight': total_weight
        }
    
    def get_parliament_status(self) -> Dict[str, Any]:
        """
        Trạng thái của Quốc hội AI
        """
        status = {
            'total_registered': self.total_models,
            'active_models': self.active_models,
            'parliament_efficiency': self.active_models / max(self.total_models, 1),
            'loaded_models': {}
        }
        
        for model_id, model_info in self.model_registry.items():
            status['loaded_models'][model_id] = {
                'name': model_info.name,
                'type': model_info.model_type,
                'weight': self.model_weights.get(model_id, 0.0),
                'expected_accuracy': model_info.expected_accuracy,
                'status': model_info.status
            }
        
        return status
    
    def get_top_performers(self, n: int = 5) -> List[Dict]:
        """
        Lấy top performers trong parliament
        """
        sorted_models = sorted(
            [(mid, info) for mid, info in self.model_registry.items()],
            key=lambda x: x[1].expected_accuracy,
            reverse=True
        )
        
        return [
            {
                'model_id': mid,
                'name': info.name,
                'expected_accuracy': info.expected_accuracy,
                'weight': self.model_weights.get(mid, 0.0),
                'status': info.status
            }
            for mid, info in sorted_models[:n]
        ]

if __name__ == "__main__":
    # Demo Enhanced Ensemble
    logging.basicConfig(level=logging.INFO)
    
    print("🏛️ ENHANCED ENSEMBLE MANAGER DEMO")
    print("=" * 50)
    
    manager = EnhancedEnsembleManager()
    
    # Register all available models
    if manager.register_models():
        print(f"\n✅ Successfully registered {manager.total_models} models")
        
        # Load priority models
        if manager.load_priority_models(max_models=8):
            print(f"\n🚀 Loaded {manager.active_models} priority models")
            
            # Show parliament status
            status = manager.get_parliament_status()
            print(f"\n📊 PARLIAMENT STATUS:")
            print(f"   Efficiency: {status['parliament_efficiency']:.1%}")
            
            # Show top performers
            top_performers = manager.get_top_performers(5)
            print(f"\n🏆 TOP 5 PERFORMERS:")
            for i, performer in enumerate(top_performers, 1):
                print(f"   {i}. {performer['name']}: {performer['expected_accuracy']:.1f}% (Weight: {performer['weight']:.3f})")
        
        else:
            print("❌ Failed to load priority models")
    else:
        print("❌ Failed to register models") 