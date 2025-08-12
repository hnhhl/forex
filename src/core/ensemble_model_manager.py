"""
Ensemble Model Manager for AI3.0 Trading System
Manages multiple AI models and provides ensemble predictions with conflict resolution
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

class EnsembleModelManager:
    """
    Advanced Ensemble Model Manager
    - Loads all 4 trained models (Dense, CNN, LSTM, Hybrid)
    - Implements weighted voting based on performance
    - Handles conflicts with agreement-based decision making
    - Provides confidence scoring and risk management
    """
    
    def __init__(self, models_path: str = "trained_models/unified"):
        self.models_path = models_path
        self.models = {}
        self.model_performance = {}
        self.model_weights = {}
        self.is_loaded = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model performance from training results (accuracy)
        self.known_performance = {
            'dense': 0.7335,    # 73.35% - Best model
            'cnn': 0.5151,      # 51.51% 
            'lstm': 0.5050,     # 50.50%
            'hybrid': 0.5050    # 50.50%
        }
        
        # Calculate weights based on performance
        self._calculate_model_weights()
        
        # Load all models
        self._load_all_models()
        
    def _calculate_model_weights(self):
        """Calculate weights based on model performance"""
        total_performance = sum(self.known_performance.values())
        
        # Normalize weights (Dense gets higher weight due to superior performance)
        self.model_weights = {
            'dense': 0.4,   # 40% weight for best model
            'cnn': 0.2,     # 20% weight
            'lstm': 0.2,    # 20% weight  
            'hybrid': 0.2   # 20% weight
        }
        
        self.logger.info(f"📊 Model weights calculated: {self.model_weights}")
    
    def _load_all_models(self):
        """Load all available models from unified directory"""
        try:
            if not os.path.exists(self.models_path):
                self.logger.error(f"❌ Models directory not found: {self.models_path}")
                return
            
            model_files = [f for f in os.listdir(self.models_path) if f.endswith('.keras')]
            
            if not model_files:
                self.logger.error("❌ No .keras model files found")
                return
            
            loaded_count = 0
            for model_file in model_files:
                try:
                    # Extract model type from filename
                    model_type = model_file.replace('_unified.keras', '').replace('.keras', '')
                    
                    if model_type in self.known_performance:
                        model_path = os.path.join(self.models_path, model_file)
                        
                        # Load model
                        model = keras.models.load_model(model_path)
                        self.models[model_type] = model
                        self.model_performance[model_type] = self.known_performance[model_type]
                        
                        loaded_count += 1
                        self.logger.info(f"✅ Loaded {model_type} model: {model_file}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {model_file}: {e}")
            
            if loaded_count > 0:
                self.is_loaded = True
                self.logger.info(f"🎯 Ensemble loaded: {loaded_count}/4 models")
            else:
                self.logger.error("❌ No models loaded successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
    
    def get_ensemble_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get ensemble prediction from all loaded models
        
        Args:
            features: Input features array
            
        Returns:
            Dict with prediction, confidence, agreement score, and individual predictions
        """
        if not self.is_loaded:
            return {
                'error': 'No models loaded',
                'prediction': 0.5,
                'confidence': 0.0,
                'agreement_score': 0.0,
                'individual_predictions': {},
                'final_decision': 'HOLD'
            }
        
        try:
            individual_predictions = {}
            valid_predictions = []
            valid_weights = []
            
            # Get predictions from all models
            for model_type, model in self.models.items():
                try:
                    # Prepare input based on model architecture
                    if len(model.input_shape) == 3:  # Sequence model (LSTM/CNN)
                        sequence_length = model.input_shape[1]
                        model_input = np.tile(features, (1, sequence_length, 1))
                    else:  # Dense model
                        model_input = features.reshape(1, -1)
                    
                    # Get prediction
                    prediction = model.predict(model_input, verbose=0)
                    pred_value = float(prediction[0][0])
                    
                    individual_predictions[model_type] = pred_value
                    valid_predictions.append(pred_value)
                    valid_weights.append(self.model_weights[model_type])
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error getting prediction from {model_type}: {e}")
            
            if not valid_predictions:
                return {
                    'error': 'No valid predictions',
                    'prediction': 0.5,
                    'confidence': 0.0,
                    'agreement_score': 0.0,
                    'individual_predictions': {},
                    'final_decision': 'HOLD'
                }
            
            # Calculate ensemble metrics
            weighted_prediction = self._calculate_weighted_prediction(valid_predictions, valid_weights)
            agreement_score = self._calculate_agreement_score(valid_predictions)
            confidence = self._calculate_confidence(weighted_prediction, agreement_score)
            final_decision = self._make_final_decision(weighted_prediction, agreement_score, confidence)
            
            return {
                'prediction': weighted_prediction,
                'confidence': confidence,
                'agreement_score': agreement_score,
                'individual_predictions': individual_predictions,
                'final_decision': final_decision,
                'models_used': len(valid_predictions),
                'ensemble_method': 'weighted_voting_with_agreement'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in ensemble prediction: {e}")
            return {
                'error': str(e),
                'prediction': 0.5,
                'confidence': 0.0,
                'agreement_score': 0.0,
                'individual_predictions': {},
                'final_decision': 'HOLD'
            }
    
    def _calculate_weighted_prediction(self, predictions: List[float], weights: List[float]) -> float:
        """Calculate weighted average prediction"""
        if len(predictions) != len(weights):
            # Fallback to simple average
            return np.mean(predictions)
        
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(predictions)
    
    def _calculate_agreement_score(self, predictions: List[float]) -> float:
        """
        Calculate agreement score between models
        Higher score = more agreement, Lower score = more conflict
        """
        if len(predictions) < 2:
            return 1.0
        
        # Calculate standard deviation
        pred_std = np.std(predictions)
        
        # Convert to agreement score (0-1, higher = better agreement)
        # std of 0 = perfect agreement (score 1.0)
        # std of 0.5 = maximum disagreement (score 0.0)
        agreement_score = max(0.0, 1.0 - (pred_std * 2.0))
        
        return agreement_score
    
    def _calculate_confidence(self, prediction: float, agreement_score: float) -> float:
        """
        Calculate confidence based on prediction strength and model agreement
        """
        # Base confidence from prediction strength
        prediction_strength = abs(prediction - 0.5) * 2  # 0-1 scale
        base_confidence = 50 + prediction_strength * 30  # 50-80 range
        
        # Boost confidence with agreement
        agreement_boost = agreement_score * 20  # 0-20 boost
        
        # Final confidence
        confidence = min(95, base_confidence + agreement_boost)
        
        return confidence
    
    def _make_final_decision(self, prediction: float, agreement_score: float, confidence: float) -> str:
        """
        Make final trading decision based on ensemble results and risk management
        """
        # Risk management based on agreement
        if agreement_score < 0.3:  # Very low agreement
            return 'HOLD'  # Too risky to trade
        
        # Decision thresholds (more conservative with lower agreement)
        if agreement_score > 0.7:  # High agreement
            buy_threshold = 0.6
            sell_threshold = 0.4
        elif agreement_score > 0.4:  # Medium agreement
            buy_threshold = 0.65
            sell_threshold = 0.35
        else:  # Low agreement
            buy_threshold = 0.7
            sell_threshold = 0.3
        
        # Make decision
        if prediction > buy_threshold:
            return 'BUY'
        elif prediction < sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            'is_loaded': self.is_loaded,
            'models_loaded': list(self.models.keys()),
            'model_count': len(self.models),
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'models_path': self.models_path
        }
    
    def get_detailed_analysis(self, features: np.ndarray) -> Dict[str, Any]:
        """Get detailed analysis for debugging and monitoring"""
        result = self.get_ensemble_prediction(features)
        
        if 'error' in result:
            return result
        
        # Add detailed breakdown
        individual_preds = result['individual_predictions']
        
        analysis = {
            **result,
            'detailed_breakdown': {
                'individual_decisions': {},
                'weight_contributions': {},
                'conflict_analysis': {}
            }
        }
        
        # Individual decisions
        for model_type, pred in individual_preds.items():
            if pred > 0.6:
                decision = 'BUY'
            elif pred < 0.4:
                decision = 'SELL'
            else:
                decision = 'HOLD'
            
            analysis['detailed_breakdown']['individual_decisions'][model_type] = {
                'prediction': pred,
                'decision': decision,
                'weight': self.model_weights.get(model_type, 0),
                'performance': self.model_performance.get(model_type, 0)
            }
        
        # Weight contributions
        for model_type, pred in individual_preds.items():
            weight = self.model_weights.get(model_type, 0)
            contribution = pred * weight
            analysis['detailed_breakdown']['weight_contributions'][model_type] = contribution
        
        # Conflict analysis
        predictions_list = list(individual_preds.values())
        if len(predictions_list) > 1:
            analysis['detailed_breakdown']['conflict_analysis'] = {
                'prediction_range': max(predictions_list) - min(predictions_list),
                'standard_deviation': np.std(predictions_list),
                'disagreement_level': 'HIGH' if np.std(predictions_list) > 0.25 else 
                                    'MEDIUM' if np.std(predictions_list) > 0.15 else 'LOW'
            }
        
        return analysis

# Utility function for easy integration
def create_ensemble_manager(models_path: str = "trained_models/unified") -> EnsembleModelManager:
    """Create and return configured ensemble manager"""
    return EnsembleModelManager(models_path) 