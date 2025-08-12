"""
Unified Prediction Logic for AI3.0 Ultimate XAU System
SINGLE SOURCE OF TRUTH for prediction processing
Used by BOTH Training and Production systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from tensorflow import keras

logger = logging.getLogger(__name__)


class UnifiedPredictionLogic:
    """
    Unified Prediction Processing Logic
    Standard prediction workflow for both Training and Production
    """
    
    # STANDARD PREDICTION THRESHOLDS
    PREDICTION_THRESHOLDS = {
        'buy_threshold': 0.6,
        'sell_threshold': 0.4,
        'confidence_multiplier': 90,
        'base_confidence': 50,
        'max_confidence': 95
    }
    
    # VOLATILITY-BASED DYNAMIC THRESHOLDS
    VOLATILITY_THRESHOLDS = {
        'low': {'buy': 0.58, 'sell': 0.42},      # Low volatility
        'medium': {'buy': 0.60, 'sell': 0.40},   # Medium volatility  
        'high': {'buy': 0.65, 'sell': 0.35}     # High volatility
    }
    
    def __init__(self):
        logger.info("UnifiedPredictionLogic initialized with standard thresholds")
    
    def process_model_prediction(self, 
                                model: keras.Model,
                                features: np.ndarray,
                                model_type: str = "unknown") -> Dict[str, Any]:
        """
        Process model prediction with unified logic
        
        Args:
            model: Trained Keras model
            features: Input features (19 features standard)
            model_type: Type of model ('lstm', 'cnn', 'dense', 'hybrid')
            
        Returns:
            Dict: Standardized prediction result
        """
        try:
            # Validate input
            if not self._validate_input(features, model):
                return self._get_error_prediction("Invalid input")
            
            # Prepare input based on model architecture
            model_input = self._prepare_model_input(features, model, model_type)
            
            # Get raw prediction
            raw_prediction = model.predict(model_input, verbose=0)
            prediction_value = float(raw_prediction[0][0])
            
            # Calculate confidence
            confidence = self._calculate_confidence(prediction_value)
            
            # Generate trading signal
            signal_type = self._determine_signal_type(prediction_value, features)
            
            # Create standardized result
            result = {
                'prediction_value': round(prediction_value, 4),
                'confidence': round(confidence, 2),
                'signal_type': signal_type,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'features_used': features.shape[-1] if len(features.shape) > 1 else len(features),
                'processing_status': 'success'
            }
            
            logger.debug(f"Prediction processed: {signal_type} ({confidence:.1f}%) from {model_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            return self._get_error_prediction(str(e))
    
    def process_ensemble_predictions(self, 
                                   predictions: List[Dict[str, Any]],
                                   weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Process ensemble of predictions with unified logic
        
        Args:
            predictions: List of individual model predictions
            weights: Optional weights for ensemble averaging
            
        Returns:
            Dict: Ensemble prediction result
        """
        try:
            if not predictions:
                return self._get_error_prediction("No predictions provided")
            
            # Filter successful predictions
            valid_predictions = [p for p in predictions if p.get('processing_status') == 'success']
            
            if not valid_predictions:
                return self._get_error_prediction("No valid predictions")
            
            # Calculate weighted ensemble
            if weights is None:
                weights = {p['model_type']: 1.0 for p in valid_predictions}
            
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for prediction in valid_predictions:
                model_type = prediction['model_type']
                weight = weights.get(model_type, 1.0)
                
                weighted_prediction += prediction['prediction_value'] * weight
                weighted_confidence += prediction['confidence'] * weight
                total_weight += weight
            
            if total_weight == 0:
                return self._get_error_prediction("Zero total weight")
            
            # Normalize
            ensemble_prediction = weighted_prediction / total_weight
            ensemble_confidence = weighted_confidence / total_weight
            
            # Determine ensemble signal
            ensemble_signal = self._determine_signal_type(ensemble_prediction)
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(valid_predictions)
            
            result = {
                'ensemble_prediction': round(ensemble_prediction, 4),
                'ensemble_confidence': round(ensemble_confidence, 2),
                'ensemble_signal': ensemble_signal,
                'consensus_score': round(consensus_score, 3),
                'individual_predictions': valid_predictions,
                'models_used': len(valid_predictions),
                'timestamp': datetime.now().isoformat(),
                'processing_status': 'success'
            }
            
            logger.info(f"Ensemble processed: {ensemble_signal} ({ensemble_confidence:.1f}%) from {len(valid_predictions)} models")
            return result
            
        except Exception as e:
            logger.error(f"Error processing ensemble: {e}")
            return self._get_error_prediction(str(e))
    
    def create_trading_signal(self,
                            prediction_result: Dict[str, Any],
                            current_price: float,
                            symbol: str = "XAUUSD",
                            volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        Create standardized trading signal from prediction
        
        Args:
            prediction_result: Result from process_model_prediction
            current_price: Current market price
            symbol: Trading symbol
            volatility: Market volatility (optional, for dynamic thresholds)
            
        Returns:
            Dict: Standardized trading signal
        """
        try:
            if prediction_result.get('processing_status') != 'success':
                return self._get_error_signal("Invalid prediction result")
            
            # Get prediction values
            prediction_value = prediction_result['prediction_value']
            confidence = prediction_result['confidence']
            signal_type = prediction_result['signal_type']
            source = f"unified_{prediction_result['model_type']}"
            
            # Calculate position sizing
            position_size = self._calculate_position_size(confidence)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(current_price, signal_type, volatility)
            
            # Create trading signal
            signal = {
                'action': signal_type,
                'confidence': confidence,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': round(current_price, 2),
                'prediction_value': prediction_value,
                'source': source,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price, stop_loss, take_profit),
                'volatility_regime': self._classify_volatility_regime(volatility) if volatility else 'unknown',
                'signal_quality': self._assess_signal_quality(confidence, prediction_value),
                'metadata': {
                    'feature_engine': 'unified_19_features',
                    'prediction_logic': 'unified_standard',
                    'processing_time': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Trading signal created: {signal_type} @ {current_price} ({confidence:.1f}%)")
            return signal
            
        except Exception as e:
            logger.error(f"Error creating trading signal: {e}")
            return self._get_error_signal(str(e))
    
    def _validate_input(self, features: np.ndarray, model: keras.Model) -> bool:
        """Validate input features and model compatibility"""
        try:
            # Check features shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Check model input shape compatibility
            model_input_shape = model.input_shape
            
            if len(model_input_shape) == 3:  # Sequence model
                return features.shape[-1] == model_input_shape[-1]
            elif len(model_input_shape) == 2:  # Dense model
                return features.shape[-1] == model_input_shape[-1]
            
            return False
            
        except Exception:
            return False
    
    def _prepare_model_input(self, features: np.ndarray, model: keras.Model, model_type: str) -> np.ndarray:
        """Prepare input based on model architecture"""
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            model_input_shape = model.input_shape
            
            if len(model_input_shape) == 3:  # Sequence model (LSTM, CNN, Hybrid)
                sequence_length = model_input_shape[1]
                # For real-time prediction, repeat current features to create sequence
                # In production, this should use historical data
                model_input = np.tile(features, (1, sequence_length, 1))
            else:  # Dense model
                model_input = features
            
            return model_input
            
        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            raise
    
    def _calculate_confidence(self, prediction_value: float) -> float:
        """Calculate confidence score from prediction value"""
        # Distance from neutral (0.5)
        distance_from_neutral = abs(prediction_value - 0.5)
        
        # Convert to confidence percentage
        confidence = self.PREDICTION_THRESHOLDS['base_confidence'] + \
                    (distance_from_neutral * self.PREDICTION_THRESHOLDS['confidence_multiplier'])
        
        return min(confidence, self.PREDICTION_THRESHOLDS['max_confidence'])
    
    def _determine_signal_type(self, prediction_value: float, features: Optional[np.ndarray] = None) -> str:
        """Determine signal type based on prediction value and optional volatility"""
        volatility = None
        if features is not None and len(features) >= 13:  # Volatility is 13th feature
            volatility = float(features[12])
        
        # Get dynamic thresholds based on volatility
        thresholds = self._get_dynamic_thresholds(volatility)
        
        if prediction_value > thresholds['buy']:
            return 'BUY'
        elif prediction_value < thresholds['sell']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _get_dynamic_thresholds(self, volatility: Optional[float] = None) -> Dict[str, float]:
        """Get dynamic thresholds based on volatility"""
        if volatility is None:
            return {
                'buy': self.PREDICTION_THRESHOLDS['buy_threshold'],
                'sell': self.PREDICTION_THRESHOLDS['sell_threshold']
            }
        
        if volatility < 0.5:
            return self.VOLATILITY_THRESHOLDS['low']
        elif volatility > 1.0:
            return self.VOLATILITY_THRESHOLDS['high']
        else:
            return self.VOLATILITY_THRESHOLDS['medium']
    
    def _calculate_consensus_score(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate consensus score among predictions"""
        if len(predictions) <= 1:
            return 1.0
        
        signals = [p['signal_type'] for p in predictions]
        
        # Count signal types
        signal_counts = {}
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Calculate consensus (percentage of majority)
        max_count = max(signal_counts.values())
        consensus = max_count / len(signals)
        
        return consensus
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        # Base position size
        base_size = 0.01
        
        # Scale by confidence (higher confidence = larger position)
        if confidence >= 80:
            multiplier = 2.0
        elif confidence >= 70:
            multiplier = 1.5
        elif confidence >= 60:
            multiplier = 1.0
        else:
            multiplier = 0.5
        
        return round(base_size * multiplier, 2)
    
    def _calculate_risk_levels(self, current_price: float, signal_type: str, 
                             volatility: Optional[float] = None) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        # Base risk percentage
        base_risk = 0.025  # 2.5%
        
        # Adjust for volatility
        if volatility is not None:
            if volatility > 1.0:  # High volatility
                risk_multiplier = 1.5
            elif volatility < 0.5:  # Low volatility
                risk_multiplier = 0.8
            else:  # Medium volatility
                risk_multiplier = 1.0
        else:
            risk_multiplier = 1.0
        
        adjusted_risk = base_risk * risk_multiplier
        
        if signal_type == 'BUY':
            stop_loss = round(current_price * (1 - adjusted_risk), 2)
            take_profit = round(current_price * (1 + adjusted_risk * 2), 2)  # 2:1 RR
        elif signal_type == 'SELL':
            stop_loss = round(current_price * (1 + adjusted_risk), 2)
            take_profit = round(current_price * (1 - adjusted_risk * 2), 2)  # 2:1 RR
        else:  # HOLD
            stop_loss = current_price
            take_profit = current_price
        
        return stop_loss, take_profit
    
    def _calculate_risk_reward_ratio(self, current_price: float, stop_loss: float, take_profit: float) -> float:
        """Calculate risk-reward ratio"""
        try:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            
            if risk == 0:
                return 0.0
            
            return round(reward / risk, 2)
            
        except Exception:
            return 0.0
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.5:
            return 'low'
        elif volatility > 1.0:
            return 'high'
        else:
            return 'medium'
    
    def _assess_signal_quality(self, confidence: float, prediction_value: float) -> str:
        """Assess overall signal quality"""
        if confidence >= 80 and abs(prediction_value - 0.5) >= 0.3:
            return 'excellent'
        elif confidence >= 70 and abs(prediction_value - 0.5) >= 0.2:
            return 'good'
        elif confidence >= 60 and abs(prediction_value - 0.5) >= 0.1:
            return 'fair'
        else:
            return 'poor'
    
    def _get_error_prediction(self, error_message: str) -> Dict[str, Any]:
        """Get standardized error prediction"""
        return {
            'prediction_value': 0.5,
            'confidence': 0.0,
            'signal_type': 'HOLD',
            'model_type': 'error',
            'timestamp': datetime.now().isoformat(),
            'processing_status': 'error',
            'error_message': error_message
        }
    
    def _get_error_signal(self, error_message: str) -> Dict[str, Any]:
        """Get standardized error signal"""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'symbol': 'ERROR',
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'signal_quality': 'error'
        }
    
    def get_prediction_thresholds(self) -> Dict[str, Any]:
        """Get current prediction thresholds"""
        return {
            'standard_thresholds': self.PREDICTION_THRESHOLDS.copy(),
            'volatility_thresholds': self.VOLATILITY_THRESHOLDS.copy()
        }
    
    def update_prediction_thresholds(self, new_thresholds: Dict[str, Any]) -> bool:
        """Update prediction thresholds (for optimization)"""
        try:
            if 'standard' in new_thresholds:
                self.PREDICTION_THRESHOLDS.update(new_thresholds['standard'])
            
            if 'volatility' in new_thresholds:
                self.VOLATILITY_THRESHOLDS.update(new_thresholds['volatility'])
            
            logger.info("Prediction thresholds updated")
            return True
            
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
            return False 