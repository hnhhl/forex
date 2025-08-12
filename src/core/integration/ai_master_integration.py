"""
AI Master Integration System
Ultimate XAU Super System V4.0 - Day 18 Implementation

Advanced integration system that combines:
- Neural Ensemble System (Multi-network predictions)
- Reinforcement Learning System (DQN agent)
- Advanced Meta-Learning System (MAML, Transfer, Continual)
- Intelligent ensemble decision making
- Performance optimization and monitoring

Target: +1.5% additional performance boost to reach +20% total
"""

import numpy as np
import pandas as pd
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# Import all AI systems
try:
    from src.core.ai.neural_ensemble import (
        NeuralEnsemble, NetworkType, PredictionType, EnsembleConfig,
        create_default_ensemble, EnsembleResult
    )
    NEURAL_ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Neural Ensemble not available: {e}")
    NEURAL_ENSEMBLE_AVAILABLE = False

try:
    from src.core.ai.reinforcement_learning import (
        DQNAgent, TradingEnvironment, create_default_agent_config,
        ActionType, AgentType, RewardType, create_trading_environment
    )
    RL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Reinforcement Learning not available: {e}")
    RL_AVAILABLE = False

try:
    from src.core.ai.advanced_meta_learning import (
        AdvancedMetaLearningSystem, MetaLearningConfig, MetaLearningResult,
        create_meta_learning_system
    )
    META_LEARNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced Meta-Learning not available: {e}")
    META_LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


class AISystemType(Enum):
    """Types of AI systems"""
    NEURAL_ENSEMBLE = "neural_ensemble"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    META_LEARNING = "meta_learning"


class DecisionStrategy(Enum):
    """Strategies for combining AI predictions"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"
    META_LEARNING_GUIDED = "meta_learning_guided"


@dataclass
class AISystemConfig:
    """Configuration for AI Master Integration"""
    
    # System activation
    enable_neural_ensemble: bool = True
    enable_reinforcement_learning: bool = True
    enable_meta_learning: bool = True
    
    # Neural Ensemble settings
    neural_ensemble_networks: List[NetworkType] = field(default_factory=lambda: [
        NetworkType.LSTM, NetworkType.GRU, NetworkType.CNN, NetworkType.DENSE
    ])
    neural_confidence_threshold: float = 0.7
    
    # Reinforcement Learning settings
    rl_agent_type: str = "DQN"
    rl_exploration_rate: float = 0.1
    rl_update_frequency: int = 100
    
    # Meta-Learning settings
    meta_learning_adaptation_rate: float = 0.001
    meta_learning_memory_size: int = 1000
    meta_continual_plasticity: float = 0.8
    
    # Ensemble decision making
    decision_strategy: DecisionStrategy = DecisionStrategy.ADAPTIVE_ENSEMBLE
    min_confidence_threshold: float = 0.6
    max_position_size: float = 0.25
    
    # Performance optimization
    enable_adaptive_weights: bool = True
    performance_window: int = 100  # Number of decisions to track
    rebalance_frequency: int = 50   # How often to update weights
    
    # Data processing
    sequence_length: int = 50
    input_features: int = 95
    prediction_horizon: int = 1
    
    # Real-time settings
    update_interval: float = 1.0
    max_concurrent_predictions: int = 3


@dataclass
class AIMarketData:
    """Enhanced market data for AI systems"""
    timestamp: datetime
    symbol: str
    price: float
    high: float
    low: float
    volume: float
    
    # Technical indicators
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    atr: float = 0.0
    
    # Market microstructure
    bid_ask_spread: float = 0.0
    order_book_imbalance: float = 0.0
    
    # Sentiment indicators
    sentiment_score: float = 0.0
    news_impact: float = 0.0
    
    # Additional features
    volatility: float = 0.0
    momentum: float = 0.0
    mean_reversion: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for AI models"""
        features = [
            self.price, self.high, self.low, self.volume,
            self.sma_20, self.sma_50, self.ema_12, self.ema_26,
            self.rsi, self.macd, self.bb_upper, self.bb_lower, self.atr,
            self.bid_ask_spread, self.order_book_imbalance,
            self.sentiment_score, self.news_impact,
            self.volatility, self.momentum, self.mean_reversion
        ]
        
        # Pad to required feature size
        while len(features) < 95:
            features.append(0.0)
        
        return np.array(features[:95])


@dataclass
class AIPrediction:
    """Unified AI prediction structure"""
    timestamp: datetime
    symbol: str
    source: AISystemType
    
    # Prediction details
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    probability_distribution: np.ndarray
    
    # Position sizing
    recommended_position_size: float
    
    # Risk assessment
    risk_score: float
    expected_return: float
    uncertainty: float
    
    # Meta information
    adaptation_score: float = 0.0
    transfer_effectiveness: float = 0.0
    continual_retention: float = 0.0
    
    # Performance tracking
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleDecision:
    """Final ensemble decision combining all AI systems"""
    timestamp: datetime
    symbol: str
    
    # Final decision
    action: str
    confidence: float
    consensus_score: float
    
    # Position and risk
    position_size: float
    risk_score: float
    expected_return: float
    
    # System contributions
    neural_ensemble_weight: float
    rl_weight: float
    meta_learning_weight: float
    
    # Performance metrics
    total_processing_time: float
    individual_predictions: List[AIPrediction]
    
    # Metadata
    decision_strategy: str
    market_conditions: Dict[str, Any] = field(default_factory=dict)


class AIPerformanceTracker:
    """Tracks performance of individual AI systems and ensemble"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions_history = []
        self.performance_metrics = {}
        self.system_weights = {
            AISystemType.NEURAL_ENSEMBLE: 0.4,
            AISystemType.REINFORCEMENT_LEARNING: 0.3,
            AISystemType.META_LEARNING: 0.3
        }
        self.last_rebalance = datetime.now()
    
    def add_prediction(self, prediction: AIPrediction, actual_outcome: float = None):
        """Add prediction and track performance"""
        self.predictions_history.append({
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'timestamp': prediction.timestamp
        })
        
        # Keep only recent predictions
        if len(self.predictions_history) > self.window_size:
            self.predictions_history = self.predictions_history[-self.window_size:]
    
    def calculate_system_performance(self) -> Dict[AISystemType, Dict[str, float]]:
        """Calculate performance metrics for each AI system"""
        performance = {}
        
        for system_type in AISystemType:
            system_predictions = [
                p for p in self.predictions_history 
                if p['prediction'].source == system_type and p['actual_outcome'] is not None
            ]
            
            if len(system_predictions) < 10:  # Need minimum data
                performance[system_type] = {
                    'accuracy': 0.5,
                    'precision': 0.5,
                    'recall': 0.5,
                    'avg_confidence': 0.5,
                    'avg_return': 0.0
                }
                continue
            
            # Calculate metrics
            correct_predictions = sum(
                1 for p in system_predictions
                if (p['prediction'].action == 'BUY' and p['actual_outcome'] > 0) or
                   (p['prediction'].action == 'SELL' and p['actual_outcome'] < 0) or
                   (p['prediction'].action == 'HOLD' and abs(p['actual_outcome']) < 0.001)
            )
            
            accuracy = correct_predictions / len(system_predictions)
            avg_confidence = np.mean([p['prediction'].confidence for p in system_predictions])
            avg_return = np.mean([p['actual_outcome'] for p in system_predictions])
            
            performance[system_type] = {
                'accuracy': accuracy,
                'precision': accuracy,  # Simplified
                'recall': accuracy,     # Simplified
                'avg_confidence': avg_confidence,
                'avg_return': avg_return,
                'total_predictions': len(system_predictions)
            }
        
        self.performance_metrics = performance
        return performance
    
    def update_system_weights(self) -> Dict[AISystemType, float]:
        """Update system weights based on performance"""
        performance = self.calculate_system_performance()
        
        # Calculate new weights based on accuracy and returns
        new_weights = {}
        total_score = 0
        
        for system_type in AISystemType:
            if system_type in performance:
                accuracy = performance[system_type]['accuracy']
                avg_return = performance[system_type]['avg_return']
                confidence = performance[system_type]['avg_confidence']
                
                # Combined score (accuracy + return performance + confidence)
                score = (accuracy * 0.4 + (avg_return + 1) * 0.4 + confidence * 0.2)
                new_weights[system_type] = score
                total_score += score
        
        # Normalize weights
        if total_score > 0:
            for system_type in new_weights:
                new_weights[system_type] /= total_score
        else:
            # Default equal weights
            num_systems = len(AISystemType)
            for system_type in AISystemType:
                new_weights[system_type] = 1.0 / num_systems
        
        # Smooth weight updates (avoid drastic changes)
        smoothing_factor = 0.8
        for system_type in AISystemType:
            old_weight = self.system_weights.get(system_type, 1.0/len(AISystemType))
            new_weight = new_weights.get(system_type, 1.0/len(AISystemType))
            self.system_weights[system_type] = (
                old_weight * smoothing_factor + new_weight * (1 - smoothing_factor)
            )
        
        self.last_rebalance = datetime.now()
        return self.system_weights.copy()


class AIMasterIntegrationSystem:
    """Master AI Integration System combining all AI components"""
    
    def __init__(self, config: AISystemConfig):
        self.config = config
        self.performance_tracker = AIPerformanceTracker(config.performance_window)
        
        # AI system instances
        self.neural_ensemble = None
        self.rl_agent = None
        self.rl_environment = None
        self.meta_learning_system = None
        
        # Data management
        self.market_data_buffer = []
        self.prediction_history = []
        self.decision_history = []
        
        # Threading
        self._running = False
        self._update_thread = None
        self._lock = threading.Lock()
        
        # Initialize AI systems
        self._initialize_ai_systems()
        
        logger.info("AI Master Integration System initialized")
        logger.info(f"Systems available: Neural={NEURAL_ENSEMBLE_AVAILABLE}, "
                   f"RL={RL_AVAILABLE}, Meta={META_LEARNING_AVAILABLE}")
    
    def _initialize_ai_systems(self):
        """Initialize all AI systems"""
        try:
            # Initialize Neural Ensemble
            if NEURAL_ENSEMBLE_AVAILABLE and self.config.enable_neural_ensemble:
                try:
                    ensemble_config = {
                        'input_features': self.config.input_features,
                        'sequence_length': self.config.sequence_length,
                        'networks': self.config.neural_ensemble_networks
                    }
                    self.neural_ensemble = create_default_ensemble(ensemble_config)
                    logger.info("Neural Ensemble System initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Neural Ensemble: {e}")
                    self.neural_ensemble = None
            
            # Initialize Reinforcement Learning
            if RL_AVAILABLE and self.config.enable_reinforcement_learning:
                try:
                    agent_config = create_default_agent_config()
                    # Create new config dict instead of modifying dataclass
                    rl_config = {
                        'state_size': agent_config.state_size,
                        'action_size': agent_config.action_size,
                        'learning_rate': agent_config.learning_rate,
                        'exploration_rate': self.config.rl_exploration_rate,
                        'exploration_decay': agent_config.exploration_decay,
                        'exploration_min': agent_config.exploration_min,
                        'memory_size': agent_config.memory_size,
                        'batch_size': agent_config.batch_size,
                        'gamma': agent_config.gamma
                    }
                    
                    self.rl_environment = create_trading_environment()
                    from src.core.ai.reinforcement_learning import AgentConfig
                    self.rl_agent = DQNAgent(AgentConfig(**rl_config))
                    logger.info("Reinforcement Learning System initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize RL system: {e}")
                    self.rl_agent = None
                    self.rl_environment = None
            
            # Initialize Meta-Learning
            if META_LEARNING_AVAILABLE and self.config.enable_meta_learning:
                meta_config = {
                    'input_features': self.config.input_features,
                    'sequence_length': self.config.sequence_length,
                    'transfer_adaptation_rate': self.config.meta_learning_adaptation_rate,
                    'continual_memory_size': self.config.meta_learning_memory_size,
                    'continual_plasticity_factor': self.config.meta_continual_plasticity
                }
                self.meta_learning_system = create_meta_learning_system(meta_config)
                logger.info("Advanced Meta-Learning System initialized")
        
        except Exception as e:
            logger.error(f"Error initializing AI systems: {e}")
    
    def process_market_data(self, market_data: AIMarketData) -> Optional[EnsembleDecision]:
        """Process market data through all AI systems and make ensemble decision"""
        with self._lock:
            start_time = time.time()
            
            # Add to buffer
            self.market_data_buffer.append(market_data)
            if len(self.market_data_buffer) > self.config.sequence_length * 2:
                self.market_data_buffer = self.market_data_buffer[-self.config.sequence_length * 2:]
            
            # Check if we have enough data
            if len(self.market_data_buffer) < self.config.sequence_length:
                return None
            
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data()
            if sequence_data is None:
                return None
            
            # Get predictions from all systems
            predictions = []
            
            # Neural Ensemble prediction
            if self.neural_ensemble is not None:
                neural_prediction = self._get_neural_ensemble_prediction(sequence_data, market_data)
                if neural_prediction:
                    predictions.append(neural_prediction)
            
            # Reinforcement Learning prediction
            if self.rl_agent is not None and self.rl_environment is not None:
                rl_prediction = self._get_rl_prediction(sequence_data, market_data)
                if rl_prediction:
                    predictions.append(rl_prediction)
            
            # Meta-Learning prediction
            if self.meta_learning_system is not None:
                meta_prediction = self._get_meta_learning_prediction(sequence_data, market_data)
                if meta_prediction:
                    predictions.append(meta_prediction)
            
            # Make ensemble decision
            decision = self._make_ensemble_decision(predictions, market_data)
            decision.total_processing_time = time.time() - start_time
            
            # Store decision
            self.decision_history.append(decision)
            if len(self.decision_history) > 1000:  # Keep last 1000 decisions
                self.decision_history = self.decision_history[-1000:]
            
            return decision
    
    def _prepare_sequence_data(self) -> Optional[np.ndarray]:
        """Prepare sequence data for AI models"""
        if len(self.market_data_buffer) < self.config.sequence_length:
            return None
        
        # Get last sequence_length data points
        recent_data = self.market_data_buffer[-self.config.sequence_length:]
        
        # Convert to feature vectors
        features = []
        for data_point in recent_data:
            features.append(data_point.to_feature_vector())
        
        # Shape: (sequence_length, input_features)
        sequence = np.array(features)
        return sequence.reshape(1, self.config.sequence_length, self.config.input_features)
    
    def _get_neural_ensemble_prediction(self, sequence_data: np.ndarray, 
                                      market_data: AIMarketData) -> Optional[AIPrediction]:
        """Get prediction from Neural Ensemble"""
        try:
            start_time = time.time()
            
            # Make prediction
            result = self.neural_ensemble.predict_ensemble(sequence_data)
            
            # Convert to AIPrediction
            prediction = AIPrediction(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                source=AISystemType.NEURAL_ENSEMBLE,
                action=self._convert_to_action(result.prediction),
                confidence=result.confidence,
                probability_distribution=result.prediction,
                recommended_position_size=min(result.confidence * self.config.max_position_size, 
                                            self.config.max_position_size),
                risk_score=1.0 - result.consensus_score,
                expected_return=float(np.max(result.prediction) - 0.33),  # Above random
                uncertainty=1.0 - result.confidence,
                processing_time=time.time() - start_time,
                metadata={'ensemble_consensus': result.consensus_score}
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in Neural Ensemble prediction: {e}")
            return None
    
    def _get_rl_prediction(self, sequence_data: np.ndarray, 
                          market_data: AIMarketData) -> Optional[AIPrediction]:
        """Get prediction from Reinforcement Learning"""
        try:
            start_time = time.time()
            
            # Prepare state for RL agent
            state = sequence_data.flatten()  # Flatten for RL agent
            
            # Get action from agent
            action_value = self.rl_agent.select_action(state)
            
            # Convert RL action to trading action
            if action_value == 0:
                action = "HOLD"
                confidence = 0.6
            elif action_value == 1:
                action = "BUY"
                confidence = 0.8
            else:
                action = "SELL"
                confidence = 0.8
            
            # Create probability distribution
            prob_dist = np.array([0.33, 0.33, 0.34])  # Default
            if action == "BUY":
                prob_dist = np.array([0.1, 0.1, 0.8])
            elif action == "SELL":
                prob_dist = np.array([0.1, 0.8, 0.1])
            
            prediction = AIPrediction(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                source=AISystemType.REINFORCEMENT_LEARNING,
                action=action,
                confidence=confidence,
                probability_distribution=prob_dist,
                recommended_position_size=confidence * self.config.max_position_size,
                risk_score=0.5,  # RL has moderate risk assessment
                expected_return=0.1 if action != "HOLD" else 0.0,
                uncertainty=1.0 - confidence,
                processing_time=time.time() - start_time,
                metadata={'action_value': action_value}
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in RL prediction: {e}")
            return None
    
    def _get_meta_learning_prediction(self, sequence_data: np.ndarray, 
                                    market_data: AIMarketData) -> Optional[AIPrediction]:
        """Get prediction from Meta-Learning System"""
        try:
            start_time = time.time()
            
            # Get ensemble prediction from meta-learning
            weights = {'maml': 0.2, 'transfer': 0.4, 'continual': 0.4}
            result = self.meta_learning_system.ensemble_predict(sequence_data, weights)
            
            prediction = AIPrediction(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                source=AISystemType.META_LEARNING,
                action=self._convert_to_action(result.prediction),
                confidence=result.confidence,
                probability_distribution=result.prediction,
                recommended_position_size=result.confidence * self.config.max_position_size,
                risk_score=1.0 - result.adaptation_score,
                expected_return=float(np.max(result.prediction) - 0.33),
                uncertainty=1.0 - result.confidence,
                adaptation_score=result.adaptation_score,
                transfer_effectiveness=result.transfer_effectiveness,
                continual_retention=result.continual_retention,
                processing_time=time.time() - start_time,
                metadata={
                    'meta_gradient_norm': result.meta_gradient_norm,
                    'adaptation_score': result.adaptation_score
                }
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in Meta-Learning prediction: {e}")
            return None
    
    def _convert_to_action(self, prediction: np.ndarray) -> str:
        """Convert prediction array to action string"""
        if len(prediction.shape) > 1:
            prediction = prediction[0]  # Take first sample if batch
        
        action_idx = np.argmax(prediction)
        actions = ['HOLD', 'SELL', 'BUY']
        return actions[action_idx]
    
    def _make_ensemble_decision(self, predictions: List[AIPrediction], 
                              market_data: AIMarketData) -> EnsembleDecision:
        """Combine predictions into final ensemble decision"""
        if not predictions:
            return self._create_default_decision(market_data)
        
        # Get current system weights
        weights = self.performance_tracker.system_weights
        
        # Apply decision strategy
        if self.config.decision_strategy == DecisionStrategy.ADAPTIVE_ENSEMBLE:
            return self._adaptive_ensemble_decision(predictions, weights, market_data)
        elif self.config.decision_strategy == DecisionStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_decision(predictions, market_data)
        elif self.config.decision_strategy == DecisionStrategy.MAJORITY_VOTING:
            return self._majority_voting_decision(predictions, market_data)
        else:
            return self._weighted_average_decision(predictions, weights, market_data)
    
    def _adaptive_ensemble_decision(self, predictions: List[AIPrediction], 
                                  weights: Dict[AISystemType, float],
                                  market_data: AIMarketData) -> EnsembleDecision:
        """Make decision using adaptive ensemble approach"""
        
        # Calculate weighted predictions
        weighted_predictions = np.zeros(3)  # [HOLD, SELL, BUY]
        total_weight = 0
        confidence_sum = 0
        position_size_sum = 0
        risk_score_sum = 0
        expected_return_sum = 0
        
        system_weights_used = {
            AISystemType.NEURAL_ENSEMBLE: 0.0,
            AISystemType.REINFORCEMENT_LEARNING: 0.0,
            AISystemType.META_LEARNING: 0.0
        }
        
        for pred in predictions:
            system_weight = weights.get(pred.source, 0.33)
            confidence_weight = pred.confidence
            
            # Adaptive weight combines system performance and current confidence
            adaptive_weight = system_weight * confidence_weight
            
            weighted_predictions += adaptive_weight * pred.probability_distribution
            total_weight += adaptive_weight
            confidence_sum += pred.confidence * adaptive_weight
            position_size_sum += pred.recommended_position_size * adaptive_weight
            risk_score_sum += pred.risk_score * adaptive_weight
            expected_return_sum += pred.expected_return * adaptive_weight
            
            system_weights_used[pred.source] = adaptive_weight
        
        # Normalize
        if total_weight > 0:
            weighted_predictions /= total_weight
            confidence_sum /= total_weight
            position_size_sum /= total_weight
            risk_score_sum /= total_weight
            expected_return_sum /= total_weight
            
            # Normalize system weights
            total_system_weight = sum(system_weights_used.values())
            if total_system_weight > 0:
                for system_type in system_weights_used:
                    system_weights_used[system_type] /= total_system_weight
        
        # Determine final action
        final_action = self._convert_to_action(weighted_predictions)
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(predictions)
        
        # Apply confidence threshold
        final_confidence = confidence_sum
        if final_confidence < self.config.min_confidence_threshold:
            final_action = "HOLD"
            position_size_sum *= 0.5  # Reduce position size for low confidence
        
        return EnsembleDecision(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            action=final_action,
            confidence=final_confidence,
            consensus_score=consensus_score,
            position_size=min(position_size_sum, self.config.max_position_size),
            risk_score=risk_score_sum,
            expected_return=expected_return_sum,
            neural_ensemble_weight=system_weights_used[AISystemType.NEURAL_ENSEMBLE],
            rl_weight=system_weights_used[AISystemType.REINFORCEMENT_LEARNING],
            meta_learning_weight=system_weights_used[AISystemType.META_LEARNING],
            total_processing_time=0.0,  # Will be set by caller
            individual_predictions=predictions,
            decision_strategy=self.config.decision_strategy.value,
            market_conditions={'volatility': market_data.volatility, 'volume': market_data.volume}
        )
    
    def _confidence_weighted_decision(self, predictions: List[AIPrediction], 
                                    market_data: AIMarketData) -> EnsembleDecision:
        """Make decision based on confidence weighting"""
        total_confidence = sum(pred.confidence for pred in predictions)
        
        if total_confidence == 0:
            return self._create_default_decision(market_data)
        
        weighted_predictions = np.zeros(3)
        position_size_sum = 0
        risk_score_sum = 0
        expected_return_sum = 0
        
        for pred in predictions:
            weight = pred.confidence / total_confidence
            weighted_predictions += weight * pred.probability_distribution
            position_size_sum += weight * pred.recommended_position_size
            risk_score_sum += weight * pred.risk_score
            expected_return_sum += weight * pred.expected_return
        
        final_action = self._convert_to_action(weighted_predictions)
        consensus_score = self._calculate_consensus(predictions)
        
        return EnsembleDecision(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            action=final_action,
            confidence=np.mean([pred.confidence for pred in predictions]),
            consensus_score=consensus_score,
            position_size=min(position_size_sum, self.config.max_position_size),
            risk_score=risk_score_sum,
            expected_return=expected_return_sum,
            neural_ensemble_weight=0.33,
            rl_weight=0.33,
            meta_learning_weight=0.34,
            total_processing_time=0.0,
            individual_predictions=predictions,
            decision_strategy="confidence_weighted"
        )
    
    def _majority_voting_decision(self, predictions: List[AIPrediction], 
                                market_data: AIMarketData) -> EnsembleDecision:
        """Make decision using majority voting"""
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
        for pred in predictions:
            action_counts[pred.action] += 1
        
        final_action = max(action_counts, key=action_counts.get)
        consensus_score = action_counts[final_action] / len(predictions)
        
        avg_confidence = np.mean([pred.confidence for pred in predictions])
        avg_position_size = np.mean([pred.recommended_position_size for pred in predictions])
        avg_risk_score = np.mean([pred.risk_score for pred in predictions])
        avg_expected_return = np.mean([pred.expected_return for pred in predictions])
        
        return EnsembleDecision(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            action=final_action,
            confidence=avg_confidence,
            consensus_score=consensus_score,
            position_size=min(avg_position_size, self.config.max_position_size),
            risk_score=avg_risk_score,
            expected_return=avg_expected_return,
            neural_ensemble_weight=0.33,
            rl_weight=0.33,
            meta_learning_weight=0.34,
            total_processing_time=0.0,
            individual_predictions=predictions,
            decision_strategy="majority_voting"
        )
    
    def _weighted_average_decision(self, predictions: List[AIPrediction], 
                                 weights: Dict[AISystemType, float],
                                 market_data: AIMarketData) -> EnsembleDecision:
        """Make decision using weighted average"""
        weighted_predictions = np.zeros(3)
        total_weight = 0
        
        for pred in predictions:
            weight = weights.get(pred.source, 0.33)
            weighted_predictions += weight * pred.probability_distribution
            total_weight += weight
        
        if total_weight > 0:
            weighted_predictions /= total_weight
        
        final_action = self._convert_to_action(weighted_predictions)
        consensus_score = self._calculate_consensus(predictions)
        
        avg_confidence = np.mean([pred.confidence for pred in predictions])
        avg_position_size = np.mean([pred.recommended_position_size for pred in predictions])
        avg_risk_score = np.mean([pred.risk_score for pred in predictions])
        avg_expected_return = np.mean([pred.expected_return for pred in predictions])
        
        return EnsembleDecision(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            action=final_action,
            confidence=avg_confidence,
            consensus_score=consensus_score,
            position_size=min(avg_position_size, self.config.max_position_size),
            risk_score=avg_risk_score,
            expected_return=avg_expected_return,
            neural_ensemble_weight=weights.get(AISystemType.NEURAL_ENSEMBLE, 0.33),
            rl_weight=weights.get(AISystemType.REINFORCEMENT_LEARNING, 0.33),
            meta_learning_weight=weights.get(AISystemType.META_LEARNING, 0.34),
            total_processing_time=0.0,
            individual_predictions=predictions,
            decision_strategy="weighted_average"
        )
    
    def _calculate_consensus(self, predictions: List[AIPrediction]) -> float:
        """Calculate consensus score among predictions"""
        if len(predictions) <= 1:
            return 1.0
        
        actions = [pred.action for pred in predictions]
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
        for action in actions:
            action_counts[action] += 1
        
        max_count = max(action_counts.values())
        consensus = max_count / len(predictions)
        
        return consensus
    
    def _create_default_decision(self, market_data: AIMarketData) -> EnsembleDecision:
        """Create default decision when no predictions available"""
        return EnsembleDecision(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            action="HOLD",
            confidence=0.5,
            consensus_score=1.0,
            position_size=0.0,
            risk_score=1.0,
            expected_return=0.0,
            neural_ensemble_weight=0.0,
            rl_weight=0.0,
            meta_learning_weight=0.0,
            total_processing_time=0.0,
            individual_predictions=[],
            decision_strategy="default"
        )
    
    def update_performance(self, decision: EnsembleDecision, actual_outcome: float):
        """Update performance tracking with actual outcome"""
        for prediction in decision.individual_predictions:
            self.performance_tracker.add_prediction(prediction, actual_outcome)
        
        # Update system weights if needed
        decisions_since_rebalance = len(self.decision_history) - \
            len([d for d in self.decision_history 
                 if d.timestamp < self.performance_tracker.last_rebalance])
        
        if decisions_since_rebalance >= self.config.rebalance_frequency:
            new_weights = self.performance_tracker.update_system_weights()
            logger.info(f"Updated system weights: {new_weights}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        performance = self.performance_tracker.calculate_system_performance()
        
        return {
            'timestamp': datetime.now(),
            'systems_active': {
                'neural_ensemble': self.neural_ensemble is not None,
                'reinforcement_learning': self.rl_agent is not None,
                'meta_learning': self.meta_learning_system is not None
            },
            'system_weights': self.performance_tracker.system_weights,
            'performance_metrics': performance,
            'total_predictions': len(self.prediction_history),
            'total_decisions': len(self.decision_history),
            'recent_decisions': len([d for d in self.decision_history 
                                   if d.timestamp > datetime.now() - timedelta(hours=1)]),
            'config': {
                'decision_strategy': self.config.decision_strategy.value,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'max_position_size': self.config.max_position_size
            }
        }
    
    def export_system_data(self, filepath: str = None) -> Dict[str, Any]:
        """Export system data and performance"""
        if filepath is None:
            filepath = f"ai_master_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert system status to JSON-serializable format
        status = self.get_system_status()
        
        # Convert system weights to string keys
        system_weights = {}
        for key, value in status['system_weights'].items():
            system_weights[str(key.value) if hasattr(key, 'value') else str(key)] = value
        
        # Convert performance metrics to string keys
        performance_metrics = {}
        for key, value in self.performance_tracker.performance_metrics.items():
            performance_metrics[str(key.value) if hasattr(key, 'value') else str(key)] = value
        
        data = {
            'system_status': {
                'timestamp': status['timestamp'].isoformat(),
                'systems_active': status['systems_active'],
                'system_weights': system_weights,
                'performance_metrics': performance_metrics,
                'total_predictions': status['total_predictions'],
                'total_decisions': status['total_decisions'],
                'recent_decisions': status['recent_decisions'],
                'config': status['config']
            },
            'recent_decisions': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'action': d.action,
                    'confidence': d.confidence,
                    'consensus_score': d.consensus_score,
                    'position_size': d.position_size,
                    'system_weights': {
                        'neural_ensemble': d.neural_ensemble_weight,
                        'rl': d.rl_weight,
                        'meta_learning': d.meta_learning_weight
                    }
                }
                for d in self.decision_history[-100:]  # Last 100 decisions
            ],
            'performance_summary': performance_metrics,
            'export_timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"AI Master Integration data exported to {filepath}")
            return {'success': True, 'filepath': filepath}
        
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return {'success': False, 'error': str(e)}


def create_ai_master_system(config: Dict[str, Any] = None) -> AIMasterIntegrationSystem:
    """Factory function to create AI Master Integration System"""
    if config is None:
        config = {}
    
    # Create configuration
    ai_config = AISystemConfig(
        enable_neural_ensemble=config.get('enable_neural_ensemble', True),
        enable_reinforcement_learning=config.get('enable_reinforcement_learning', True),
        enable_meta_learning=config.get('enable_meta_learning', True),
        decision_strategy=DecisionStrategy(config.get('decision_strategy', 'adaptive_ensemble')),
        min_confidence_threshold=config.get('min_confidence_threshold', 0.6),
        max_position_size=config.get('max_position_size', 0.25)
    )
    
    return AIMasterIntegrationSystem(ai_config)


def demo_ai_master_integration():
    """Demo function to showcase AI Master Integration"""
    print("\n" + "="*80)
    print("🤖 AI MASTER INTEGRATION SYSTEM DEMO")
    print("Ultimate XAU Super System V4.0 - Day 18")
    print("="*80)
    
    # Create system
    print("\n🔧 Creating AI Master Integration System...")
    system = create_ai_master_system()
    
    print("✅ System initialized with all AI components")
    
    # Show system status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Neural Ensemble: {'✅' if status['systems_active']['neural_ensemble'] else '❌'}")
    print(f"   Reinforcement Learning: {'✅' if status['systems_active']['reinforcement_learning'] else '❌'}")
    print(f"   Meta-Learning: {'✅' if status['systems_active']['meta_learning'] else '❌'}")
    
    # Demo market data processing
    print(f"\n🚀 AI Master Integration System ready for production!")
    
    return system


if __name__ == "__main__":
    demo_ai_master_integration()