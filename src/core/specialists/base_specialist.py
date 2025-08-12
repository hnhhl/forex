"""
Base Specialist Class
================================================================================
Foundation class cho Multi-Perspective Ensemble System
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpecialistVote:
    """Vote structure từ mỗi specialist"""
    
    specialist_name: str
    vote: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
    technical_data: Dict[str, Any]
    
    def __post_init__(self):
        """Validate vote data"""
        if self.vote not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Invalid vote: {self.vote}. Must be BUY, SELL, or HOLD")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be between 0.0 and 1.0")


class BaseSpecialist(ABC):
    """Base class cho tất cả specialists"""
    
    def __init__(self, name: str, category: str, description: str):
        self.name = name
        self.category = category  # Technical, Sentiment, Pattern, Risk, Momentum, Volatility
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Performance tracking
        self.vote_history = []
        self.accuracy_history = []
        self.confidence_calibration = []
        
        # Configuration
        self.enabled = True
        self.weight = 1.0
        self.min_data_points = 20
        
        self.logger.info(f"{self.name} specialist initialized in {self.category} category")
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, current_price: float, **kwargs) -> SpecialistVote:
        """
        Analyze data và generate vote
        
        Args:
            data: Historical price data
            current_price: Current market price
            **kwargs: Additional parameters
        
        Returns:
            SpecialistVote: Vote với confidence và reasoning
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on analysis
        
        Args:
            analysis_result: Result từ technical analysis
        
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def generate_reasoning(self, analysis_result: Dict[str, Any], vote: str) -> str:
        """
        Generate human-readable reasoning cho vote
        
        Args:
            analysis_result: Analysis data
            vote: Vote decision
        
        Returns:
            str: Detailed reasoning
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality"""
        try:
            if data is None or data.empty:
                self.logger.warning(f"{self.name}: Empty data provided")
                return False
            
            if len(data) < self.min_data_points:
                self.logger.warning(f"{self.name}: Insufficient data points: {len(data)} < {self.min_data_points}")
                return False
            
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"{self.name}: Missing required columns: {missing_columns}")
                return False
            
            # Check for NaN values
            if data[required_columns].isnull().any().any():
                self.logger.warning(f"{self.name}: Data contains NaN values")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"{self.name}: Data validation error: {e}")
            return False
    
    def update_performance(self, actual_outcome: str, predicted_vote: str, confidence: float):
        """Update performance metrics"""
        try:
            # Calculate accuracy
            is_correct = (actual_outcome == predicted_vote)
            self.accuracy_history.append(is_correct)
            
            # Update confidence calibration
            self.confidence_calibration.append({
                'confidence': confidence,
                'correct': is_correct,
                'timestamp': datetime.now()
            })
            
            # Keep only recent history (last 100 votes)
            if len(self.accuracy_history) > 100:
                self.accuracy_history = self.accuracy_history[-100:]
            
            if len(self.confidence_calibration) > 100:
                self.confidence_calibration = self.confidence_calibration[-100:]
            
            self.logger.debug(f"{self.name}: Performance updated. Recent accuracy: {self.get_recent_accuracy():.2%}")
            
        except Exception as e:
            self.logger.error(f"{self.name}: Error updating performance: {e}")
    
    def get_recent_accuracy(self, window: int = 20) -> float:
        """Get recent accuracy over specified window"""
        if not self.accuracy_history:
            return 0.5  # Default neutral accuracy
        
        recent_results = self.accuracy_history[-window:]
        return sum(recent_results) / len(recent_results)
    
    def get_confidence_calibration(self) -> Dict[str, float]:
        """Get confidence calibration metrics"""
        if not self.confidence_calibration:
            return {'calibration_error': 0.0, 'avg_confidence': 0.5}
        
        try:
            # Calculate calibration error
            total_error = 0.0
            total_weight = 0.0
            
            for record in self.confidence_calibration:
                confidence = record['confidence']
                actual = 1.0 if record['correct'] else 0.0
                error = abs(confidence - actual)
                
                total_error += error * confidence  # Weight by confidence
                total_weight += confidence
            
            calibration_error = total_error / total_weight if total_weight > 0 else 0.0
            avg_confidence = np.mean([r['confidence'] for r in self.confidence_calibration])
            
            return {
                'calibration_error': calibration_error,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            self.logger.error(f"{self.name}: Error calculating calibration: {e}")
            return {'calibration_error': 0.0, 'avg_confidence': 0.5}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'name': self.name,
            'category': self.category,
            'enabled': self.enabled,
            'weight': self.weight,
            'total_votes': len(self.vote_history),
            'recent_accuracy': self.get_recent_accuracy(),
            'overall_accuracy': self.get_recent_accuracy(len(self.accuracy_history)) if self.accuracy_history else 0.5,
            'confidence_calibration': self.get_confidence_calibration(),
            'last_vote_time': self.vote_history[-1]['timestamp'] if self.vote_history else None
        }
    
    def enable(self):
        """Enable specialist"""
        self.enabled = True
        self.logger.info(f"{self.name} specialist enabled")
    
    def disable(self):
        """Disable specialist"""
        self.enabled = False
        self.logger.info(f"{self.name} specialist disabled")
    
    def set_weight(self, weight: float):
        """Set specialist weight"""
        if 0.0 <= weight <= 2.0:  # Allow up to 2x weight
            self.weight = weight
            self.logger.info(f"{self.name} weight set to {weight}")
        else:
            self.logger.warning(f"{self.name}: Invalid weight {weight}. Must be between 0.0 and 2.0")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category}): Accuracy={self.get_recent_accuracy():.1%}, Weight={self.weight}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', category='{self.category}', enabled={self.enabled})>" 