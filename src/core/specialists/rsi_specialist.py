"""
RSI Specialist
================================================================================
Technical Specialist chuyên về RSI (Relative Strength Index)
Thuộc Technical Category trong Multi-Perspective Ensemble System
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from .base_specialist import BaseSpecialist, SpecialistVote

logger = logging.getLogger(__name__)


class RSISpecialist(BaseSpecialist):
    """
    RSI Specialist - Chuyên gia phân tích RSI
    
    RSI Strategy:
    - RSI < 30: Oversold → BUY signal
    - RSI > 70: Overbought → SELL signal  
    - RSI 30-70: HOLD
    """
    
    def __init__(self, period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70):
        super().__init__(
            name="RSI_Specialist",
            category="Technical",
            description=f"RSI analysis với period={period}, oversold<{oversold_threshold}, overbought>{overbought_threshold}"
        )
        
        self.period = period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.min_data_points = max(20, period * 2)
        
        logger.info(f"RSI Specialist initialized: period={period}, thresholds=({oversold_threshold}, {overbought_threshold})")
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.period
        
        try:
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def analyze(self, data: pd.DataFrame, current_price: float, **kwargs) -> SpecialistVote:
        """Analyze RSI và generate vote"""
        
        if not self.enabled:
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning="Specialist is disabled",
                timestamp=datetime.now(),
                technical_data={}
            )
        
        if not self.validate_data(data):
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning="Invalid or insufficient data",
                timestamp=datetime.now(),
                technical_data={}
            )
        
        try:
            rsi = self.calculate_rsi(data['close'])
            
            if rsi.empty or rsi.isna().all():
                return SpecialistVote(
                    specialist_name=self.name,
                    vote="HOLD",
                    confidence=0.0,
                    reasoning="Unable to calculate RSI",
                    timestamp=datetime.now(),
                    technical_data={}
                )
            
            current_rsi = rsi.iloc[-1]
            
            analysis_result = {
                'current_rsi': current_rsi,
                'rsi_series': rsi,
                'current_price': current_price,
                'period': self.period,
                'oversold_threshold': self.oversold_threshold,
                'overbought_threshold': self.overbought_threshold
            }
            
            vote = self.generate_vote_decision(analysis_result)
            confidence = self.calculate_confidence(analysis_result)
            reasoning = self.generate_reasoning(analysis_result, vote)
            
            specialist_vote = SpecialistVote(
                specialist_name=self.name,
                vote=vote,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
                technical_data=analysis_result
            )
            
            self.vote_history.append({
                'vote': vote,
                'confidence': confidence,
                'rsi_value': current_rsi,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"RSI Analysis: RSI={current_rsi:.2f}, Vote={vote}, Confidence={confidence:.2f}")
            
            return specialist_vote
            
        except Exception as e:
            self.logger.error(f"Error in RSI analysis: {e}")
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                timestamp=datetime.now(),
                technical_data={}
            )
    
    def generate_vote_decision(self, analysis_result: Dict[str, Any]) -> str:
        """Generate vote decision based on RSI analysis"""
        current_rsi = analysis_result['current_rsi']
        
        if current_rsi <= self.oversold_threshold:
            return "BUY"
        elif current_rsi >= self.overbought_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on RSI analysis"""
        current_rsi = analysis_result['current_rsi']
        
        if current_rsi <= 20:
            base_confidence = 0.9
        elif current_rsi <= self.oversold_threshold:
            base_confidence = 0.7
        elif current_rsi >= 80:
            base_confidence = 0.9
        elif current_rsi >= self.overbought_threshold:
            base_confidence = 0.7
        elif 40 <= current_rsi <= 60:
            base_confidence = 0.3
        else:
            base_confidence = 0.5
        
        recent_accuracy = self.get_recent_accuracy()
        if recent_accuracy > 0.6:
            base_confidence *= 1.1
        elif recent_accuracy < 0.4:
            base_confidence *= 0.8
        
        return min(1.0, max(0.1, base_confidence))
    
    def generate_reasoning(self, analysis_result: Dict[str, Any], vote: str) -> str:
        """Generate human-readable reasoning"""
        current_rsi = analysis_result['current_rsi']
        reasoning_parts = []
        
        if current_rsi <= 20:
            reasoning_parts.append(f"RSI extremely oversold at {current_rsi:.1f}")
        elif current_rsi <= self.oversold_threshold:
            reasoning_parts.append(f"RSI oversold at {current_rsi:.1f}")
        elif current_rsi >= 80:
            reasoning_parts.append(f"RSI extremely overbought at {current_rsi:.1f}")
        elif current_rsi >= self.overbought_threshold:
            reasoning_parts.append(f"RSI overbought at {current_rsi:.1f}")
        else:
            reasoning_parts.append(f"RSI neutral at {current_rsi:.1f}")
        
        vote_reasoning = {
            "BUY": "Suggesting BUY - oversold conditions",
            "SELL": "Suggesting SELL - overbought conditions", 
            "HOLD": "Suggesting HOLD - neutral RSI conditions"
        }
        
        reasoning_parts.append(vote_reasoning.get(vote, ""))
        
        recent_accuracy = self.get_recent_accuracy()
        if len(self.accuracy_history) > 5:
            reasoning_parts.append(f"Recent accuracy: {recent_accuracy:.1%}")
        
        return ". ".join(filter(None, reasoning_parts))


def create_rsi_specialist(period: int = 14, oversold: float = 30, overbought: float = 70) -> RSISpecialist:
    """Factory function to create RSI Specialist"""
    return RSISpecialist(period=period, oversold_threshold=oversold, overbought_threshold=overbought)
