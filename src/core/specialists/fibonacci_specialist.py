"""
Fibonacci Specialist
================================================================================
Technical Specialist chuyên về Fibonacci Retracements và Extensions
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging

from .base_specialist import BaseSpecialist, SpecialistVote

logger = logging.getLogger(__name__)


class FibonacciSpecialist(BaseSpecialist):
    """Fibonacci Specialist - Chuyên gia phân tích Fibonacci"""
    
    def __init__(self, lookback_period: int = 50, min_swing_size: float = 0.01):
        super().__init__(
            name="Fibonacci_Specialist",
            category="Technical",
            description=f"Fibonacci retracement analysis với lookback={lookback_period}"
        )
        
        self.lookback_period = lookback_period
        self.min_swing_size = min_swing_size
        self.fibonacci_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.min_data_points = max(30, lookback_period)
        
        logger.info(f"Fibonacci Specialist initialized")
    
    def find_swing_points(self, data: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        """Find swing highs and lows"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            swing_highs = []
            swing_lows = []
            
            # Simple swing detection
            for i in range(5, len(highs) - 5):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    swing_highs.append((i, highs[i], data.index[i]))
                
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    swing_lows.append((i, lows[i], data.index[i]))
            
            return swing_highs, swing_lows
            
        except Exception as e:
            self.logger.error(f"Error finding swing points: {e}")
            return [], []
    
    def analyze(self, data: pd.DataFrame, current_price: float, **kwargs) -> SpecialistVote:
        """Analyze Fibonacci levels và generate vote"""
        
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
            # Find swing points
            swing_highs, swing_lows = self.find_swing_points(data)
            
            # Simple Fibonacci analysis
            signal_strength = 0.0
            vote = "HOLD"
            
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                recent_high = max(swing_highs, key=lambda x: x[0])
                recent_low = max(swing_lows, key=lambda x: x[0])
                
                # Calculate simple Fibonacci levels
                high_price = recent_high[1]
                low_price = recent_low[1]
                price_range = high_price - low_price
                
                # 38.2% and 61.8% retracement levels
                fib_382 = high_price - (price_range * 0.382)
                fib_618 = high_price - (price_range * 0.618)
                
                # Check if price is near Fibonacci levels
                distance_382 = abs(current_price - fib_382) / current_price
                distance_618 = abs(current_price - fib_618) / current_price
                
                if distance_382 < 0.005:  # Within 0.5%
                    signal_strength = 0.6
                    vote = "BUY" if current_price > fib_382 else "SELL"
                elif distance_618 < 0.005:
                    signal_strength = 0.7
                    vote = "BUY" if current_price > fib_618 else "SELL"
            
            analysis_result = {
                'current_price': current_price,
                'signal_strength': signal_strength,
                'swing_highs_count': len(swing_highs),
                'swing_lows_count': len(swing_lows)
            }
            
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
                'signal_strength': signal_strength,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Fibonacci Analysis: Vote={vote}, Confidence={confidence:.2f}")
            
            return specialist_vote
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci analysis: {e}")
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                timestamp=datetime.now(),
                technical_data={}
            )
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        signal_strength = analysis_result.get('signal_strength', 0.0)
        base_confidence = abs(signal_strength)
        
        recent_accuracy = self.get_recent_accuracy()
        if recent_accuracy > 0.6:
            base_confidence *= 1.1
        elif recent_accuracy < 0.4:
            base_confidence *= 0.8
        
        return min(1.0, max(0.1, base_confidence))
    
    def generate_reasoning(self, analysis_result: Dict[str, Any], vote: str) -> str:
        """Generate reasoning"""
        signal_strength = analysis_result.get('signal_strength', 0.0)
        swing_highs = analysis_result.get('swing_highs_count', 0)
        swing_lows = analysis_result.get('swing_lows_count', 0)
        
        reasoning = f"Fibonacci analysis: {swing_highs} swing highs, {swing_lows} swing lows detected"
        
        if signal_strength > 0:
            reasoning += f". Signal strength: {signal_strength:.2f}"
        
        vote_reasoning = {
            "BUY": "Suggesting BUY - Fibonacci support levels",
            "SELL": "Suggesting SELL - Fibonacci resistance levels",
            "HOLD": "Suggesting HOLD - no clear Fibonacci signals"
        }
        
        return f"{reasoning}. {vote_reasoning.get(vote, '')}"


def create_fibonacci_specialist(lookback: int = 50, min_swing: float = 0.01) -> FibonacciSpecialist:
    """Factory function to create Fibonacci Specialist"""
    return FibonacciSpecialist(lookback_period=lookback, min_swing_size=min_swing)
