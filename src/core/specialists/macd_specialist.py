"""
MACD Specialist
================================================================================
Technical Specialist chuyên về MACD (Moving Average Convergence Divergence)
Thuộc Technical Category trong Multi-Perspective Ensemble System
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

from .base_specialist import BaseSpecialist, SpecialistVote

logger = logging.getLogger(__name__)


class MACDSpecialist(BaseSpecialist):
    """MACD Specialist - Chuyên gia phân tích MACD"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(
            name="MACD_Specialist",
            category="Technical",
            description=f"MACD analysis với periods=({fast_period},{slow_period},{signal_period})"
        )
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.min_data_points = max(30, slow_period + signal_period)
        
        logger.info(f"MACD Specialist initialized: periods=({fast_period},{slow_period},{signal_period})")
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD components"""
        try:
            ema_fast = prices.ewm(span=self.fast_period).mean()
            ema_slow = prices.ewm(span=self.slow_period).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period).mean()
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    def analyze(self, data: pd.DataFrame, current_price: float, **kwargs) -> SpecialistVote:
        """Analyze MACD và generate vote"""
        
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
            macd_line, signal_line, histogram = self.calculate_macd(data['close'])
            
            if macd_line.empty or signal_line.empty:
                return SpecialistVote(
                    specialist_name=self.name,
                    vote="HOLD",
                    confidence=0.0,
                    reasoning="Unable to calculate MACD",
                    timestamp=datetime.now(),
                    technical_data={}
                )
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            analysis_result = {
                'current_macd': current_macd,
                'current_signal': current_signal,
                'current_histogram': current_histogram,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram
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
                'macd_value': current_macd,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"MACD Analysis: MACD={current_macd:.4f}, Signal={current_signal:.4f}, Vote={vote}")
            
            return specialist_vote
            
        except Exception as e:
            self.logger.error(f"Error in MACD analysis: {e}")
            return SpecialistVote(
                specialist_name=self.name,
                vote="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                timestamp=datetime.now(),
                technical_data={}
            )
    
    def generate_vote_decision(self, analysis_result: Dict[str, Any]) -> str:
        """Generate vote decision based on MACD analysis"""
        current_macd = analysis_result['current_macd']
        current_signal = analysis_result['current_signal']
        
        # Check for crossovers
        if len(analysis_result['macd_line']) >= 2:
            prev_macd = analysis_result['macd_line'].iloc[-2]
            prev_signal = analysis_result['signal_line'].iloc[-2]
            
            # Bullish crossover
            if prev_macd <= prev_signal and current_macd > current_signal:
                return "BUY"
            # Bearish crossover
            elif prev_macd >= prev_signal and current_macd < current_signal:
                return "SELL"
        
        # Current position
        if current_macd > current_signal and current_macd > 0:
            return "BUY"
        elif current_macd < current_signal and current_macd < 0:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        current_macd = analysis_result['current_macd']
        current_signal = analysis_result['current_signal']
        
        signal_separation = abs(current_macd - current_signal)
        base_confidence = min(0.8, signal_separation * 1000)  # Scale appropriately
        
        recent_accuracy = self.get_recent_accuracy()
        if recent_accuracy > 0.6:
            base_confidence *= 1.1
        elif recent_accuracy < 0.4:
            base_confidence *= 0.8
        
        return min(1.0, max(0.1, base_confidence))
    
    def generate_reasoning(self, analysis_result: Dict[str, Any], vote: str) -> str:
        """Generate reasoning"""
        current_macd = analysis_result['current_macd']
        current_signal = analysis_result['current_signal']
        
        if current_macd > current_signal:
            position = f"MACD above Signal ({current_macd:.4f} > {current_signal:.4f})"
        else:
            position = f"MACD below Signal ({current_macd:.4f} < {current_signal:.4f})"
        
        vote_reasoning = {
            "BUY": "Suggesting BUY - bullish MACD signals",
            "SELL": "Suggesting SELL - bearish MACD signals",
            "HOLD": "Suggesting HOLD - mixed MACD signals"
        }
        
        return f"{position}. {vote_reasoning.get(vote, '')}"


def create_macd_specialist(fast: int = 12, slow: int = 26, signal: int = 9) -> MACDSpecialist:
    """Factory function to create MACD Specialist"""
    return MACDSpecialist(fast_period=fast, slow_period=slow, signal_period=signal)
