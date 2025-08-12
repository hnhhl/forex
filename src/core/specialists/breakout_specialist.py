from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BreakoutSpecialist(BaseSpecialist):
    def __init__(self, lookback_period: int = 20, volume_threshold: float = 1.5):
        super().__init__(
            name="Breakout_Specialist",
            category="Momentum", 
            description="Breakout analysis"
        )
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        logger.info("Breakout Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.lookback_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            # Simple breakout detection
            recent_high = data["high"].tail(self.lookback_period).max()
            recent_low = data["low"].tail(self.lookback_period).min()
            
            if current_price > recent_high * 1.01:  # 1% breakout
                vote = "BUY"
                confidence = 0.7
                reasoning = "Resistance breakout detected. Suggesting BUY"
            elif current_price < recent_low * 0.99:  # 1% breakdown
                vote = "SELL"
                confidence = 0.7
                reasoning = "Support breakdown detected. Suggesting SELL"
            else:
                vote = "HOLD"
                confidence = 0.3
                reasoning = "No breakout detected"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {})
            
        except Exception as e:
            logger.error(f"Error in breakout analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Breakout analysis: {vote}"

def create_breakout_specialist(period=20, volume_threshold=1.5):
    return BreakoutSpecialist(period, volume_threshold)