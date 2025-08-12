from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TrendSpecialist(BaseSpecialist):
    def __init__(self, trend_period: int = 20, strength_threshold: float = 0.6):
        super().__init__(
            name="Trend_Specialist",
            category="Momentum", 
            description=f"Trend analysis với period={trend_period}"
        )
        self.trend_period = trend_period
        self.strength_threshold = strength_threshold
        logger.info("Trend Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.trend_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            prices = data["close"]
            
            # Simple trend analysis using moving averages
            ma_short = prices.rolling(window=10).mean().iloc[-1]
            ma_long = prices.rolling(window=20).mean().iloc[-1]
            
            if ma_short > ma_long * 1.01:  # 1% threshold
                vote = "BUY"
                confidence = 0.7
                reasoning = "Uptrend detected: Short MA > Long MA. Suggesting BUY"
            elif ma_short < ma_long * 0.99:  # 1% threshold
                vote = "SELL"
                confidence = 0.7
                reasoning = "Downtrend detected: Short MA < Long MA. Suggesting SELL"
            else:
                vote = "HOLD"
                confidence = 0.4
                reasoning = "Sideways trend detected. Suggesting HOLD"
            
            analysis_result = {
                "current_price": current_price,
                "ma_short": ma_short,
                "ma_long": ma_long
            }
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), analysis_result)
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Trend analysis: {vote}"

def create_trend_specialist(period=20, threshold=0.6):
    return TrendSpecialist(period, threshold)
