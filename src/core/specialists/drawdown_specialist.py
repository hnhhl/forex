from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DrawdownSpecialist(BaseSpecialist):
    def __init__(self, max_drawdown_threshold: float = 0.05, lookback_period: int = 30):
        super().__init__(
            name="Drawdown_Specialist",
            category="Risk", 
            description="Drawdown analysis"
        )
        self.max_drawdown_threshold = max_drawdown_threshold
        self.lookback_period = lookback_period
        logger.info("Drawdown Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.lookback_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            prices = data["close"].tail(self.lookback_period)
            peak = prices.expanding(min_periods=1).max()
            drawdown = (prices - peak) / peak
            current_drawdown = drawdown.iloc[-1]
            
            if abs(current_drawdown) > self.max_drawdown_threshold:
                vote = "SELL"
                confidence = 0.7
                reasoning = f"High drawdown risk. Suggesting SELL"
            else:
                vote = "BUY"
                confidence = 0.5
                reasoning = f"Low drawdown risk. Safe to BUY"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {})
            
        except Exception as e:
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Drawdown analysis: {vote}"

def create_drawdown_specialist(max_threshold=0.05, lookback=30):
    return DrawdownSpecialist(max_threshold, lookback)