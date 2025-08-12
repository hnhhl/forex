from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PositionSizeSpecialist(BaseSpecialist):
    def __init__(self, risk_per_trade: float = 0.02, max_position_size: float = 0.1):
        super().__init__(
            name="Position_Size_Specialist",
            category="Risk", 
            description="Position sizing analysis"
        )
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        logger.info("Position Size Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < 20:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            # Simple position sizing based on volatility
            prices = data["close"]
            volatility = prices.pct_change().std()
            
            if volatility > 0.03:  # High volatility
                vote = "HOLD"
                confidence = 0.6
                reasoning = "High volatility. Reduce position size. Suggesting HOLD"
            elif volatility > 0.01:  # Medium volatility
                vote = "BUY"
                confidence = 0.5
                reasoning = "Medium volatility. Normal position size. Suggesting BUY"
            else:  # Low volatility
                vote = "BUY"
                confidence = 0.7
                reasoning = "Low volatility. Can increase position size. Suggesting BUY"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {})
            
        except Exception as e:
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.5
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Position sizing analysis: {vote}"

def create_position_size_specialist(risk_per_trade=0.02, max_size=0.1):
    return PositionSizeSpecialist(risk_per_trade, max_size)