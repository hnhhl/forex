from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MeanReversionSpecialist(BaseSpecialist):
    def __init__(self, lookback_period: int = 20, deviation_threshold: float = 2.0):
        super().__init__(
            name="Mean_Reversion_Specialist",
            category="Momentum", 
            description=f"Mean reversion analysis"
        )
        self.lookback_period = lookback_period
        self.deviation_threshold = deviation_threshold
        logger.info("Mean Reversion Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.lookback_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            prices = data["close"]
            
            # Calculate Z-score for mean reversion
            recent_prices = prices.tail(self.lookback_period)
            mean_price = recent_prices.mean()
            std_price = recent_prices.std()
            
            if std_price == 0:
                return SpecialistVote(self.name, "HOLD", 0.0, "No volatility", datetime.now(), {})
            
            z_score = (current_price - mean_price) / std_price
            
            # Generate mean reversion signals
            if z_score > self.deviation_threshold:
                vote = "SELL"  # Price too high, expect reversion
                confidence = min(0.8, abs(z_score) / 3)
                reasoning = f"Mean reversion SELL: Z-score={z_score:.2f} (overbought)"
            elif z_score < -self.deviation_threshold:
                vote = "BUY"  # Price too low, expect reversion
                confidence = min(0.8, abs(z_score) / 3)
                reasoning = f"Mean reversion BUY: Z-score={z_score:.2f} (oversold)"
            else:
                vote = "HOLD"
                confidence = 0.3
                reasoning = f"Price near mean: Z-score={z_score:.2f}"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {"z_score": z_score})
            
        except Exception as e:
            logger.error(f"Error in mean reversion analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.5
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Mean reversion analysis: {vote}"

def create_mean_reversion_specialist(period=20, threshold=2.0):
    return MeanReversionSpecialist(period, threshold)
