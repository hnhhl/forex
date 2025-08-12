from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ATRSpecialist(BaseSpecialist):
    def __init__(self, atr_period: int = 14, volatility_threshold: float = 1.5):
        super().__init__(
            name="ATR_Specialist",
            category="Volatility", 
            description="ATR volatility analysis"
        )
        self.atr_period = atr_period
        self.volatility_threshold = volatility_threshold
        logger.info("ATR Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.atr_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            # Simple volatility analysis
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
            
            # Volatility based voting
            if atr > 0:
                atr_percentage = (atr / current_price) * 100
                
                if atr_percentage > 2.0:  # High volatility
                    vote = "HOLD"
                    confidence = 0.6
                    reasoning = f"High volatility detected (ATR: {atr_percentage:.2f}%). Suggesting HOLD"
                else:
                    vote = "BUY"
                    confidence = 0.5
                    reasoning = f"Normal volatility (ATR: {atr_percentage:.2f}%). Suggesting BUY"
            else:
                vote = "HOLD"
                confidence = 0.3
                reasoning = "Cannot calculate ATR"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {"atr": atr})
            
        except Exception as e:
            logger.error(f"Error in ATR analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.5
    
    def generate_reasoning(self, analysis_result, vote):
        return f"ATR volatility analysis: {vote}"

def create_atr_specialist(period=14, threshold=1.5):
    return ATRSpecialist(period, threshold)