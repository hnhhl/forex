from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BollingerSpecialist(BaseSpecialist):
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        super().__init__(
            name="Bollinger_Specialist",
            category="Volatility", 
            description="Bollinger Bands analysis"
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        logger.info("Bollinger Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.bb_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            prices = data["close"]
            
            # Calculate Bollinger Bands
            sma = prices.rolling(window=self.bb_period).mean().iloc[-1]
            std = prices.rolling(window=self.bb_period).std().iloc[-1]
            
            upper_band = sma + (std * self.bb_std)
            lower_band = sma - (std * self.bb_std)
            
            # Generate vote based on Bollinger Band position
            if current_price > upper_band:
                vote = "SELL"
                confidence = 0.7
                reasoning = "Price above upper Bollinger Band. Overbought. Suggesting SELL"
            elif current_price < lower_band:
                vote = "BUY"
                confidence = 0.7
                reasoning = "Price below lower Bollinger Band. Oversold. Suggesting BUY"
            else:
                vote = "HOLD"
                confidence = 0.4
                reasoning = "Price within Bollinger Bands. Suggesting HOLD"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {
                "sma": sma, "upper_band": upper_band, "lower_band": lower_band
            })
            
        except Exception as e:
            logger.error(f"Error in Bollinger analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Bollinger Bands analysis: {vote}"

def create_bollinger_specialist(period=20, std=2.0):
    return BollingerSpecialist(period, std)