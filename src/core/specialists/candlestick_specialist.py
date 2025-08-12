from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CandlestickSpecialist(BaseSpecialist):
    def __init__(self, min_body_ratio: float = 0.6):
        super().__init__(
            name="Candlestick_Specialist",
            category="Pattern", 
            description=f"Candlestick pattern analysis"
        )
        self.min_body_ratio = min_body_ratio
        logger.info("Candlestick Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < 3:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            # Simple candlestick analysis
            last_candle = data.iloc[-1]
            
            # Check if bullish or bearish candle
            if last_candle["close"] > last_candle["open"]:
                vote = "BUY"
                confidence = 0.6
                reasoning = "Bullish candlestick detected. Suggesting BUY"
            elif last_candle["close"] < last_candle["open"]:
                vote = "SELL"
                confidence = 0.6
                reasoning = "Bearish candlestick detected. Suggesting SELL"
            else:
                vote = "HOLD"
                confidence = 0.3
                reasoning = "Doji candlestick detected. Suggesting HOLD"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {})
            
        except Exception as e:
            logger.error(f"Error in candlestick analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Candlestick analysis: {vote}"

def create_candlestick_specialist(min_body_ratio=0.6):
    return CandlestickSpecialist(min_body_ratio)
