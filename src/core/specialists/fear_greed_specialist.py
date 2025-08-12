from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FearGreedSpecialist(BaseSpecialist):
    def __init__(self, fear_threshold: float = 20, greed_threshold: float = 80):
        super().__init__(
            name="Fear_Greed_Specialist",
            category="Sentiment", 
            description=f"Fear/Greed index analysis với thresholds={fear_threshold}/{greed_threshold}"
        )
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        logger.info("Fear Greed Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        # Simulate Fear/Greed index (0-100)
        fear_greed_index = np.random.uniform(0, 100)
        
        if fear_greed_index < self.fear_threshold:
            vote = "BUY"  # Contrarian: Buy when fearful
            reasoning = f"Extreme fear detected (index={fear_greed_index:.1f}). Contrarian BUY signal"
        elif fear_greed_index > self.greed_threshold:
            vote = "SELL"  # Contrarian: Sell when greedy
            reasoning = f"Extreme greed detected (index={fear_greed_index:.1f}). Contrarian SELL signal"
        else:
            vote = "HOLD"
            reasoning = f"Neutral sentiment (index={fear_greed_index:.1f}). HOLD position"
        
        # Confidence higher for extreme values
        if fear_greed_index < 25 or fear_greed_index > 75:
            confidence = 0.7
        else:
            confidence = 0.4
        
        return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {"fear_greed_index": fear_greed_index})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Fear/Greed analysis: {vote}"

def create_fear_greed_specialist(fear_threshold=20, greed_threshold=80):
    return FearGreedSpecialist(fear_threshold, greed_threshold)
