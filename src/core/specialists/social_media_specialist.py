from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SocialMediaSpecialist(BaseSpecialist):
    def __init__(self, sentiment_threshold: float = 0.5):
        super().__init__(
            name="Social_Media_Specialist",
            category="Sentiment", 
            description=f"Social media sentiment analysis với threshold={sentiment_threshold}"
        )
        self.sentiment_threshold = sentiment_threshold
        logger.info("Social Media Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        # Simulate social media sentiment
        social_sentiment = np.random.normal(0, 0.4)
        
        if social_sentiment > self.sentiment_threshold:
            vote = "BUY"
        elif social_sentiment < -self.sentiment_threshold:
            vote = "SELL"
        else:
            vote = "HOLD"
        
        confidence = 0.5 + abs(social_sentiment) * 0.4
        reasoning = f"Social media sentiment: {social_sentiment:.3f}. Suggesting {vote}"
        
        return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {"social_sentiment": social_sentiment})
    
    def calculate_confidence(self, analysis_result):
        return 0.5
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Social media sentiment analysis: {vote}"

def create_social_media_specialist(threshold=0.5):
    return SocialMediaSpecialist(threshold)
