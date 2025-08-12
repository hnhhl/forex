from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NewsSentimentSpecialist(BaseSpecialist):
    def __init__(self, sentiment_threshold: float = 0.6):
        super().__init__(
            name="News_Sentiment_Specialist",
            category="Sentiment", 
            description=f"News sentiment analysis với threshold={sentiment_threshold}"
        )
        self.sentiment_threshold = sentiment_threshold
        logger.info("News Sentiment Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        # Simulate sentiment
        sentiment = np.random.normal(0, 0.3)
        
        if sentiment > self.sentiment_threshold:
            vote = "BUY"
        elif sentiment < -self.sentiment_threshold:
            vote = "SELL"
        else:
            vote = "HOLD"
        
        confidence = 0.6 + abs(sentiment) * 0.3
        reasoning = f"News sentiment: {sentiment:.3f}. Suggesting {vote}"
        
        return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {"sentiment": sentiment})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"News sentiment analysis: {vote}"

def create_news_sentiment_specialist(threshold=0.6):
    return NewsSentimentSpecialist(threshold)
