from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VolatilityClusteringSpecialist(BaseSpecialist):
    def __init__(self, vol_period: int = 20, cluster_threshold: float = 1.5):
        super().__init__(
            name="Volatility_Clustering_Specialist",
            category="Volatility", 
            description="Volatility clustering analysis"
        )
        self.vol_period = vol_period
        self.cluster_threshold = cluster_threshold
        logger.info("Volatility Clustering Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.vol_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            prices = data["close"]
            
            # Calculate volatility
            returns = prices.pct_change().dropna()
            current_vol = returns.tail(10).std()
            avg_vol = returns.tail(self.vol_period).std()
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
            else:
                vol_ratio = 1.0
            
            # Generate vote based on volatility clustering
            if vol_ratio > self.cluster_threshold:
                vote = "HOLD"
                confidence = 0.6
                reasoning = f"High volatility clustering detected (ratio: {vol_ratio:.2f}). Suggesting HOLD"
            elif vol_ratio < 1 / self.cluster_threshold:
                vote = "BUY"
                confidence = 0.5
                reasoning = f"Low volatility detected (ratio: {vol_ratio:.2f}). Expecting expansion. Suggesting BUY"
            else:
                vote = "HOLD"
                confidence = 0.3
                reasoning = f"Normal volatility (ratio: {vol_ratio:.2f}). Suggesting HOLD"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {"vol_ratio": vol_ratio})
            
        except Exception as e:
            logger.error(f"Error in volatility clustering analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.5
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Volatility clustering analysis: {vote}"

def create_volatility_clustering_specialist(period=20, threshold=1.5):
    return VolatilityClusteringSpecialist(period, threshold)