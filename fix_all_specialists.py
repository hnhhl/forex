"""
Fix All Specialists - Clean Syntax Errors
================================================================================
Tạo lại tất cả specialists với clean syntax
"""

import os

def create_clean_specialists():
    """Create clean versions of all specialists"""
    
    specialists_dir = "src/core/specialists"
    
    # Clean specialists data
    specialists = {
        "breakout_specialist.py": '''from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BreakoutSpecialist(BaseSpecialist):
    def __init__(self, lookback_period: int = 20, volume_threshold: float = 1.5):
        super().__init__(
            name="Breakout_Specialist",
            category="Momentum", 
            description="Breakout analysis"
        )
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        logger.info("Breakout Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.lookback_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            # Simple breakout detection
            recent_high = data["high"].tail(self.lookback_period).max()
            recent_low = data["low"].tail(self.lookback_period).min()
            
            if current_price > recent_high * 1.01:  # 1% breakout
                vote = "BUY"
                confidence = 0.7
                reasoning = "Resistance breakout detected. Suggesting BUY"
            elif current_price < recent_low * 0.99:  # 1% breakdown
                vote = "SELL"
                confidence = 0.7
                reasoning = "Support breakdown detected. Suggesting SELL"
            else:
                vote = "HOLD"
                confidence = 0.3
                reasoning = "No breakout detected"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {})
            
        except Exception as e:
            logger.error(f"Error in breakout analysis: {e}")
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Breakout analysis: {vote}"

def create_breakout_specialist(period=20, volume_threshold=1.5):
    return BreakoutSpecialist(period, volume_threshold)''',

        "atr_specialist.py": '''from .base_specialist import BaseSpecialist, SpecialistVote
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
    return ATRSpecialist(period, threshold)''',

        "bollinger_specialist.py": '''from .base_specialist import BaseSpecialist, SpecialistVote
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
    return BollingerSpecialist(period, std)''',

        "volatility_clustering_specialist.py": '''from .base_specialist import BaseSpecialist, SpecialistVote
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
    return VolatilityClusteringSpecialist(period, threshold)'''
    }
    
    # Write all specialists
    for filename, content in specialists.items():
        filepath = os.path.join(specialists_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created {filename}")
    
    print(f"\n🎉 Fixed {len(specialists)} specialists!")

if __name__ == "__main__":
    create_clean_specialists() 