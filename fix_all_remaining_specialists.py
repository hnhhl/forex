"""
Fix All Remaining Specialists
================================================================================
"""

import os

def fix_all_specialists():
    specialists_dir = "src/core/specialists"
    
    # All specialists with clean syntax
    specialists = {
        "drawdown_specialist.py": '''from .base_specialist import BaseSpecialist, SpecialistVote
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DrawdownSpecialist(BaseSpecialist):
    def __init__(self, max_drawdown_threshold: float = 0.05, lookback_period: int = 30):
        super().__init__(
            name="Drawdown_Specialist",
            category="Risk", 
            description="Drawdown analysis"
        )
        self.max_drawdown_threshold = max_drawdown_threshold
        self.lookback_period = lookback_period
        logger.info("Drawdown Specialist initialized")
    
    def analyze(self, data, current_price, **kwargs):
        if not self.enabled:
            return SpecialistVote(self.name, "HOLD", 0.0, "Disabled", datetime.now(), {})
        
        if len(data) < self.lookback_period:
            return SpecialistVote(self.name, "HOLD", 0.0, "Insufficient data", datetime.now(), {})
        
        try:
            prices = data["close"].tail(self.lookback_period)
            peak = prices.expanding(min_periods=1).max()
            drawdown = (prices - peak) / peak
            current_drawdown = drawdown.iloc[-1]
            
            if abs(current_drawdown) > self.max_drawdown_threshold:
                vote = "SELL"
                confidence = 0.7
                reasoning = f"High drawdown risk. Suggesting SELL"
            else:
                vote = "BUY"
                confidence = 0.5
                reasoning = f"Low drawdown risk. Safe to BUY"
            
            return SpecialistVote(self.name, vote, confidence, reasoning, datetime.now(), {})
            
        except Exception as e:
            return SpecialistVote(self.name, "HOLD", 0.0, "Analysis error", datetime.now(), {})
    
    def calculate_confidence(self, analysis_result):
        return 0.6
    
    def generate_reasoning(self, analysis_result, vote):
        return f"Drawdown analysis: {vote}"

def create_drawdown_specialist(max_threshold=0.05, lookback=30):
    return DrawdownSpecialist(max_threshold, lookback)''',

        "position_size_specialist.py": '''from .base_specialist import BaseSpecialist, SpecialistVote
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
    return PositionSizeSpecialist(risk_per_trade, max_size)'''
    }
    
    # Write all specialists
    for filename, content in specialists.items():
        filepath = os.path.join(specialists_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {filename}")
    
    print(f"\n🎉 Fixed {len(specialists)} remaining specialists!")

if __name__ == "__main__":
    fix_all_specialists() 