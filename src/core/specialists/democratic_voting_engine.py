"""
Democratic Voting Engine
================================================================================
Core engine cho Multi-Perspective Ensemble System
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import logging

from .base_specialist import BaseSpecialist, SpecialistVote
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)




@dataclass
class DemocraticResult:
    """Final democratic voting result"""
    final_vote: str
    final_confidence: float
    consensus_strength: float
    total_specialists: int
    active_specialists: int
    vote_distribution: Dict[str, int]
    reasoning: str
    individual_votes: List[SpecialistVote]
    timestamp: datetime


class DemocraticVotingEngine:
    """Democratic Voting Engine with Performance Tracking"""
    
    def __init__(self, consensus_threshold: float = 0.67):
        self.consensus_threshold = consensus_threshold
        self.specialists = []
        self.performance_tracker = PerformanceTracker()
        self.logger = logging.getLogger(__name__)
        self.voting_history = []
        
        logger.info(f"Democratic Voting Engine initialized with Performance Tracker")
    
    def add_specialist(self, specialist: BaseSpecialist):
        """Add specialist to voting engine"""
        self.specialists.append(specialist)
        logger.info(f"Added specialist: {specialist.name}")
    
    def conduct_vote(self, data, **kwargs) -> DemocraticResult:
        """Conduct democratic vote với all active specialists"""
        
        try:
            # Convert numpy array to DataFrame if needed
            if isinstance(data, np.ndarray):
                # Create DataFrame with standard columns
                df_data = pd.DataFrame([{
                    'close': data[0] if len(data) > 0 else 2000.0,
                    'high': data[1] if len(data) > 1 else 2001.0,
                    'low': data[2] if len(data) > 2 else 1999.0,
                    'volume': data[3] if len(data) > 3 else 1000.0,
                    'rsi': data[4] if len(data) > 4 else 50.0,
                    'macd': data[5] if len(data) > 5 else 0.0,
                    'bb_upper': data[6] if len(data) > 6 else 2020.0,
                    'bb_lower': data[7] if len(data) > 7 else 1980.0,
                    'sma_20': data[8] if len(data) > 8 else 2000.0,
                    'ema_20': data[9] if len(data) > 9 else 2000.0
                }])
                current_price = data[0] if len(data) > 0 else 2000.0
            else:
                df_data = data
                current_price = kwargs.get('current_price', 2000.0)
            
            individual_votes = []
            active_specialists = 0
            
            for specialist in self.specialists:
                if specialist.enabled:
                    try:
                        vote = specialist.analyze(df_data, current_price, **kwargs)
                        individual_votes.append(vote)
                        active_specialists += 1
                    except Exception as e:
                        self.logger.error(f"Error getting vote from {specialist.name}: {e}")
            
            if not individual_votes:
                return self._create_default_result("No active specialists")
            
            # Count votes
            vote_counts = Counter([vote.vote for vote in individual_votes])
            final_vote = vote_counts.most_common(1)[0][0]
            
            # Calculate confidence
            final_confidence = np.mean([v.confidence for v in individual_votes if v.vote == final_vote])
            
            # Calculate consensus strength
            consensus_strength = vote_counts[final_vote] / len(individual_votes)
            
            # Generate reasoning
            reasoning = f"Democratic vote: {final_vote} ({vote_counts[final_vote]}/{len(individual_votes)} specialists agree)"
            
            result = DemocraticResult(
                final_vote=final_vote,
                final_confidence=final_confidence,
                consensus_strength=consensus_strength,
                total_specialists=len(self.specialists),
                active_specialists=active_specialists,
                vote_distribution=dict(vote_counts),
                reasoning=reasoning,
                individual_votes=individual_votes,
                timestamp=datetime.now()
            )
            
            self.voting_history.append(result)
            self.logger.info(f"Democratic Vote: {final_vote} (confidence: {final_confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in democratic voting: {e}")
            return self._create_default_result(f"Voting error: {str(e)}")
    
    def conduct_vote_legacy(self, specialists: List[BaseSpecialist], data: pd.DataFrame, 
                    current_price: float, **kwargs) -> DemocraticResult:
        """Legacy conduct vote method for backwards compatibility"""
        return self.conduct_vote(data, current_price=current_price, **kwargs)
    
    def _create_default_result(self, reason: str) -> DemocraticResult:
        """Create default result when voting fails"""
        return DemocraticResult(
            final_vote="HOLD",
            final_confidence=0.0,
            consensus_strength=0.0,
            total_specialists=0,
            active_specialists=0,
            vote_distribution={"HOLD": 1},
            reasoning=f"Default HOLD decision: {reason}",
            individual_votes=[],
            timestamp=datetime.now()
        )
    
    def get_voting_summary(self) -> Dict[str, Any]:
        """Get summary of recent voting history"""
        if not self.voting_history:
            return {"total_votes": 0, "avg_consensus": 0.0}
        
        recent_votes = self.voting_history[-10:]
        return {
            "total_votes": len(self.voting_history),
            "avg_consensus": np.mean([v.consensus_strength for v in recent_votes]),
            "avg_confidence": np.mean([v.final_confidence for v in recent_votes])
        }


def create_democratic_voting_engine(consensus_threshold: float = 0.67) -> DemocraticVotingEngine:
    """Factory function to create Democratic Voting Engine"""
    return DemocraticVotingEngine(consensus_threshold=consensus_threshold)
