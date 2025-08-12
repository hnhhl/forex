"""
Performance Tracker for Multi-Perspective Ensemble
================================================================================
Theo dõi hiệu suất từng specialist và tối ưu hóa trọng số động
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpecialistPerformance:
    """Performance metrics for individual specialist"""
    name: str
    category: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    current_weight: float = 1.0
    suggested_weight: float = 1.0
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)


class PerformanceTracker:
    """Track and optimize specialist performance"""
    
    def __init__(self):
        self.specialist_performance: Dict[str, SpecialistPerformance] = {}
        self.global_metrics = {
            'total_votes': 0,
            'successful_votes': 0,
            'overall_accuracy': 0.0
        }
        
        logger.info("Performance Tracker initialized")
    
    def register_specialist(self, name: str, category: str):
        """Register a new specialist for tracking"""
        if name not in self.specialist_performance:
            self.specialist_performance[name] = SpecialistPerformance(
                name=name,
                category=category
            )
            
            logger.info(f"Registered specialist: {name} in category {category}")
    
    def calculate_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on performance"""
        weights = {}
        
        for name, perf in self.specialist_performance.items():
            base_weight = max(0.1, perf.accuracy if perf.accuracy > 0 else 0.5)
            weights[name] = min(2.0, base_weight)
            perf.suggested_weight = weights[name]
        
        return weights
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'timestamp': datetime.now(),
            'global_metrics': self.global_metrics,
            'specialist_count': len(self.specialist_performance),
            'dynamic_weights': self.calculate_dynamic_weights()
        }
