"""
AI Phases - Hệ thống nâng cao hiệu suất với 6 phases

Package này chứa 6 phases phát triển để nâng cao hiệu suất hệ thống:
- Phase 1: Online Learning Engine (+2.5%)
- Phase 2: Advanced Backtest Framework (+1.5%)
- Phase 3: Adaptive Intelligence (+3.0%)
- Phase 4: Multi-Market Learning (+2.0%)
- Phase 5: Real-Time Enhancement (+1.5%)
- Phase 6: Future Evolution (+1.5%)

Tổng performance boost: +12.0%
"""

from .phase1_online_learning import Phase1OnlineLearningEngine
from .phase2_backtest_framework import Phase2BacktestFramework
from .phase3_adaptive_intelligence import Phase3AdaptiveIntelligence
from .phase4_multi_market_learning import Phase4MultiMarketLearning
from .phase5_realtime_enhancement import Phase5RealTimeEnhancement
from .phase6_future_evolution import Phase6FutureEvolution
from .utils.progress_tracker import PhaseProgressTracker

__version__ = '1.0.0'

# Expose main classes
__all__ = [
    'Phase1OnlineLearningEngine',
    'Phase2BacktestFramework',
    'Phase3AdaptiveIntelligence',
    'Phase4MultiMarketLearning',
    'Phase5RealTimeEnhancement',
    'Phase6FutureEvolution',
    'PhaseProgressTracker'
] 