from .base_specialist import BaseSpecialist, SpecialistVote
from .rsi_specialist import RSISpecialist, create_rsi_specialist
from .macd_specialist import MACDSpecialist, create_macd_specialist
from .fibonacci_specialist import FibonacciSpecialist, create_fibonacci_specialist
from .chart_pattern_specialist import ChartPatternSpecialist, create_chart_pattern_specialist
from .var_risk_specialist import VaRRiskSpecialist, create_var_risk_specialist
from .democratic_voting_engine import DemocraticVotingEngine, create_democratic_voting_engine

# Week 2 - All 13 New Specialists (FIXED)
from .news_sentiment_specialist import NewsSentimentSpecialist, create_news_sentiment_specialist
from .social_media_specialist import SocialMediaSpecialist, create_social_media_specialist
from .fear_greed_specialist import FearGreedSpecialist, create_fear_greed_specialist
from .candlestick_specialist import CandlestickSpecialist, create_candlestick_specialist
from .wave_specialist import WaveSpecialist, create_wave_specialist
from .drawdown_specialist import DrawdownSpecialist, create_drawdown_specialist
from .position_size_specialist import PositionSizeSpecialist, create_position_size_specialist
from .trend_specialist import TrendSpecialist, create_trend_specialist
from .mean_reversion_specialist import MeanReversionSpecialist, create_mean_reversion_specialist
from .breakout_specialist import BreakoutSpecialist, create_breakout_specialist
from .atr_specialist import ATRSpecialist, create_atr_specialist
from .bollinger_specialist import BollingerSpecialist, create_bollinger_specialist
from .volatility_clustering_specialist import VolatilityClusteringSpecialist, create_volatility_clustering_specialist

__all__ = [
    "BaseSpecialist",
    "SpecialistVote",
    "RSISpecialist",
    "MACDSpecialist", 
    "FibonacciSpecialist",
    "ChartPatternSpecialist",
    "VaRRiskSpecialist",
    "DemocraticVotingEngine",
    "create_rsi_specialist",
    "create_macd_specialist",
    "create_fibonacci_specialist",
    "create_chart_pattern_specialist",
    "create_var_risk_specialist",
    "create_democratic_voting_engine",
    # Week 2 - All 13 Specialists
    "NewsSentimentSpecialist",
    "SocialMediaSpecialist", 
    "FearGreedSpecialist",
    "CandlestickSpecialist",
    "WaveSpecialist",
    "DrawdownSpecialist",
    "PositionSizeSpecialist",
    "TrendSpecialist",
    "MeanReversionSpecialist",
    "BreakoutSpecialist",
    "ATRSpecialist",
    "BollingerSpecialist",
    "VolatilityClusteringSpecialist",
    "create_news_sentiment_specialist",
    "create_social_media_specialist",
    "create_fear_greed_specialist",
    "create_candlestick_specialist",
    "create_wave_specialist",
    "create_drawdown_specialist",
    "create_position_size_specialist",
    "create_trend_specialist",
    "create_mean_reversion_specialist",
    "create_breakout_specialist",
    "create_atr_specialist",
    "create_bollinger_specialist",
    "create_volatility_clustering_specialist"
]
