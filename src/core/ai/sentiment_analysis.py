"""
Sentiment Analysis Engine
Ultimate XAU Super System V4.0 - Phase 2 Component 3

This module provides comprehensive sentiment analysis capabilities for XAU trading:
- News sentiment analysis from multiple sources
- Social media sentiment tracking (Twitter, Reddit, etc.)
- Market sentiment indicators (Fear & Greed, VIX correlation)
- Real-time sentiment processing and aggregation
- Sentiment-based trading signals generation
"""

import numpy as np
import pandas as pd
import logging
import re
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available - using basic sentiment analysis")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER Sentiment not available - using alternative methods")

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FINANCIAL_BLOGS = "financial_blogs"
    MARKET_DATA = "market_data"
    FEAR_GREED_INDEX = "fear_greed_index"
    VIX = "vix"
    ANALYST_REPORTS = "analyst_reports"


class SentimentPolarity(Enum):
    """Sentiment polarity classification"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class SentimentMethod(Enum):
    """Sentiment analysis methods"""
    TEXTBLOB = "textblob"
    VADER = "vader"
    LEXICON_BASED = "lexicon_based"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"


@dataclass
class SentimentData:
    """Individual sentiment data point"""
    timestamp: datetime
    source: SentimentSource
    text: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float      # 0.0 to 1.0
    polarity: SentimentPolarity
    
    # Metadata
    author: Optional[str] = None
    url: Optional[str] = None
    engagement: Optional[int] = None  # likes, shares, etc.
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'text': self.text,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'polarity': self.polarity.value,
            'author': self.author,
            'url': self.url,
            'engagement': self.engagement,
            'keywords': self.keywords
        }


@dataclass
class SentimentSignal:
    """Aggregated sentiment signal for trading"""
    timestamp: datetime
    overall_sentiment: float  # -1.0 to 1.0
    confidence: float        # 0.0 to 1.0
    
    # Source breakdown
    news_sentiment: float
    social_sentiment: float
    market_sentiment: float
    
    # Signal strength
    signal_strength: float   # 0.0 to 1.0
    recommendation: str      # BUY, SELL, HOLD
    
    # Supporting data
    data_points: int
    sources_count: int
    keywords_trending: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_sentiment': self.overall_sentiment,
            'confidence': self.confidence,
            'news_sentiment': self.news_sentiment,
            'social_sentiment': self.social_sentiment,
            'market_sentiment': self.market_sentiment,
            'signal_strength': self.signal_strength,
            'recommendation': self.recommendation,
            'data_points': self.data_points,
            'sources_count': self.sources_count,
            'keywords_trending': self.keywords_trending
        }


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    # Analysis settings
    method: SentimentMethod = SentimentMethod.ENSEMBLE
    update_frequency: float = 300.0  # 5 minutes
    lookback_hours: int = 24
    
    # Data sources
    enable_news: bool = True
    enable_social: bool = True
    enable_market_data: bool = True
    
    # Signal generation
    sentiment_threshold: float = 0.3  # Minimum sentiment for signal
    confidence_threshold: float = 0.6  # Minimum confidence for signal
    min_data_points: int = 10         # Minimum data points for signal
    
    # Keywords for XAU/Gold
    gold_keywords: List[str] = field(default_factory=lambda: [
        'gold', 'xau', 'precious metals', 'bullion', 'gold price',
        'gold market', 'gold trading', 'gold investment', 'gold futures',
        'gold etf', 'inflation hedge', 'safe haven', 'monetary policy'
    ])
    
    # Weights for different sources
    news_weight: float = 0.4
    social_weight: float = 0.3
    market_weight: float = 0.3


class SentimentAnalyzer:
    """Core sentiment analysis engine"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        
        # Initialize analyzers
        self.textblob_analyzer = None
        self.vader_analyzer = None
        self._initialize_analyzers()
        
        # Gold-specific lexicon
        self.gold_lexicon = self._build_gold_lexicon()
        
        logger.info(f"SentimentAnalyzer initialized with method: {config.method.value}")
    
    def _initialize_analyzers(self):
        """Initialize sentiment analysis tools"""
        if TEXTBLOB_AVAILABLE:
            self.textblob_analyzer = TextBlob
            logger.info("TextBlob analyzer initialized")
        
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER analyzer initialized")
    
    def _build_gold_lexicon(self) -> Dict[str, float]:
        """Build gold-specific sentiment lexicon"""
        return {
            # Positive gold sentiment
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'soar': 0.9,
            'breakout': 0.7, 'uptrend': 0.6, 'strong': 0.6, 'robust': 0.6,
            'safe haven': 0.8, 'hedge': 0.5, 'inflation protection': 0.7,
            'buy': 0.8, 'accumulate': 0.6, 'oversold': 0.5,
            
            # Negative gold sentiment
            'bearish': -0.8, 'decline': -0.6, 'fall': -0.6, 'drop': -0.7,
            'crash': -0.9, 'plunge': -0.8, 'weakness': -0.6, 'sell-off': -0.8,
            'overbought': -0.5, 'resistance': -0.4, 'profit-taking': -0.3,
            'sell': -0.8, 'dump': -0.7, 'correction': -0.5,
            
            # Market context
            'fed': 0.0, 'interest rates': 0.0, 'inflation': 0.3,
            'dollar': -0.2, 'usd': -0.2, 'treasury': -0.1,
            'recession': 0.6, 'uncertainty': 0.4, 'volatility': 0.2
        }
    
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text
        Returns: (sentiment_score, confidence)
        """
        if not text or not text.strip():
            return 0.0, 0.0
        
        text = text.lower().strip()
        
        if self.config.method == SentimentMethod.TEXTBLOB and self.textblob_analyzer:
            return self._analyze_textblob(text)
        elif self.config.method == SentimentMethod.VADER and self.vader_analyzer:
            return self._analyze_vader(text)
        elif self.config.method == SentimentMethod.LEXICON_BASED:
            return self._analyze_lexicon(text)
        elif self.config.method == SentimentMethod.ENSEMBLE:
            return self._analyze_ensemble(text)
        else:
            # Fallback to lexicon-based
            return self._analyze_lexicon(text)
    
    def _analyze_textblob(self, text: str) -> Tuple[float, float]:
        """Analyze using TextBlob"""
        try:
            blob = self.textblob_analyzer(text)
            sentiment = blob.sentiment.polarity  # -1 to 1
            confidence = abs(blob.sentiment.subjectivity)  # 0 to 1
            return float(sentiment), float(confidence)
        except Exception as e:
            logger.error(f"TextBlob analysis error: {e}")
            return 0.0, 0.0
    
    def _analyze_vader(self, text: str) -> Tuple[float, float]:
        """Analyze using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            sentiment = scores['compound']  # -1 to 1
            confidence = max(scores['pos'], scores['neg'], scores['neu'])
            return float(sentiment), float(confidence)
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return 0.0, 0.0
    
    def _analyze_lexicon(self, text: str) -> Tuple[float, float]:
        """Analyze using gold-specific lexicon"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        sentiment_scores = []
        matched_words = 0
        
        for word in words:
            if word in self.gold_lexicon:
                sentiment_scores.append(self.gold_lexicon[word])
                matched_words += 1
        
        if not sentiment_scores:
            return 0.0, 0.0
        
        sentiment = np.mean(sentiment_scores)
        confidence = min(matched_words / len(words), 1.0)
        
        return float(sentiment), float(confidence)
    
    def _analyze_ensemble(self, text: str) -> Tuple[float, float]:
        """Analyze using ensemble of methods"""
        results = []
        
        # TextBlob
        if self.textblob_analyzer:
            sentiment, confidence = self._analyze_textblob(text)
            if confidence > 0:
                results.append((sentiment, confidence, 0.3))
        
        # VADER
        if self.vader_analyzer:
            sentiment, confidence = self._analyze_vader(text)
            if confidence > 0:
                results.append((sentiment, confidence, 0.4))
        
        # Lexicon
        sentiment, confidence = self._analyze_lexicon(text)
        if confidence > 0:
            results.append((sentiment, confidence, 0.3))
        
        if not results:
            return 0.0, 0.0
        
        # Weighted average
        total_weight = sum(weight for _, _, weight in results)
        weighted_sentiment = sum(sent * conf * weight for sent, conf, weight in results) / total_weight
        avg_confidence = sum(conf * weight for _, conf, weight in results) / total_weight
        
        return float(weighted_sentiment), float(avg_confidence)
    
    def classify_polarity(self, sentiment_score: float) -> SentimentPolarity:
        """Classify sentiment score into polarity"""
        if sentiment_score <= -0.6:
            return SentimentPolarity.VERY_NEGATIVE
        elif sentiment_score <= -0.2:
            return SentimentPolarity.NEGATIVE
        elif sentiment_score >= 0.6:
            return SentimentPolarity.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            return SentimentPolarity.POSITIVE
        else:
            return SentimentPolarity.NEUTRAL
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.config.gold_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords


class SentimentDataCollector:
    """Collects sentiment data from various sources"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.analyzer = SentimentAnalyzer(config)
        
        # Data storage
        self.sentiment_data = deque(maxlen=10000)  # Store last 10k data points
        self._lock = threading.Lock()
        
        logger.info("SentimentDataCollector initialized")
    
    def collect_news_sentiment(self) -> List[SentimentData]:
        """Collect sentiment from news sources"""
        # Mock news data for demonstration
        mock_news = [
            "Gold prices surge as inflation fears mount amid economic uncertainty",
            "Federal Reserve signals potential rate cuts, boosting gold demand",
            "Analysts predict gold rally as safe haven demand increases",
            "Gold ETF inflows reach highest level in months",
            "Central banks continue gold accumulation amid dollar weakness"
        ]
        
        sentiment_data = []
        
        for i, news_text in enumerate(mock_news):
            sentiment_score, confidence = self.analyzer.analyze_text(news_text)
            polarity = self.analyzer.classify_polarity(sentiment_score)
            keywords = self.analyzer.extract_keywords(news_text)
            
            data = SentimentData(
                timestamp=datetime.now() - timedelta(minutes=i*30),
                source=SentimentSource.NEWS,
                text=news_text,
                sentiment_score=sentiment_score,
                confidence=confidence,
                polarity=polarity,
                keywords=keywords,
                engagement=np.random.randint(100, 1000)
            )
            
            sentiment_data.append(data)
        
        return sentiment_data
    
    def collect_social_sentiment(self) -> List[SentimentData]:
        """Collect sentiment from social media"""
        # Mock social media data
        mock_social = [
            "Just bought more #gold, this rally is just getting started! 🚀",
            "Gold looking weak here, might be time to take profits",
            "#XAU breaking resistance, bullish setup confirmed",
            "Fed pivot incoming, gold to the moon! 📈",
            "Dollar strength killing gold momentum, bearish outlook"
        ]
        
        sentiment_data = []
        
        for i, social_text in enumerate(mock_social):
            sentiment_score, confidence = self.analyzer.analyze_text(social_text)
            polarity = self.analyzer.classify_polarity(sentiment_score)
            keywords = self.analyzer.extract_keywords(social_text)
            
            # Simulate different sources
            source = SentimentSource.TWITTER if i % 2 == 0 else SentimentSource.REDDIT
            
            data = SentimentData(
                timestamp=datetime.now() - timedelta(minutes=i*15),
                source=source,
                text=social_text,
                sentiment_score=sentiment_score,
                confidence=confidence,
                polarity=polarity,
                keywords=keywords,
                engagement=np.random.randint(10, 500),
                author=f"user_{i+1}"
            )
            
            sentiment_data.append(data)
        
        return sentiment_data
    
    def collect_market_sentiment(self) -> List[SentimentData]:
        """Collect sentiment from market indicators"""
        # Mock market sentiment indicators
        fear_greed_score = np.random.uniform(20, 80)  # Fear & Greed Index
        vix_level = np.random.uniform(15, 35)         # VIX level
        
        sentiment_data = []
        
        # Fear & Greed Index
        if fear_greed_score < 25:
            fg_sentiment = 0.6  # Extreme fear = good for gold
            fg_text = f"Fear & Greed Index at {fear_greed_score:.1f} - Extreme Fear"
        elif fear_greed_score < 45:
            fg_sentiment = 0.3
            fg_text = f"Fear & Greed Index at {fear_greed_score:.1f} - Fear"
        elif fear_greed_score < 55:
            fg_sentiment = 0.0
            fg_text = f"Fear & Greed Index at {fear_greed_score:.1f} - Neutral"
        elif fear_greed_score < 75:
            fg_sentiment = -0.3
            fg_text = f"Fear & Greed Index at {fear_greed_score:.1f} - Greed"
        else:
            fg_sentiment = -0.6  # Extreme greed = bad for gold
            fg_text = f"Fear & Greed Index at {fear_greed_score:.1f} - Extreme Greed"
        
        fg_data = SentimentData(
            timestamp=datetime.now(),
            source=SentimentSource.FEAR_GREED_INDEX,
            text=fg_text,
            sentiment_score=fg_sentiment,
            confidence=0.8,
            polarity=self.analyzer.classify_polarity(fg_sentiment),
            keywords=['fear', 'greed', 'market sentiment']
        )
        sentiment_data.append(fg_data)
        
        # VIX Level
        if vix_level > 30:
            vix_sentiment = 0.7  # High VIX = good for gold
            vix_text = f"VIX at {vix_level:.1f} - High volatility, risk-off sentiment"
        elif vix_level > 20:
            vix_sentiment = 0.3
            vix_text = f"VIX at {vix_level:.1f} - Moderate volatility"
        else:
            vix_sentiment = -0.2  # Low VIX = risk-on, bad for gold
            vix_text = f"VIX at {vix_level:.1f} - Low volatility, risk-on sentiment"
        
        vix_data = SentimentData(
            timestamp=datetime.now(),
            source=SentimentSource.VIX,
            text=vix_text,
            sentiment_score=vix_sentiment,
            confidence=0.9,
            polarity=self.analyzer.classify_polarity(vix_sentiment),
            keywords=['vix', 'volatility', 'risk sentiment']
        )
        sentiment_data.append(vix_data)
        
        return sentiment_data
    
    def collect_all_sentiment_data(self) -> List[SentimentData]:
        """Collect sentiment data from all enabled sources"""
        all_data = []
        
        try:
            if self.config.enable_news:
                news_data = self.collect_news_sentiment()
                all_data.extend(news_data)
                logger.info(f"Collected {len(news_data)} news sentiment data points")
            
            if self.config.enable_social:
                social_data = self.collect_social_sentiment()
                all_data.extend(social_data)
                logger.info(f"Collected {len(social_data)} social sentiment data points")
            
            if self.config.enable_market_data:
                market_data = self.collect_market_sentiment()
                all_data.extend(market_data)
                logger.info(f"Collected {len(market_data)} market sentiment data points")
            
            # Store data
            with self._lock:
                self.sentiment_data.extend(all_data)
            
            logger.info(f"Total sentiment data points collected: {len(all_data)}")
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
        
        return all_data
    
    def get_recent_data(self, hours: int = None) -> List[SentimentData]:
        """Get recent sentiment data"""
        if hours is None:
            hours = self.config.lookback_hours
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_data = [
                data for data in self.sentiment_data
                if data.timestamp > cutoff_time
            ]
        
        return recent_data


class SentimentSignalGenerator:
    """Generates trading signals from sentiment data"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        logger.info("SentimentSignalGenerator initialized")
    
    def generate_signal(self, sentiment_data: List[SentimentData]) -> Optional[SentimentSignal]:
        """Generate trading signal from sentiment data"""
        if len(sentiment_data) < self.config.min_data_points:
            logger.warning(f"Insufficient data points: {len(sentiment_data)} < {self.config.min_data_points}")
            return None
        
        try:
            # Aggregate sentiment by source
            news_sentiments = [d.sentiment_score for d in sentiment_data if d.source == SentimentSource.NEWS]
            social_sentiments = []
            for d in sentiment_data:
                if d.source in [SentimentSource.TWITTER, SentimentSource.REDDIT]:
                    social_sentiments.append(d.sentiment_score)
            
            market_sentiments = []
            for d in sentiment_data:
                if d.source in [SentimentSource.FEAR_GREED_INDEX, SentimentSource.VIX]:
                    market_sentiments.append(d.sentiment_score)
            
            # Calculate average sentiments
            news_sentiment = np.mean(news_sentiments) if news_sentiments else 0.0
            social_sentiment = np.mean(social_sentiments) if social_sentiments else 0.0
            market_sentiment = np.mean(market_sentiments) if market_sentiments else 0.0
            
            # Calculate overall sentiment (weighted average)
            overall_sentiment = (
                news_sentiment * self.config.news_weight +
                social_sentiment * self.config.social_weight +
                market_sentiment * self.config.market_weight
            )
            
            # Calculate confidence based on data consistency
            all_sentiments = [d.sentiment_score for d in sentiment_data]
            sentiment_std = np.std(all_sentiments)
            confidence = max(0.0, 1.0 - sentiment_std)  # Lower std = higher confidence
            
            # Adjust confidence based on data volume
            data_volume_factor = min(len(sentiment_data) / (self.config.min_data_points * 2), 1.0)
            confidence *= data_volume_factor
            
            # Calculate signal strength
            signal_strength = abs(overall_sentiment) * confidence
            
            # Generate recommendation
            if overall_sentiment > self.config.sentiment_threshold and confidence > self.config.confidence_threshold:
                recommendation = "BUY"
            elif overall_sentiment < -self.config.sentiment_threshold and confidence > self.config.confidence_threshold:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            # Extract trending keywords
            all_keywords = []
            for data in sentiment_data:
                all_keywords.extend(data.keywords)
            
            keyword_counts = defaultdict(int)
            for keyword in all_keywords:
                keyword_counts[keyword] += 1
            
            trending_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            trending_keywords = [kw for kw, _ in trending_keywords]
            
            # Count unique sources
            sources = set(d.source for d in sentiment_data)
            
            signal = SentimentSignal(
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                market_sentiment=market_sentiment,
                signal_strength=signal_strength,
                recommendation=recommendation,
                data_points=len(sentiment_data),
                sources_count=len(sources),
                keywords_trending=trending_keywords
            )
            
            logger.info(f"Generated sentiment signal: {recommendation} (sentiment: {overall_sentiment:.3f}, confidence: {confidence:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {e}")
            return None


class SentimentEngine:
    """Main sentiment analysis engine"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        
        # Initialize components
        self.data_collector = SentimentDataCollector(self.config)
        self.signal_generator = SentimentSignalGenerator(self.config)
        
        # Signal history
        self.signal_history = deque(maxlen=1000)
        
        # Real-time processing
        self._running = False
        self._update_thread = None
        self._lock = threading.Lock()
        
        logger.info("SentimentEngine initialized")
    
    def analyze_current_sentiment(self) -> Optional[SentimentSignal]:
        """Analyze current market sentiment and generate signal"""
        try:
            # Collect fresh sentiment data
            sentiment_data = self.data_collector.collect_all_sentiment_data()
            
            if not sentiment_data:
                logger.warning("No sentiment data collected")
                return None
            
            # Generate signal
            signal = self.signal_generator.generate_signal(sentiment_data)
            
            if signal:
                with self._lock:
                    self.signal_history.append(signal)
                
                logger.info(f"Sentiment analysis complete: {signal.recommendation}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
    
    def get_sentiment_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment analysis summary"""
        try:
            recent_data = self.data_collector.get_recent_data(hours)
            
            if not recent_data:
                return {
                    'status': 'no_data',
                    'message': 'No recent sentiment data available'
                }
            
            # Calculate summary statistics
            sentiments = [d.sentiment_score for d in recent_data]
            confidences = [d.confidence for d in recent_data]
            
            # Source breakdown
            source_counts = defaultdict(int)
            source_sentiments = defaultdict(list)
            
            for data in recent_data:
                source_counts[data.source.value] += 1
                source_sentiments[data.source.value].append(data.sentiment_score)
            
            source_avg_sentiments = {
                source: np.mean(sentiments) 
                for source, sentiments in source_sentiments.items()
            }
            
            # Polarity distribution
            polarity_counts = defaultdict(int)
            for data in recent_data:
                polarity_counts[data.polarity.value] += 1
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'data_points': len(recent_data),
                'overall_sentiment': float(np.mean(sentiments)),
                'sentiment_std': float(np.std(sentiments)),
                'average_confidence': float(np.mean(confidences)),
                'source_counts': dict(source_counts),
                'source_sentiments': {k: float(v) for k, v in source_avg_sentiments.items()},
                'polarity_distribution': dict(polarity_counts),
                'sentiment_range': {
                    'min': float(np.min(sentiments)),
                    'max': float(np.max(sentiments)),
                    'median': float(np.median(sentiments))
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def start_real_time_analysis(self):
        """Start real-time sentiment analysis"""
        if self._running:
            logger.warning("Real-time analysis already running")
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._real_time_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        logger.info("Real-time sentiment analysis started")
    
    def stop_real_time_analysis(self):
        """Stop real-time sentiment analysis"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
        
        logger.info("Real-time sentiment analysis stopped")
    
    def _real_time_loop(self):
        """Real-time analysis loop"""
        while self._running:
            try:
                # Analyze current sentiment
                signal = self.analyze_current_sentiment()
                
                if signal:
                    logger.info(f"Real-time sentiment signal: {signal.recommendation} "
                              f"(strength: {signal.signal_strength:.3f})")
                
                # Wait for next update
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in real-time sentiment loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def get_recent_signals(self, count: int = 10) -> List[SentimentSignal]:
        """Get recent sentiment signals"""
        with self._lock:
            return list(self.signal_history)[-count:]
    
    def export_sentiment_data(self, filepath: str, hours: int = 24):
        """Export sentiment data to file"""
        try:
            recent_data = self.data_collector.get_recent_data(hours)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'data_count': len(recent_data),
                'sentiment_data': [data.to_dict() for data in recent_data],
                'recent_signals': [signal.to_dict() for signal in self.get_recent_signals()],
                'summary': self.get_sentiment_summary(hours)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Sentiment data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting sentiment data: {e}")


# Factory functions
def create_default_sentiment_config() -> SentimentConfig:
    """Create default sentiment analysis configuration"""
    return SentimentConfig(
        method=SentimentMethod.ENSEMBLE,
        update_frequency=300.0,  # 5 minutes
        lookback_hours=24,
        sentiment_threshold=0.3,
        confidence_threshold=0.6,
        min_data_points=10
    )


def create_aggressive_sentiment_config() -> SentimentConfig:
    """Create aggressive sentiment analysis configuration"""
    return SentimentConfig(
        method=SentimentMethod.ENSEMBLE,
        update_frequency=60.0,   # 1 minute
        lookback_hours=12,
        sentiment_threshold=0.2,  # Lower threshold
        confidence_threshold=0.5, # Lower confidence requirement
        min_data_points=5,        # Fewer data points needed
        social_weight=0.5,        # Higher social media weight
        news_weight=0.3,
        market_weight=0.2
    )


def create_conservative_sentiment_config() -> SentimentConfig:
    """Create conservative sentiment analysis configuration"""
    return SentimentConfig(
        method=SentimentMethod.ENSEMBLE,
        update_frequency=900.0,   # 15 minutes
        lookback_hours=48,
        sentiment_threshold=0.5,  # Higher threshold
        confidence_threshold=0.8, # Higher confidence requirement
        min_data_points=20,       # More data points needed
        news_weight=0.6,          # Higher news weight
        social_weight=0.2,
        market_weight=0.2
    )


if __name__ == "__main__":
    # Example usage
    print("🔍 Sentiment Analysis Engine")
    print("Ultimate XAU Super System V4.0 - Phase 2 Component 3")
    
    # Create sentiment engine
    config = create_default_sentiment_config()
    engine = SentimentEngine(config)
    
    print(f"\n✅ Sentiment Engine initialized")
    print(f"   Method: {config.method.value}")
    print(f"   Update Frequency: {config.update_frequency}s")
    print(f"   Lookback Hours: {config.lookback_hours}")
    print(f"   TextBlob Available: {TEXTBLOB_AVAILABLE}")
    print(f"   VADER Available: {VADER_AVAILABLE}")
    
    # Analyze current sentiment
    print(f"\n📊 Analyzing Current Sentiment...")
    signal = engine.analyze_current_sentiment()
    
    if signal:
        print(f"   Overall Sentiment: {signal.overall_sentiment:.3f}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Recommendation: {signal.recommendation}")
        print(f"   Signal Strength: {signal.signal_strength:.3f}")
        print(f"   Data Points: {signal.data_points}")
        print(f"   Sources: {signal.sources_count}")
        print(f"   Trending Keywords: {', '.join(signal.keywords_trending[:3])}")
    
    # Get sentiment summary
    print(f"\n📈 Sentiment Summary (24h):")
    summary = engine.get_sentiment_summary()
    
    if summary.get('status') != 'error':
        print(f"   Data Points: {summary['data_points']}")
        print(f"   Overall Sentiment: {summary['overall_sentiment']:.3f}")
        print(f"   Average Confidence: {summary['average_confidence']:.3f}")
        print(f"   Sources: {list(summary['source_counts'].keys())}")
    
    print(f"\n🚀 Sentiment Analysis Engine ready!")