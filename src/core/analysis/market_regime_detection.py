"""
Market Regime Detection System
Ultimate XAU Super System V4.0 - Day 25 Implementation

Advanced market regime detection capabilities:
- Multi-state regime classification (trending, ranging, volatile)
- Real-time regime change detection
- Adaptive strategy selection based on regimes
- Machine learning-enhanced regime prediction
- Integration with multi-timeframe analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    UNKNOWN = "unknown"


class RegimeStrength(Enum):
    """Regime strength classification"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class RegimeConfig:
    """Configuration for market regime detection"""
    
    # Detection parameters
    lookback_period: int = 50
    volatility_window: int = 20
    trend_window: int = 30
    
    # Regime thresholds
    trend_threshold: float = 0.02
    volatility_threshold: float = 0.015
    breakout_threshold: float = 0.025
    
    # Machine learning settings
    enable_ml_prediction: bool = True
    feature_window: int = 20
    retrain_frequency: int = 100
    
    # Real-time settings
    enable_real_time: bool = True
    update_frequency: float = 1.0
    
    # Advanced settings
    enable_multi_timeframe: bool = True
    regime_smoothing: bool = True
    confidence_threshold: float = 0.7


@dataclass
class RegimeResult:
    """Result container for regime detection"""
    
    timestamp: datetime
    regime: MarketRegime
    strength: RegimeStrength
    confidence: float
    
    # Regime metrics
    trend_score: float = 0.0
    volatility_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    
    # Prediction data
    regime_probability: Dict[str, float] = field(default_factory=dict)
    regime_duration: int = 0
    regime_stability: float = 0.0
    
    # Change detection
    regime_changed: bool = False
    previous_regime: Optional[MarketRegime] = None
    change_confidence: float = 0.0
    
    # Additional metrics
    calculation_time: float = 0.0
    data_quality: float = 1.0


class RegimeDetector:
    """Core regime detection engine"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.regime_history = []
        self.last_regime = None
        self.regime_start_time = None
        
    def detect_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Detect current market regime"""
        try:
            start_time = datetime.now()
            
            if len(data) < self.config.lookback_period:
                return self._create_unknown_regime(start_time)
            
            # Calculate regime features
            features = self._calculate_regime_features(data)
            
            # Determine regime using rule-based approach
            regime = self._classify_regime_rules(features)
            
            # Calculate regime strength and confidence
            strength = self._calculate_regime_strength(features, regime)
            confidence = self._calculate_confidence(features, regime)
            
            # Detect regime changes
            regime_changed = self._detect_regime_change(regime)
            
            # Create result
            result = RegimeResult(
                timestamp=datetime.now(),
                regime=regime,
                strength=strength,
                confidence=confidence,
                trend_score=features['trend_score'],
                volatility_score=features['volatility_score'],
                momentum_score=features['momentum_score'],
                volume_score=features['volume_score'],
                regime_changed=regime_changed,
                previous_regime=self.last_regime,
                calculation_time=(datetime.now() - start_time).total_seconds(),
                data_quality=self._assess_data_quality(data)
            )
            
            # Update history
            self._update_regime_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return self._create_unknown_regime(start_time)
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for regime classification"""
        features = {}
        
        try:
            # Price-based features
            returns = data['close'].pct_change().dropna()
            
            # Trend features
            trend_window = min(self.config.trend_window, len(data))
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-trend_window]) / data['close'].iloc[-trend_window]
            features['trend_score'] = price_change
            
            # Moving average trend
            if len(data) >= 20:
                ma_short = data['close'].rolling(10).mean().iloc[-1]
                ma_long = data['close'].rolling(20).mean().iloc[-1]
                features['ma_trend'] = (ma_short - ma_long) / ma_long
            else:
                features['ma_trend'] = 0.0
            
            # Volatility features
            vol_window = min(self.config.volatility_window, len(returns))
            if vol_window > 1:
                features['volatility_score'] = returns.tail(vol_window).std()
                features['volatility_normalized'] = features['volatility_score'] / returns.std()
            else:
                features['volatility_score'] = 0.0
                features['volatility_normalized'] = 1.0
            
            # Momentum features
            if len(data) >= 14:
                rsi = self._calculate_rsi(data['close'], 14)
                features['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                features['momentum_score'] = (features['rsi'] - 50) / 50
            else:
                features['momentum_score'] = 0.0
            
            # Volume features
            if 'volume' in data.columns and len(data) >= 10:
                vol_ma = data['volume'].rolling(10).mean()
                current_vol = data['volume'].iloc[-1]
                avg_vol = vol_ma.iloc[-1]
                features['volume_score'] = (current_vol / avg_vol - 1) if avg_vol > 0 else 0
            else:
                features['volume_score'] = 0.0
            
            # Range features
            if len(data) >= 10:
                high_low_range = (data['high'] - data['low']) / data['close']
                features['range_score'] = high_low_range.tail(10).mean()
            else:
                features['range_score'] = 0.0
            
            # Breakout features
            if len(data) >= 20:
                recent_high = data['high'].tail(20).max()
                recent_low = data['low'].tail(20).min()
                current_price = data['close'].iloc[-1]
                
                if recent_high > recent_low:
                    position_in_range = (current_price - recent_low) / (recent_high - recent_low)
                    features['breakout_score'] = max(0, min(1, position_in_range))
                else:
                    features['breakout_score'] = 0.5
            else:
                features['breakout_score'] = 0.5
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            # Return default features
            features = {
                'trend_score': 0.0,
                'ma_trend': 0.0,
                'volatility_score': 0.0,
                'volatility_normalized': 1.0,
                'momentum_score': 0.0,
                'volume_score': 0.0,
                'range_score': 0.0,
                'breakout_score': 0.5
            }
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _classify_regime_rules(self, features: Dict[str, float]) -> MarketRegime:
        """Classify regime using rule-based approach"""
        try:
            trend_score = features['trend_score']
            volatility_score = features['volatility_score']
            momentum_score = features['momentum_score']
            breakout_score = features['breakout_score']
            
            # High volatility regime
            if volatility_score > self.config.volatility_threshold * 2:
                return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility regime
            if volatility_score < self.config.volatility_threshold * 0.5:
                return MarketRegime.LOW_VOLATILITY
            
            # Breakout regime
            if (breakout_score > 0.9 or breakout_score < 0.1) and abs(trend_score) > self.config.breakout_threshold:
                return MarketRegime.BREAKOUT
            
            # Trending regimes
            if abs(trend_score) > self.config.trend_threshold:
                if trend_score > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            
            # Ranging regime (default)
            return MarketRegime.RANGING
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return MarketRegime.UNKNOWN
    
    def _calculate_regime_strength(self, features: Dict[str, float], regime: MarketRegime) -> RegimeStrength:
        """Calculate regime strength"""
        try:
            if regime == MarketRegime.UNKNOWN:
                return RegimeStrength.VERY_WEAK
            
            # Calculate strength based on regime type
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                strength_score = abs(features['trend_score']) / self.config.trend_threshold
            elif regime == MarketRegime.HIGH_VOLATILITY:
                strength_score = features['volatility_score'] / (self.config.volatility_threshold * 2)
            elif regime == MarketRegime.LOW_VOLATILITY:
                strength_score = 1.0 - (features['volatility_score'] / (self.config.volatility_threshold * 0.5))
            elif regime == MarketRegime.BREAKOUT:
                strength_score = abs(features['trend_score']) / self.config.breakout_threshold
            else:  # RANGING, CONSOLIDATION
                strength_score = 1.0 - abs(features['trend_score']) / self.config.trend_threshold
            
            # Convert to enum
            strength_score = max(0.0, min(2.0, strength_score))
            
            if strength_score >= 1.8:
                return RegimeStrength.VERY_STRONG
            elif strength_score >= 1.4:
                return RegimeStrength.STRONG
            elif strength_score >= 1.0:
                return RegimeStrength.MODERATE
            elif strength_score >= 0.6:
                return RegimeStrength.WEAK
            else:
                return RegimeStrength.VERY_WEAK
                
        except Exception as e:
            logger.error(f"Error calculating strength: {e}")
            return RegimeStrength.WEAK
    
    def _calculate_confidence(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate confidence in regime classification"""
        try:
            if regime == MarketRegime.UNKNOWN:
                return 0.0
            
            # Base confidence from feature consistency
            trend_strength = abs(features['trend_score'])
            vol_consistency = 1.0 - abs(features['volatility_normalized'] - 1.0)
            momentum_alignment = abs(features['momentum_score'])
            
            base_confidence = (trend_strength + vol_consistency + momentum_alignment) / 3
            
            # Adjust based on regime type
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                confidence = min(1.0, base_confidence + trend_strength)
            elif regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
                confidence = min(1.0, base_confidence + vol_consistency)
            else:
                confidence = base_confidence
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _detect_regime_change(self, current_regime: MarketRegime) -> bool:
        """Detect if regime has changed"""
        if self.last_regime is None:
            self.last_regime = current_regime
            self.regime_start_time = datetime.now()
            return True
        
        if current_regime != self.last_regime:
            self.last_regime = current_regime
            self.regime_start_time = datetime.now()
            return True
        
        return False
    
    def _update_regime_history(self, result: RegimeResult):
        """Update regime history"""
        self.regime_history.append(result)
        
        # Keep only recent history
        max_history = 1000
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality"""
        if data.empty:
            return 0.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Check for reasonable price ranges
        if 'close' in data.columns:
            price_std = data['close'].std()
            price_mean = data['close'].mean()
            cv = price_std / price_mean if price_mean > 0 else 1.0
            
            # Quality decreases with extreme volatility
            volatility_quality = 1.0 - min(1.0, cv)
        else:
            volatility_quality = 0.5
        
        quality_score = (1.0 - missing_ratio) * volatility_quality
        return max(0.0, min(1.0, quality_score))
    
    def _create_unknown_regime(self, start_time: datetime) -> RegimeResult:
        """Create unknown regime result"""
        return RegimeResult(
            timestamp=datetime.now(),
            regime=MarketRegime.UNKNOWN,
            strength=RegimeStrength.VERY_WEAK,
            confidence=0.0,
            calculation_time=(datetime.now() - start_time).total_seconds(),
            data_quality=0.0
        )


class MLRegimePredictor:
    """Machine learning-based regime prediction"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_ml_features(self, data: pd.DataFrame, detector: RegimeDetector) -> np.ndarray:
        """Prepare features for ML model"""
        features_list = []
        
        # Use rolling window to create feature vectors
        for i in range(self.config.feature_window, len(data)):
            window_data = data.iloc[i-self.config.feature_window:i]
            features = detector._calculate_regime_features(window_data)
            
            # Additional technical features
            if len(window_data) >= 14:
                # Price statistics
                price_changes = window_data['close'].pct_change().dropna()
                features['returns_mean'] = price_changes.mean()
                features['returns_std'] = price_changes.std()
                features['returns_skew'] = price_changes.skew()
                
                # Price levels
                features['price_position'] = (window_data['close'].iloc[-1] - window_data['close'].min()) / (window_data['close'].max() - window_data['close'].min())
                
                # Moving averages
                if len(window_data) >= 10:
                    ma5 = window_data['close'].rolling(5).mean().iloc[-1]
                    ma10 = window_data['close'].rolling(10).mean().iloc[-1]
                    features['ma5_ma10_ratio'] = ma5 / ma10 if ma10 > 0 else 1.0
            
            features_list.append(list(features.values()))
            
            # Store feature names from first iteration
            if not self.feature_columns:
                self.feature_columns = list(features.keys())
        
        return np.array(features_list) if features_list else np.array([])
    
    def train_model(self, data: pd.DataFrame, detector: RegimeDetector) -> bool:
        """Train ML model on historical data"""
        try:
            if len(data) < self.config.feature_window * 2:
                return False
            
            # Prepare features
            X = self.prepare_ml_features(data, detector)
            if len(X) == 0:
                return False
            
            # Generate labels by detecting regimes
            y = []
            for i in range(self.config.feature_window, len(data)):
                window_data = data.iloc[i-self.config.lookback_period:i+1]
                regime_result = detector.detect_regime(window_data)
                y.append(regime_result.regime.value)
            
            y = np.array(y)
            
            # Ensure we have enough samples
            if len(X) != len(y) or len(X) < 10:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"ML model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False
    
    def predict_regime(self, data: pd.DataFrame, detector: RegimeDetector) -> Dict[str, float]:
        """Predict regime probabilities"""
        if not self.is_trained:
            return {}
        
        try:
            # Prepare features for current window
            if len(data) < self.config.feature_window:
                return {}
            
            window_data = data.tail(self.config.feature_window)
            features = detector._calculate_regime_features(window_data)
            
            # Add additional features (same as training)
            if len(window_data) >= 14:
                price_changes = window_data['close'].pct_change().dropna()
                features['returns_mean'] = price_changes.mean()
                features['returns_std'] = price_changes.std()
                features['returns_skew'] = price_changes.skew()
                
                features['price_position'] = (window_data['close'].iloc[-1] - window_data['close'].min()) / (window_data['close'].max() - window_data['close'].min())
                
                if len(window_data) >= 10:
                    ma5 = window_data['close'].rolling(5).mean().iloc[-1]
                    ma10 = window_data['close'].rolling(10).mean().iloc[-1]
                    features['ma5_ma10_ratio'] = ma5 / ma10 if ma10 > 0 else 1.0
            
            # Convert to array and scale
            feature_vector = np.array([list(features.values())])
            
            # Ensure feature consistency
            if feature_vector.shape[1] != len(self.feature_columns):
                return {}
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            regime_classes = self.model.classes_
            
            return dict(zip(regime_classes, probabilities))
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return {}


class RegimeChangeDetector:
    """Advanced regime change detection"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.change_history = []
        
    def detect_regime_change(self, current_result: RegimeResult, 
                           recent_results: List[RegimeResult]) -> Tuple[bool, float]:
        """Detect regime changes with confidence scoring"""
        
        if len(recent_results) < 3:
            return False, 0.0
        
        # Check for regime stability
        recent_regimes = [r.regime for r in recent_results[-5:]]
        current_regime = current_result.regime
        
        # Count regime occurrences
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate change probability
        if current_regime in regime_counts:
            stability = regime_counts[current_regime] / len(recent_regimes)
        else:
            stability = 0.0
        
        # Change detected if stability is low and confidence is high
        change_detected = stability < 0.6 and current_result.confidence > self.config.confidence_threshold
        change_confidence = (1.0 - stability) * current_result.confidence
        
        return change_detected, change_confidence


class MarketRegimeDetection:
    """Main market regime detection system"""
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        
        # Initialize components
        self.detector = RegimeDetector(self.config)
        self.ml_predictor = MLRegimePredictor(self.config) if self.config.enable_ml_prediction else None
        self.change_detector = RegimeChangeDetector(self.config)
        
        # State management
        self.regime_history = []
        self.last_training_time = None
        
        logger.info("Market Regime Detection system initialized")
    
    def analyze_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Comprehensive regime analysis"""
        try:
            start_time = datetime.now()
            
            # Basic regime detection
            result = self.detector.detect_regime(data)
            
            # ML prediction if enabled
            if self.ml_predictor and self.ml_predictor.is_trained:
                ml_predictions = self.ml_predictor.predict_regime(data, self.detector)
                result.regime_probability = ml_predictions
            
            # Enhanced change detection
            if len(self.regime_history) > 0:
                change_detected, change_confidence = self.change_detector.detect_regime_change(
                    result, self.regime_history
                )
                result.regime_changed = change_detected
                result.change_confidence = change_confidence
            
            # Calculate regime duration
            if len(self.regime_history) > 0:
                result.regime_duration = self._calculate_regime_duration(result.regime)
                result.regime_stability = self._calculate_regime_stability(result.regime)
            
            # Update history
            self.regime_history.append(result)
            self._maintain_history()
            
            # Retrain ML model if needed
            if self.ml_predictor and self._should_retrain():
                self._retrain_ml_model(data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in regime analysis: {e}")
            return self.detector._create_unknown_regime(datetime.now())
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long current regime has been active"""
        duration = 0
        for result in reversed(self.regime_history):
            if result.regime == current_regime:
                duration += 1
            else:
                break
        return duration
    
    def _calculate_regime_stability(self, current_regime: MarketRegime) -> float:
        """Calculate stability of current regime"""
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_results = self.regime_history[-10:]
        same_regime_count = sum(1 for r in recent_results if r.regime == current_regime)
        
        return same_regime_count / len(recent_results)
    
    def _maintain_history(self):
        """Maintain regime history size"""
        max_history = 1000
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]
    
    def _should_retrain(self) -> bool:
        """Check if ML model should be retrained"""
        if not self.ml_predictor:
            return False
        
        if self.last_training_time is None:
            return True
        
        # Retrain based on frequency or significant regime changes
        time_since_training = len(self.regime_history) - (self.last_training_time or 0)
        
        return time_since_training >= self.config.retrain_frequency
    
    def _retrain_ml_model(self, data: pd.DataFrame):
        """Retrain ML model with recent data"""
        try:
            if len(data) >= self.config.feature_window * 2:
                success = self.ml_predictor.train_model(data, self.detector)
                if success:
                    self.last_training_time = len(self.regime_history)
                    logger.info("ML model retrained successfully")
        except Exception as e:
            logger.error(f"Error retraining ML model: {e}")
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics"""
        if not self.regime_history:
            return {}
        
        # Regime distribution
        regime_counts = {}
        for result in self.regime_history:
            regime = result.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Average confidence by regime
        regime_confidence = {}
        for regime_type in regime_counts.keys():
            confidences = [r.confidence for r in self.regime_history if r.regime.value == regime_type]
            regime_confidence[regime_type] = np.mean(confidences) if confidences else 0.0
        
        # Recent performance
        recent_results = self.regime_history[-50:] if len(self.regime_history) >= 50 else self.regime_history
        avg_confidence = np.mean([r.confidence for r in recent_results])
        avg_calculation_time = np.mean([r.calculation_time for r in recent_results])
        
        return {
            'total_analyses': len(self.regime_history),
            'regime_distribution': regime_counts,
            'regime_confidence': regime_confidence,
            'average_confidence': avg_confidence,
            'average_calculation_time': avg_calculation_time,
            'ml_model_trained': self.ml_predictor.is_trained if self.ml_predictor else False,
            'current_regime': self.regime_history[-1].regime.value if self.regime_history else 'unknown'
        }


def create_market_regime_detection(custom_config: Dict = None) -> MarketRegimeDetection:
    """Factory function to create market regime detection system"""
    
    config = RegimeConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return MarketRegimeDetection(config)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create detection system
    system = create_market_regime_detection({
        'lookback_period': 30,
        'enable_ml_prediction': True
    })
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # Simulate different market regimes
    prices = [2000]
    for i in range(1, 200):
        if i < 50:  # Trending up
            change = np.random.normal(0.001, 0.01)
        elif i < 100:  # Ranging
            change = np.random.normal(0, 0.005)
        elif i < 150:  # High volatility
            change = np.random.normal(0, 0.02)
        else:  # Trending down
            change = np.random.normal(-0.001, 0.01)
        
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [p * 0.999 for p in prices],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    data.set_index('timestamp', inplace=True)
    
    # Analyze regimes
    results = []
    for i in range(50, len(data)):
        window_data = data.iloc[:i+1]
        result = system.analyze_regime(window_data)
        results.append(result)
    
    print("Market Regime Detection Results:")
    print(f"Total analyses: {len(results)}")
    
    # Show regime distribution
    regime_counts = {}
    for result in results:
        regime = result.regime.value
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} ({count/len(results)*100:.1f}%)")
    
    # Show recent results
    print(f"\nRecent Results:")
    for result in results[-5:]:
        print(f"  {result.timestamp.strftime('%H:%M')}: {result.regime.value} "
              f"(confidence: {result.confidence:.3f}, strength: {result.strength.name})")
    
    print(f"\nSystem Statistics:")
    stats = system.get_regime_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}") 