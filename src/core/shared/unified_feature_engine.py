"""
Unified Feature Engine for AI3.0 Ultimate XAU System
SINGLE SOURCE OF TRUTH for feature engineering
Used by BOTH Training and Production systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class UnifiedFeatureEngine:
    """
    Unified Feature Engineering Engine
    19 Features Standard for both Training and Production
    """
    
    # STANDARD 19 FEATURES DEFINITION
    FEATURE_NAMES = [
        # Moving Averages (8 features)
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20', 'ema_50',
        
        # Technical Indicators (4 features)  
        'rsi', 'macd', 'macd_signal', 'bb_position',
        
        # Market Analysis (3 features)
        'volatility', 'price_momentum', 'volume_ratio',
        
        # Regime Detection (2 features)
        'volatility_regime', 'trend_strength',
        
        # Temporal Features (2 features)
        'hour', 'day_of_week'
    ]
    
    def __init__(self):
        self.feature_count = len(self.FEATURE_NAMES)
        logger.info(f"UnifiedFeatureEngine initialized with {self.feature_count} standard features")
    
    def prepare_features_from_market_data(self, market_data) -> np.ndarray:
        """
        Prepare 19 features from MarketData object (Production use)
        
        Args:
            market_data: MarketData object with price, technical_indicators, timestamp
            
        Returns:
            np.ndarray: Array of 19 features
        """
        try:
            features = []
            tech_indicators = getattr(market_data, 'technical_indicators', {})
            
            # Moving averages (8 features)
            features.extend([
                tech_indicators.get('sma_5', market_data.price),
                tech_indicators.get('sma_10', market_data.price),
                tech_indicators.get('sma_20', market_data.price),
                tech_indicators.get('sma_50', market_data.price),
                tech_indicators.get('ema_5', market_data.price),
                tech_indicators.get('ema_10', market_data.price),
                tech_indicators.get('ema_20', market_data.price),
                tech_indicators.get('ema_50', market_data.price)
            ])
            
            # Technical indicators (4 features)
            features.extend([
                tech_indicators.get('rsi', 50.0),
                tech_indicators.get('macd', 0.0),
                tech_indicators.get('macd_signal', 0.0),
                tech_indicators.get('bb_position', 0.5)
            ])
            
            # Market analysis (3 features)
            features.extend([
                tech_indicators.get('volatility', 0.5),
                tech_indicators.get('price_momentum', 0.0),
                tech_indicators.get('volume_ratio', 1.0)
            ])
            
            # Regime detection (2 features)
            features.extend([
                tech_indicators.get('volatility_regime', 1.0),
                tech_indicators.get('trend_strength', 0.5)
            ])
            
            # Temporal features (2 features)
            timestamp = getattr(market_data, 'timestamp', datetime.now())
            hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
            day_of_week = timestamp.weekday() if hasattr(timestamp, 'weekday') else 2
            features.extend([hour, day_of_week])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error preparing features from market data: {e}")
            # Return default features
            return self._get_default_features()
    
    def prepare_features_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare 19 features from DataFrame (Training use)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with 19 engineered features
        """
        try:
            # Create copy to avoid modifying original
            data = df.copy()
            
            # Normalize column names to lowercase
            data.columns = data.columns.str.lower()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}. Available: {list(data.columns)}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Handle volume if not present
            if 'volume' not in data.columns:
                if 'tick_volume' in data.columns:
                    data['volume'] = data['tick_volume']
                else:
                    data['volume'] = (data['high'] - data['low']) * 1000
            
            # Calculate features
            features_df = pd.DataFrame(index=data.index)
            
            # Moving Averages (8 features)
            features_df['sma_5'] = data['close'].rolling(5).mean()
            features_df['sma_10'] = data['close'].rolling(10).mean()
            features_df['sma_20'] = data['close'].rolling(20).mean()
            features_df['sma_50'] = data['close'].rolling(50).mean()
            
            features_df['ema_5'] = data['close'].ewm(span=5).mean()
            features_df['ema_10'] = data['close'].ewm(span=10).mean()
            features_df['ema_20'] = data['close'].ewm(span=20).mean()
            features_df['ema_50'] = data['close'].ewm(span=50).mean()
            
            # Technical Indicators (4 features)
            features_df['rsi'] = self._calculate_rsi(data['close'])
            macd_line, macd_signal = self._calculate_macd(data['close'])
            features_df['macd'] = macd_line
            features_df['macd_signal'] = macd_signal
            features_df['bb_position'] = self._calculate_bollinger_position(data['close'])
            
            # Market Analysis (3 features)
            features_df['volatility'] = data['close'].pct_change().rolling(20).std() * 100
            features_df['price_momentum'] = data['close'].pct_change(10)
            features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Regime Detection (2 features)
            features_df['volatility_regime'] = self._calculate_volatility_regime(features_df['volatility'])
            features_df['trend_strength'] = self._calculate_trend_strength(data['close'])
            
            # Temporal Features (2 features)
            if hasattr(data.index, 'hour'):
                features_df['hour'] = data.index.hour
                features_df['day_of_week'] = data.index.dayofweek
            else:
                features_df['hour'] = 12  # Default noon
                features_df['day_of_week'] = 2  # Default Wednesday
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure exactly 19 features
            assert len(features_df.columns) == 19, f"Expected 19 features, got {len(features_df.columns)}"
            
            logger.info(f"Successfully created {len(features_df.columns)} features for {len(features_df)} records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features from dataframe: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        return macd_line, macd_signal
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return (prices - lower_band) / (upper_band - lower_band)
    
    def _calculate_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Calculate volatility regime (low=0, medium=1, high=2)"""
        vol_median = volatility.rolling(100).median()
        conditions = [
            volatility < vol_median * 0.8,
            volatility > vol_median * 1.2
        ]
        choices = [0, 2]  # Low, High
        return pd.Series(np.select(conditions, choices, default=1), index=volatility.index)
    
    def _calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate trend strength using ADX-like calculation"""
        high_low = (prices.rolling(2).max() - prices.rolling(2).min())
        close_close = abs(prices.diff())
        tr = pd.concat([high_low, close_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = prices.diff().where(prices.diff() > 0, 0)
        minus_dm = (-prices.diff()).where(prices.diff() < 0, 0)
        
        plus_di = (plus_dm.rolling(14).mean() / atr) * 100
        minus_di = (minus_dm.rolling(14).mean() / atr) * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        return dx.rolling(14).mean() / 100  # Normalize to 0-1
    
    def _get_default_features(self) -> np.ndarray:
        """Get default feature values when calculation fails"""
        return np.array([
            # Moving averages (use price=2000 as default)
            2000.0, 2000.0, 2000.0, 2000.0,  # SMA
            2000.0, 2000.0, 2000.0, 2000.0,  # EMA
            # Technical indicators
            50.0, 0.0, 0.0, 0.5,  # RSI, MACD, MACD_signal, BB_position
            # Market analysis
            0.5, 0.0, 1.0,  # Volatility, momentum, volume_ratio
            # Regime detection
            1.0, 0.5,  # Volatility_regime, trend_strength
            # Temporal
            12, 2  # Hour, day_of_week
        ], dtype=np.float32)
    
    def validate_features(self, features: Union[np.ndarray, pd.DataFrame]) -> bool:
        """Validate that features match the 19-feature standard"""
        if isinstance(features, pd.DataFrame):
            return len(features.columns) == 19
        elif isinstance(features, np.ndarray):
            return features.shape[-1] == 19
        return False
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.FEATURE_NAMES.copy()
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get comprehensive feature information"""
        return {
            'feature_count': self.feature_count,
            'feature_names': self.FEATURE_NAMES,
            'categories': {
                'moving_averages': self.FEATURE_NAMES[:8],
                'technical_indicators': self.FEATURE_NAMES[8:12],
                'market_analysis': self.FEATURE_NAMES[12:15],
                'regime_detection': self.FEATURE_NAMES[15:17],
                'temporal': self.FEATURE_NAMES[17:19]
            },
            'description': 'Unified 19-feature standard for AI3.0 Ultimate XAU System'
        } 