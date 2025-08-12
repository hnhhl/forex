"""
Machine Learning Enhanced Trading Signals
Ultimate XAU Super System V4.0 - Day 29 Implementation

Advanced ML-powered trading signal generation:
- AI Signal Generation with multiple ML models
- Ensemble Methods for robust predictions
- Feature Engineering for market patterns
- Model Validation and backtesting framework
- Real-time Inference for live trading
- Signal Confidence and Risk Assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MLModelType(Enum):
    """Types of ML models"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SUPPORT_VECTOR = "support_vector"
    NEURAL_NETWORK = "neural_network"


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class FeatureType(Enum):
    """Types of features for ML models"""
    TECHNICAL_INDICATORS = "technical_indicators"
    PRICE_PATTERNS = "price_patterns"
    VOLUME_PATTERNS = "volume_patterns"
    VOLATILITY_FEATURES = "volatility_features"
    MOMENTUM_FEATURES = "momentum_features"
    MARKET_MICROSTRUCTURE = "market_microstructure"


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"


@dataclass
class MLConfig:
    """Configuration for ML trading signals"""
    
    # Model selection
    ml_models: List[MLModelType] = field(default_factory=lambda: [
        MLModelType.RANDOM_FOREST,
        MLModelType.GRADIENT_BOOSTING,
        MLModelType.LINEAR_REGRESSION,
        MLModelType.RIDGE_REGRESSION
    ])
    
    # Feature engineering
    feature_types: List[FeatureType] = field(default_factory=lambda: [
        FeatureType.TECHNICAL_INDICATORS,
        FeatureType.PRICE_PATTERNS,
        FeatureType.MOMENTUM_FEATURES,
        FeatureType.VOLATILITY_FEATURES
    ])
    
    # Training parameters
    lookback_period: int = 60  # Days for feature calculation
    prediction_horizon: int = 5  # Days ahead to predict
    train_test_split: float = 0.8  # 80% training, 20% testing
    validation_method: str = "time_series"  # time_series, cross_validation
    
    # Model parameters
    random_forest_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    })
    
    gradient_boosting_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    })
    
    # Ensemble settings
    ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE
    enable_ensemble: bool = True
    
    # Signal generation
    signal_confidence_threshold: float = 0.6  # Minimum confidence for signals
    signal_strength_levels: int = 5  # Number of signal strength levels
    
    # Risk management
    max_position_size: float = 1.0  # Maximum position size
    risk_adjustment: bool = True
    volatility_scaling: bool = True
    
    # Real-time settings
    real_time_inference: bool = True
    model_update_frequency: int = 24  # Hours between model updates
    feature_update_frequency: int = 1  # Hours between feature updates


@dataclass
class MLFeatures:
    """Container for ML features"""
    
    timestamp: datetime
    
    # Technical indicators
    sma_short: float = 0.0
    sma_long: float = 0.0
    ema_short: float = 0.0
    ema_long: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    
    # Price patterns
    returns_1d: float = 0.0
    returns_5d: float = 0.0
    returns_10d: float = 0.0
    price_momentum: float = 0.0
    price_acceleration: float = 0.0
    
    # Volatility features
    volatility_1d: float = 0.0
    volatility_5d: float = 0.0
    volatility_10d: float = 0.0
    volatility_ratio: float = 1.0
    
    # Volume features
    volume_sma: float = 0.0
    volume_ratio: float = 1.0
    
    # Market microstructure
    bid_ask_spread: float = 0.0
    order_flow: float = 0.0
    
    # Target variable
    target_return: Optional[float] = None


@dataclass
class MLSignal:
    """ML-generated trading signal"""
    
    timestamp: datetime
    signal_type: SignalType
    
    # Signal strength and confidence
    strength: float = 0.0  # 0-1 scale
    confidence: float = 0.0  # 0-1 scale
    
    # Model predictions
    predicted_return: float = 0.0
    predicted_volatility: float = 0.0
    prediction_horizon: int = 5  # Days
    
    # Individual model contributions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    risk_adjusted_signal: float = 0.0
    max_drawdown_prediction: float = 0.0
    var_prediction: float = 0.0
    
    # Execution parameters
    recommended_position_size: float = 0.0
    stop_loss_level: float = 0.0
    take_profit_level: float = 0.0
    
    # Model diagnostics
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    
    model_name: str
    timestamp: datetime
    
    # Accuracy metrics
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    
    # Financial metrics
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    
    # Statistical tests
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Validation results
    cross_val_scores: List[float] = field(default_factory=list)
    out_of_sample_performance: float = 0.0
    
    # Feature analysis
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_stability: float = 0.0


class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def create_features(self, price_data: pd.DataFrame, 
                       volume_data: pd.Series = None) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        
        try:
            features_df = pd.DataFrame(index=price_data.index)
            
            # Technical indicators
            if FeatureType.TECHNICAL_INDICATORS in self.config.feature_types:
                tech_features = self._create_technical_features(price_data)
                features_df = pd.concat([features_df, tech_features], axis=1)
            
            # Price patterns
            if FeatureType.PRICE_PATTERNS in self.config.feature_types:
                price_features = self._create_price_features(price_data)
                features_df = pd.concat([features_df, price_features], axis=1)
            
            # Momentum features
            if FeatureType.MOMENTUM_FEATURES in self.config.feature_types:
                momentum_features = self._create_momentum_features(price_data)
                features_df = pd.concat([features_df, momentum_features], axis=1)
            
            # Volatility features
            if FeatureType.VOLATILITY_FEATURES in self.config.feature_types:
                vol_features = self._create_volatility_features(price_data)
                features_df = pd.concat([features_df, vol_features], axis=1)
            
            # Volume features (if available)
            if volume_data is not None and FeatureType.VOLUME_PATTERNS in self.config.feature_types:
                volume_features = self._create_volume_features(volume_data)
                features_df = pd.concat([features_df, volume_features], axis=1)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def _create_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Assume price_data has 'close' column or use first column
        prices = price_data.iloc[:, 0] if 'close' not in price_data.columns else price_data['close']
        
        # Moving averages
        features['sma_5'] = prices.rolling(5).mean()
        features['sma_10'] = prices.rolling(10).mean()
        features['sma_20'] = prices.rolling(20).mean()
        features['ema_5'] = prices.ewm(span=5).mean()
        features['ema_10'] = prices.ewm(span=10).mean()
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        features['bollinger_upper'] = sma_20 + (std_20 * 2)
        features['bollinger_lower'] = sma_20 - (std_20 * 2)
        features['bollinger_width'] = features['bollinger_upper'] - features['bollinger_lower']
        features['bollinger_position'] = (prices - features['bollinger_lower']) / features['bollinger_width']
        
        return features
    
    def _create_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create price pattern features"""
        
        features = pd.DataFrame(index=price_data.index)
        prices = price_data.iloc[:, 0]
        
        # Returns
        features['returns_1d'] = prices.pct_change(1)
        features['returns_3d'] = prices.pct_change(3)
        features['returns_5d'] = prices.pct_change(5)
        features['returns_10d'] = prices.pct_change(10)
        
        # Price momentum
        features['momentum_5d'] = prices / prices.shift(5) - 1
        features['momentum_10d'] = prices / prices.shift(10) - 1
        features['momentum_20d'] = prices / prices.shift(20) - 1
        
        # Price acceleration
        features['acceleration_5d'] = features['returns_1d'] - features['returns_1d'].shift(5)
        features['acceleration_10d'] = features['returns_1d'] - features['returns_1d'].shift(10)
        
        # Price relative to moving averages
        features['price_vs_sma_5'] = prices / prices.rolling(5).mean() - 1
        features['price_vs_sma_20'] = prices / prices.rolling(20).mean() - 1
        
        return features
    
    def _create_momentum_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features"""
        
        features = pd.DataFrame(index=price_data.index)
        prices = price_data.iloc[:, 0]
        returns = prices.pct_change()
        
        # Rolling momentum
        features['momentum_3d'] = returns.rolling(3).sum()
        features['momentum_5d'] = returns.rolling(5).sum()
        features['momentum_10d'] = returns.rolling(10).sum()
        
        # Momentum strength
        features['positive_momentum_3d'] = (returns > 0).rolling(3).sum()
        features['positive_momentum_5d'] = (returns > 0).rolling(5).sum()
        
        # Rate of change
        features['roc_5d'] = (prices - prices.shift(5)) / prices.shift(5)
        features['roc_10d'] = (prices - prices.shift(10)) / prices.shift(10)
        
        return features
    
    def _create_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        
        features = pd.DataFrame(index=price_data.index)
        prices = price_data.iloc[:, 0]
        returns = prices.pct_change()
        
        # Rolling volatility
        features['volatility_5d'] = returns.rolling(5).std()
        features['volatility_10d'] = returns.rolling(10).std()
        features['volatility_20d'] = returns.rolling(20).std()
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
        features['vol_ratio_10_20'] = features['volatility_10d'] / features['volatility_20d']
        
        # GARCH-like features
        features['volatility_mean_reversion'] = features['volatility_5d'] - features['volatility_20d']
        
        return features
    
    def _create_volume_features(self, volume_data: pd.Series) -> pd.DataFrame:
        """Create volume-based features"""
        
        features = pd.DataFrame(index=volume_data.index)
        
        # Volume moving averages
        features['volume_sma_5'] = volume_data.rolling(5).mean()
        features['volume_sma_20'] = volume_data.rolling(20).mean()
        
        # Volume ratios
        features['volume_ratio'] = volume_data / features['volume_sma_20']
        features['volume_trend'] = features['volume_sma_5'] / features['volume_sma_20']
        
        return features


class MLModelManager:
    """Manages multiple ML models for ensemble predictions"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_performances = {}
        self.scaler = StandardScaler()
        
    def initialize_models(self):
        """Initialize all configured ML models"""
        
        try:
            for model_type in self.config.ml_models:
                if model_type == MLModelType.LINEAR_REGRESSION:
                    self.models[model_type.value] = LinearRegression()
                
                elif model_type == MLModelType.RIDGE_REGRESSION:
                    self.models[model_type.value] = Ridge(alpha=1.0, random_state=42)
                
                elif model_type == MLModelType.LASSO_REGRESSION:
                    self.models[model_type.value] = Lasso(alpha=1.0, random_state=42)
                
                elif model_type == MLModelType.RANDOM_FOREST:
                    self.models[model_type.value] = RandomForestRegressor(
                        **self.config.random_forest_params
                    )
                
                elif model_type == MLModelType.GRADIENT_BOOSTING:
                    self.models[model_type.value] = GradientBoostingRegressor(
                        **self.config.gradient_boosting_params
                    )
                
                elif model_type == MLModelType.SUPPORT_VECTOR:
                    self.models[model_type.value] = SVR(kernel='rbf', C=1.0, gamma='scale')
            
            self.logger.info(f"Initialized {len(self.models)} ML models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, ModelPerformance]:
        """Train all models and evaluate performance"""
        
        try:
            if len(self.models) == 0:
                self.initialize_models()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data for training/testing
            split_index = int(len(X) * self.config.train_test_split)
            
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            performances = {}
            
            for model_name, model in self.models.items():
                try:
                    # Train model
                    train_start = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - train_start
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate performance metrics
                    performance = self._calculate_performance(
                        model_name, y_train, y_pred_train, y_test, y_pred_test
                    )
                    
                    # Cross validation
                    if len(X_train) > 50:  # Only if enough data
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        performance.cross_val_scores = cv_scores.tolist()
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        importance_dict = dict(zip(X.columns, model.feature_importances_))
                        performance.feature_importance = importance_dict
                    elif hasattr(model, 'coef_'):
                        importance_dict = dict(zip(X.columns, abs(model.coef_)))
                        performance.feature_importance = importance_dict
                    
                    performances[model_name] = performance
                    self.model_performances[model_name] = performance
                    
                    self.logger.info(f"Trained {model_name}: R² = {performance.r2_score:.3f}, "
                                   f"Train time = {train_time:.3f}s")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
                    continue
            
            return performances
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return {}
    
    def _calculate_performance(self, model_name: str, y_train: pd.Series, y_pred_train: np.ndarray,
                             y_test: pd.Series, y_pred_test: np.ndarray) -> ModelPerformance:
        """Calculate comprehensive model performance metrics"""
        
        # Basic metrics on test set
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        # Financial metrics (simplified)
        correct_direction = ((y_test > 0) == (y_pred_test > 0)).mean()
        
        # Create performance object
        performance = ModelPerformance(
            model_name=model_name,
            timestamp=datetime.now(),
            mse=mse,
            mae=mae,
            r2_score=r2,
            win_rate=correct_direction,
            out_of_sample_performance=r2
        )
        
        return performance
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Generate ensemble prediction from all models"""
        
        try:
            if len(self.models) == 0:
                return 0.0, {}
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            predictions = {}
            weights = {}
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]  # Single prediction
                    predictions[model_name] = pred
                    
                    # Weight based on model performance
                    if model_name in self.model_performances:
                        performance = self.model_performances[model_name]
                        weights[model_name] = max(0, performance.r2_score)  # Use R² as weight
                    else:
                        weights[model_name] = 1.0
                        
                except Exception as e:
                    self.logger.error(f"Error predicting with {model_name}: {e}")
                    continue
            
            if not predictions:
                return 0.0, {}
            
            # Ensemble combination
            if self.config.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    ensemble_pred = sum(pred * weights[name] for name, pred in predictions.items()) / total_weight
                else:
                    ensemble_pred = np.mean(list(predictions.values()))
            else:
                ensemble_pred = np.mean(list(predictions.values()))
            
            return ensemble_pred, predictions
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return 0.0, {}


class SignalGenerator:
    """Generates trading signals from ML predictions"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(self, prediction: float, 
                       model_predictions: Dict[str, float],
                       current_price: float,
                       volatility: float = 0.02) -> MLSignal:
        """Generate trading signal from ML prediction"""
        
        try:
            # Determine signal type based on prediction
            if prediction > 0.02:  # 2% threshold
                signal_type = SignalType.STRONG_BUY if prediction > 0.05 else SignalType.BUY
            elif prediction < -0.02:
                signal_type = SignalType.STRONG_SELL if prediction < -0.05 else SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Calculate signal strength (0-1)
            strength = min(1.0, abs(prediction) / 0.1)  # Scale to 10% max
            
            # Calculate confidence based on model agreement
            pred_values = list(model_predictions.values())
            if len(pred_values) > 1:
                pred_std = np.std(pred_values)
                confidence = max(0.0, 1.0 - pred_std / 0.05)  # Higher agreement = higher confidence
            else:
                confidence = 0.5  # Default confidence
            
            # Risk-adjusted signal
            risk_adjusted_signal = prediction / (volatility + 0.01)  # Adjust for volatility
            
            # Position sizing
            base_position = min(1.0, abs(prediction) / 0.05)  # Scale to 5% max prediction
            volatility_adjusted_position = base_position / (volatility * 10)  # Reduce size in high vol
            recommended_position = min(self.config.max_position_size, volatility_adjusted_position)
            
            # Stop loss and take profit levels
            stop_loss_distance = volatility * 2  # 2x daily volatility
            take_profit_distance = abs(prediction) * 0.5  # 50% of expected return
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss_level = current_price * (1 - stop_loss_distance)
                take_profit_level = current_price * (1 + take_profit_distance)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss_level = current_price * (1 + stop_loss_distance)
                take_profit_level = current_price * (1 - take_profit_distance)
            else:
                stop_loss_level = current_price
                take_profit_level = current_price
            
            signal = MLSignal(
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                predicted_return=prediction,
                predicted_volatility=volatility,
                prediction_horizon=self.config.prediction_horizon,
                model_predictions=model_predictions,
                risk_adjusted_signal=risk_adjusted_signal,
                recommended_position_size=recommended_position,
                stop_loss_level=stop_loss_level,
                take_profit_level=take_profit_level
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return MLSignal(
                timestamp=datetime.now(),
                signal_type=SignalType.HOLD,
                strength=0.0,
                confidence=0.0
            )


class MLEnhancedTradingSignals:
    """Main ML-enhanced trading signals system"""
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_manager = MLModelManager(self.config)
        self.signal_generator = SignalGenerator(self.config)
        
        # State management
        self.signal_history = []
        self.model_performance_history = []
        self.feature_history = []
        
        self.logger.info("ML Enhanced Trading Signals system initialized")
    
    def train_system(self, price_data: pd.DataFrame, 
                    volume_data: pd.Series = None) -> Dict[str, ModelPerformance]:
        """Train the complete ML system"""
        
        try:
            # Create features
            features_df = self.feature_engineer.create_features(price_data, volume_data)
            
            if features_df.empty:
                self.logger.error("No features created")
                return {}
            
            # Create target variable (future returns)
            prices = price_data.iloc[:, 0]
            target = prices.pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
            
            # Align features and target
            aligned_data = pd.concat([features_df, target], axis=1, join='inner')
            aligned_data.columns = list(features_df.columns) + ['target']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 50:
                self.logger.error("Insufficient data for training")
                return {}
            
            X = aligned_data.iloc[:, :-1]
            y = aligned_data.iloc[:, -1]
            
            # Train models
            performances = self.model_manager.train_models(X, y)
            
            # Store performance history
            self.model_performance_history.extend(performances.values())
            
            self.logger.info(f"Training completed. {len(performances)} models trained.")
            
            return performances
            
        except Exception as e:
            self.logger.error(f"Error training system: {e}")
            return {}
    
    def generate_trading_signal(self, current_price_data: pd.DataFrame,
                              current_volume_data: pd.Series = None) -> MLSignal:
        """Generate trading signal for current market conditions"""
        
        try:
            # Create features for current data
            features_df = self.feature_engineer.create_features(current_price_data, current_volume_data)
            
            if features_df.empty:
                return MLSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.HOLD,
                    strength=0.0,
                    confidence=0.0
                )
            
            # Get latest feature vector
            latest_features = features_df.iloc[[-1]]  # Last row as DataFrame
            
            # Generate ensemble prediction
            ensemble_pred, model_preds = self.model_manager.predict_ensemble(latest_features)
            
            # Get current price and volatility
            current_price = current_price_data.iloc[-1, 0]
            recent_returns = current_price_data.iloc[:, 0].pct_change().tail(20)
            current_volatility = recent_returns.std()
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                ensemble_pred, model_preds, current_price, current_volatility
            )
            
            # Add feature importance from best performing model
            if self.model_manager.model_performances:
                best_model = max(self.model_manager.model_performances.keys(),
                               key=lambda k: self.model_manager.model_performances[k].r2_score)
                if best_model in self.model_manager.model_performances:
                    signal.feature_importance = self.model_manager.model_performances[best_model].feature_importance
            
            # Store signal history
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return MLSignal(
                timestamp=datetime.now(),
                signal_type=SignalType.HOLD,
                strength=0.0,
                confidence=0.0
            )
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics"""
        
        try:
            diagnostics = {
                "timestamp": datetime.now(),
                "models_trained": len(self.model_manager.models),
                "features_available": len(self.feature_engineer.config.feature_types),
                "signals_generated": len(self.signal_history),
                "model_performances": {}
            }
            
            # Add model performance summary
            for model_name, performance in self.model_manager.model_performances.items():
                diagnostics["model_performances"][model_name] = {
                    "r2_score": performance.r2_score,
                    "mse": performance.mse,
                    "win_rate": performance.win_rate,
                    "feature_count": len(performance.feature_importance)
                }
            
            # Signal distribution
            if self.signal_history:
                signal_types = [s.signal_type.value for s in self.signal_history]
                signal_distribution = {st: signal_types.count(st) for st in set(signal_types)}
                diagnostics["signal_distribution"] = signal_distribution
                
                # Average confidence and strength
                diagnostics["average_confidence"] = np.mean([s.confidence for s in self.signal_history])
                diagnostics["average_strength"] = np.mean([s.strength for s in self.signal_history])
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error getting diagnostics: {e}")
            return {"error": "Diagnostics generation failed"}


def create_ml_enhanced_trading_signals(custom_config: Dict = None) -> MLEnhancedTradingSignals:
    """Factory function to create ML trading signals system"""
    
    config = MLConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return MLEnhancedTradingSignals(config)


if __name__ == "__main__":
    # Example usage
    print("ML Enhanced Trading Signals System")
    
    # Create system
    ml_system = create_ml_enhanced_trading_signals({
        'ml_models': [MLModelType.RANDOM_FOREST, MLModelType.GRADIENT_BOOSTING],
        'prediction_horizon': 5,
        'ensemble_method': EnsembleMethod.WEIGHTED_AVERAGE
    })
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='1D')
    
    # Simulate price data with trend and noise
    returns = np.random.normal(0.001, 0.02, 252)
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    price_data = pd.DataFrame({'close': prices[1:]}, index=dates)
    
    # Train system
    print("\nTraining ML system...")
    performances = ml_system.train_system(price_data)
    
    for model_name, performance in performances.items():
        print(f"{model_name}: R² = {performance.r2_score:.3f}, Win Rate = {performance.win_rate:.1%}")
    
    # Generate signal
    recent_data = price_data.tail(60)  # Last 60 days for signal generation
    signal = ml_system.generate_trading_signal(recent_data)
    
    print(f"\nTrading Signal:")
    print(f"Type: {signal.signal_type.value}")
    print(f"Strength: {signal.strength:.2f}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Predicted Return: {signal.predicted_return:.2%}")
    print(f"Recommended Position: {signal.recommended_position_size:.2f}")
    
    # Get diagnostics
    diagnostics = ml_system.get_model_diagnostics()
    print(f"\nSystem Diagnostics:")
    print(f"Models Trained: {diagnostics['models_trained']}")
    print(f"Signals Generated: {diagnostics['signals_generated']}")
    print(f"Average Confidence: {diagnostics.get('average_confidence', 0):.2f}")