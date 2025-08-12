"""
Ultimate XAU Super System V4.0 - Day 32: Advanced AI Ensemble & Optimization
Phát triển hệ thống AI ensemble tiên tiến với optimization tự động và adaptive learning.

Author: AI Assistant
Date: 2024-12-20
Version: 4.0.32
"""

import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import threading

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelType(Enum):
    """Định nghĩa các loại AI model."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"

class EnsembleStrategy(Enum):
    """Các chiến lược ensemble."""
    VOTING = "voting"
    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_WEIGHTING = "dynamic_weighting"

@dataclass
class EnsembleConfig:
    """Cấu hình cho ensemble system."""
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.DYNAMIC_WEIGHTING
    models: List[AIModelType] = field(default_factory=lambda: [
        AIModelType.RANDOM_FOREST, 
        AIModelType.GRADIENT_BOOSTING
    ])
    confidence_threshold: float = 0.7
    retraining_frequency: int = 100
    performance_window: int = 50
    weight_decay: float = 0.95
    min_model_weight: float = 0.1
    max_model_weight: float = 0.6
    adaptive_learning: bool = True

@dataclass
class EnsembleResult:
    """Kết quả ensemble."""
    prediction: float
    confidence: float
    model_contributions: Dict[str, float]
    individual_predictions: Dict[str, float]
    ensemble_weights: Dict[str, float]
    signal_strength: float
    processing_time: float

class AdvancedAIEnsembleOptimization:
    """
    Hệ thống AI Ensemble tiên tiến với optimization tự động và adaptive learning.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models = {}
        self.model_weights = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Metrics tracking
        self.metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'model_accuracies': {},
            'ensemble_accuracy': 0.0,
            'optimization_history': [],
            'retraining_count': 0
        }
        
        logger.info(f"AdvancedAIEnsembleOptimization initialized with {len(self.config.models)} models")
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị features cho training."""
        data = data.copy()
        
        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20]:
            data[f'ma_{window}'] = data['close'].rolling(window).mean()
            data[f'ma_ratio_{window}'] = data['close'] / data[f'ma_{window}']
        
        # Volatility
        data['volatility_5'] = data['returns'].rolling(5).std()
        data['volatility_20'] = data['returns'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_window = 20
        bb_std = data['close'].rolling(bb_window).std()
        bb_mean = data['close'].rolling(bb_window).mean()
        data['bb_upper'] = bb_mean + 2 * bb_std
        data['bb_lower'] = bb_mean - 2 * bb_std
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Select features
        feature_columns = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20',
            'rsi', 'macd', 'macd_signal', 'bb_position'
        ] + [f'ma_ratio_{w}' for w in [5, 10, 20]]
        
        # Create target (next period return)
        target = data['returns'].shift(-1)
        
        # Remove NaN values
        valid_mask = ~(data[feature_columns].isna().any(axis=1) | target.isna())
        
        X = data[feature_columns][valid_mask].values
        y = target[valid_mask].values
        
        return X, y
    
    def optimize_model_simple(self, model_type: AIModelType, X: np.ndarray, y: np.ndarray):
        """Tối ưu hóa model đơn giản với grid search."""
        start_time = time.time()
        
        best_score = -np.inf
        best_params = {}
        
        if model_type == AIModelType.RANDOM_FOREST:
            param_grid = [
                {'n_estimators': 50, 'max_depth': 8, 'min_samples_split': 2},
                {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
                {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 3}
            ]
            
            for params in param_grid:
                try:
                    model = RandomForestRegressor(**params, random_state=42)
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                except:
                    continue
                    
        elif model_type == AIModelType.GRADIENT_BOOSTING:
            param_grid = [
                {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 6},
                {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 8},
                {'n_estimators': 150, 'learning_rate': 0.02, 'max_depth': 10}
            ]
            
            for params in param_grid:
                try:
                    model = GradientBoostingRegressor(**params, random_state=42)
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                except:
                    continue
        
        optimization_time = time.time() - start_time
        
        if not best_params:
            # Default params if optimization fails
            if model_type == AIModelType.RANDOM_FOREST:
                best_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42}
            elif model_type == AIModelType.GRADIENT_BOOSTING:
                best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
        
        return best_params, best_score, optimization_time
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train tất cả models với optimization."""
        logger.info("Starting model training with optimization...")
        
        optimization_results = {}
        
        with self.lock:
            self.models.clear()
            
            for model_type in self.config.models:
                try:
                    # Optimize model
                    best_params, best_score, opt_time = self.optimize_model_simple(model_type, X, y)
                    optimization_results[model_type.value] = {
                        'best_score': best_score,
                        'optimization_time': opt_time,
                        'best_params': best_params
                    }
                    
                    # Train with best params
                    if model_type == AIModelType.RANDOM_FOREST:
                        model = RandomForestRegressor(**best_params)
                    elif model_type == AIModelType.GRADIENT_BOOSTING:
                        model = GradientBoostingRegressor(**best_params)
                    else:
                        continue
                    
                    model.fit(X, y)
                    self.models[model_type] = model
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {e}")
                    continue
            
            # Initialize equal weights
            n_models = len(self.models)
            if n_models > 0:
                equal_weight = 1.0 / n_models
                self.model_weights = {model_type: equal_weight for model_type in self.models.keys()}
        
        logger.info(f"Training completed. {len(self.models)} models trained.")
        return optimization_results
    
    def predict_individual_models(self, X: np.ndarray) -> Dict[AIModelType, Tuple[float, float]]:
        """Predict từ từng model riêng lẻ."""
        predictions = {}
        
        with self.lock:
            for model_type, model in self.models.items():
                try:
                    pred = model.predict(X.reshape(1, -1))[0]
                    
                    # Estimate confidence based on model type
                    if hasattr(model, 'feature_importances_'):
                        confidence = min(np.mean(model.feature_importances_) + 0.4, 1.0)
                    else:
                        confidence = 0.7
                    
                    predictions[model_type] = (pred, confidence)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_type}: {e}")
                    predictions[model_type] = (0.0, 0.0)
        
        return predictions
    
    def ensemble_predict(self, X: np.ndarray) -> EnsembleResult:
        """Thực hiện ensemble prediction."""
        start_time = time.time()
        
        # Get individual predictions
        individual_predictions = self.predict_individual_models(X)
        
        if not individual_predictions:
            return EnsembleResult(
                prediction=0.0,
                confidence=0.0,
                model_contributions={},
                individual_predictions={},
                ensemble_weights={},
                signal_strength=0.0,
                processing_time=time.time() - start_time
            )
        
        # Calculate ensemble prediction using dynamic weighting
        total_weighted_pred = 0.0
        total_weighted_conf = 0.0
        total_weight = 0.0
        
        model_contributions = {}
        
        for model_type, (pred, conf) in individual_predictions.items():
            weight = self.model_weights.get(model_type, 1.0 / len(individual_predictions))
            
            # Adjust weight by confidence
            adjusted_weight = weight * conf
            
            total_weighted_pred += pred * adjusted_weight
            total_weighted_conf += conf * adjusted_weight
            total_weight += adjusted_weight
            
            model_contributions[model_type.value] = pred * adjusted_weight
        
        # Normalize
        if total_weight > 0:
            ensemble_prediction = total_weighted_pred / total_weight
            ensemble_confidence = total_weighted_conf / total_weight
        else:
            ensemble_prediction = 0.0
            ensemble_confidence = 0.0
        
        # Calculate signal strength
        signal_strength = ensemble_confidence * len(individual_predictions) / len(self.config.models)
        
        processing_time = time.time() - start_time
        
        # Create weights dict for output
        ensemble_weights = {}
        if total_weight > 0:
            for model_type, (pred, conf) in individual_predictions.items():
                weight = self.model_weights.get(model_type, 1.0 / len(individual_predictions))
                adjusted_weight = weight * conf
                ensemble_weights[model_type.value] = adjusted_weight / total_weight
        
        # Update metrics
        self.metrics['total_predictions'] += 1
        self.metrics['average_confidence'] = (
            (self.metrics['average_confidence'] * (self.metrics['total_predictions'] - 1) + ensemble_confidence) / 
            self.metrics['total_predictions']
        )
        self.metrics['average_processing_time'] = (
            (self.metrics['average_processing_time'] * (self.metrics['total_predictions'] - 1) + processing_time) / 
            self.metrics['total_predictions']
        )
        
        return EnsembleResult(
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            model_contributions=model_contributions,
            individual_predictions={k.value: v for k, v in individual_predictions.items()},
            ensemble_weights=ensemble_weights,
            signal_strength=signal_strength,
            processing_time=processing_time
        )
    
    def update_model_weights(self, performances: Dict[AIModelType, float]):
        """Cập nhật model weights dựa trên performance."""
        if not performances:
            return
        
        with self.lock:
            total_performance = sum(performances.values())
            
            if total_performance > 0:
                new_weights = {}
                for model_type, performance in performances.items():
                    weight = performance / total_performance
                    
                    # Apply constraints
                    weight = max(weight, self.config.min_model_weight)
                    weight = min(weight, self.config.max_model_weight)
                    
                    new_weights[model_type] = weight
                
                # Normalize
                total_weight = sum(new_weights.values())
                if total_weight > 0:
                    new_weights = {k: v / total_weight for k, v in new_weights.items()}
                
                # Apply decay and update
                for model_type in self.model_weights:
                    if model_type in new_weights:
                        old_weight = self.model_weights[model_type]
                        new_weight = new_weights[model_type]
                        
                        self.model_weights[model_type] = (
                            self.config.weight_decay * old_weight + 
                            (1 - self.config.weight_decay) * new_weight
                        )
        
        logger.info(f"Model weights updated: {self.model_weights}")
    
    def full_system_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test toàn bộ hệ thống."""
        logger.info("Starting full system test...")
        start_time = time.time()
        
        try:
            # Prepare data
            X, y = self.prepare_features(data)
            
            if len(X) < 100:
                return {
                    'status': 'ERROR',
                    'message': 'Insufficient data for testing',
                    'execution_time': time.time() - start_time
                }
            
            # Split data
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Train models with optimization
            optimization_results = self.train_models(X_train, y_train)
            
            if not self.models:
                return {
                    'status': 'ERROR',
                    'message': 'No models trained successfully',
                    'execution_time': time.time() - start_time
                }
            
            # Test predictions
            test_results = []
            prediction_times = []
            
            for i in range(min(len(X_test), 50)):
                pred_start = time.time()
                result = self.ensemble_predict(X_test[i])
                pred_time = time.time() - pred_start
                
                prediction_times.append(pred_time)
                test_results.append({
                    'prediction': result.prediction,
                    'actual': y_test[i],
                    'confidence': result.confidence,
                    'signal_strength': result.signal_strength,
                    'processing_time': pred_time
                })
            
            # Calculate metrics
            predictions = [r['prediction'] for r in test_results]
            actuals = [r['actual'] for r in test_results]
            
            if len(predictions) == 0:
                return {
                    'status': 'ERROR',
                    'message': 'No predictions generated',
                    'execution_time': time.time() - start_time
                }
            
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            # Direction accuracy
            direction_correct = sum(
                1 for p, a in zip(predictions, actuals) 
                if (p > 0 and a > 0) or (p <= 0 and a <= 0)
            )
            direction_accuracy = direction_correct / len(predictions)
            
            avg_confidence = np.mean([r['confidence'] for r in test_results])
            avg_processing_time = np.mean(prediction_times)
            
            execution_time = time.time() - start_time
            
            # Calculate scores with improved weighting
            performance_score = min(direction_accuracy * 100, 100)
            speed_score = min(100, max(0, (0.1 - avg_processing_time) / 0.1 * 100))
            optimization_score = min(100, len(optimization_results) / len(self.config.models) * 100)
            confidence_score = min(avg_confidence * 100, 100)
            
            # Weighted overall score for Day 32
            overall_score = (
                performance_score * 0.35 + 
                speed_score * 0.25 + 
                optimization_score * 0.25 + 
                confidence_score * 0.15
            )
            
            results = {
                'day': 32,
                'system_name': 'Ultimate XAU Super System V4.0',
                'module_name': 'Advanced AI Ensemble & Optimization',
                'completion_date': datetime.now().strftime('%Y-%m-%d'),
                'version': '4.0.32',
                'phase': 'Phase 4: Advanced AI Systems',
                'status': 'SUCCESS',
                'execution_time': execution_time,
                'overall_score': overall_score,
                'performance_metrics': {
                    'direction_accuracy': direction_accuracy,
                    'mse': mse,
                    'r2_score': r2,
                    'average_confidence': avg_confidence,
                    'average_processing_time': avg_processing_time,
                    'performance_score': performance_score,
                    'speed_score': speed_score,
                    'optimization_score': optimization_score,
                    'confidence_score': confidence_score
                },
                'optimization_results': optimization_results,
                'ensemble_metrics': {
                    'total_models': len(self.models),
                    'model_weights': {k.value: v for k, v in self.model_weights.items()},
                    'predictions_made': len(test_results),
                    'ensemble_strategy': self.config.ensemble_strategy.value
                },
                'test_details': {
                    'samples_tested': len(test_results),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': X.shape[1]
                },
                'adaptive_learning': {
                    'enabled': self.config.adaptive_learning,
                    'confidence_threshold': self.config.confidence_threshold,
                    'weight_decay': self.config.weight_decay
                }
            }
            
            logger.info(f"Full system test completed. Overall score: {overall_score:.1f}/100")
            return results
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            return {
                'status': 'ERROR',
                'message': str(e),
                'execution_time': time.time() - start_time
            }

def demo_advanced_ai_ensemble():
    """Demo function để test hệ thống."""
    print("=== ULTIMATE XAU SUPER SYSTEM V4.0 - DAY 32 ===")
    print("Advanced AI Ensemble & Optimization System Demo")
    print("=" * 50)
    
    try:
        # Tạo sample data
        print("\n1. Generating sample market data...")
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='D')
        
        # Tạo realistic price data với trend
        initial_price = 2000
        trend = 0.0002  # Slight upward trend
        returns = np.random.normal(trend, 0.015, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            # Add some autocorrelation
            if len(prices) > 1:
                momentum = (prices[-1] - prices[-2]) / prices[-2] * 0.1
                ret += momentum
            prices.append(prices[-1] * (1 + ret))
        
        # Add realistic price features
        data = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        print(f"✅ Generated {len(data)} days of market data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Khởi tạo hệ thống
        print("\n2. Initializing Advanced AI Ensemble System...")
        config = EnsembleConfig(
            models=[AIModelType.RANDOM_FOREST, AIModelType.GRADIENT_BOOSTING],
            confidence_threshold=0.7,
            adaptive_learning=True,
            ensemble_strategy=EnsembleStrategy.DYNAMIC_WEIGHTING
        )
        
        system = AdvancedAIEnsembleOptimization(config)
        print("✅ System initialized successfully")
        
        # Chạy full test
        print("\n3. Running comprehensive system test...")
        print("   - Feature engineering...")
        print("   - Model optimization...")
        print("   - Ensemble training...")
        print("   - Performance evaluation...")
        
        results = system.full_system_test(data)
        
        if results['status'] == 'SUCCESS':
            print("✅ System test completed successfully!")
            
            print(f"\n📊 DAY 32 PERFORMANCE METRICS:")
            print(f"   Overall Score: {results['overall_score']:.1f}/100")
            
            perf = results['performance_metrics']
            print(f"   Direction Accuracy: {perf['direction_accuracy']:.1%}")
            print(f"   Average Confidence: {perf['average_confidence']:.1%}")
            print(f"   Processing Time: {perf['average_processing_time']:.3f}s")
            print(f"   R² Score: {perf['r2_score']:.3f}")
            
            print(f"\n🔧 OPTIMIZATION RESULTS:")
            for model, opt_result in results['optimization_results'].items():
                print(f"   {model}: Score {opt_result['best_score']:.3f}, "
                      f"Time {opt_result['optimization_time']:.1f}s")
            
            print(f"\n🎯 ENSEMBLE METRICS:")
            ensemble = results['ensemble_metrics']
            print(f"   Models Trained: {ensemble['total_models']}")
            print(f"   Strategy: {ensemble['ensemble_strategy']}")
            print(f"   Model Weights: {ensemble['model_weights']}")
            print(f"   Predictions Made: {ensemble['predictions_made']}")
            
            print(f"\n🤖 ADAPTIVE LEARNING:")
            adaptive = results['adaptive_learning']
            print(f"   Enabled: {adaptive['enabled']}")
            print(f"   Confidence Threshold: {adaptive['confidence_threshold']}")
            print(f"   Weight Decay: {adaptive['weight_decay']}")
            
            print(f"\n⏱️  EXECUTION TIME: {results['execution_time']:.2f} seconds")
            
            # Đánh giá theo tiêu chuẩn Day 32
            if results['overall_score'] >= 85:
                grade = "XUẤT SẮC"
                status = "🎯"
            elif results['overall_score'] >= 75:
                grade = "TỐT"
                status = "✅"
            elif results['overall_score'] >= 65:
                grade = "KHANG ĐỊNH"
                status = "⚠️"
            else:
                grade = "CẦN CẢI THIỆN"
                status = "🔴"
            
            print(f"\n{status} DAY 32 COMPLETION: {grade} ({results['overall_score']:.1f}/100)")
            
            # Lưu kết quả
            with open('day32_advanced_ai_ensemble_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print("✅ Results saved to day32_advanced_ai_ensemble_results.json")
            
        else:
            print(f"❌ System test failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_advanced_ai_ensemble()
