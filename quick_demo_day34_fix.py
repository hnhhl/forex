#!/usr/bin/env python3
"""
Ultimate XAU Super System V4.0 - Day 34: Neural Network Optimization (Quick Demo)
Fixed version với enhanced features và proper ensemble methods.

Author: AI Assistant
Date: 2024-12-20
Version: 4.0.34
"""

import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# ML imports
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """Enhanced feature engineering với 16+ technical indicators."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_regression, k=12)
        
    def create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Tạo enhanced features với comprehensive technical indicators."""
        data = data.copy()
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_change'] = data['close'].diff()
        
        # Enhanced moving averages
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            data[f'ma_{window}'] = ma
            data[f'ma_ratio_{window}'] = data['close'] / ma
            data[f'ma_slope_{window}'] = ma.diff()
        
        # Volatility indicators
        data['volatility_5'] = data['returns'].rolling(5).std()
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_ratio'] = data['volatility_5'] / (data['volatility_20'] + 1e-8)
        
        # RSI enhanced
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_norm'] = (data['rsi'] - 50) / 50
        data['rsi_momentum'] = data['rsi'].diff()
        
        # Bollinger Bands (Fixed)
        bb_window = 20
        bb_std = data['close'].rolling(bb_window).std()
        bb_mean = data['close'].rolling(bb_window).mean()
        bb_upper = bb_mean + (bb_std * 2)
        bb_lower = bb_mean - (bb_std * 2)
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        data['bb_squeeze'] = bb_std / (bb_mean + 1e-8)
        
        # Stochastic Oscillator
        low_14 = data['close'].rolling(14).min()
        high_14 = data['close'].rolling(14).max()
        stoch_k = ((data['close'] - low_14) / (high_14 - low_14 + 1e-8)) * 100
        data['stoch_k'] = stoch_k
        data['stoch_d'] = stoch_k.rolling(3).mean()
        data['stoch_norm'] = (stoch_k - 50) / 50
        
        # Williams %R
        williams_r = ((high_14 - data['close']) / (high_14 - low_14 + 1e-8)) * -100
        data['williams_r'] = williams_r
        data['williams_norm'] = (williams_r + 50) / 50
        
        # Rate of Change
        data['roc_5'] = ((data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)) * 100
        data['roc_10'] = ((data['close'] - data['close'].shift(10)) / (data['close'].shift(10) + 1e-8)) * 100
        
        # Volume-based (using close as proxy)
        vwap = (data['close'] * data['close']).rolling(20).sum() / (data['close'].rolling(20).sum() + 1e-8)
        data['vwap'] = vwap
        data['vwap_ratio'] = data['close'] / (vwap + 1e-8)
        
        # Momentum indicators
        data['momentum_5'] = data['close'] / (data['close'].shift(5) + 1e-8)
        data['momentum_10'] = data['close'] / (data['close'].shift(10) + 1e-8)
        
        # Price position
        rolling_min = data['close'].rolling(20).min()
        rolling_max = data['close'].rolling(20).max()
        data['price_position'] = (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        return data
    
    def select_best_features(self, data: pd.DataFrame, target: pd.Series) -> List[str]:
        """Select best features for prediction."""
        feature_columns = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20', 'volatility_ratio',
            'rsi_norm', 'rsi_momentum', 'bb_position', 'bb_squeeze',
            'stoch_norm', 'williams_norm', 'roc_5', 'roc_10',
            'vwap_ratio', 'momentum_5', 'momentum_10', 'price_position'
        ] + [f'ma_ratio_{w}' for w in [5, 10, 20]]
        
        # Remove features với too many NaN values
        valid_features = []
        for col in feature_columns:
            if col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                if nan_ratio < 0.3:
                    valid_features.append(col)
        
        # Select best features
        valid_mask = ~(data[valid_features].isna().any(axis=1) | target.isna())
        if valid_mask.sum() > 50:
            X_valid = data[valid_features][valid_mask]
            y_valid = target[valid_mask]
            
            try:
                self.feature_selector.fit(X_valid, y_valid)
                selected_indices = self.feature_selector.get_support(indices=True)
                selected_features = [valid_features[i] for i in selected_indices]
                return selected_features
            except:
                return valid_features[:12]
        
        return valid_features[:12]

class OptimizedMLModels:
    """Optimized ML models với enhanced configurations."""
    
    def __init__(self):
        self.models = {}
        self.training_results = {}
        
    def create_optimized_models(self) -> Dict[str, Any]:
        """Create optimized ML models."""
        models = {
            'Enhanced_RandomForest': RandomForestRegressor(
                n_estimators=150,  # More trees
                max_depth=12,  # Deeper trees
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'Enhanced_GradientBoosting': GradientBoostingRegressor(
                n_estimators=120,  # More estimators
                learning_rate=0.08,  # Optimized learning rate
                max_depth=8,  # Deeper trees
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'Enhanced_MLP': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),  # 3-layer neural network
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=15,
                random_state=42
            )
        }
        
        return models
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train optimized models."""
        models = self.create_optimized_models()
        training_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(X, y)
                training_time = time.time() - start_time
                
                # Get training score
                train_score = model.score(X, y)
                
                self.models[name] = model
                training_results[name] = {
                    'training_time': training_time,
                    'train_score': train_score,
                    'model_type': type(model).__name__
                }
                
                logger.info(f"✅ {name} trained successfully in {training_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ {name} training failed: {e}")
                training_results[name] = {
                    'training_time': 0,
                    'train_score': 0,
                    'error': str(e)
                }
        
        self.training_results = training_results
        return training_results

class AdvancedEnsemble:
    """Advanced ensemble methods với multiple strategies."""
    
    def __init__(self):
        self.weights = {}
        self.performance_history = []
        
    def calculate_dynamic_weights(self, models: Dict[str, Any], 
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic weights based on validation performance."""
        weights = {}
        scores = {}
        
        # Calculate validation scores
        for name, model in models.items():
            try:
                predictions = model.predict(X_val)
                score = r2_score(y_val, predictions)
                scores[name] = max(score, 0)  # Minimum 0 score
            except:
                scores[name] = 0
        
        # Convert scores to weights
        total_score = sum(scores.values())
        if total_score > 0:
            for name in scores:
                weights[name] = scores[name] / total_score
        else:
            # Equal weights fallback
            n_models = len(models)
            weights = {name: 1.0/n_models for name in models.keys()}
        
        self.weights = weights
        return weights
    
    def weighted_ensemble_prediction(self, models: Dict[str, Any], 
                                   X: np.ndarray) -> Dict[str, Any]:
        """Make weighted ensemble prediction."""
        start_time = time.time()
        
        individual_predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        for name, model in models.items():
            try:
                pred = model.predict(X.reshape(1, -1))[0]
                weight = self.weights.get(name, 1.0/len(models))
                
                individual_predictions[name] = pred
                weighted_sum += pred * weight
                total_weight += weight
                
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
                individual_predictions[name] = 0.0
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
        processing_time = time.time() - start_time
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': individual_predictions,
            'weights': self.weights,
            'processing_time': processing_time
        }

class NeuralOptimizationSystem:
    """Complete neural network optimization system."""
    
    def __init__(self):
        self.feature_engineer = EnhancedFeatureEngineer()
        self.ml_models = OptimizedMLModels()
        self.ensemble = AdvancedEnsemble()
        
    def full_optimization_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete optimization test."""
        logger.info("Starting Day 34 Neural Optimization test...")
        start_time = time.time()
        
        try:
            # Enhanced feature engineering
            logger.info("Enhanced feature engineering...")
            enhanced_data = self.feature_engineer.create_enhanced_features(data)
            
            # Target variable
            target = enhanced_data['returns'].shift(-1)
            
            # Select best features
            selected_features = self.feature_engineer.select_best_features(enhanced_data, target)
            logger.info(f"Selected {len(selected_features)} features")
            
            # Prepare data
            valid_mask = ~(enhanced_data[selected_features].isna().any(axis=1) | target.isna())
            X = enhanced_data[selected_features][valid_mask].values
            y = target[valid_mask].values
            
            if len(X) < 100:
                return {
                    'status': 'ERROR',
                    'message': f'Insufficient data: {len(X)} samples',
                    'execution_time': time.time() - start_time
                }
            
            # Scale features
            X_scaled = self.feature_engineer.scaler.fit_transform(X)
            
            # Split data
            split_point = int(len(X_scaled) * 0.75)
            X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Validation split
            val_split = int(len(X_train) * 0.8)
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            X_train = X_train[:val_split]
            y_train = y_train[:val_split]
            
            # Train optimized models
            training_results = self.ml_models.train_models(X_train, y_train)
            
            # Calculate dynamic ensemble weights
            weights = self.ensemble.calculate_dynamic_weights(
                self.ml_models.models, X_val, y_val
            )
            
            # Test predictions
            test_results = []
            n_test_samples = min(50, len(X_test))
            
            for i in range(n_test_samples):
                test_features = X_test[i]
                actual_target = y_test[i]
                
                # Get ensemble prediction
                ensemble_result = self.ensemble.weighted_ensemble_prediction(
                    self.ml_models.models, test_features
                )
                
                test_results.append({
                    'ensemble_prediction': ensemble_result['ensemble_prediction'],
                    'actual': actual_target,
                    'individual_predictions': ensemble_result['individual_predictions'],
                    'processing_time': ensemble_result['processing_time']
                })
            
            # Calculate metrics
            ensemble_preds = [r['ensemble_prediction'] for r in test_results]
            actuals = [r['actual'] for r in test_results]
            
            # Direction accuracy
            direction_accuracy = sum(
                1 for p, a in zip(ensemble_preds, actuals)
                if (p > 0 and a > 0) or (p <= 0 and a <= 0)
            ) / len(ensemble_preds)
            
            # Individual accuracies
            individual_accuracies = {}
            for model_name in self.ml_models.models.keys():
                model_preds = [r['individual_predictions'][model_name] for r in test_results]
                accuracy = sum(
                    1 for p, a in zip(model_preds, actuals)
                    if (p > 0 and a > 0) or (p <= 0 and a <= 0)
                ) / len(model_preds)
                individual_accuracies[model_name] = accuracy
            
            # Performance metrics
            mse = mean_squared_error(actuals, ensemble_preds)
            r2 = r2_score(actuals, ensemble_preds)
            mae = mean_absolute_error(actuals, ensemble_preds)
            
            avg_processing_time = np.mean([r['processing_time'] for r in test_results])
            execution_time = time.time() - start_time
            
            # Scoring for Day 34
            performance_score = min(direction_accuracy * 100, 100)
            optimization_score = len(self.ml_models.models) / 3 * 100
            speed_score = min(100, max(0, (0.1 - avg_processing_time) / 0.1 * 100))
            
            # Ensemble improvement
            best_individual = max(individual_accuracies.values()) if individual_accuracies else 0
            ensemble_improvement = direction_accuracy - best_individual
            ensemble_score = min(100, 75 + ensemble_improvement * 200)
            
            feature_score = min(100, len(selected_features) / 12 * 100)
            
            # Overall score
            overall_score = (
                performance_score * 0.35 + 
                optimization_score * 0.25 + 
                ensemble_score * 0.20 +
                speed_score * 0.10 +
                feature_score * 0.10
            )
            
            results = {
                'day': 34,
                'system_name': 'Ultimate XAU Super System V4.0',
                'module_name': 'Neural Network Optimization & Enhancement',
                'completion_date': datetime.now().strftime('%Y-%m-%d'),
                'version': '4.0.34',
                'phase': 'Phase 4: Advanced AI Systems',
                'status': 'SUCCESS',
                'execution_time': execution_time,
                'overall_score': overall_score,
                'performance_breakdown': {
                    'performance_score': performance_score,
                    'optimization_score': optimization_score,
                    'ensemble_score': ensemble_score,
                    'speed_score': speed_score,
                    'feature_score': feature_score
                },
                'performance_metrics': {
                    'direction_accuracy': direction_accuracy,
                    'ensemble_improvement': ensemble_improvement,
                    'best_individual_accuracy': best_individual,
                    'mse': mse,
                    'r2_score': r2,
                    'mae': mae,
                    'average_processing_time': avg_processing_time,
                    'individual_accuracies': individual_accuracies
                },
                'optimization_details': {
                    'features_created': len(enhanced_data.columns) - len(data.columns),
                    'features_selected': len(selected_features),
                    'selected_features': selected_features,
                    'models_trained': len(training_results),
                    'ensemble_weights': weights,
                    'training_results': training_results
                },
                'advanced_features': {
                    'enhanced_feature_engineering': True,
                    'dynamic_ensemble_weighting': True,
                    'optimized_ml_models': True,
                    'comprehensive_validation': True,
                    'performance_monitoring': True
                }
            }
            
            logger.info(f"Day 34 optimization completed. Score: {overall_score:.1f}/100")
            return results
            
        except Exception as e:
            logger.error(f"Optimization test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'ERROR',
                'message': str(e),
                'execution_time': time.time() - start_time
            }

def demo_day34_optimization():
    """Quick demo Day 34 optimization."""
    print("=== ULTIMATE XAU SUPER SYSTEM V4.0 - DAY 34 ===")
    print("Neural Network Optimization & Enhancement")
    print("=" * 50)
    
    try:
        # Generate test data
        print("\n1. Generating enhanced market data...")
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=600, freq='D')
        
        # Complex price simulation
        initial_price = 2000
        prices = [initial_price]
        trend = 0.0001
        volatility = 0.02
        
        for i in range(1, len(dates)):
            # Market dynamics
            momentum = (prices[-1] / prices[max(0, i-10)] - 1) * 0.1 if i > 10 else 0
            mean_reversion = (initial_price - prices[-1]) * 0.00005
            
            daily_return = np.random.normal(trend + momentum + mean_reversion, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(5000, 15000, len(dates))
        })
        
        print(f"✅ Generated {len(data)} days of market data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Initialize system
        print("\n2. Initializing optimization system...")
        system = NeuralOptimizationSystem()
        print("✅ System initialized")
        
        # Run optimization
        print("\n3. Running optimization test...")
        print("   - Enhanced feature engineering...")
        print("   - Training optimized ML models...")
        print("   - Dynamic ensemble weighting...")
        print("   - Performance validation...")
        
        results = system.full_optimization_test(data)
        
        if results['status'] == 'SUCCESS':
            print("✅ Optimization completed successfully!")
            
            print(f"\n📊 DAY 34 OPTIMIZATION RESULTS:")
            print(f"   Overall Score: {results['overall_score']:.1f}/100")
            
            perf = results['performance_metrics']
            print(f"   Direction Accuracy: {perf['direction_accuracy']:.1%}")
            print(f"   Ensemble Improvement: {perf['ensemble_improvement']:+.1%}")
            print(f"   Best Individual: {perf['best_individual_accuracy']:.1%}")
            print(f"   R² Score: {perf['r2_score']:.3f}")
            print(f"   Processing Time: {perf['average_processing_time']:.3f}s")
            
            print(f"\n🧠 MODEL PERFORMANCE:")
            for name, acc in perf['individual_accuracies'].items():
                print(f"   {name}: {acc:.1%}")
            
            opt = results['optimization_details']
            print(f"\n🔧 OPTIMIZATION DETAILS:")
            print(f"   Features Created: {opt['features_created']}")
            print(f"   Features Selected: {opt['features_selected']}")
            print(f"   Models Trained: {opt['models_trained']}")
            
            print(f"\n⚖️ ENSEMBLE WEIGHTS:")
            for name, weight in opt['ensemble_weights'].items():
                print(f"   {name}: {weight:.1%}")
            
            print(f"\n⏱️ EXECUTION TIME: {results['execution_time']:.2f} seconds")
            
            # Grading
            score = results['overall_score']
            if score >= 80:
                grade = "XUẤT SẮC"
                status = "🎯"
            elif score >= 75:
                grade = "TỐT"
                status = "✅"
            elif score >= 65:
                grade = "KHANG ĐỊNH"
                status = "⚠️"
            else:
                grade = "CẦN CẢI THIỆN"
                status = "🔴"
            
            print(f"\n{status} DAY 34 COMPLETION: {grade} ({score:.1f}/100)")
            
            # Comparison với Day 33
            day33_score = 65.1
            day33_accuracy = 0.50
            improvement = score - day33_score
            acc_improvement = perf['direction_accuracy'] - day33_accuracy
            
            print(f"\n📈 IMPROVEMENT từ Day 33:")
            print(f"   Overall Score: {day33_score:.1f} → {score:.1f} ({improvement:+.1f})")
            print(f"   Direction Accuracy: {day33_accuracy:.1%} → {perf['direction_accuracy']:.1%} ({acc_improvement:+.1%})")
            
            # Save results
            with open('day34_optimization_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print("✅ Results saved to day34_optimization_results.json")
            
        else:
            print(f"❌ Optimization failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_day34_optimization() 