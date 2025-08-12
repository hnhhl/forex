#!/usr/bin/env python3
"""
Ultimate XAU Super System V4.0 - Day 36: Ensemble Optimization & Simplification
Fix Day 35 over-complexity issues và optimize ensemble performance.

Author: AI Assistant
Date: 2024-12-20
Version: 4.0.36
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Optimized model selection (top performers from Day 35)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# Advanced ensemble methods
from sklearn.ensemble import BaggingRegressor, StackingRegressor

# Try advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFeatureEngineer:
    """Simplified và optimized feature engineering dựa trên Day 35 insights."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_regression, k=10)  # Reduced to 10 best features
        
    def create_optimized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Tạo optimized feature set focusing on most effective indicators."""
        data = data.copy()
        
        # Core price features (proven effective)
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages (focus on most effective periods)
        for window in [10, 20]:  # Reduced from 4 to 2 MAs
            ma = data['close'].rolling(window).mean()
            data[f'ma_ratio_{window}'] = data['close'] / (ma + 1e-8)
            data[f'ma_distance_{window}'] = (data['close'] - ma) / (ma + 1e-8)
        
        # Volatility (key indicator)
        data['volatility_10'] = data['returns'].rolling(10).std()
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_ratio'] = data['volatility_10'] / (data['volatility_20'] + 1e-8)
        
        # RSI (simplified)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_norm'] = (data['rsi'] - 50) / 50
        
        # Bollinger Bands (simplified)
        bb_window = 20
        bb_std = data['close'].rolling(bb_window).std()
        bb_mean = data['close'].rolling(bb_window).mean()
        bb_upper = bb_mean + (bb_std * 2)
        bb_lower = bb_mean - (bb_std * 2)
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Momentum (key indicators)
        data['roc_10'] = ((data['close'] - data['close'].shift(10)) / 
                         (data['close'].shift(10) + 1e-8)) * 100
        data['momentum_10'] = data['close'] / (data['close'].shift(10) + 1e-8)
        
        # Price position
        rolling_min = data['close'].rolling(20).min()
        rolling_max = data['close'].rolling(20).max()
        data['price_position'] = (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        return data
    
    def select_best_features(self, data: pd.DataFrame, target: pd.Series) -> List[str]:
        """Select top 10 most predictive features."""
        feature_columns = [
            'returns', 'log_returns', 'volatility_10', 'volatility_20', 'volatility_ratio',
            'rsi_norm', 'bb_position', 'roc_10', 'momentum_10', 'price_position'
        ] + [f'ma_ratio_{w}' for w in [10, 20]] + [f'ma_distance_{w}' for w in [10, 20]]
        
        # Filter valid features
        valid_features = []
        for col in feature_columns:
            if col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                if nan_ratio < 0.2:  # Stricter NaN tolerance
                    valid_features.append(col)
        
        # Feature selection
        valid_mask = ~(data[valid_features].isna().any(axis=1) | target.isna())
        if valid_mask.sum() > 50:
            X_valid = data[valid_features][valid_mask]
            y_valid = target[valid_mask]
            
            try:
                # Use both statistical và mutual information
                f_selector = SelectKBest(f_regression, k=min(8, len(valid_features)))
                mi_selector = SelectKBest(mutual_info_regression, k=min(8, len(valid_features)))
                
                f_selector.fit(X_valid, y_valid)
                mi_selector.fit(X_valid, y_valid)
                
                # Combine selections
                f_features = [valid_features[i] for i in f_selector.get_support(indices=True)]
                mi_features = [valid_features[i] for i in mi_selector.get_support(indices=True)]
                
                # Union of both selections
                selected_features = list(set(f_features + mi_features))
                return selected_features[:10]  # Top 10
                
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")
                return valid_features[:10]
        
        return valid_features[:10]

class OptimizedModelSelector:
    """Select optimized model portfolio based on Day 35 analysis."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def create_optimized_models(self) -> Dict[str, Any]:
        """Create optimized model portfolio based on Day 35 performance analysis."""
        models = {}
        
        # Ridge - Top performer (72% accuracy)
        models['Ridge_Optimized'] = Ridge(
            alpha=0.5,  # Optimized regularization
            random_state=self.random_state
        )
        
        # DecisionTree - Good performer (58% accuracy)
        models['DecisionTree_Optimized'] = DecisionTreeRegressor(
            max_depth=8,  # Reduced from 10 to prevent overfitting
            min_samples_split=10,  # Increased for stability
            min_samples_leaf=5,
            random_state=self.random_state
        )
        
        # MLP_Small - Stable neural network (54% accuracy)
        models['MLP_Optimized'] = MLPRegressor(
            hidden_layer_sizes=(40,),  # Simplified architecture
            activation='relu',
            solver='adam',
            alpha=0.01,  # Increased regularization
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=self.random_state
        )
        
        # RandomForest - Reliable ensemble (52% accuracy)
        models['RandomForest_Optimized'] = RandomForestRegressor(
            n_estimators=80,  # Reduced from 100
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost - If available (50% accuracy, potential for improvement)
        if XGBOOST_AVAILABLE:
            models['XGBoost_Optimized'] = xgb.XGBRegressor(
                n_estimators=60,  # Reduced
                learning_rate=0.05,  # Reduced learning rate
                max_depth=4,  # Reduced depth
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1,  # L2 regularization
                random_state=self.random_state,
                n_jobs=-1
            )
        
        return models

class OptimizedEnsemble:
    """Simplified và optimized ensemble methods."""
    
    def __init__(self, base_models: Dict[str, Any], random_state: int = 42):
        self.base_models = base_models
        self.random_state = random_state
        self.trained_models = {}
        self.ensemble_weights = {}
        
    def train_models_with_validation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train models với validation-based performance tracking."""
        logger.info("Training optimized model portfolio...")
        
        # Split for validation
        split_point = int(len(X) * 0.8)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]
        
        model_performances = {}
        training_results = {}
        
        for name, model in self.base_models.items():
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Validate performance
                val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                val_direction_acc = sum(
                    1 for p, a in zip(val_pred, y_val)
                    if (p > 0 and a > 0) or (p <= 0 and a <= 0)
                ) / len(val_pred)
                
                # Training metrics
                train_pred = model.predict(X_train)
                train_r2 = r2_score(y_train, train_pred)
                
                self.trained_models[name] = model
                model_performances[name] = {
                    'val_r2': val_r2,
                    'val_direction_acc': val_direction_acc,
                    'train_r2': train_r2,
                    'training_time': time.time() - start_time
                }
                
                training_results[name] = {
                    'status': 'success',
                    'val_r2': val_r2,
                    'direction_acc': val_direction_acc,
                    'training_time': time.time() - start_time
                }
                
                logger.info(f"✅ {name}: R²={val_r2:.3f}, Acc={val_direction_acc:.1%}")
                
            except Exception as e:
                logger.error(f"❌ {name} training failed: {e}")
                training_results[name] = {
                    'status': 'failed',
                    'error': str(e),
                    'training_time': time.time() - start_time
                }
        
        # Calculate ensemble weights based on validation performance
        self._calculate_optimized_weights(model_performances)
        
        return training_results
    
    def _calculate_optimized_weights(self, performances: Dict[str, Dict]) -> None:
        """Calculate optimized ensemble weights based on multiple criteria."""
        weights = {}
        
        for name, perf in performances.items():
            # Base weight from direction accuracy
            acc_weight = max(0, perf['val_direction_acc'] - 0.4)  # Baseline 40%
            
            # R² score weight (positive R² gets bonus)
            r2_weight = max(0, perf['val_r2'])
            
            # Stability weight (train vs val performance similarity)
            stability = 1 - abs(perf['train_r2'] - perf['val_r2'])
            stability_weight = max(0, stability)
            
            # Combined weight
            combined_weight = (acc_weight * 0.5 + r2_weight * 0.3 + stability_weight * 0.2)
            weights[name] = combined_weight
        
        # Normalize weights với minimum threshold
        total_weight = sum(weights.values())
        if total_weight > 0:
            # Normalize
            weights = {name: w / total_weight for name, w in weights.items()}
            
            # Apply minimum contribution (prevent single model dominance)
            min_weight = 0.05  # 5% minimum per model
            max_weight = 0.5   # 50% maximum per model
            
            for name in weights:
                if weights[name] < min_weight:
                    weights[name] = min_weight
                elif weights[name] > max_weight:
                    weights[name] = max_weight
            
            # Re-normalize
            total_weight = sum(weights.values())
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Equal weights fallback
            n_models = len(performances)
            weights = {name: 1.0/n_models for name in performances.keys()}
        
        self.ensemble_weights = weights
        logger.info(f"Ensemble weights: {weights}")
    
    def create_sklearn_ensemble(self) -> Any:
        """Create sklearn-based stacking ensemble."""
        if len(self.trained_models) < 2:
            return None
        
        try:
            # Base models for stacking
            estimators = [(name, model) for name, model in self.trained_models.items()]
            
            # Meta-learner (use Ridge as it performed well)
            meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
            
            # Create stacking regressor
            stacking_ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,  # Reduced CV splits
                n_jobs=-1
            )
            
            return stacking_ensemble
            
        except Exception as e:
            logger.error(f"Sklearn ensemble creation failed: {e}")
            return None
    
    def predict_optimized_ensemble(self, X: np.ndarray) -> Dict[str, Any]:
        """Make optimized ensemble prediction."""
        start_time = time.time()
        
        # Individual predictions
        individual_predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X.reshape(1, -1))[0]
                weight = self.ensemble_weights.get(name, 0.2)  # Default 20%
                
                individual_predictions[name] = pred
                weighted_sum += pred * weight
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                individual_predictions[name] = 0.0
        
        # Weighted ensemble prediction
        weighted_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Simple average for comparison
        individual_values = [p for p in individual_predictions.values() if p != 0]
        simple_prediction = np.mean(individual_values) if individual_values else 0.0
        
        processing_time = time.time() - start_time
        
        return {
            'weighted_prediction': weighted_prediction,
            'simple_prediction': simple_prediction,
            'individual_predictions': individual_predictions,
            'ensemble_weights': self.ensemble_weights,
            'processing_time': processing_time
        }

class EnsembleOptimizationSystem:
    """Day 36: Optimized ensemble system focusing on simplicity và effectiveness."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = OptimizedFeatureEngineer()
        self.model_selector = OptimizedModelSelector(random_state)
        self.ensemble = None
        self.sklearn_ensemble = None
        
    def full_optimization_test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete optimization test."""
        logger.info("Starting Day 36 Ensemble Optimization test...")
        start_time = time.time()
        
        try:
            # Optimized feature engineering
            logger.info("Optimized feature engineering...")
            enhanced_data = self.feature_engineer.create_optimized_features(data)
            
            # Target variable
            target = enhanced_data['returns'].shift(-1)
            
            # Select best features
            selected_features = self.feature_engineer.select_best_features(enhanced_data, target)
            logger.info(f"Selected {len(selected_features)} optimized features")
            
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
            split_point = int(len(X_scaled) * 0.7)
            X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Create optimized models
            base_models = self.model_selector.create_optimized_models()
            logger.info(f"Created {len(base_models)} optimized models")
            
            # Train ensemble
            self.ensemble = OptimizedEnsemble(base_models, self.random_state)
            training_results = self.ensemble.train_models_with_validation(X_train, y_train)
            
            # Create sklearn stacking ensemble
            self.sklearn_ensemble = self.ensemble.create_sklearn_ensemble()
            if self.sklearn_ensemble:
                logger.info("Training sklearn stacking ensemble...")
                self.sklearn_ensemble.fit(X_train, y_train)
            
            # Test predictions
            test_results = []
            n_test_samples = min(50, len(X_test))
            
            for i in range(n_test_samples):
                test_features = X_test[i]
                actual_target = y_test[i]
                
                # Get ensemble predictions
                ensemble_result = self.ensemble.predict_optimized_ensemble(test_features)
                
                # Sklearn stacking prediction
                sklearn_pred = 0.0
                if self.sklearn_ensemble:
                    try:
                        sklearn_pred = self.sklearn_ensemble.predict(test_features.reshape(1, -1))[0]
                    except:
                        sklearn_pred = ensemble_result['weighted_prediction']
                
                test_results.append({
                    'weighted_prediction': ensemble_result['weighted_prediction'],
                    'simple_prediction': ensemble_result['simple_prediction'],
                    'sklearn_prediction': sklearn_pred,
                    'actual': actual_target,
                    'individual_predictions': ensemble_result['individual_predictions'],
                    'processing_time': ensemble_result['processing_time']
                })
            
            # Calculate metrics
            weighted_preds = [r['weighted_prediction'] for r in test_results]
            simple_preds = [r['simple_prediction'] for r in test_results]
            sklearn_preds = [r['sklearn_prediction'] for r in test_results]
            actuals = [r['actual'] for r in test_results]
            
            # Direction accuracies
            weighted_direction_acc = sum(
                1 for p, a in zip(weighted_preds, actuals)
                if (p > 0 and a > 0) or (p <= 0 and a <= 0)
            ) / len(weighted_preds)
            
            simple_direction_acc = sum(
                1 for p, a in zip(simple_preds, actuals)
                if (p > 0 and a > 0) or (p <= 0 and a <= 0)
            ) / len(simple_preds)
            
            sklearn_direction_acc = sum(
                1 for p, a in zip(sklearn_preds, actuals)
                if (p > 0 and a > 0) or (p <= 0 and a <= 0)
            ) / len(sklearn_preds)
            
            # Individual accuracies
            individual_accuracies = {}
            successful_models = [name for name, result in training_results.items() 
                               if result['status'] == 'success']
            
            for model_name in successful_models:
                model_preds = [r['individual_predictions'].get(model_name, 0) for r in test_results]
                accuracy = sum(
                    1 for p, a in zip(model_preds, actuals)
                    if (p > 0 and a > 0) or (p <= 0 and a <= 0)
                ) / len(model_preds)
                individual_accuracies[model_name] = accuracy
            
            # Performance metrics
            weighted_mse = mean_squared_error(actuals, weighted_preds)
            weighted_r2 = r2_score(actuals, weighted_preds)
            sklearn_r2 = r2_score(actuals, sklearn_preds)
            
            avg_processing_time = np.mean([r['processing_time'] for r in test_results])
            execution_time = time.time() - start_time
            
            # Calculate improvements
            best_individual = max(individual_accuracies.values()) if individual_accuracies else 0
            weighted_improvement = weighted_direction_acc - best_individual
            sklearn_improvement = sklearn_direction_acc - best_individual
            
            # Enhanced scoring for Day 36
            best_accuracy = max(weighted_direction_acc, sklearn_direction_acc)
            performance_score = min(best_accuracy * 100, 100)
            
            ensemble_improvement = max(weighted_improvement, sklearn_improvement)
            ensemble_score = min(100, 75 + ensemble_improvement * 400)  # Bonus for improvement
            
            model_quality_score = len(successful_models) / len(base_models) * 100
            speed_score = min(100, max(0, (0.05 - avg_processing_time) / 0.05 * 100))
            feature_score = min(100, len(selected_features) / 10 * 100)
            
            # Overall score
            overall_score = (
                performance_score * 0.40 + 
                ensemble_score * 0.25 + 
                model_quality_score * 0.15 +
                speed_score * 0.10 +
                feature_score * 0.10
            )
            
            results = {
                'day': 36,
                'system_name': 'Ultimate XAU Super System V4.0',
                'module_name': 'Ensemble Optimization & Simplification',
                'completion_date': datetime.now().strftime('%Y-%m-%d'),
                'version': '4.0.36',
                'phase': 'Phase 4: Advanced AI Systems',
                'status': 'SUCCESS',
                'execution_time': execution_time,
                'overall_score': overall_score,
                'performance_breakdown': {
                    'performance_score': performance_score,
                    'ensemble_score': ensemble_score,
                    'model_quality_score': model_quality_score,
                    'speed_score': speed_score,
                    'feature_score': feature_score
                },
                'performance_metrics': {
                    'weighted_direction_accuracy': weighted_direction_acc,
                    'simple_direction_accuracy': simple_direction_acc,
                    'sklearn_direction_accuracy': sklearn_direction_acc,
                    'best_ensemble_accuracy': best_accuracy,
                    'weighted_improvement': weighted_improvement,
                    'sklearn_improvement': sklearn_improvement,
                    'best_individual_accuracy': best_individual,
                    'weighted_r2': weighted_r2,
                    'sklearn_r2': sklearn_r2,
                    'weighted_mse': weighted_mse,
                    'average_processing_time': avg_processing_time,
                    'individual_accuracies': individual_accuracies
                },
                'optimization_details': {
                    'models_attempted': len(base_models),
                    'models_successful': len(successful_models),
                    'features_selected': len(selected_features),
                    'selected_features': selected_features,
                    'ensemble_weights': self.ensemble.ensemble_weights if self.ensemble else {},
                    'sklearn_ensemble_available': self.sklearn_ensemble is not None,
                    'training_results': training_results
                },
                'advanced_features': {
                    'optimized_model_selection': True,
                    'feature_optimization': True,
                    'validation_based_weighting': True,
                    'sklearn_stacking': self.sklearn_ensemble is not None,
                    'performance_constraints': True,
                    'weight_normalization': True
                }
            }
            
            logger.info(f"Day 36 ensemble optimization completed. Score: {overall_score:.1f}/100")
            return results
            
        except Exception as e:
            logger.error(f"Ensemble optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'ERROR',
                'message': str(e),
                'execution_time': time.time() - start_time
            }

def demo_ensemble_optimization():
    """Demo Day 36 ensemble optimization."""
    print("=== ULTIMATE XAU SUPER SYSTEM V4.0 - DAY 36 ===")
    print("Ensemble Optimization & Simplification")
    print("=" * 50)
    
    try:
        print(f"XGBoost available: {'✅' if XGBOOST_AVAILABLE else '❌'}")
        
        # Generate test data
        print("\n1. Generating optimized test data...")
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='D')
        
        # Simplified market simulation focusing on patterns
        initial_price = 2000
        prices = [initial_price]
        trend_strength = 0.0002
        
        for i in range(1, len(dates)):
            # Add trend với some noise
            trend = trend_strength * (1 + 0.1 * np.sin(i / 50))  # Cyclical trend
            volatility = 0.012 + 0.005 * abs(np.sin(i / 30))  # Variable volatility
            
            # Mean reversion component
            mean_reversion = (initial_price - prices[-1]) * 0.00001
            
            daily_return = np.random.normal(trend + mean_reversion, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(8000, 15000, len(dates))
        })
        
        print(f"✅ Generated {len(data)} days of optimized market data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"   Return volatility: {data['close'].pct_change().std():.4f}")
        
        # Initialize system
        print("\n2. Initializing Ensemble Optimization System...")
        system = EnsembleOptimizationSystem(random_state=42)
        print("✅ Optimization system initialized")
        
        # Run optimization test
        print("\n3. Running ensemble optimization test...")
        print("   - Optimized feature engineering (10 best features)...")
        print("   - Training 5 optimized models...")
        print("   - Validation-based ensemble weighting...")
        print("   - Sklearn stacking ensemble...")
        
        results = system.full_optimization_test(data)
        
        if results['status'] == 'SUCCESS':
            print("✅ Ensemble optimization completed!")
            
            print(f"\n📊 DAY 36 OPTIMIZATION RESULTS:")
            print(f"   Overall Score: {results['overall_score']:.1f}/100")
            
            perf = results['performance_metrics']
            print(f"   Best Ensemble Accuracy: {perf['best_ensemble_accuracy']:.1%}")
            print(f"   Weighted Ensemble: {perf['weighted_direction_accuracy']:.1%}")
            print(f"   Sklearn Stacking: {perf['sklearn_direction_accuracy']:.1%}")
            print(f"   Simple Average: {perf['simple_direction_accuracy']:.1%}")
            print(f"   Best Individual: {perf['best_individual_accuracy']:.1%}")
            print(f"   Ensemble Improvement: {max(perf['weighted_improvement'], perf['sklearn_improvement']):+.1%}")
            print(f"   Processing Time: {perf['average_processing_time']:.3f}s")
            
            print(f"\n🧠 OPTIMIZED MODEL PERFORMANCE:")
            for name, acc in perf['individual_accuracies'].items():
                print(f"   {name}: {acc:.1%}")
            
            opt = results['optimization_details']
            print(f"\n🔧 OPTIMIZATION DETAILS:")
            print(f"   Models Successful: {opt['models_successful']}/{opt['models_attempted']}")
            print(f"   Features Selected: {opt['features_selected']}")
            print(f"   Sklearn Stacking: {'✅' if opt['sklearn_ensemble_available'] else '❌'}")
            
            if opt['ensemble_weights']:
                print(f"\n⚖️ ENSEMBLE WEIGHTS:")
                for name, weight in opt['ensemble_weights'].items():
                    print(f"   {name}: {weight:.1%}")
            
            print(f"\n⏱️ EXECUTION TIME: {results['execution_time']:.2f} seconds")
            
            # Grading
            score = results['overall_score']
            if score >= 80:
                grade = "XUẤT SẮC"
                status = "🎯"
                message = "Ensemble optimization breakthrough!"
            elif score >= 75:
                grade = "TỐT"
                status = "✅"
                message = "Strong optimization achieved"
            elif score >= 65:
                grade = "KHANG ĐỊNH"
                status = "⚠️"
                message = "Optimization working well"
            else:
                grade = "CẦN CẢI THIỆN"
                status = "🔴"
                message = "Further optimization needed"
            
            print(f"\n{status} DAY 36 COMPLETION: {grade} ({score:.1f}/100)")
            print(f"   {message}")
            
            # Progress tracking
            day35_score = 41.7
            day34_score = 71.1
            
            print(f"\n📈 PROGRESS RECOVERY:")
            print(f"   Day 34: {day34_score:.1f}/100")
            print(f"   Day 35: {day35_score:.1f}/100 (regression)")
            print(f"   Day 36: {score:.1f}/100 ({score-day35_score:+.1f} recovery)")
            
            recovery = score - day35_score
            if recovery > 0:
                print(f"   ✅ Successfully recovered {recovery:.1f} points!")
            
            # Save results
            with open('day36_ensemble_optimization_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print("✅ Results saved to day36_ensemble_optimization_results.json")
            
        else:
            print(f"❌ Optimization failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_ensemble_optimization() 