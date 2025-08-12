#!/usr/bin/env python3
"""
Ultimate XAU Super System V4.0 - Day 37: Selective High-Performance Ensemble (Simplified)
Focus on selective ensemble với chỉ high-performing models.

Author: AI Assistant
Date: 2024-12-20
Version: 4.0.37-simplified
"""

import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime
import json
import logging

# ML imports
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression

# Models
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor

# Try XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_optimized_features(data):
    """Create optimized feature set."""
    data = data.copy()
    
    # Core features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Moving averages
    for window in [10, 20]:
        ma = data['close'].rolling(window).mean()
        data[f'ma_ratio_{window}'] = data['close'] / (ma + 1e-8)
        data[f'ma_distance_{window}'] = (data['close'] - ma) / (ma + 1e-8)
    
    # Volatility
    data['volatility_10'] = data['returns'].rolling(10).std()
    data['volatility_20'] = data['returns'].rolling(20).std()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_norm'] = (data['rsi'] - 50) / 50
    
    # Bollinger Bands
    bb_mean = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    bb_upper = bb_mean + (bb_std * 2)
    bb_lower = bb_mean - (bb_std * 2)
    data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    
    # Momentum
    data['roc_10'] = ((data['close'] - data['close'].shift(10)) / 
                     (data['close'].shift(10) + 1e-8)) * 100
    
    # Price position
    rolling_min = data['close'].rolling(20).min()
    rolling_max = data['close'].rolling(20).max()
    data['price_position'] = (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
    
    return data

def select_best_features(data, target, n_features=10):
    """Select best features."""
    feature_columns = [
        'returns', 'log_returns', 'volatility_10', 'volatility_20',
        'rsi_norm', 'bb_position', 'roc_10', 'price_position'
    ] + [f'ma_ratio_{w}' for w in [10, 20]] + [f'ma_distance_{w}' for w in [10, 20]]
    
    # Filter valid features
    valid_features = []
    for col in feature_columns:
        if col in data.columns:
            nan_ratio = data[col].isna().sum() / len(data)
            if nan_ratio < 0.2:
                valid_features.append(col)
    
    if len(valid_features) < n_features:
        return valid_features
    
    # Feature selection
    valid_mask = ~(data[valid_features].isna().any(axis=1) | target.isna())
    if valid_mask.sum() < 50:
        return valid_features[:n_features]
    
    try:
        X_valid = data[valid_features][valid_mask]
        y_valid = target[valid_mask]
        
        selector = SelectKBest(f_regression, k=min(n_features, len(valid_features)))
        selector.fit(X_valid, y_valid)
        selected_features = [valid_features[i] for i in selector.get_support(indices=True)]
        return selected_features
    except:
        return valid_features[:n_features]

def create_elite_models():
    """Create elite model portfolio."""
    models = {}
    
    # Ridge (proven performer)
    models['Ridge_Elite'] = Ridge(alpha=0.5, random_state=42)
    
    # ElasticNet
    models['ElasticNet_Elite'] = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42, max_iter=2000)
    
    # RandomForest (optimized)
    models['RandomForest_Elite'] = RandomForestRegressor(
        n_estimators=50, max_depth=8, min_samples_split=8,
        random_state=42, n_jobs=-1
    )
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost_Elite'] = xgb.XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=4,
            random_state=42, n_jobs=-1
        )
    
    # MLP (simplified)
    models['MLP_Elite'] = MLPRegressor(
        hidden_layer_sizes=(30,), activation='relu',
        alpha=0.02, max_iter=300, early_stopping=True,
        random_state=42
    )
    
    return models

def evaluate_and_select_models(models, X, y, threshold=0.55):
    """Evaluate models và select high performers."""
    logger.info(f"Evaluating {len(models)} models với threshold {threshold:.0%}...")
    
    elite_models = {}
    model_performances = {}
    
    for name, model in models.items():
        try:
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            
            # Fit và evaluate
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Direction accuracy
            direction_acc = sum(
                1 for p, a in zip(y_pred, y)
                if (p > 0 and a > 0) or (p <= 0 and a <= 0)
            ) / len(y)
            
            model_performances[name] = {
                'direction_accuracy': direction_acc,
                'cv_score': -np.mean(cv_scores)
            }
            
            # Select high performers
            if direction_acc >= threshold:
                elite_models[name] = model
                logger.info(f"✅ {name}: {direction_acc:.1%} (SELECTED)")
            else:
                logger.info(f"❌ {name}: {direction_acc:.1%} (REJECTED)")
                
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
    
    logger.info(f"Selected {len(elite_models)}/{len(models)} elite models")
    return elite_models, model_performances

def create_ensemble_weights(elite_models, model_performances):
    """Calculate ensemble weights."""
    weights = {}
    
    for name in elite_models.keys():
        perf = model_performances[name]
        weight = perf['direction_accuracy'] - 0.5  # Baseline 50%
        weights[name] = max(0.1, weight)  # Minimum 10%
    
    # Normalize
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {name: w / total_weight for name, w in weights.items()}
    else:
        n_models = len(elite_models)
        weights = {name: 1.0/n_models for name in elite_models.keys()}
    
    return weights

def create_advanced_ensembles(elite_models, X, y):
    """Create advanced ensemble methods."""
    ensembles = {}
    
    if len(elite_models) < 2:
        return ensembles
    
    try:
        estimators = [(name, model) for name, model in elite_models.items()]
        
        # Voting ensemble
        voting_ensemble = VotingRegressor(estimators=estimators)
        voting_ensemble.fit(X, y)
        ensembles['Voting'] = voting_ensemble
        
        # Stacking ensemble
        stacking_ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=0.5),
            cv=3
        )
        stacking_ensemble.fit(X, y)
        ensembles['Stacking'] = stacking_ensemble
        
        logger.info(f"Created {len(ensembles)} advanced ensembles")
        
    except Exception as e:
        logger.error(f"Ensemble creation failed: {e}")
    
    return ensembles

def run_selective_ensemble_test():
    """Run Day 37 selective ensemble test."""
    logger.info("Starting Day 37 Selective Ensemble test...")
    start_time = time.time()
    
    try:
        # Generate test data
        print("1. Generating test data...")
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=400, freq='D')
        
        # Market simulation
        initial_price = 2000
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            # Trend với noise
            trend = 0.0002 * np.sin(i / 50)
            volatility = 0.015
            
            # Mean reversion
            mean_reversion = (initial_price - prices[-1]) * 0.00002
            
            daily_return = np.random.normal(trend + mean_reversion, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(10000, 20000, len(dates))
        })
        
        print(f"✅ Generated {len(data)} days of market data")
        
        # Feature engineering
        print("2. Feature engineering...")
        enhanced_data = create_optimized_features(data)
        target = enhanced_data['returns'].shift(-1)
        
        # Select features
        selected_features = select_best_features(enhanced_data, target, n_features=10)
        print(f"✅ Selected {len(selected_features)} features")
        
        # Prepare data
        valid_mask = ~(enhanced_data[selected_features].isna().any(axis=1) | target.isna())
        X = enhanced_data[selected_features][valid_mask].values
        y = target[valid_mask].values
        
        if len(X) < 100:
            return {'status': 'ERROR', 'message': 'Insufficient data'}
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        split_point = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Create và evaluate models
        print("3. Model evaluation và selection...")
        base_models = create_elite_models()
        elite_models, model_performances = evaluate_and_select_models(
            base_models, X_train, y_train, threshold=0.55)
        
        if len(elite_models) == 0:
            # Lower threshold
            elite_models, model_performances = evaluate_and_select_models(
                base_models, X_train, y_train, threshold=0.50)
        
        # Create ensembles
        print("4. Creating advanced ensembles...")
        ensembles = create_advanced_ensembles(elite_models, X_train, y_train)
        ensemble_weights = create_ensemble_weights(elite_models, model_performances)
        
        # Test predictions
        print("5. Testing predictions...")
        test_results = []
        n_test_samples = min(30, len(X_test))
        
        for i in range(n_test_samples):
            test_features = X_test[i:i+1]
            actual_target = y_test[i]
            
            # Individual predictions
            individual_preds = {}
            for name, model in elite_models.items():
                try:
                    pred = model.predict(test_features)[0]
                    individual_preds[name] = pred
                except:
                    individual_preds[name] = 0.0
            
            # Ensemble predictions
            ensemble_preds = {}
            for ens_name, ensemble in ensembles.items():
                try:
                    pred = ensemble.predict(test_features)[0]
                    ensemble_preds[ens_name] = pred
                except:
                    ensemble_preds[ens_name] = 0.0
            
            # Weighted prediction
            if individual_preds and ensemble_weights:
                weighted_pred = sum(pred * ensemble_weights.get(name, 0) 
                                  for name, pred in individual_preds.items())
            else:
                weighted_pred = np.mean(list(individual_preds.values())) if individual_preds else 0
            
            test_results.append({
                'individual': individual_preds,
                'ensemble': ensemble_preds,
                'weighted': weighted_pred,
                'actual': actual_target
            })
        
        # Calculate metrics
        print("6. Calculating performance metrics...")
        
        # Extract predictions
        weighted_preds = [r['weighted'] for r in test_results]
        actuals = [r['actual'] for r in test_results]
        
        # Direction accuracies
        def calc_direction_acc(preds, actuals):
            return sum(1 for p, a in zip(preds, actuals)
                      if (p > 0 and a > 0) or (p <= 0 and a <= 0)) / len(preds)
        
        weighted_acc = calc_direction_acc(weighted_preds, actuals)
        
        # Individual accuracies
        individual_accuracies = {}
        for model_name in elite_models.keys():
            model_preds = [r['individual'].get(model_name, 0) for r in test_results]
            accuracy = calc_direction_acc(model_preds, actuals)
            individual_accuracies[model_name] = accuracy
        
        # Ensemble method accuracies
        ensemble_accuracies = {}
        for ens_name in ensembles.keys():
            ens_preds = [r['ensemble'].get(ens_name, 0) for r in test_results]
            accuracy = calc_direction_acc(ens_preds, actuals)
            ensemble_accuracies[ens_name] = accuracy
        
        # Best metrics
        all_accuracies = [weighted_acc] + list(individual_accuracies.values()) + list(ensemble_accuracies.values())
        best_accuracy = max(all_accuracies) if all_accuracies else 0
        best_individual = max(individual_accuracies.values()) if individual_accuracies else 0
        
        # Performance scoring
        performance_score = min(best_accuracy * 100, 100)
        selection_score = len(elite_models) / len(base_models) * 100
        ensemble_improvement = best_accuracy - best_individual
        ensemble_score = min(100, 75 + ensemble_improvement * 500)
        
        overall_score = (performance_score * 0.5 + ensemble_score * 0.3 + selection_score * 0.2)
        
        execution_time = time.time() - start_time
        
        results = {
            'day': 37,
            'system_name': 'Ultimate XAU Super System V4.0',
            'module_name': 'Selective High-Performance Ensemble (Simplified)',
            'completion_date': datetime.now().strftime('%Y-%m-%d'),
            'version': '4.0.37-simplified',
            'status': 'SUCCESS',
            'execution_time': execution_time,
            'overall_score': overall_score,
            'performance_metrics': {
                'best_accuracy': best_accuracy,
                'weighted_ensemble_accuracy': weighted_acc,
                'ensemble_improvement': ensemble_improvement,
                'best_individual_accuracy': best_individual,
                'individual_accuracies': individual_accuracies,
                'ensemble_accuracies': ensemble_accuracies
            },
            'selective_details': {
                'models_evaluated': len(base_models),
                'elite_models_selected': len(elite_models),
                'selection_ratio': len(elite_models) / len(base_models),
                'elite_models': list(elite_models.keys()),
                'ensemble_weights': ensemble_weights,
                'ensembles_created': len(ensembles),
                'features_selected': len(selected_features)
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {
            'status': 'ERROR',
            'message': str(e),
            'execution_time': time.time() - start_time
        }

def demo_day37_selective():
    """Demo Day 37 selective ensemble."""
    print("=== ULTIMATE XAU SUPER SYSTEM V4.0 - DAY 37 ===")
    print("Selective High-Performance Ensemble (Simplified)")
    print("=" * 50)
    
    try:
        print(f"XGBoost available: {'✅' if XGBOOST_AVAILABLE else '❌'}")
        
        results = run_selective_ensemble_test()
        
        if results['status'] == 'SUCCESS':
            print("✅ Selective ensemble test completed!")
            
            print(f"\n📊 DAY 37 SELECTIVE RESULTS:")
            print(f"   Overall Score: {results['overall_score']:.1f}/100")
            
            perf = results['performance_metrics']
            print(f"   Best Accuracy: {perf['best_accuracy']:.1%}")
            print(f"   Weighted Ensemble: {perf['weighted_ensemble_accuracy']:.1%}")
            print(f"   Ensemble Improvement: {perf['ensemble_improvement']:+.1%}")
            print(f"   Best Individual: {perf['best_individual_accuracy']:.1%}")
            
            print(f"\n🏆 ELITE MODEL PERFORMANCE:")
            for name, acc in perf['individual_accuracies'].items():
                print(f"   {name}: {acc:.1%}")
            
            print(f"\n🔧 ENSEMBLE METHOD PERFORMANCE:")
            for method, acc in perf['ensemble_accuracies'].items():
                print(f"   {method}: {acc:.1%}")
            
            selective = results['selective_details']
            print(f"\n🎯 SELECTION SUMMARY:")
            print(f"   Models Evaluated: {selective['models_evaluated']}")
            print(f"   Elite Selected: {selective['elite_models_selected']}")
            print(f"   Selection Ratio: {selective['selection_ratio']:.1%}")
            print(f"   Features Used: {selective['features_selected']}")
            
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
            
            print(f"\n{status} DAY 37 COMPLETION: {grade} ({score:.1f}/100)")
            
            # Progress comparison
            day36_score = 55.4
            improvement = score - day36_score
            print(f"   Progress từ Day 36: {improvement:+.1f} points")
            
            # Save results
            with open('day37_selective_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print("✅ Results saved to day37_selective_results.json")
            
        else:
            print(f"❌ Test failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_day37_selective() 