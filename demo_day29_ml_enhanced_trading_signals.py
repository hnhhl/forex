"""
Demo Day 29: ML Enhanced Trading Signals System
Ultimate XAU Super System V4.0 - Comprehensive Machine Learning Trading Signals

Features being demonstrated:
1. AI Signal Generation với 4 ML models
2. Feature Engineering với advanced technical indicators
3. Ensemble Model Predictions với weighted averaging
4. Real-time Signal Generation và confidence scoring
5. Risk-Adjusted Position Sizing và model diagnostics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import warnings

# Import our ML Enhanced Trading Signals components
from src.core.analysis.ml_enhanced_trading_signals import (
    MLEnhancedTradingSignals,
    MLConfig,
    MLModelType,
    FeatureType,
    EnsembleMethod,
    SignalType,
    create_ml_enhanced_trading_signals
)

warnings.filterwarnings('ignore')

class Day29MLDemo:
    """Demo class for ML Enhanced Trading Signals"""
    
    def __init__(self):
        print("=" * 70)
        print("🤖 Day 29: ML Enhanced Trading Signals Demo 🤖")
        print("Ultimate XAU Super System V4.0 - Advanced Machine Learning")
        print("=" * 70)
        
        # Initialize results storage
        self.demo_results = {
            "demo_info": {
                "name": "Day 29 ML Enhanced Trading Signals Demo",
                "version": "4.0.29",
                "timestamp": datetime.now().isoformat(),
                "description": "Comprehensive ML-powered trading signal system"
            },
            "modules": {}
        }
        
        # Generate realistic market data
        self.market_data = self._generate_market_data()
        
    def _generate_market_data(self) -> dict:
        """Generate comprehensive market data for ML training"""
        
        print("\n📊 Generating Market Data...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate 1 year of daily data
        periods = 365
        dates = pd.date_range('2023-01-01', periods=periods, freq='1D')
        
        # Create realistic price movements with trend, volatility clustering
        base_trend = 0.0001  # Small upward trend
        volatility_regimes = np.random.choice([0.01, 0.02, 0.03], periods, p=[0.7, 0.2, 0.1])
        
        returns = []
        for i in range(periods):
            # Add momentum and mean reversion effects
            momentum = np.random.normal(0, 0.3) * (returns[-1] if returns else 0)
            mean_reversion = -0.1 * (sum(returns[-5:]) if len(returns) >= 5 else 0)
            
            daily_return = (base_trend + 
                          momentum + 
                          mean_reversion + 
                          np.random.normal(0, volatility_regimes[i]))
            returns.append(daily_return)
        
        # Generate prices
        prices = [2000.0]  # Starting price for XAU
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create comprehensive market data
        market_data = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices[1:]],
            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices[1:]],
            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices[1:]],
            'close': prices[1:],
            'volume': np.random.lognormal(10, 0.5, periods)
        }, index=dates)
        
        # Ensure OHLC logic (High >= Open,Close; Low <= Open,Close)
        for i in range(len(market_data)):
            o, h, l, c = market_data.iloc[i][['open', 'high', 'low', 'close']]
            market_data.iloc[i, 1] = max(h, o, c)  # high
            market_data.iloc[i, 2] = min(l, o, c)  # low
        
        print(f"✅ Generated {len(market_data)} days of market data")
        print(f"   Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
        print(f"   Average volume: {market_data['volume'].mean():,.0f}")
        
        return market_data
    
    def run_ai_signal_generation_demo(self):
        """Demo 1: AI Signal Generation với multiple ML models"""
        
        print("\n" + "="*50)
        print("🧠 Module 1: AI Signal Generation")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Create ML system with comprehensive configuration
            config_dict = {
                'ml_models': [
                    MLModelType.RANDOM_FOREST,
                    MLModelType.GRADIENT_BOOSTING,
                    MLModelType.LINEAR_REGRESSION,
                    MLModelType.RIDGE_REGRESSION
                ],
                'feature_types': [
                    FeatureType.TECHNICAL_INDICATORS,
                    FeatureType.PRICE_PATTERNS,
                    FeatureType.MOMENTUM_FEATURES,
                    FeatureType.VOLATILITY_FEATURES
                ],
                'prediction_horizon': 5,
                'ensemble_method': EnsembleMethod.WEIGHTED_AVERAGE,
                'lookback_period': 60
            }
            
            ml_system = create_ml_enhanced_trading_signals(config_dict)
            
            # Prepare training data (first 80% of data)
            train_data = self.market_data.iloc[:int(len(self.market_data) * 0.8)]
            price_data = train_data[['close']]
            volume_data = train_data['volume']
            
            # Train the ML system
            print("🔄 Training ML models...")
            performances = ml_system.train_system(price_data, volume_data)
            
            execution_time = time.time() - start_time
            
            # Calculate performance metrics
            model_count = len(performances)
            avg_r2 = np.mean([p.r2_score for p in performances.values()]) if performances else 0
            avg_win_rate = np.mean([p.win_rate for p in performances.values()]) if performances else 0
            feature_count = len(config_dict['feature_types'])
            
            # Performance scoring
            performance_score = min(100, max(0, (
                (model_count / 4) * 25 +  # Model diversity (25 points)
                (max(0, avg_r2) * 100) * 0.3 +  # R² score (30 points)
                (avg_win_rate * 100) * 0.2 +  # Win rate (20 points)
                (min(execution_time, 10) / 10) * 25  # Speed (25 points, faster = better)
            )))
            
            # Store results
            self.demo_results["modules"]["ai_signal_generation"] = {
                "performance_score": round(performance_score, 1),
                "execution_time": round(execution_time, 3),
                "models_trained": model_count,
                "features_engineered": feature_count,
                "average_r2_score": round(avg_r2, 3),
                "average_win_rate": round(avg_win_rate * 100, 1),
                "model_performances": {
                    name: {
                        "r2_score": round(perf.r2_score, 3),
                        "mse": round(perf.mse, 6),
                        "win_rate": round(perf.win_rate * 100, 1)
                    } for name, perf in performances.items()
                }
            }
            
            print(f"✅ AI Signal Generation: {performance_score:.1f}/100")
            print(f"   Models trained: {model_count}")
            print(f"   Average R² Score: {avg_r2:.3f}")
            print(f"   Average Win Rate: {avg_win_rate:.1%}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return ml_system
            
        except Exception as e:
            print(f"❌ Error in AI Signal Generation: {e}")
            self.demo_results["modules"]["ai_signal_generation"] = {
                "performance_score": 0,
                "error": str(e)
            }
            return None
    
    def run_feature_engineering_demo(self, ml_system):
        """Demo 2: Feature Engineering với advanced indicators"""
        
        print("\n" + "="*50)
        print("🔧 Module 2: Feature Engineering")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Use recent data for feature analysis
            recent_data = self.market_data.tail(100)
            price_data = recent_data[['close']]
            volume_data = recent_data['volume']
            
            # Generate features
            print("🔄 Creating advanced features...")
            features_df = ml_system.feature_engineer.create_features(price_data, volume_data)
            
            execution_time = time.time() - start_time
            
            # Analyze feature quality
            feature_count = len(features_df.columns)
            data_completeness = (features_df.count().sum() / features_df.size) * 100
            
            # Feature correlation analysis
            correlation_matrix = features_df.corr()
            avg_correlation = correlation_matrix.abs().mean().mean()
            
            # Feature variance analysis
            feature_variances = features_df.var()
            low_variance_features = (feature_variances < 0.0001).sum()
            
            # Performance scoring
            performance_score = min(100, max(0, (
                (feature_count / 25) * 25 +  # Feature diversity (25 points)
                data_completeness * 0.3 +  # Data completeness (30 points)
                (1 - avg_correlation) * 25 +  # Feature independence (25 points)
                ((feature_count - low_variance_features) / feature_count) * 20  # Feature quality (20 points)
            )))
            
            # Store results
            self.demo_results["modules"]["feature_engineering"] = {
                "performance_score": round(performance_score, 1),
                "execution_time": round(execution_time, 3),
                "features_created": feature_count,
                "data_completeness": round(data_completeness, 1),
                "average_correlation": round(avg_correlation, 3),
                "low_variance_features": low_variance_features,
                "feature_types": {
                    "technical_indicators": len([c for c in features_df.columns if any(x in c for x in ['sma', 'ema', 'rsi', 'macd', 'bollinger'])]),
                    "price_patterns": len([c for c in features_df.columns if any(x in c for x in ['returns', 'momentum', 'acceleration'])]),
                    "volatility_features": len([c for c in features_df.columns if 'volatility' in c or 'vol_' in c]),
                    "volume_features": len([c for c in features_df.columns if 'volume' in c])
                }
            }
            
            print(f"✅ Feature Engineering: {performance_score:.1f}/100")
            print(f"   Features created: {feature_count}")
            print(f"   Data completeness: {data_completeness:.1f}%")
            print(f"   Average correlation: {avg_correlation:.3f}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return features_df
            
        except Exception as e:
            print(f"❌ Error in Feature Engineering: {e}")
            self.demo_results["modules"]["feature_engineering"] = {
                "performance_score": 0,
                "error": str(e)
            }
            return None
    
    def run_ensemble_predictions_demo(self, ml_system):
        """Demo 3: Ensemble Model Predictions"""
        
        print("\n" + "="*50)
        print("🎯 Module 3: Ensemble Model Predictions")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Use test data (last 20% of data)
            test_data = self.market_data.tail(int(len(self.market_data) * 0.2))
            
            predictions = []
            prediction_details = []
            
            print("🔄 Generating ensemble predictions...")
            
            # Generate predictions for multiple time points
            for i in range(5, len(test_data), 10):  # Every 10 days
                current_data = self.market_data.iloc[:len(self.market_data) - len(test_data) + i]
                recent_window = current_data.tail(60)  # Last 60 days
                
                # Create features for prediction
                features_df = ml_system.feature_engineer.create_features(recent_window[['close']], recent_window['volume'])
                
                if not features_df.empty:
                    # Get ensemble prediction
                    latest_features = features_df.iloc[[-1]]
                    ensemble_pred, model_preds = ml_system.model_manager.predict_ensemble(latest_features)
                    
                    predictions.append({
                        'timestamp': test_data.index[i],
                        'ensemble_prediction': ensemble_pred,
                        'individual_predictions': model_preds,
                        'actual_price': test_data.iloc[i]['close'],
                        'previous_price': test_data.iloc[i-1]['close']
                    })
                    
                    prediction_details.append(ensemble_pred)
            
            execution_time = time.time() - start_time
            
            # Analyze prediction quality
            if predictions:
                ensemble_preds = [p['ensemble_prediction'] for p in predictions]
                pred_variance = np.var(ensemble_preds)
                pred_mean = np.mean(ensemble_preds)
                
                # Calculate prediction accuracy (direction)
                correct_directions = 0
                for pred in predictions:
                    actual_return = (pred['actual_price'] / pred['previous_price']) - 1
                    predicted_direction = pred['ensemble_prediction'] > 0
                    actual_direction = actual_return > 0
                    if predicted_direction == actual_direction:
                        correct_directions += 1
                
                direction_accuracy = correct_directions / len(predictions) * 100
            else:
                pred_variance = 0
                pred_mean = 0
                direction_accuracy = 0
            
            # Performance scoring
            performance_score = min(100, max(0, (
                (len(predictions) / 5) * 20 +  # Prediction count (20 points)
                direction_accuracy * 0.4 +  # Direction accuracy (40 points)
                (1 / (pred_variance + 0.001)) * 0.2 +  # Prediction consistency (20 points)
                (max(0, 10 - execution_time) / 10) * 20  # Speed (20 points)
            )))
            
            # Store results
            self.demo_results["modules"]["ensemble_predictions"] = {
                "performance_score": round(performance_score, 1),
                "execution_time": round(execution_time, 3),
                "predictions_generated": len(predictions),
                "direction_accuracy": round(direction_accuracy, 1),
                "prediction_variance": round(pred_variance, 6),
                "prediction_mean": round(pred_mean, 6),
                "model_agreement": round(1 - pred_variance, 3) if pred_variance < 1 else 0.0
            }
            
            print(f"✅ Ensemble Predictions: {performance_score:.1f}/100")
            print(f"   Predictions generated: {len(predictions)}")
            print(f"   Direction accuracy: {direction_accuracy:.1f}%")
            print(f"   Model agreement: {1 - min(pred_variance, 1):.3f}")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error in Ensemble Predictions: {e}")
            self.demo_results["modules"]["ensemble_predictions"] = {
                "performance_score": 0,
                "error": str(e)
            }
            return []
    
    def run_realtime_signal_generation_demo(self, ml_system):
        """Demo 4: Real-time Signal Generation"""
        
        print("\n" + "="*50)
        print("⚡ Module 4: Real-time Signal Generation")
        print("="*50)
        
        start_time = time.time()
        
        try:
            signals = []
            signal_performance = []
            
            print("🔄 Generating real-time trading signals...")
            
            # Generate signals for the last 10 market days
            test_data = self.market_data.tail(15)
            
            for i in range(5, len(test_data)):
                # Current market snapshot
                current_data = self.market_data.iloc[:len(self.market_data) - len(test_data) + i]
                recent_window = current_data.tail(60)
                
                # Generate trading signal
                signal = ml_system.generate_trading_signal(recent_window[['close']], recent_window['volume'])
                
                if signal:
                    signals.append({
                        'timestamp': test_data.index[i],
                        'signal_type': signal.signal_type.value,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'predicted_return': signal.predicted_return,
                        'recommended_position': signal.recommended_position_size,
                        'current_price': test_data.iloc[i]['close']
                    })
                    
                    # Evaluate signal quality
                    if i < len(test_data) - 1:  # If we have next day data
                        actual_return = (test_data.iloc[i+1]['close'] / test_data.iloc[i]['close']) - 1
                        predicted_return = signal.predicted_return
                        
                        # Calculate signal performance
                        if signal.signal_type != SignalType.HOLD:
                            direction_correct = (predicted_return > 0) == (actual_return > 0)
                            magnitude_error = abs(predicted_return - actual_return)
                            
                            signal_performance.append({
                                'direction_correct': direction_correct,
                                'magnitude_error': magnitude_error,
                                'confidence': signal.confidence,
                                'strength': signal.strength
                            })
            
            execution_time = time.time() - start_time
            
            # Analyze signal quality
            if signals:
                avg_confidence = np.mean([s['confidence'] for s in signals])
                avg_strength = np.mean([s['strength'] for s in signals])
                signal_distribution = {}
                for signal in signals:
                    signal_type = signal['signal_type']
                    signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1
                
                # Signal performance analysis
                if signal_performance:
                    direction_accuracy = np.mean([sp['direction_correct'] for sp in signal_performance]) * 100
                    avg_magnitude_error = np.mean([sp['magnitude_error'] for sp in signal_performance])
                else:
                    direction_accuracy = 0
                    avg_magnitude_error = 0
            else:
                avg_confidence = 0
                avg_strength = 0
                signal_distribution = {}
                direction_accuracy = 0
                avg_magnitude_error = 0
            
            # Performance scoring
            performance_score = min(100, max(0, (
                (len(signals) / 10) * 20 +  # Signal count (20 points)
                direction_accuracy * 0.3 +  # Direction accuracy (30 points)
                avg_confidence * 25 +  # Average confidence (25 points)
                (max(0, 1 - avg_magnitude_error) * 25)  # Magnitude accuracy (25 points)
            )))
            
            # Store results
            self.demo_results["modules"]["realtime_signal_generation"] = {
                "performance_score": round(performance_score, 1),
                "execution_time": round(execution_time, 3),
                "signals_generated": len(signals),
                "average_confidence": round(avg_confidence, 3),
                "average_strength": round(avg_strength, 3),
                "direction_accuracy": round(direction_accuracy, 1),
                "magnitude_error": round(avg_magnitude_error, 6),
                "signal_distribution": signal_distribution
            }
            
            print(f"✅ Real-time Signal Generation: {performance_score:.1f}/100")
            print(f"   Signals generated: {len(signals)}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Direction accuracy: {direction_accuracy:.1f}%")
            print(f"   Execution time: {execution_time:.3f}s")
            
            return signals
            
        except Exception as e:
            print(f"❌ Error in Real-time Signal Generation: {e}")
            self.demo_results["modules"]["realtime_signal_generation"] = {
                "performance_score": 0,
                "error": str(e)
            }
            return []
    
    def run_risk_position_sizing_demo(self, ml_system, signals):
        """Demo 5: Risk-Adjusted Position Sizing"""
        
        print("\n" + "="*50)
        print("🛡️ Module 5: Risk-Adjusted Position Sizing")
        print("="*50)
        
        start_time = time.time()
        
        try:
            position_analysis = []
            risk_metrics = []
            
            print("🔄 Analyzing risk-adjusted position sizing...")
            
            if not signals:
                print("⚠️ No signals available for position sizing analysis")
                return
            
            # Portfolio parameters
            initial_capital = 100000
            max_risk_per_trade = 0.02  # 2% max risk per trade
            current_capital = initial_capital
            
            for i, signal in enumerate(signals):
                if signal['signal_type'] != 'hold':
                    # Calculate position size based on signal and risk
                    base_position = signal['recommended_position']
                    confidence_adjustment = signal['confidence']
                    strength_adjustment = signal['strength']
                    
                    # Risk-adjusted position size
                    risk_adjusted_position = (base_position * 
                                            confidence_adjustment * 
                                            strength_adjustment * 
                                            max_risk_per_trade)
                    
                    # Calculate maximum position based on capital
                    max_position_value = current_capital * risk_adjusted_position
                    
                    # Position analysis
                    position_analysis.append({
                        'timestamp': signal['timestamp'],
                        'signal_type': signal['signal_type'],
                        'base_position': base_position,
                        'risk_adjusted_position': risk_adjusted_position,
                        'position_value': max_position_value,
                        'confidence': signal['confidence'],
                        'strength': signal['strength']
                    })
                    
                    # Risk metrics
                    position_risk = risk_adjusted_position * current_capital
                    var_estimate = position_risk * 1.65  # 95% VaR approximation
                    
                    risk_metrics.append({
                        'position_risk': position_risk,
                        'var_estimate': var_estimate,
                        'risk_ratio': position_risk / current_capital,
                        'confidence_weighted_risk': position_risk * signal['confidence']
                    })
            
            execution_time = time.time() - start_time
            
            # Analyze position sizing quality
            if position_analysis:
                avg_position_size = np.mean([p['risk_adjusted_position'] for p in position_analysis])
                position_variance = np.var([p['risk_adjusted_position'] for p in position_analysis])
                avg_risk_ratio = np.mean([r['risk_ratio'] for r in risk_metrics])
                max_risk_ratio = max([r['risk_ratio'] for r in risk_metrics])
                
                # Risk management score
                risk_management_score = (
                    (max_risk_ratio <= max_risk_per_trade) * 30 +  # Risk limit adherence
                    (avg_risk_ratio < max_risk_per_trade * 0.5) * 20 +  # Conservative sizing
                    (position_variance < 0.01) * 30 +  # Position consistency
                    (len(position_analysis) > 0) * 20  # Position generation
                )
            else:
                avg_position_size = 0
                avg_risk_ratio = 0
                max_risk_ratio = 0
                risk_management_score = 0
            
            # Performance scoring
            performance_score = min(100, max(0, (
                risk_management_score * 0.6 +  # Risk management (60 points)
                (len(position_analysis) / len(signals)) * 30 +  # Position coverage (30 points)
                (max(0, 5 - execution_time) / 5) * 10  # Speed (10 points)
            )))
            
            # Store results
            self.demo_results["modules"]["risk_position_sizing"] = {
                "performance_score": round(performance_score, 1),
                "execution_time": round(execution_time, 3),
                "positions_analyzed": len(position_analysis),
                "average_position_size": round(avg_position_size, 4),
                "average_risk_ratio": round(avg_risk_ratio * 100, 2),
                "max_risk_ratio": round(max_risk_ratio * 100, 2),
                "risk_limit_adherence": max_risk_ratio <= max_risk_per_trade,
                "total_portfolio_risk": round(sum([r['position_risk'] for r in risk_metrics]), 2)
            }
            
            print(f"✅ Risk-Adjusted Position Sizing: {performance_score:.1f}/100")
            print(f"   Positions analyzed: {len(position_analysis)}")
            print(f"   Average risk ratio: {avg_risk_ratio * 100:.2f}%")
            print(f"   Risk limit adherence: {max_risk_ratio <= max_risk_per_trade}")
            print(f"   Execution time: {execution_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Error in Risk-Adjusted Position Sizing: {e}")
            self.demo_results["modules"]["risk_position_sizing"] = {
                "performance_score": 0,
                "error": str(e)
            }
    
    def calculate_overall_performance(self):
        """Calculate overall system performance"""
        
        print("\n" + "="*70)
        print("📊 OVERALL PERFORMANCE ASSESSMENT")
        print("="*70)
        
        module_scores = []
        total_execution_time = 0
        
        for module_name, results in self.demo_results["modules"].items():
            if "performance_score" in results and "error" not in results:
                score = results["performance_score"]
                exec_time = results.get("execution_time", 0)
                
                module_scores.append(score)
                total_execution_time += exec_time
                
                print(f"✅ {module_name.replace('_', ' ').title()}: {score:.1f}/100")
            else:
                print(f"❌ {module_name.replace('_', ' ').title()}: FAILED")
        
        # Calculate overall performance
        if module_scores:
            overall_score = np.mean(module_scores)
            
            # Performance grade
            if overall_score >= 90:
                grade = "EXCELLENT 🥇"
                status = "Production Ready - Outstanding Performance"
            elif overall_score >= 80:
                grade = "VERY GOOD 🥈"
                status = "Production Ready - High Performance"
            elif overall_score >= 70:
                grade = "GOOD 🥉"
                status = "Production Ready - Satisfactory Performance"
            elif overall_score >= 60:
                grade = "SATISFACTORY ⭐"
                status = "Production Ready - Needs Optimization"
            else:
                grade = "NEEDS IMPROVEMENT ⚠️"
                status = "Requires Development - Below Standards"
        else:
            overall_score = 0
            grade = "SYSTEM FAILURE ❌"
            status = "Critical Issues - Not Ready"
        
        # Store final results
        self.demo_results["overall_performance"] = {
            "overall_score": round(overall_score, 1),
            "grade": grade,
            "status": status,
            "total_execution_time": round(total_execution_time, 3),
            "modules_completed": len(module_scores),
            "modules_failed": len(self.demo_results["modules"]) - len(module_scores),
            "performance_breakdown": {
                module: results.get("performance_score", 0) 
                for module, results in self.demo_results["modules"].items()
            }
        }
        
        print(f"\n🎯 OVERALL SCORE: {overall_score:.1f}/100")
        print(f"🏆 GRADE: {grade}")
        print(f"📈 STATUS: {status}")
        print(f"⏱️ TOTAL EXECUTION TIME: {total_execution_time:.3f}s")
        print(f"✅ MODULES COMPLETED: {len(module_scores)}/5")
        
        return overall_score
    
    def save_results(self):
        """Save demo results to file"""
        
        try:
            filename = "day29_ml_enhanced_trading_signals_results.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.demo_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def run_complete_demo(self):
        """Run complete Day 29 ML Enhanced Trading Signals demo"""
        
        try:
            # Run all demo modules
            ml_system = self.run_ai_signal_generation_demo()
            
            if ml_system:
                features_df = self.run_feature_engineering_demo(ml_system)
                predictions = self.run_ensemble_predictions_demo(ml_system)
                signals = self.run_realtime_signal_generation_demo(ml_system)
                self.run_risk_position_sizing_demo(ml_system, signals)
            
            # Calculate overall performance
            overall_score = self.calculate_overall_performance()
            
            # Save results
            self.save_results()
            
            return overall_score
            
        except Exception as e:
            print(f"❌ Error in complete demo: {e}")
            return 0


def main():
    """Main demo execution"""
    
    print("Starting Day 29: ML Enhanced Trading Signals Demo...")
    
    # Create and run demo
    demo = Day29MLDemo()
    overall_score = demo.run_complete_demo()
    
    print(f"\n🎬 Demo completed with overall score: {overall_score:.1f}/100")
    
    return overall_score


if __name__ == "__main__":
    main()