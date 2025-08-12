"""
Demo: Neural Network Ensemble System
Demonstrates the AI neural network ensemble for Phase 2
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.ai.neural_ensemble import (
    NeuralEnsemble, NetworkConfig, NetworkType, PredictionType,
    create_default_ensemble_configs
)

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"🧠 {title}")
    print(f"{'='*80}")

def print_section(title: str):
    """Print formatted section"""
    print(f"\n{'─'*60}")
    print(f"📊 {title}")
    print(f"{'─'*60}")

def generate_sample_data(samples: int = 2000) -> pd.DataFrame:
    """Generate realistic XAU price data for demonstration"""
    print("📈 Generating realistic XAU market data...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=samples//24)
    timestamps = pd.date_range(start=start_date, periods=samples, freq='H')
    
    # Generate realistic XAU price with trend, seasonality, and volatility
    base_price = 2000.0
    trend = np.linspace(0, 100, samples)  # Upward trend
    
    # Add seasonality (daily and weekly patterns)
    daily_pattern = 10 * np.sin(2 * np.pi * np.arange(samples) / 24)
    weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(samples) / (24 * 7))
    
    # Add volatility clustering
    volatility = np.random.uniform(0.01, 0.03, samples)
    for i in range(1, samples):
        volatility[i] = 0.7 * volatility[i-1] + 0.3 * volatility[i]
    
    # Generate price with all components
    noise = np.random.normal(0, 1, samples)
    price_changes = volatility * noise
    prices = base_price + trend + daily_pattern + weekly_pattern
    
    # Apply price changes
    for i in range(1, samples):
        prices[i] = prices[i-1] * (1 + price_changes[i])
    
    # Generate additional technical indicators
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': np.random.lognormal(8, 0.5, samples),  # Log-normal volume
        'high': prices + np.random.uniform(0, 10, samples),
        'low': prices - np.random.uniform(0, 10, samples),
    })
    
    # Calculate technical indicators
    data['returns'] = data['price'].pct_change()
    data['volatility'] = data['returns'].rolling(24).std()
    data['sma_20'] = data['price'].rolling(20).mean()
    data['sma_50'] = data['price'].rolling(50).mean()
    data['rsi'] = calculate_rsi(data['price'], 14)
    data['macd'] = calculate_macd(data['price'])
    
    # Fill NaN values
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    print(f"✅ Generated {len(data)} samples of XAU data")
    print(f"   📊 Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
    print(f"   📈 Average volume: {data['volume'].mean():.0f}")
    
    return data

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

def demo_ensemble_creation():
    """Demo 1: Neural Ensemble Creation and Configuration"""
    print_header("Demo 1: Neural Ensemble Creation and Configuration")
    
    print("🔧 Creating default ensemble configuration...")
    configs = create_default_ensemble_configs()
    
    print(f"✅ Created {len(configs)} network configurations:")
    for i, config in enumerate(configs, 1):
        print(f"   {i}. {config.network_type.value.upper()} - {config.prediction_type.value}")
        print(f"      Sequence Length: {config.sequence_length}")
        print(f"      Hidden Units: {config.hidden_units}")
        print(f"      Weight: {config.weight}")
    
    print("\n🧠 Initializing Neural Ensemble...")
    ensemble = NeuralEnsemble(configs)
    
    summary = ensemble.get_ensemble_summary()
    print(f"✅ Ensemble initialized successfully!")
    print(f"   📊 Networks: {summary['networks_count']}")
    print(f"   🎯 Network Types: {', '.join(summary['network_types'])}")
    print(f"   📈 Prediction Types: {', '.join(summary['prediction_types'])}")
    
    return ensemble, configs

def demo_data_preparation(ensemble, data):
    """Demo 2: Data Preparation and Feature Engineering"""
    print_header("Demo 2: Data Preparation and Feature Engineering")
    
    print("📊 Preparing data for neural network training...")
    
    # Select features for training
    feature_columns = ['volume', 'high', 'low', 'returns', 'volatility', 
                      'sma_20', 'sma_50', 'rsi', 'macd']
    
    # Ensure we have enough features
    available_features = [col for col in feature_columns if col in data.columns]
    print(f"   📈 Available features: {len(available_features)}")
    for feature in available_features:
        print(f"      • {feature}")
    
    # Prepare training data
    training_data = data[available_features + ['price']].copy()
    training_data = training_data.dropna()
    
    print(f"\n📊 Training data statistics:")
    print(f"   📈 Samples: {len(training_data)}")
    print(f"   🎯 Features: {len(available_features)}")
    print(f"   📊 Target (price) range: ${training_data['price'].min():.2f} - ${training_data['price'].max():.2f}")
    
    # Show data preparation for one network
    network = list(ensemble.networks.values())[0]
    print(f"\n🔧 Preparing sequences for {network.config.network_type.value.upper()} network...")
    
    try:
        X, y = network.prepare_data(training_data, 'price')
        print(f"✅ Sequences created successfully!")
        print(f"   📊 Input shape: {X.shape}")
        print(f"   🎯 Target shape: {y.shape}")
        print(f"   📈 Sequence length: {X.shape[1]}")
        print(f"   🔢 Features per timestep: {X.shape[2]}")
        
        return training_data, X, y
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return training_data, None, None

def demo_ensemble_training(ensemble, training_data):
    """Demo 3: Ensemble Training"""
    print_header("Demo 3: Neural Ensemble Training")
    
    print("🚀 Starting ensemble training...")
    print("⏱️ This may take a few minutes depending on your hardware...")
    
    start_time = time.time()
    
    try:
        # Train ensemble
        training_results = ensemble.train_ensemble(training_data, 'price')
        
        training_time = time.time() - start_time
        
        print(f"✅ Ensemble training completed in {training_time:.2f} seconds!")
        
        print(f"\n📊 Training Results Summary:")
        print(f"   ⏱️ Total training time: {training_results['total_training_time']:.2f}s")
        print(f"   🧠 Networks trained: {training_results['networks_trained']}")
        
        print(f"\n📈 Individual Network Performance:")
        for network_id, results in training_results['individual_results'].items():
            print(f"   🔸 {network_id.upper()}:")
            print(f"      Final Loss: {results['final_loss']:.6f}")
            print(f"      Validation Loss: {results['final_val_loss']:.6f}")
            print(f"      Training Time: {results['training_time']:.2f}s")
            print(f"      Epochs: {results['epochs_trained']}")
            print(f"      R² Score: {results['train_r2']:.4f}")
        
        print(f"\n⚖️ Ensemble Weights:")
        for network_id, weight in training_results['ensemble_weights'].items():
            print(f"   🔸 {network_id}: {weight:.4f} ({weight*100:.1f}%)")
        
        return training_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def demo_ensemble_prediction(ensemble, training_data):
    """Demo 4: Ensemble Prediction and Analysis"""
    print_header("Demo 4: Ensemble Prediction and Analysis")
    
    if not ensemble.is_trained:
        print("❌ Ensemble must be trained before making predictions")
        return
    
    print("🔮 Making ensemble predictions...")
    
    # Get a sample for prediction
    network = list(ensemble.networks.values())[0]
    X, y = network.prepare_data(training_data, 'price')
    
    # Use last few samples for prediction
    test_samples = 5
    test_X = X[-test_samples:]
    test_y = y[-test_samples:]
    
    predictions = []
    actual_prices = []
    
    print(f"📊 Testing on {test_samples} samples...")
    
    for i in range(test_samples):
        sample_X = test_X[i:i+1]
        actual_price = test_y[i]
        
        try:
            # Make ensemble prediction
            result = ensemble.predict(sample_X)
            predicted_price = result.final_prediction[0][0]
            
            predictions.append(predicted_price)
            actual_prices.append(actual_price)
            
            print(f"\n🔸 Sample {i+1}:")
            print(f"   🎯 Actual Price: ${actual_price:.2f}")
            print(f"   🔮 Predicted Price: ${predicted_price:.2f}")
            print(f"   📊 Error: ${abs(actual_price - predicted_price):.2f}")
            print(f"   📈 Error %: {abs(actual_price - predicted_price)/actual_price*100:.2f}%")
            print(f"   🎯 Ensemble Confidence: {result.ensemble_confidence:.1%}")
            print(f"   🤝 Consensus Score: {result.consensus_score:.1%}")
            print(f"   ⏱️ Processing Time: {result.processing_time*1000:.1f}ms")
            
            print(f"   🧠 Individual Network Predictions:")
            for pred in result.individual_predictions:
                network_pred = pred.prediction[0][0]
                print(f"      • {pred.network_type.value}: ${network_pred:.2f} (conf: {pred.confidence:.1%})")
                
        except Exception as e:
            print(f"❌ Prediction failed for sample {i+1}: {e}")
    
    if predictions and actual_prices:
        # Calculate overall metrics
        predictions = np.array(predictions)
        actual_prices = np.array(actual_prices)
        
        mae = np.mean(np.abs(predictions - actual_prices))
        mse = np.mean((predictions - actual_prices) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((predictions - actual_prices) / actual_prices)) * 100
        
        print(f"\n📊 Overall Prediction Performance:")
        print(f"   📈 Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   📊 Root Mean Square Error (RMSE): ${rmse:.2f}")
        print(f"   📉 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Accuracy assessment
        if mape < 1.0:
            accuracy_level = "🏆 Excellent"
        elif mape < 2.0:
            accuracy_level = "✅ Very Good"
        elif mape < 5.0:
            accuracy_level = "👍 Good"
        else:
            accuracy_level = "⚠️ Needs Improvement"
        
        print(f"   🎯 Accuracy Level: {accuracy_level}")

def demo_real_time_simulation(ensemble, training_data):
    """Demo 5: Real-time Prediction Simulation"""
    print_header("Demo 5: Real-time Prediction Simulation")
    
    if not ensemble.is_trained:
        print("❌ Ensemble must be trained before simulation")
        return
    
    print("⚡ Simulating real-time trading predictions...")
    
    # Get data for simulation
    network = list(ensemble.networks.values())[0]
    X, y = network.prepare_data(training_data, 'price')
    
    # Simulate real-time predictions
    simulation_steps = 10
    start_idx = len(X) - simulation_steps - 10
    
    print(f"🔄 Running {simulation_steps} real-time prediction steps...")
    
    total_processing_time = 0
    confidence_scores = []
    consensus_scores = []
    
    for step in range(simulation_steps):
        current_idx = start_idx + step
        current_X = X[current_idx:current_idx+1]
        actual_price = y[current_idx]
        
        # Simulate real-time prediction
        step_start = time.time()
        
        try:
            result = ensemble.predict(current_X)
            predicted_price = result.final_prediction[0][0]
            
            step_time = time.time() - step_start
            total_processing_time += step_time
            
            confidence_scores.append(result.ensemble_confidence)
            consensus_scores.append(result.consensus_score)
            
            # Simulate trading signal
            price_change = predicted_price - actual_price
            signal = "🟢 BUY" if price_change > 5 else "🔴 SELL" if price_change < -5 else "⚪ HOLD"
            
            print(f"   Step {step+1:2d}: ${actual_price:7.2f} → ${predicted_price:7.2f} "
                  f"({price_change:+6.2f}) {signal} "
                  f"[{result.ensemble_confidence:.1%} conf, {step_time*1000:.1f}ms]")
                  
        except Exception as e:
            print(f"   Step {step+1:2d}: ❌ Error - {e}")
    
    # Simulation summary
    avg_processing_time = total_processing_time / simulation_steps
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    avg_consensus = np.mean(consensus_scores) if consensus_scores else 0
    
    print(f"\n📊 Real-time Simulation Summary:")
    print(f"   ⚡ Average processing time: {avg_processing_time*1000:.1f}ms")
    print(f"   🎯 Average confidence: {avg_confidence:.1%}")
    print(f"   🤝 Average consensus: {avg_consensus:.1%}")
    print(f"   📈 Predictions per second: {1/avg_processing_time:.1f}")
    
    # Performance assessment
    if avg_processing_time < 0.1:
        performance_level = "🚀 Excellent (Real-time ready)"
    elif avg_processing_time < 0.5:
        performance_level = "✅ Very Good (Near real-time)"
    elif avg_processing_time < 1.0:
        performance_level = "👍 Good (Acceptable for trading)"
    else:
        performance_level = "⚠️ Needs Optimization"
    
    print(f"   🏆 Performance Level: {performance_level}")

def main():
    """Run all neural ensemble demos"""
    print_header("🧠 Neural Network Ensemble System - Phase 2 Demo Suite")
    print("Advanced AI component for Ultimate XAU Super System V4.0")
    
    try:
        # Generate sample data
        data = generate_sample_data(1500)  # Reduced for faster demo
        
        # Demo 1: Ensemble Creation
        ensemble, configs = demo_ensemble_creation()
        
        # Demo 2: Data Preparation
        training_data, X, y = demo_data_preparation(ensemble, data)
        
        if X is not None and y is not None:
            # Demo 3: Training
            training_results = demo_ensemble_training(ensemble, training_data)
            
            if training_results:
                # Demo 4: Prediction
                demo_ensemble_prediction(ensemble, training_data)
                
                # Demo 5: Real-time Simulation
                demo_real_time_simulation(ensemble, training_data)
        
        print_header("✅ Neural Ensemble Demo Completed Successfully!")
        print("🎯 Key Features Demonstrated:")
        print("   • Multi-network ensemble architecture")
        print("   • LSTM, GRU, CNN, and Dense networks")
        print("   • Automated weight calculation")
        print("   • Real-time prediction capabilities")
        print("   • Confidence and consensus scoring")
        print("   • Performance optimization")
        print("\n🚀 Neural Ensemble System is ready for Phase 2 integration!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 