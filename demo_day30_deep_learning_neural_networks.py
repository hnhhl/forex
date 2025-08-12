"""
Demo Day 30: Deep Learning Neural Networks
Ultimate XAU Super System V4.0

Kiểm tra toàn diện hệ thống Deep Learning Neural Networks:
1. Neural Network Architecture Training
2. Deep Feature Extraction & Engineering  
3. LSTM Time Series Modeling
4. Ensemble Deep Learning Predictions
5. Real-time Deep Learning Inference

Đạt mục tiêu: Triển khai mạng nơ-ron học sâu cho trading XAU
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Day 30 modules
sys.path.append('src')
from core.analysis.deep_learning_neural_networks import (
    DeepLearningNeuralNetworks, NetworkConfig, NetworkType, ActivationFunction,
    DeepFeatureExtractor, DeepLearningPredictor, EnsembleDeepLearning,
    create_default_configs, create_ensemble_config, analyze_prediction_performance
)

def generate_market_data(days: int = 30) -> pd.DataFrame:
    """Tạo dữ liệu thị trường mô phỏng cho XAU"""
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days*24, freq='H')
    
    # Tạo dữ liệu giá vàng với trend và volatility thực tế
    base_price = 2000
    trend = np.random.normal(0, 0.001, len(dates))
    volatility = np.random.normal(0, 0.02, len(dates))
    
    prices = [base_price]
    for i in range(1, len(dates)):
        price_change = trend[i] + volatility[i]
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1500))  # Giá tối thiểu
    
    # Tạo OHLC data
    data = []
    for i in range(len(dates)):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        close = price + np.random.normal(0, price * 0.001)
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'timestamp': dates[i],
            'open': price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def demo_neural_network_training():
    """Demo 1: Neural Network Architecture Training"""
    print("\n" + "="*70)
    print("📊 DEMO 1: NEURAL NETWORK ARCHITECTURE TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    # Tạo dữ liệu training
    training_data = generate_market_data(30)
    print(f"📈 Đã tạo {len(training_data)} mẫu dữ liệu training")
    
    # Khởi tạo hệ thống
    dl_system = DeepLearningNeuralNetworks()
    
    # Tạo các cấu hình mạng khác nhau
    configs = create_default_configs()
    
    # LSTM Networks - sử dụng 18 features cố định
    lstm_config1 = NetworkConfig(
        network_type=NetworkType.LSTM,
        input_size=18,  # 18 features cố định
        hidden_size=64,
        num_layers=2,
        sequence_length=20,
        learning_rate=0.001,
        epochs=50
    )
    
    lstm_config2 = NetworkConfig(
        network_type=NetworkType.LSTM,
        input_size=18,  # 18 features cố định
        hidden_size=128,
        num_layers=3,
        sequence_length=15,
        learning_rate=0.0005,
        epochs=75
    )
    
    # Dense Networks - flattened sequences
    dense_config1 = NetworkConfig(
        network_type=NetworkType.DENSE,
        input_size=360,  # 20 * 18 features flattened
        hidden_size=256,
        num_layers=4,
        dropout_rate=0.3,
        activation=ActivationFunction.RELU,
        learning_rate=0.001,
        epochs=100
    )
    
    dense_config2 = NetworkConfig(
        network_type=NetworkType.DENSE,
        input_size=270,  # 15 * 18 features flattened
        hidden_size=512,
        num_layers=5,
        dropout_rate=0.2,
        activation=ActivationFunction.LEAKY_RELU,
        learning_rate=0.0005,
        epochs=150
    )
    
    # Tạo và huấn luyện các predictors
    networks = {
        'LSTM_64': lstm_config1,
        'LSTM_128': lstm_config2,
        'Dense_256': dense_config1,
        'Dense_512': dense_config2
    }
    
    training_results = {}
    for name, config in networks.items():
        try:
            predictor = dl_system.create_predictor(name, config)
            performance = predictor.train(training_data)
            training_results[name] = performance
            print(f"✅ {name}: Accuracy={performance.accuracy:.3f}, Loss={performance.training_loss:.3f}")
        except Exception as e:
            print(f"❌ Lỗi training {name}: {e}")
    
    execution_time = time.time() - start_time
    
    # Đánh giá
    total_networks = len(networks)
    successful_trainings = len(training_results)
    avg_accuracy = np.mean([r.accuracy for r in training_results.values()]) if training_results else 0
    
    score = (successful_trainings / total_networks) * 100 * min(avg_accuracy * 2, 1)
    
    print(f"\n📊 KẾT QUẢ DEMO 1:")
    print(f"Networks trained: {successful_trainings}/{total_networks}")
    print(f"Average accuracy: {avg_accuracy:.3f}")
    print(f"Execution time: {execution_time:.2f}s")
    print(f"Score: {score:.1f}/100")
    
    return {
        'score': score,
        'execution_time': execution_time,
        'networks_trained': successful_trainings,
        'avg_accuracy': avg_accuracy,
        'details': training_results
    }

def demo_deep_feature_extraction():
    """Demo 2: Deep Feature Extraction & Engineering"""
    print("\n" + "="*70)
    print("🔍 DEMO 2: DEEP FEATURE EXTRACTION & ENGINEERING")
    print("="*70)
    
    start_time = time.time()
    
    # Tạo dữ liệu thị trường
    market_data = generate_market_data(45)
    print(f"📈 Dữ liệu đầu vào: {len(market_data)} samples")
    
    # Khởi tạo feature extractor
    extractor = DeepFeatureExtractor(sequence_length=20)
    
    try:
        # Trích xuất features
        features = extractor.extract_features(market_data)
        
        print(f"\n🔧 FEATURE EXTRACTION RESULTS:")
        print(f"Sequence length: {features.sequence_length}")
        print(f"Total sequences: {len(features.market_state_features)}")
        print(f"Features per sequence: {len(features.feature_names)}")
        print(f"Price sequences shape: {features.price_sequences.shape}")
        print(f"Technical indicators shape: {features.technical_indicators.shape}")
        print(f"Pattern features shape: {features.pattern_features.shape}")
        print(f"Volume sequences shape: {features.volume_sequences.shape}")
        
        # Phân tích chất lượng features
        feature_stats = {
            'completeness': 1.0 - (np.isnan(features.market_state_features).sum() / features.market_state_features.size),
            'variance': np.var(features.market_state_features),
            'mean_correlation': np.corrcoef(features.market_state_features.reshape(-1, features.market_state_features.shape[-1]).T).mean(),
            'sequence_coverage': len(features.market_state_features) / len(market_data)
        }
        
        print(f"\n📊 FEATURE QUALITY ANALYSIS:")
        print(f"Data completeness: {feature_stats['completeness']:.3f}")
        print(f"Feature variance: {feature_stats['variance']:.6f}")
        print(f"Sequence coverage: {feature_stats['sequence_coverage']:.3f}")
        
        execution_time = time.time() - start_time
        
        # Scoring
        completeness_score = feature_stats['completeness'] * 40
        coverage_score = feature_stats['sequence_coverage'] * 30
        complexity_score = min(len(features.feature_names) / 20, 1) * 30
        
        total_score = completeness_score + coverage_score + complexity_score
        
        print(f"\n📊 KẾT QUẢ DEMO 2:")
        print(f"Features extracted: {len(features.feature_names)}")
        print(f"Sequences created: {len(features.market_state_features)}")
        print(f"Data completeness: {feature_stats['completeness']:.1%}")
        print(f"Execution time: {execution_time:.3f}s")
        print(f"Score: {total_score:.1f}/100")
        
        return {
            'score': total_score,
            'execution_time': execution_time,
            'features_count': len(features.feature_names),
            'sequences_count': len(features.market_state_features),
            'completeness': feature_stats['completeness'],
            'features_object': features
        }
        
    except Exception as e:
        print(f"❌ Lỗi trong feature extraction: {e}")
        return {
            'score': 0,
            'execution_time': time.time() - start_time,
            'error': str(e)
        }

def demo_lstm_time_series_modeling():
    """Demo 3: LSTM Time Series Modeling"""
    print("\n" + "="*70)
    print("🕒 DEMO 3: LSTM TIME SERIES MODELING")
    print("="*70)
    
    start_time = time.time()
    
    # Tạo dữ liệu time series
    time_series_data = generate_market_data(60)
    print(f"📈 Time series data: {len(time_series_data)} points")
    
    # Sử dụng configs mặc định từ module để tránh dimension mismatch
    default_configs = create_default_configs()
    lstm_configs = [
        default_configs[0],  # LSTM config with 18 features, sequence_length=20
        NetworkConfig(
            network_type=NetworkType.LSTM,
            input_size=18,  # 18 features cố định  
            hidden_size=128,
            num_layers=2,
            sequence_length=20,  # Sử dụng cùng sequence_length
            learning_rate=0.0005,
            epochs=75
        )
    ]
    
    models_performance = {}
    predictions_made = []
    
    for i, config in enumerate(lstm_configs):
        model_name = f"LSTM_TS_{i+1}"
        try:
            # Tạo và huấn luyện mô hình
            predictor = DeepLearningPredictor(config)
            performance = predictor.train(time_series_data)
            
            # Tạo predictions
            predictions = []
            for j in range(min(4, len(time_series_data) // 20)):
                sample_start = j * 20
                sample_end = sample_start + 40  # Đủ data cho sequence_length=20
                sample_data = time_series_data.iloc[sample_start:sample_end]
                if len(sample_data) >= 25:  # Đảm bảo đủ data cho feature extraction
                    pred = predictor.predict(sample_data)
                    predictions.append(pred)
                    predictions_made.append(pred)
            
            models_performance[model_name] = {
                'performance': performance,
                'predictions': predictions
            }
            
            print(f"✅ {model_name}: Accuracy={performance.accuracy:.3f}, Predictions={len(predictions)}")
            
        except Exception as e:
            print(f"❌ Lỗi với {model_name}: {e}")
    
    execution_time = time.time() - start_time
    
    # Đánh giá tổng thể
    successful_models = len(models_performance)
    total_predictions = len(predictions_made)
    avg_accuracy = np.mean([mp['performance'].accuracy for mp in models_performance.values()]) if models_performance else 0
    avg_confidence = np.mean([p.confidence for p in predictions_made]) if predictions_made else 0
    
    # Scoring
    model_score = (successful_models / len(lstm_configs)) * 40
    prediction_score = min(total_predictions / 8, 1) * 30
    accuracy_score = avg_accuracy * 30
    
    total_score = model_score + prediction_score + accuracy_score
    
    print(f"\n📊 KẾT QUẢ DEMO 3:")
    print(f"LSTM models trained: {successful_models}/{len(lstm_configs)}")
    print(f"Total predictions: {total_predictions}")
    print(f"Average accuracy: {avg_accuracy:.3f}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Execution time: {execution_time:.2f}s")
    print(f"Score: {total_score:.1f}/100")
    
    return {
        'score': total_score,
        'execution_time': execution_time,
        'models_trained': successful_models,
        'predictions_made': total_predictions,
        'avg_accuracy': avg_accuracy,
        'avg_confidence': avg_confidence
    }

def demo_ensemble_deep_learning():
    """Demo 4: Ensemble Deep Learning Predictions"""
    print("\n" + "="*70)
    print("🎯 DEMO 4: ENSEMBLE DEEP LEARNING PREDICTIONS")
    print("="*70)
    
    start_time = time.time()
    
    # Tạo dữ liệu cho ensemble
    ensemble_data = generate_market_data(40)
    print(f"📈 Ensemble data: {len(ensemble_data)} samples")
    
    # Sử dụng configs mặc định để tránh dimension mismatch
    default_configs = create_default_configs()
    ensemble_configs = default_configs  # LSTM và Dense configs đã đúng
    
    ensemble_results = {}
    
    try:
        # Tạo ensemble systems
        ensemble1 = EnsembleDeepLearning(ensemble_configs)
        ensemble2 = EnsembleDeepLearning([ensemble_configs[0]])  # Single LSTM ensemble
        
        ensembles = {
            'Multi_Model': ensemble1,
            'LSTM_Only': ensemble2
        }
        
        for name, ensemble in ensembles.items():
            try:
                # Huấn luyện ensemble
                performances = ensemble.train(ensemble_data)
                
                # Tạo predictions với data đủ lớn
                predictions = []
                for i in range(3):
                    sample_start = i * 20
                    sample_end = sample_start + 40
                    sample_data = ensemble_data.iloc[sample_start:sample_end]
                    if len(sample_data) >= 25:
                        pred = ensemble.predict(sample_data)
                        predictions.append(pred)
                
                ensemble_results[name] = {
                    'performances': performances,
                    'predictions': predictions,
                    'avg_accuracy': np.mean([p.accuracy for p in performances])
                }
                
                print(f"✅ {name}: Models={len(performances)}, Avg_Accuracy={ensemble_results[name]['avg_accuracy']:.3f}")
                
            except Exception as e:
                print(f"❌ Lỗi với ensemble {name}: {e}")
    
    except Exception as e:
        print(f"❌ Lỗi khởi tạo ensemble: {e}")
    
    execution_time = time.time() - start_time
    
    # Đánh giá
    successful_ensembles = len(ensemble_results)
    total_ensemble_predictions = sum([len(er['predictions']) for er in ensemble_results.values()])
    avg_ensemble_accuracy = np.mean([er['avg_accuracy'] for er in ensemble_results.values()]) if ensemble_results else 0
    
    # Scoring
    ensemble_score = (successful_ensembles / 2) * 40
    prediction_score = min(total_ensemble_predictions / 6, 1) * 30
    accuracy_score = avg_ensemble_accuracy * 30
    
    total_score = ensemble_score + prediction_score + accuracy_score
    
    print(f"\n📊 KẾT QUẢ DEMO 4:")
    print(f"Ensemble models: {successful_ensembles}/2")
    print(f"Ensemble predictions: {total_ensemble_predictions}")
    print(f"Average ensemble accuracy: {avg_ensemble_accuracy:.3f}")
    print(f"Execution time: {execution_time:.2f}s")
    print(f"Score: {total_score:.1f}/100")
    
    return {
        'score': total_score,
        'execution_time': execution_time,
        'ensembles_created': successful_ensembles,
        'predictions_made': total_ensemble_predictions,
        'avg_accuracy': avg_ensemble_accuracy
    }

def demo_realtime_inference():
    """Demo 5: Real-time Deep Learning Inference"""
    print("\n" + "="*70)
    print("⚡ DEMO 5: REAL-TIME DEEP LEARNING INFERENCE")
    print("="*70)
    
    start_time = time.time()
    
    # Tạo streaming data simulation
    base_data = generate_market_data(20)
    print(f"📊 Base data for inference: {len(base_data)} points")
    
    # Sử dụng config mặc định cho Dense network để tránh dimension mismatch
    default_configs = create_default_configs()
    fast_config = default_configs[1]  # Dense config với 360 input size (20 * 18)
    fast_config.epochs = 25  # Giảm epochs cho demo
    
    inference_times = []
    predictions = []
    
    try:
        # Tạo và huấn luyện model nhanh
        fast_predictor = DeepLearningPredictor(fast_config)
        training_performance = fast_predictor.train(base_data)
        
        print(f"🚀 Fast model trained - Accuracy: {training_performance.accuracy:.3f}")
        
        # Mô phỏng real-time inference
        for i in range(10):
            # Tạo "streaming" data point mới
            new_data_point = {
                'timestamp': datetime.now(),
                'open': np.random.uniform(1950, 2050),
                'high': np.random.uniform(1960, 2060),
                'low': np.random.uniform(1940, 2040),
                'close': np.random.uniform(1950, 2050),
                'volume': np.random.uniform(1000, 3000)
            }
            
            # Cập nhật data window
            current_data = base_data.copy()
            current_data = pd.concat([current_data, pd.DataFrame([new_data_point])], ignore_index=True)
            current_data = current_data.tail(50)  # Keep only recent data
            
            # Đo thời gian inference
            inference_start = time.time()
            prediction = fast_predictor.predict(current_data)
            inference_time = (time.time() - inference_start) * 1000  # Convert to ms
            
            inference_times.append(inference_time)
            predictions.append(prediction)
            
            print(f"⚡ Inference {i+1}: {inference_time:.2f}ms, Prediction: {prediction.prediction:.3f}, Direction: {prediction.direction}")
    
    except Exception as e:
        print(f"❌ Lỗi trong real-time inference: {e}")
    
    execution_time = time.time() - start_time
    
    # Đánh giá performance
    successful_inferences = len(predictions)
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_confidence = np.mean([p.confidence for p in predictions]) if predictions else 0
    
    # Scoring (real-time requirements)
    speed_score = max(0, (100 - avg_inference_time) / 100 * 40)  # Prefer < 100ms
    success_score = (successful_inferences / 10) * 30
    confidence_score = avg_confidence * 30
    
    total_score = speed_score + success_score + confidence_score
    
    print(f"\n📊 KẾT QUẢ DEMO 5:")
    print(f"Real-time inferences: {successful_inferences}/10")
    print(f"Average inference time: {avg_inference_time:.2f}ms")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Total execution time: {execution_time:.2f}s")
    print(f"Score: {total_score:.1f}/100")
    
    return {
        'score': total_score,
        'execution_time': execution_time,
        'inferences_made': successful_inferences,
        'avg_inference_time': avg_inference_time,
        'avg_confidence': avg_confidence
    }

def calculate_overall_score(demo_results):
    """Tính điểm tổng thể cho Day 30"""
    weights = {
        'demo1': 0.25,  # Neural Network Training
        'demo2': 0.20,  # Feature Extraction
        'demo3': 0.25,  # LSTM Modeling
        'demo4': 0.15,  # Ensemble Learning
        'demo5': 0.15   # Real-time Inference
    }
    
    total_score = 0
    total_weight = 0
    
    for demo_key, weight in weights.items():
        if demo_key in demo_results and 'score' in demo_results[demo_key]:
            total_score += demo_results[demo_key]['score'] * weight
            total_weight += weight
    
    final_score = total_score / total_weight if total_weight > 0 else 0
    
    # Tính grade
    if final_score >= 85:
        grade = "EXCELLENT"
        status = "🥇"
    elif final_score >= 70:
        grade = "VERY GOOD"
        status = "🥈"
    elif final_score >= 55:
        grade = "GOOD"
        status = "🥉"
    elif final_score >= 40:
        grade = "ACCEPTABLE"
        status = "⚠️"
    else:
        grade = "NEEDS IMPROVEMENT"
        status = "❌"
    
    return final_score, grade, status

def main():
    """Chạy tất cả demos cho Day 30"""
    print("🧠" + "="*69)
    print("🧠 ULTIMATE XAU SUPER SYSTEM V4.0 - DAY 30 DEEP LEARNING NEURAL NETWORKS")
    print("🧠" + "="*69)
    print("🎯 Mục tiêu: Triển khai mạng nơ-ron học sâu cho trading XAU")
    print("📅 Ngày: 30/56 - Phase 3: Advanced AI & Machine Learning")
    print("⏱️  Bắt đầu:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    # Khởi tạo kết quả
    demo_results = {}
    total_start_time = time.time()
    
    try:
        # Chạy các demos
        demo_results['demo1'] = demo_neural_network_training()
        demo_results['demo2'] = demo_deep_feature_extraction()
        demo_results['demo3'] = demo_lstm_time_series_modeling()
        demo_results['demo4'] = demo_ensemble_deep_learning()
        demo_results['demo5'] = demo_realtime_inference()
        
        # Tính điểm tổng thể
        final_score, grade, status = calculate_overall_score(demo_results)
        total_execution_time = time.time() - total_start_time
        
        # Báo cáo tổng kết
        print("\n" + "🏆"*70)
        print("🏆 TỔNG KẾT DAY 30: DEEP LEARNING NEURAL NETWORKS")
        print("🏆" + "="*69)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        for i, (demo_key, result) in enumerate(demo_results.items(), 1):
            score = result.get('score', 0)
            exec_time = result.get('execution_time', 0)
            print(f"Demo {i}: {score:5.1f}/100 ({exec_time:6.2f}s)")
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"Final Score: {final_score:.1f}/100")
        print(f"Grade: {grade} {status}")
        print(f"Total Execution Time: {total_execution_time:.2f}s")
        
        # Detailed metrics
        print(f"\n📈 DETAILED METRICS:")
        if 'demo1' in demo_results:
            print(f"Networks Trained: {demo_results['demo1'].get('networks_trained', 0)}")
        if 'demo2' in demo_results:
            print(f"Features Extracted: {demo_results['demo2'].get('features_count', 0)}")
        if 'demo3' in demo_results:
            print(f"LSTM Predictions: {demo_results['demo3'].get('predictions_made', 0)}")
        if 'demo4' in demo_results:
            print(f"Ensemble Models: {demo_results['demo4'].get('ensembles_created', 0)}")
        if 'demo5' in demo_results:
            print(f"Real-time Inferences: {demo_results['demo5'].get('inferences_made', 0)}")
            print(f"Avg Inference Time: {demo_results['demo5'].get('avg_inference_time', 0):.2f}ms")
        
        print(f"\n🎉 Day 30 Deep Learning Neural Networks hoàn thành!")
        print(f"Status: {grade} - Ready for Day 31")
        print("="*70)
        
        return {
            'day': 30,
            'final_score': final_score,
            'grade': grade,
            'status': status,
            'execution_time': total_execution_time,
            'demo_results': demo_results,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ LỖI TRONG DEMO DAY 30: {e}")
        return {
            'day': 30,
            'final_score': 0,
            'grade': "FAILED",
            'status': "❌",
            'execution_time': time.time() - total_start_time,
            'error': str(e),
            'success': False
        }

if __name__ == "__main__":
    results = main()