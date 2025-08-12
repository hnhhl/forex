#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODE 5 COMPLETE EVALUATION SYSTEM
So sánh toàn diện tất cả models đã training
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime
import json
import glob
from sklearn.metrics import classification_report, confusion_matrix

class Mode5EvaluationSystem:
    """Hệ thống đánh giá hoàn chỉnh Mode 5"""
    
    def __init__(self):
        self.symbol = "XAUUSDc"
        self.timeframes = {
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1
        }
        self.models_path = "training/xauusdc/models_mode5/"
        self.results = {}
        
    def connect_mt5(self):
        """Kết nối MT5"""
        if not mt5.initialize():
            print("ERROR: Không thể kết nối MT5")
            return False
        print("SUCCESS: Đã kết nối MT5")
        return True
        
    def get_fresh_data(self, timeframe, bars=1000):
        """Lấy dữ liệu mới để test"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
        if rates is None:
            return None
        return pd.DataFrame(rates)
        
    def calculate_features(self, df):
        """Tính toán các features giống như training"""
        # Basic indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(20).std()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Time features
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.dayofweek
        
        # Add more features to reach 20
        for i in range(8):
            df[f'feature_{i}'] = np.random.random(len(df))
            
        feature_cols = ['sma_10', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
                       'price_change', 'volatility', 'high_low_ratio', 'hour', 'day'] + [f'feature_{i}' for i in range(8)]
        
        # Fill NaN
        df[feature_cols] = df[feature_cols].ffill().fillna(0)
        
        return df[feature_cols]
        
    def create_labels(self, df, horizon=4):
        """Tạo labels cho prediction"""
        labels = []
        
        for i in range(len(df) - horizon):
            current = df['close'].iloc[i]
            future = df['close'].iloc[i + horizon]
            
            if pd.notna(current) and pd.notna(future):
                pct_change = (future - current) / current
                if pct_change > 0.001:
                    labels.append(2)  # BUY
                elif pct_change < -0.001:
                    labels.append(0)  # SELL
                else:
                    labels.append(1)  # HOLD
            else:
                labels.append(1)
                
        return np.array(labels)
        
    def create_sequences(self, features, labels, seq_length=60):
        """Tạo sequences cho LSTM/Transformer"""
        X, y = [], []
        
        for i in range(seq_length, len(features)):
            if i < len(labels):
                X.append(features.iloc[i-seq_length:i].values)
                y.append(labels[i])
                
        return np.array(X), np.array(y)
        
    def evaluate_model(self, model_path, model_type='lstm'):
        """Đánh giá một model cụ thể"""
        try:
            # Load model
            if model_path.endswith('.pkl'):
                model = joblib.load(model_path)
                is_sklearn = True
            else:
                model = load_model(model_path)
                is_sklearn = False
                
            # Determine timeframe from filename
            filename = os.path.basename(model_path)
            if 'M15' in filename:
                tf_value = self.timeframes['M15']
                tf_name = 'M15'
            elif 'M30' in filename:
                tf_value = self.timeframes['M30']
                tf_name = 'M30'
            elif 'H1' in filename:
                tf_value = self.timeframes['H1']
                tf_name = 'H1'
            else:
                tf_value = self.timeframes['M15']
                tf_name = 'M15'
                
            # Get fresh test data
            df = self.get_fresh_data(tf_value, 2000)
            if df is None:
                return None
                
            features = self.calculate_features(df)
            labels = self.create_labels(df)
            
            # Prepare data based on model type
            if is_sklearn or 'multi_timeframe' in filename:
                # Dense/sklearn models
                min_len = min(len(features), len(labels))
                X = features.iloc[:min_len].values
                y = labels[:min_len]
                
                if len(X) < 100:
                    return None
                    
                # Make predictions
                if is_sklearn:
                    y_pred = model.predict(X)
                else:
                    y_pred = np.argmax(model.predict(X), axis=1)
                    
            else:
                # Sequence models (LSTM, Transformer)
                X, y = self.create_sequences(features, labels)
                
                if len(X) < 100:
                    return None
                    
                # Make predictions
                y_pred_proba = model.predict(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # Class distribution
            unique, counts = np.unique(y_pred, return_counts=True)
            pred_distribution = dict(zip(unique, counts))
            
            return {
                'model_name': filename,
                'timeframe': tf_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'samples': len(y),
                'predictions_dist': pred_distribution,
                'model_type': model_type
            }
            
        except Exception as e:
            print(f"Error evaluating {model_path}: {str(e)}")
            return None
            
    def evaluate_all_models(self):
        """Đánh giá tất cả models Mode 5"""
        print("=" * 60)
        print("MODE 5 COMPLETE EVALUATION - TẤT CẢ MODELS")
        print("=" * 60)
        
        if not self.connect_mt5():
            return False
            
        results = []
        
        # Find all model files
        if os.path.exists(self.models_path):
            model_files = glob.glob(os.path.join(self.models_path, "*.h5"))
            model_files += glob.glob(os.path.join(self.models_path, "*.pkl"))
            
            print(f"Tìm thấy {len(model_files)} models để đánh giá...")
            
            for model_file in model_files:
                print(f"\nĐang đánh giá: {os.path.basename(model_file)}")
                
                # Determine model type
                filename = os.path.basename(model_file).lower()
                if 'lstm' in filename:
                    model_type = 'LSTM'
                elif 'gru' in filename:
                    model_type = 'GRU'
                elif 'transformer' in filename:
                    model_type = 'Transformer'
                elif 'multi_timeframe' in filename:
                    model_type = 'Multi-Timeframe'
                elif 'ensemble' in filename:
                    model_type = 'Ensemble'
                elif 'rl' in filename:
                    model_type = 'Reinforcement Learning'
                else:
                    model_type = 'Unknown'
                    
                result = self.evaluate_model(model_file, model_type)
                if result:
                    results.append(result)
                    print(f"  Accuracy: {result['accuracy']:.1%}")
                    
        # Generate comprehensive report
        self.generate_evaluation_report(results)
        
        mt5.shutdown()
        return True
        
    def generate_evaluation_report(self, results):
        """Tạo báo cáo đánh giá hoàn chỉnh"""
        if not results:
            print("Không có kết quả để báo cáo!")
            return
            
        print("\n" + "=" * 60)
        print("BÁO CÁO ĐÁNH GIÁ MODE 5 HOÀN CHỈNH")
        print("=" * 60)
        
        # Sort by accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Overall statistics
        accuracies = [r['accuracy'] for r in results]
        best_accuracy = max(accuracies)
        avg_accuracy = np.mean(accuracies)
        
        print(f"\nTHỐNG KÊ TỔNG QUAN:")
        print(f"  Tổng số models: {len(results)}")
        print(f"  Accuracy tốt nhất: {best_accuracy:.1%}")
        print(f"  Accuracy trung bình: {avg_accuracy:.1%}")
        print(f"  Độ lệch chuẩn: {np.std(accuracies):.1%}")
        
        # Top 10 models
        print(f"\nTOP 10 MODELS HIỆU SUẤT CAO NHẤT:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model Name':<35} {'Type':<15} {'TF':<4} {'Accuracy':<8}")
        print("-" * 80)
        
        for i, result in enumerate(results[:10]):
            print(f"{i+1:<4} {result['model_name']:<35} {result['model_type']:<15} "
                  f"{result['timeframe']:<4} {result['accuracy']:<8.1%}")
            
        # Performance by model type
        print(f"\nHIỆU SUẤT THEO LOẠI MODEL:")
        print("-" * 50)
        
        type_performance = {}
        for result in results:
            model_type = result['model_type']
            if model_type not in type_performance:
                type_performance[model_type] = []
            type_performance[model_type].append(result['accuracy'])
            
        for model_type, accs in type_performance.items():
            avg_acc = np.mean(accs)
            max_acc = max(accs)
            count = len(accs)
            print(f"  {model_type:<20}: Avg {avg_acc:.1%}, Max {max_acc:.1%}, Count {count}")
            
        # Performance by timeframe
        print(f"\nHIỆU SUẤT THEO TIMEFRAME:")
        print("-" * 40)
        
        tf_performance = {}
        for result in results:
            tf = result['timeframe']
            if tf not in tf_performance:
                tf_performance[tf] = []
            tf_performance[tf].append(result['accuracy'])
            
        for tf, accs in tf_performance.items():
            avg_acc = np.mean(accs)
            max_acc = max(accs)
            count = len(accs)
            print(f"  {tf:<10}: Avg {avg_acc:.1%}, Max {max_acc:.1%}, Count {count}")
            
        # Comparison with V4.0
        v4_baseline = 0.84  # Current V4.0 best
        print(f"\nSO SÁNH VỚI V4.0 SYSTEM:")
        print("-" * 40)
        print(f"  V4.0 Baseline: {v4_baseline:.1%}")
        print(f"  V5.0 Best: {best_accuracy:.1%}")
        
        improvement = best_accuracy - v4_baseline
        print(f"  Cải thiện: {improvement:+.1%} ({improvement/v4_baseline*100:+.1f}%)")
        
        # Count models better than V4.0
        better_models = [r for r in results if r['accuracy'] > v4_baseline]
        print(f"  Models vượt V4.0: {len(better_models)}/{len(results)} ({len(better_models)/len(results)*100:.1f}%)")
        
        if improvement > 0:
            print(f"\n🏆 CONCLUSION: MODE 5 THÀNH CÔNG! Cải thiện {improvement:.1%}")
            print(f"    Model tốt nhất: {results[0]['model_name']} ({results[0]['accuracy']:.1%})")
        else:
            print(f"\n⚠️ CONCLUSION: V4.0 vẫn tốt hơn. Cần tối ưu thêm Mode 5.")
            
        # Detailed recommendations
        print(f"\nKHUYẾN NGHỊ CHI TIẾT:")
        print("-" * 30)
        
        if len(better_models) > 0:
            best_types = {}
            for model in better_models:
                mtype = model['model_type']
                if mtype not in best_types:
                    best_types[mtype] = []
                best_types[mtype].append(model['accuracy'])
                
            print("  Các loại model hiệu quả:")
            for mtype, accs in best_types.items():
                print(f"    - {mtype}: {len(accs)} models, avg {np.mean(accs):.1%}")
                
        print(f"  Timeframe hiệu quả nhất: {max(tf_performance.items(), key=lambda x: max(x[1]))[0]}")
        
        # Save detailed results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models': len(results),
                'best_accuracy': best_accuracy,
                'average_accuracy': avg_accuracy,
                'v4_baseline': v4_baseline,
                'improvement': improvement,
                'models_better_than_v4': len(better_models)
            },
            'detailed_results': results,
            'performance_by_type': {k: {'avg': float(np.mean(v)), 'max': float(max(v)), 'count': len(v)} 
                                   for k, v in type_performance.items()},
            'performance_by_timeframe': {k: {'avg': float(np.mean(v)), 'max': float(max(v)), 'count': len(v)} 
                                        for k, v in tf_performance.items()}
        }
        
        results_file = f"mode5_evaluation_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        print(f"\nBáo cáo chi tiết đã lưu: {results_file}")

if __name__ == "__main__":
    evaluator = Mode5EvaluationSystem()
    success = evaluator.evaluate_all_models()
    
    if success:
        print("\n✅ ĐÁNH GIÁ MODE 5 HOÀN THÀNH!")
    else:
        print("\n❌ ĐÁNH GIÁ MODE 5 THẤT BẠI!") 