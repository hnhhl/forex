#!/usr/bin/env python3
"""
SMART TRAINING THỰC TẾ - KẾT QUẢ TRAINING QUA TỪNG PHASE
Sử dụng dữ liệu MT5 thực tế và đo lường performance cụ thể
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class RealSmartTraining:
    """Smart Training thực tế với dữ liệu MT5"""
    
    def __init__(self):
        self.data_path = "data/maximum_mt5_v2"
        self.results_path = "real_training_results"
        os.makedirs(self.results_path, exist_ok=True)
        
        # Training state
        self.raw_data = None
        self.processed_data = None
        self.models = {}
        self.phase_results = {}
        
    def load_real_mt5_data(self):
        """Load dữ liệu MT5 thực tế"""
        print("📊 LOADING REAL MT5 DATA")
        print("=" * 50)
        
        try:
            # Load multiple timeframes
            timeframes = ['H1', 'H4', 'D1']  # Focus on stable timeframes
            all_data = []
            
            for tf in timeframes:
                csv_file = f"{self.data_path}/XAUUSDc_{tf}_20250618_115847.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df['timeframe'] = tf
                    all_data.append(df)
                    print(f"   ✓ {tf}: {len(df)} records loaded")
                else:
                    print(f"   ❌ {tf}: File not found")
            
            if all_data:
                self.raw_data = pd.concat(all_data, ignore_index=True)
                print(f"\n📈 Total raw data: {len(self.raw_data)} records")
                print(f"   Timeframes: {self.raw_data['timeframe'].value_counts().to_dict()}")
                return True
            else:
                print("❌ No data loaded")
                return False
                
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def phase1_data_intelligence(self):
        """Phase 1: Data Intelligence & Feature Engineering"""
        print("\n🧠 PHASE 1: DATA INTELLIGENCE & FEATURE ENGINEERING")
        print("=" * 60)
        
        if self.raw_data is None:
            print("❌ No raw data available")
            return False
        
        try:
            data = self.raw_data.copy()
            
            # 1. Basic features
            print("🔧 Creating basic features...")
            data['price_change'] = data['close'].pct_change()
            data['volatility'] = (data['high'] - data['low']) / data['close']
            data['volume_ma'] = data['volume'].rolling(20).mean()
            
            # 2. Technical indicators
            print("📈 Adding technical indicators...")
            # SMA
            for period in [5, 10, 20, 50]:
                data[f'sma_{period}'] = data['close'].rolling(period).mean()
                data[f'price_vs_sma_{period}'] = data['close'] / data[f'sma_{period}'] - 1
            
            # EMA
            for period in [12, 26]:
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # 3. Create target variable (next period direction)
            data['future_price'] = data['close'].shift(-1)
            data['target'] = (data['future_price'] > data['close']).astype(int)
            
            # 4. Select features and clean data
            feature_cols = [col for col in data.columns if col not in 
                          ['time', 'open', 'high', 'low', 'close', 'volume', 'timeframe', 'future_price']]
            
            # Remove rows with NaN
            data_clean = data[feature_cols + ['target']].dropna()
            
            print(f"✅ Feature engineering completed:")
            print(f"   Features created: {len(feature_cols)}")
            print(f"   Clean records: {len(data_clean)}")
            print(f"   Target distribution: {data_clean['target'].value_counts().to_dict()}")
            
            self.processed_data = data_clean
            
            # Phase 1 results
            self.phase_results['phase1'] = {
                'raw_records': len(self.raw_data),
                'processed_records': len(data_clean),
                'features_count': len(feature_cols),
                'target_balance': data_clean['target'].mean(),
                'data_quality_score': 1 - (data[feature_cols].isnull().sum().sum() / (len(data) * len(feature_cols)))
            }
            
            print(f"📊 Phase 1 Results:")
            for key, value in self.phase_results['phase1'].items():
                print(f"   {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            return False
    
    def phase2_curriculum_learning(self):
        """Phase 2: Curriculum Learning - Train từ dễ đến khó"""
        print("\n📚 PHASE 2: CURRICULUM LEARNING")
        print("=" * 60)
        
        if self.processed_data is None:
            print("❌ No processed data available")
            return False
        
        try:
            data = self.processed_data.copy()
            
            # 1. Sort by volatility (dễ = low volatility, khó = high volatility)
            if 'volatility' in data.columns:
                data = data.sort_values('volatility')
                print("✓ Data sorted by volatility (easy → hard)")
            
            # 2. Create curriculum batches
            total_samples = len(data)
            batch_sizes = [
                int(total_samples * 0.2),  # 20% easiest
                int(total_samples * 0.4),  # 40% medium
                int(total_samples * 0.6),  # 60% harder
                total_samples                # 100% all data
            ]
            
            curriculum_results = []
            
            for i, batch_size in enumerate(batch_sizes):
                print(f"\n📖 Curriculum Stage {i+1}: Training on {batch_size} samples")
                
                # Get batch data
                batch_data = data.iloc[:batch_size]
                X = batch_data.drop('target', axis=1)
                y = batch_data['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if SKLEARN_AVAILABLE:
                    # Train simple model
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    
                    curriculum_results.append({
                        'stage': i+1,
                        'samples': batch_size,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'avg_volatility': batch_data['volatility'].mean() if 'volatility' in batch_data.columns else 0
                    })
                    
                    print(f"   Train Accuracy: {train_acc:.4f}")
                    print(f"   Test Accuracy: {test_acc:.4f}")
                else:
                    print("   ⚠️ Sklearn not available, skipping model training")
            
            # Calculate curriculum effectiveness
            if curriculum_results:
                convergence_improvement = (curriculum_results[-1]['test_accuracy'] - curriculum_results[0]['test_accuracy']) / curriculum_results[0]['test_accuracy']
                print(f"\n📈 Curriculum Learning Effectiveness: {convergence_improvement:.2%} improvement")
            
            self.phase_results['phase2'] = {
                'curriculum_stages': len(batch_sizes),
                'curriculum_results': curriculum_results,
                'convergence_improvement': convergence_improvement if curriculum_results else 0
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 2 failed: {e}")
            return False
    
    def phase3_ensemble_training(self):
        """Phase 3: Train multiple models và ensemble"""
        print("\n🤝 PHASE 3: ENSEMBLE MODEL TRAINING")
        print("=" * 60)
        
        if self.processed_data is None:
            print("❌ No processed data available")
            return False
        
        try:
            data = self.processed_data.copy()
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models_performance = {}
            
            # 1. Random Forest
            if SKLEARN_AVAILABLE:
                print("🌲 Training Random Forest...")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_acc = accuracy_score(y_test, rf_pred)
                models_performance['RandomForest'] = rf_acc
                self.models['RandomForest'] = rf_model
                print(f"   Random Forest Accuracy: {rf_acc:.4f}")
            
            # 2. XGBoost
            if SKLEARN_AVAILABLE:
                print("🚀 Training XGBoost...")
                xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_test)
                xgb_acc = accuracy_score(y_test, xgb_pred)
                models_performance['XGBoost'] = xgb_acc
                self.models['XGBoost'] = xgb_model
                print(f"   XGBoost Accuracy: {xgb_acc:.4f}")
            
            # 3. LSTM (if TensorFlow available)
            if TENSORFLOW_AVAILABLE and len(X_train_scaled) > 100:
                print("🧠 Training LSTM...")
                try:
                    # Reshape for LSTM
                    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                    
                    lstm_model = Sequential([
                        LSTM(50, input_shape=(1, X_train_scaled.shape[1])),
                        Dropout(0.2),
                        Dense(25),
                        Dense(1, activation='sigmoid')
                    ])
                    
                    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
                    
                    lstm_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
                    lstm_acc = accuracy_score(y_test, lstm_pred)
                    models_performance['LSTM'] = lstm_acc
                    self.models['LSTM'] = lstm_model
                    print(f"   LSTM Accuracy: {lstm_acc:.4f}")
                except Exception as e:
                    print(f"   ⚠️ LSTM training failed: {e}")
            
            # 4. Ensemble prediction
            if len(models_performance) > 1:
                print("\n🤝 Creating Ensemble...")
                ensemble_preds = []
                
                for model_name, model in self.models.items():
                    if model_name == 'LSTM':
                        pred = (model.predict(X_test_lstm) > 0.5).astype(int).flatten()
                    else:
                        pred = model.predict(X_test)
                    ensemble_preds.append(pred)
                
                # Majority voting
                ensemble_pred = np.round(np.mean(ensemble_preds, axis=0)).astype(int)
                ensemble_acc = accuracy_score(y_test, ensemble_pred)
                models_performance['Ensemble'] = ensemble_acc
                print(f"   Ensemble Accuracy: {ensemble_acc:.4f}")
            
            # Find best model
            best_model = max(models_performance, key=models_performance.get)
            best_accuracy = models_performance[best_model]
            
            print(f"\n🏆 Best Model: {best_model} ({best_accuracy:.4f})")
            
            self.phase_results['phase3'] = {
                'models_trained': list(models_performance.keys()),
                'individual_accuracies': models_performance,
                'best_model': best_model,
                'best_accuracy': best_accuracy,
                'ensemble_improvement': models_performance.get('Ensemble', 0) - max([acc for name, acc in models_performance.items() if name != 'Ensemble'])
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
            return False
    
    def phase4_performance_validation(self):
        """Phase 4: Validate performance với cross-validation"""
        print("\n✅ PHASE 4: PERFORMANCE VALIDATION")
        print("=" * 60)
        
        if not self.models or self.processed_data is None:
            print("❌ No models or data available")
            return False
        
        try:
            data = self.processed_data.copy()
            X = data.drop('target', axis=1)
            y = data['target']
            
            validation_results = {}
            
            for model_name, model in self.models.items():
                if model_name == 'LSTM':
                    continue  # Skip LSTM for cross-validation (complex reshaping)
                
                print(f"🔍 Validating {model_name}...")
                
                # 5-fold cross validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                
                validation_results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                print(f"   CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.phase_results['phase4'] = validation_results
            
            return True
            
        except Exception as e:
            print(f"❌ Phase 4 failed: {e}")
            return False
    
    def save_results(self):
        """Save tất cả kết quả training"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{self.results_path}/real_smart_training_results_{timestamp}.json"
            
            # Prepare results summary
            results_summary = {
                'timestamp': timestamp,
                'training_phases': self.phase_results,
                'final_summary': self.generate_final_summary()
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Results saved: {results_file}")
            return results_file
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return None
    
    def generate_final_summary(self):
        """Generate final summary"""
        summary = {}
        
        if 'phase1' in self.phase_results:
            summary['data_processing'] = {
                'raw_records': self.phase_results['phase1']['raw_records'],
                'processed_records': self.phase_results['phase1']['processed_records'],
                'features_created': self.phase_results['phase1']['features_count'],
                'data_quality': f"{self.phase_results['phase1']['data_quality_score']:.2%}"
            }
        
        if 'phase2' in self.phase_results:
            summary['curriculum_learning'] = {
                'stages_completed': self.phase_results['phase2']['curriculum_stages'],
                'convergence_improvement': f"{self.phase_results['phase2']['convergence_improvement']:.2%}"
            }
        
        if 'phase3' in self.phase_results:
            summary['ensemble_training'] = {
                'models_trained': self.phase_results['phase3']['models_trained'],
                'best_model': self.phase_results['phase3']['best_model'],
                'best_accuracy': f"{self.phase_results['phase3']['best_accuracy']:.4f}",
                'ensemble_improvement': f"{self.phase_results['phase3']['ensemble_improvement']:.4f}"
            }
        
        if 'phase4' in self.phase_results:
            best_cv_model = max(self.phase_results['phase4'], 
                              key=lambda x: self.phase_results['phase4'][x]['cv_mean'])
            summary['validation'] = {
                'best_cv_model': best_cv_model,
                'best_cv_score': f"{self.phase_results['phase4'][best_cv_model]['cv_mean']:.4f}",
                'cv_stability': f"{self.phase_results['phase4'][best_cv_model]['cv_std']:.4f}"
            }
        
        return summary
    
    def run_complete_training(self):
        """Run complete smart training pipeline"""
        print("🚀 REAL SMART TRAINING EXECUTION")
        print("=" * 80)
        
        success_phases = 0
        
        # Phase 1: Data Intelligence
        if self.load_real_mt5_data() and self.phase1_data_intelligence():
            success_phases += 1
        
        # Phase 2: Curriculum Learning
        if self.phase2_curriculum_learning():
            success_phases += 1
        
        # Phase 3: Ensemble Training
        if self.phase3_ensemble_training():
            success_phases += 1
        
        # Phase 4: Performance Validation
        if self.phase4_performance_validation():
            success_phases += 1
        
        # Save results
        results_file = self.save_results()
        
        # Final summary
        print(f"\n🎯 TRAINING COMPLETED")
        print("=" * 50)
        print(f"Phases completed: {success_phases}/4")
        
        if success_phases == 4:
            summary = self.generate_final_summary()
            print("\n📊 FINAL RESULTS:")
            for section, data in summary.items():
                print(f"\n{section.upper()}:")
                for key, value in data.items():
                    print(f"   {key}: {value}")
        
        return success_phases == 4

def main():
    trainer = RealSmartTraining()
    success = trainer.run_complete_training()
    
    if success:
        print("\n✅ REAL SMART TRAINING SUCCESSFUL")
    else:
        print("\n❌ TRAINING INCOMPLETE")

if __name__ == "__main__":
    main() 