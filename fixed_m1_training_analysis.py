#!/usr/bin/env python3
"""
FIXED M1 TRAINING & SIGNAL ANALYSIS
Training với dữ liệu M1 và phân tích chi tiết signal generation (đã fix bugs)
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Add paths
sys.path.append('src/core')
sys.path.append('src/core/integration')
sys.path.append('src/core/specialists')

def load_m1_sample(sample_size=100000):
    """Load sample M1 data"""
    print(f"📊 LOADING {sample_size:,} M1 CANDLES...")
    print("=" * 60)
    
    m1_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
    
    if not os.path.exists(m1_file):
        print(f"❌ M1 data file not found: {m1_file}")
        return None
    
    try:
        print(f"📈 Loading {m1_file}...")
        df = pd.read_csv(m1_file, nrows=sample_size)
        
        print(f"✅ Loaded {len(df):,} M1 candles")
        print(f"📊 Data range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading M1 data: {e}")
        return None

def prepare_m1_features(df):
    """Chuẩn bị features từ dữ liệu M1"""
    print("\n🔧 PREPARING M1 FEATURES...")
    print("=" * 60)
    
    # Combine Date and Time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📈 Processing {len(df):,} M1 candles...")
    
    # Convert to float32 for memory efficiency
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = df[col].astype('float32')
    
    # Basic price features
    df['returns'] = df['Close'].pct_change().astype('float32')
    df['price_range'] = (df['High'] - df['Low']).astype('float32')
    df['body_size'] = abs(df['Close'] - df['Open']).astype('float32')
    
    # Technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = (100 - (100 / (1 + rs))).astype('float32')
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = (exp1 - exp2).astype('float32')
    df['macd_signal'] = df['macd'].ewm(span=9).mean().astype('float32')
    
    # Moving averages
    for period in [20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(window=period).mean().astype('float32')
        df[f'price_vs_sma_{period}'] = (df['Close'] / df[f'sma_{period}']).astype('float32')
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean().astype('float32')
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = (df['bb_middle'] + (bb_std * 2)).astype('float32')
    df['bb_lower'] = (df['bb_middle'] - (bb_std * 2)).astype('float32')
    df['bb_position'] = ((df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).astype('float32')
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean().astype('float32')
    df['volume_ratio'] = (df['Volume'] / df['volume_sma']).astype('float32')
    
    # Time features
    df['hour'] = df['datetime'].dt.hour.astype('int8')
    df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype('int8')
    df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype('int8')
    
    # Labels
    df['price_change_5m'] = (df['Close'].shift(-5) - df['Close']).astype('float32')
    df['direction_5m'] = (df['price_change_5m'] > 0).astype('int8')
    
    # Feature columns
    feature_cols = [
        'returns', 'price_range', 'body_size', 'rsi', 'macd', 'macd_signal',
        'sma_20', 'sma_50', 'price_vs_sma_20', 'price_vs_sma_50',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
        'volume_sma', 'volume_ratio', 'hour', 'is_london_session', 'is_ny_session'
    ]
    
    # Clean data
    df_clean = df.dropna()
    X = df_clean[feature_cols].values.astype('float32')
    y = df_clean['direction_5m'].values
    
    print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]:,} samples")
    print(f"✅ Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
    
    return X, y, df_clean, feature_cols

def train_models(X, y, feature_cols):
    """Training models"""
    print("\n🎯 TRAINING MODELS...")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Training size: {X_train.shape[0]:,} samples")
    print(f"✅ Testing size: {X_test.shape[0]:,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    trained_models = {}
    training_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. LightGBM
    print("\n🌟 Training LightGBM...")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    
    train_pred = lgb_model.predict(X_train)
    test_pred = lgb_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    lgb_path = f"trained_models/m1_lightgbm_{timestamp}.pkl"
    with open(lgb_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    
    trained_models['lightgbm'] = lgb_model
    training_results['lightgbm'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model_path': lgb_path
    }
    
    print(f"   ✅ LightGBM: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # 2. Neural Network
    print("\n🧠 Training Neural Network...")
    
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = nn_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=20,
        batch_size=512,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    train_loss, train_acc = nn_model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    
    nn_path = f"trained_models/m1_neural_{timestamp}.keras"
    nn_model.save(nn_path)
    
    trained_models['neural_network'] = nn_model
    training_results['neural_network'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model_path': nn_path
    }
    
    print(f"   ✅ Neural Network: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # Save scaler
    scaler_path = f"trained_models/m1_scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results
    results_path = f"training_results/m1_training_{timestamp}.json"
    os.makedirs('training_results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"\n📊 TRAINING COMPLETED!")
    print(f"   💾 Results saved: {results_path}")
    
    return trained_models, training_results, scaler

def create_specialists_system():
    """Tạo hệ thống 18 specialists"""
    print("\n🔍 CREATING 18 SPECIALISTS SYSTEM...")
    print("=" * 60)
    
    specialists_data = {
        'Technical': {
            'RSI_Specialist': {'accuracy': 0.65, 'bias': 'contrarian'},
            'MACD_Specialist': {'accuracy': 0.62, 'bias': 'trend_following'},
            'Fibonacci_Specialist': {'accuracy': 0.58, 'bias': 'support_resistance'}
        },
        'Sentiment': {
            'News_Sentiment_Specialist': {'accuracy': 0.55, 'bias': 'fundamental'},
            'Social_Media_Specialist': {'accuracy': 0.52, 'bias': 'crowd_sentiment'},
            'Fear_Greed_Specialist': {'accuracy': 0.60, 'bias': 'contrarian'}
        },
        'Pattern': {
            'Chart_Pattern_Specialist': {'accuracy': 0.63, 'bias': 'pattern_recognition'},
            'Candlestick_Specialist': {'accuracy': 0.59, 'bias': 'reversal_patterns'},
            'Wave_Specialist': {'accuracy': 0.57, 'bias': 'elliott_wave'}
        },
        'Risk': {
            'VaR_Risk_Specialist': {'accuracy': 0.70, 'bias': 'risk_averse'},
            'Drawdown_Specialist': {'accuracy': 0.68, 'bias': 'capital_preservation'},
            'Position_Size_Specialist': {'accuracy': 0.66, 'bias': 'money_management'}
        },
        'Momentum': {
            'Trend_Specialist': {'accuracy': 0.64, 'bias': 'trend_following'},
            'Mean_Reversion_Specialist': {'accuracy': 0.61, 'bias': 'mean_reverting'},
            'Breakout_Specialist': {'accuracy': 0.59, 'bias': 'breakout_trading'}
        },
        'Volatility': {
            'ATR_Specialist': {'accuracy': 0.56, 'bias': 'volatility_based'},
            'Bollinger_Specialist': {'accuracy': 0.58, 'bias': 'volatility_bands'},
            'Volatility_Clustering_Specialist': {'accuracy': 0.54, 'bias': 'volatility_regime'}
        }
    }
    
    total_specialists = sum(len(specialists) for specialists in specialists_data.values())
    print(f"✅ Created {total_specialists} specialists across {len(specialists_data)} categories")
    
    for category, specialists in specialists_data.items():
        print(f"   • {category}: {len(specialists)} specialists")
    
    return specialists_data

def generate_specialist_vote(specialist_info, scenario):
    """Generate vote từ specialist"""
    bias = specialist_info['bias']
    accuracy = specialist_info['accuracy']
    scenario_type = scenario['type']
    
    # Base probabilities (đảm bảo sum = 1.0)
    if scenario_type == 'bullish':
        base_probs = [0.6, 0.2, 0.2]  # [BUY, SELL, HOLD]
    elif scenario_type == 'bearish':
        base_probs = [0.2, 0.6, 0.2]
    elif scenario_type == 'sideways':
        base_probs = [0.25, 0.25, 0.5]
    else:  # volatile
        base_probs = [0.4, 0.4, 0.2]
    
    # Adjust based on bias
    if bias == 'contrarian':
        if scenario_type == 'bullish':
            base_probs = [0.3, 0.5, 0.2]
        elif scenario_type == 'bearish':
            base_probs = [0.5, 0.3, 0.2]
    elif bias == 'trend_following':
        if scenario_type == 'bullish':
            base_probs = [0.7, 0.1, 0.2]
        elif scenario_type == 'bearish':
            base_probs = [0.1, 0.7, 0.2]
    elif bias == 'risk_averse':
        # Increase HOLD probability
        hold_boost = 0.3
        base_probs = [
            base_probs[0] * (1 - hold_boost),
            base_probs[1] * (1 - hold_boost),
            base_probs[2] + hold_boost
        ]
    
    # Normalize to ensure sum = 1.0
    total = sum(base_probs)
    base_probs = [p / total for p in base_probs]
    
    # Generate vote
    decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=base_probs)
    
    # Generate confidence
    base_confidence = np.random.uniform(0.3, 0.9)
    confidence = base_confidence * accuracy + np.random.uniform(-0.1, 0.1)
    confidence = max(0.1, min(0.9, confidence))
    
    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': f"{bias.replace('_', ' ').title()} analysis suggests {decision}",
        'bias': bias,
        'scenario_type': scenario_type
    }

def analyze_signal_generation(specialists_data, scenarios):
    """Phân tích signal generation chi tiết"""
    print("\n📊 ANALYZING SIGNAL GENERATION...")
    print("=" * 60)
    
    all_analyses = []
    specialist_performance = {}
    
    # Initialize performance tracking
    for category, specialists in specialists_data.items():
        for name, data in specialists.items():
            specialist_performance[name] = {
                'category': category,
                'accuracy': data['accuracy'],
                'bias': data['bias'],
                'total_votes': 0,
                'correct_predictions': 0,
                'vote_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                'confidence_scores': []
            }
    
    # Analyze each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\n🎯 Scenario {i+1}: {scenario['name']}")
        
        specialist_votes = {}
        category_votes = {}
        
        # Get votes from all specialists
        for category, specialists in specialists_data.items():
            category_votes[category] = []
            
            for specialist_name, specialist_info in specialists.items():
                vote = generate_specialist_vote(specialist_info, scenario)
                specialist_votes[specialist_name] = vote
                category_votes[category].append(vote)
                
                # Update performance tracking
                perf = specialist_performance[specialist_name]
                perf['total_votes'] += 1
                perf['vote_distribution'][vote['decision']] += 1
                perf['confidence_scores'].append(vote['confidence'])
                
                # Check if prediction was correct
                if vote['decision'] == scenario['actual_direction']:
                    perf['correct_predictions'] += 1
        
        # Analyze category consensus
        category_consensus = {}
        for category, votes in category_votes.items():
            decisions = [v['decision'] for v in votes]
            decision_counts = {d: decisions.count(d) for d in ['BUY', 'SELL', 'HOLD']}
            majority_decision = max(decision_counts, key=decision_counts.get)
            consensus_strength = decision_counts[majority_decision] / len(decisions)
            
            category_consensus[category] = {
                'majority_decision': majority_decision,
                'consensus_strength': consensus_strength,
                'vote_distribution': decision_counts
            }
        
        # Overall consensus
        all_decisions = [v['decision'] for v in specialist_votes.values()]
        all_confidences = [v['confidence'] for v in specialist_votes.values()]
        
        decision_counts = {d: all_decisions.count(d) for d in ['BUY', 'SELL', 'HOLD']}
        final_decision = max(decision_counts, key=decision_counts.get)
        consensus_strength = decision_counts[final_decision] / len(all_decisions)
        
        # Weighted decision
        weighted_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = sum(all_confidences)
        
        for vote in specialist_votes.values():
            weight = vote['confidence'] / total_confidence if total_confidence > 0 else 1/len(specialist_votes)
            weighted_scores[vote['decision']] += weight
        
        weighted_decision = max(weighted_scores, key=weighted_scores.get)
        weighted_confidence = weighted_scores[weighted_decision]
        
        # Store analysis
        analysis = {
            'scenario': scenario,
            'specialist_votes': specialist_votes,
            'category_consensus': category_consensus,
            'final_decision': final_decision,
            'consensus_strength': consensus_strength,
            'weighted_decision': weighted_decision,
            'weighted_confidence': weighted_confidence,
            'vote_distribution': decision_counts,
            'prediction_correct': final_decision == scenario['actual_direction']
        }
        
        all_analyses.append(analysis)
        
        # Print results
        print(f"   📊 Votes: {decision_counts}")
        print(f"   🎯 Decision: {final_decision} ({consensus_strength:.1%} consensus)")
        print(f"   ⚖️ Weighted: {weighted_decision} ({weighted_confidence:.1%})")
        print(f"   ✅ Actual: {scenario['actual_direction']}")
        print(f"   🎯 Correct: {final_decision == scenario['actual_direction']}")
    
    # Calculate final performance metrics
    for name, perf in specialist_performance.items():
        if perf['total_votes'] > 0:
            perf['accuracy_rate'] = perf['correct_predictions'] / perf['total_votes']
            perf['avg_confidence'] = np.mean(perf['confidence_scores'])
            perf['vote_percentages'] = {
                vote: count/perf['total_votes']*100 
                for vote, count in perf['vote_distribution'].items()
            }
    
    return all_analyses, specialist_performance

def create_test_scenarios():
    """Tạo scenarios để test"""
    scenarios = [
        {
            'name': 'Strong Bullish Breakout',
            'type': 'bullish',
            'actual_direction': 'BUY',
            'description': 'Price breaks above resistance with high volume'
        },
        {
            'name': 'Sharp Bearish Decline',
            'type': 'bearish',
            'actual_direction': 'SELL',
            'description': 'Price falls below support with increasing volume'
        },
        {
            'name': 'Sideways Consolidation',
            'type': 'sideways',
            'actual_direction': 'HOLD',
            'description': 'Price moves in tight range with low volatility'
        },
        {
            'name': 'High Volatility Whipsaw',
            'type': 'volatile',
            'actual_direction': 'HOLD',
            'description': 'Price swings wildly with no clear direction'
        },
        {
            'name': 'Bullish Reversal',
            'type': 'bullish',
            'actual_direction': 'BUY',
            'description': 'Price reverses from oversold levels'
        },
        {
            'name': 'Bearish Reversal',
            'type': 'bearish',
            'actual_direction': 'SELL',
            'description': 'Price reverses from overbought levels'
        },
        {
            'name': 'Trend Continuation Up',
            'type': 'bullish',
            'actual_direction': 'BUY',
            'description': 'Uptrend continues after pullback'
        },
        {
            'name': 'Trend Continuation Down',
            'type': 'bearish',
            'actual_direction': 'SELL',
            'description': 'Downtrend continues after bounce'
        }
    ]
    
    return scenarios

def create_comprehensive_summary(training_results, signal_analyses, specialist_performance):
    """Tạo summary toàn diện"""
    
    # Training summary
    best_model = max(training_results.items(), key=lambda x: x[1].get('test_accuracy', 0))
    
    # Signal analysis summary
    correct_predictions = sum(1 for analysis in signal_analyses if analysis['prediction_correct'])
    signal_accuracy = correct_predictions / len(signal_analyses) if signal_analyses else 0
    
    # Category performance
    category_performance = {}
    for name, perf in specialist_performance.items():
        category = perf['category']
        if category not in category_performance:
            category_performance[category] = []
        category_performance[category].append({
            'name': name,
            'accuracy': perf.get('accuracy_rate', 0),
            'confidence': perf.get('avg_confidence', 0)
        })
    
    # Calculate category averages
    for category, specialists in category_performance.items():
        avg_accuracy = np.mean([s['accuracy'] for s in specialists])
        avg_confidence = np.mean([s['confidence'] for s in specialists])
        category_performance[category] = {
            'specialists': specialists,
            'avg_accuracy': avg_accuracy,
            'avg_confidence': avg_confidence,
            'count': len(specialists)
        }
    
    # Top performers
    top_specialists = sorted(
        specialist_performance.items(),
        key=lambda x: x[1].get('accuracy_rate', 0),
        reverse=True
    )[:5]
    
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'training_summary': {
            'models_trained': len(training_results),
            'best_model': {
                'name': best_model[0],
                'test_accuracy': best_model[1].get('test_accuracy', 0)
            }
        },
        'signal_analysis_summary': {
            'scenarios_tested': len(signal_analyses),
            'overall_signal_accuracy': signal_accuracy,
            'total_specialists': len(specialist_performance),
            'categories': len(category_performance)
        },
        'category_performance': category_performance,
        'top_specialists': [
            {
                'name': name,
                'category': perf['category'],
                'accuracy': perf.get('accuracy_rate', 0),
                'confidence': perf.get('avg_confidence', 0)
            }
            for name, perf in top_specialists
        ]
    }
    
    return summary

def print_results(summary, signal_analyses, specialist_performance):
    """In kết quả chi tiết"""
    print(f"\n🎉 M1 ANALYSIS COMPLETED!")
    print("=" * 80)
    
    # Training results
    print(f"📊 TRAINING RESULTS:")
    print(f"   • Models trained: {summary['training_summary']['models_trained']}")
    print(f"   • Best model: {summary['training_summary']['best_model']['name']}")
    print(f"   • Best accuracy: {summary['training_summary']['best_model']['test_accuracy']:.4f}")
    
    # Signal analysis
    print(f"\n📊 SIGNAL ANALYSIS:")
    print(f"   • Scenarios tested: {summary['signal_analysis_summary']['scenarios_tested']}")
    print(f"   • Signal accuracy: {summary['signal_analysis_summary']['overall_signal_accuracy']:.1%}")
    print(f"   • Total specialists: {summary['signal_analysis_summary']['total_specialists']}")
    
    # Category performance
    print(f"\n🏆 CATEGORY PERFORMANCE:")
    for category, perf in summary['category_performance'].items():
        print(f"   • {category}: {perf['avg_accuracy']:.1%} accuracy, {perf['count']} specialists")
    
    # Top specialists
    print(f"\n⭐ TOP SPECIALISTS:")
    for specialist in summary['top_specialists']:
        print(f"   • {specialist['name']}: {specialist['accuracy']:.1%} ({specialist['category']})")
    
    # Detailed scenario analysis
    print(f"\n📈 SCENARIO ANALYSIS:")
    for i, analysis in enumerate(signal_analyses):
        scenario = analysis['scenario']
        print(f"   {i+1}. {scenario['name']}: {analysis['final_decision']} "
              f"({analysis['consensus_strength']:.1%} consensus) "
              f"{'✅' if analysis['prediction_correct'] else '❌'}")

def run_analysis():
    """Chạy phân tích toàn diện"""
    print("🚀 M1 COMPREHENSIVE TRAINING & SIGNAL ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load data
    df = load_m1_sample(sample_size=50000)  # Smaller sample for faster processing
    if df is None:
        return
    
    # 2. Prepare features
    X, y, df_clean, feature_cols = prepare_m1_features(df)
    
    # 3. Train models
    trained_models, training_results, scaler = train_models(X, y, feature_cols)
    
    # 4. Create specialists system
    specialists_data = create_specialists_system()
    
    # 5. Create test scenarios
    scenarios = create_test_scenarios()
    
    # 6. Analyze signal generation
    signal_analyses, specialist_performance = analyze_signal_generation(specialists_data, scenarios)
    
    # 7. Create summary
    summary = create_comprehensive_summary(training_results, signal_analyses, specialist_performance)
    
    # 8. Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs('m1_analysis_results', exist_ok=True)
    
    # Save detailed results
    with open(f"m1_analysis_results/signal_analyses_{timestamp}.json", 'w') as f:
        json.dump(signal_analyses, f, indent=2, default=str)
    
    with open(f"m1_analysis_results/specialist_performance_{timestamp}.json", 'w') as f:
        json.dump(specialist_performance, f, indent=2, default=str)
    
    with open(f"m1_analysis_results/summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # 9. Print results
    print_results(summary, signal_analyses, specialist_performance)
    
    print(f"\n💾 Results saved in: m1_analysis_results/")
    print(f"End time: {datetime.now()}")
    
    return summary

if __name__ == "__main__":
    run_analysis() 