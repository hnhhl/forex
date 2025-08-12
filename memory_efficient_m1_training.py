#!/usr/bin/env python3
"""
MEMORY EFFICIENT M1 TRAINING & SIGNAL ANALYSIS
Training với 1 triệu nến M1 sử dụng memory efficient approach
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

# Add paths
sys.path.append('src/core')
sys.path.append('src/core/integration')
sys.path.append('src/core/specialists')

def load_m1_sample(sample_size=100000):
    """Load sample M1 data để test memory efficiency"""
    print(f"📊 LOADING {sample_size:,} M1 CANDLES SAMPLE...")
    print("=" * 60)
    
    m1_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
    
    if not os.path.exists(m1_file):
        print(f"❌ M1 data file not found: {m1_file}")
        return None
    
    try:
        print(f"📈 Loading {m1_file}...")
        # Load in chunks to save memory
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_csv(m1_file, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= sample_size:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        df = df.tail(sample_size)  # Take latest samples
        
        print(f"✅ Loaded {len(df):,} M1 candles")
        print(f"📊 Data range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading M1 data: {e}")
        return None

def prepare_m1_features_efficient(df):
    """Chuẩn bị features với memory efficient approach"""
    print("\n🔧 PREPARING M1 FEATURES (MEMORY EFFICIENT)...")
    print("=" * 60)
    
    # Combine Date and Time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"📈 Processing {len(df):,} M1 candles...")
    
    # Use float32 to save memory
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = df[col].astype('float32')
    
    # Essential features only (reduced set)
    df['returns'] = df['Close'].pct_change().astype('float32')
    df['price_range'] = (df['High'] - df['Low']).astype('float32')
    df['body_size'] = abs(df['Close'] - df['Open']).astype('float32')
    
    # Fast technical indicators (essential only)
    # RSI (14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = (100 - (100 / (1 + rs))).astype('float32')
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = (exp1 - exp2).astype('float32')
    df['macd_signal'] = df['macd'].ewm(span=9).mean().astype('float32')
    
    # Moving averages (key periods only)
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
    
    # Time features (essential)
    df['hour'] = df['datetime'].dt.hour.astype('int8')
    df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype('int8')
    df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype('int8')
    
    # Labels (multiple timeframes)
    df['price_change_1m'] = (df['Close'].shift(-1) - df['Close']).astype('float32')
    df['direction_1m'] = (df['price_change_1m'] > 0).astype('int8')
    
    df['price_change_5m'] = (df['Close'].shift(-5) - df['Close']).astype('float32')
    df['direction_5m'] = (df['price_change_5m'] > 0).astype('int8')
    
    df['price_change_15m'] = (df['Close'].shift(-15) - df['Close']).astype('float32')
    df['direction_15m'] = (df['price_change_15m'] > 0).astype('int8')
    
    # Feature columns (reduced set)
    feature_cols = [
        'returns', 'price_range', 'body_size', 'rsi', 'macd', 'macd_signal',
        'sma_20', 'sma_50', 'price_vs_sma_20', 'price_vs_sma_50',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
        'volume_sma', 'volume_ratio', 'hour', 'is_london_session', 'is_ny_session'
    ]
    
    # Final cleanup
    df_clean = df.dropna()
    X = df_clean[feature_cols].values.astype('float32')
    y = {
        'direction_1m': df_clean['direction_1m'].values,
        'direction_5m': df_clean['direction_5m'].values,
        'direction_15m': df_clean['direction_15m'].values
    }
    
    print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]:,} samples")
    print(f"✅ Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
    
    return X, y, df_clean, feature_cols

def train_efficient_models(X, y, feature_cols):
    """Training models với memory efficient approach"""
    print("\n🎯 TRAINING EFFICIENT MODELS...")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    
    # Split data
    X_train, X_test, y_train_5m, y_test_5m = train_test_split(
        X, y['direction_5m'], test_size=0.2, random_state=42, stratify=y['direction_5m']
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
    
    # 1. LightGBM (most efficient)
    print("\n🌟 Training LightGBM...")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train_5m)
    
    # Evaluate
    train_pred = lgb_model.predict(X_train)
    test_pred = lgb_model.predict(X_test)
    
    train_acc = accuracy_score(y_train_5m, train_pred)
    test_acc = accuracy_score(y_test_5m, test_pred)
    
    # Save
    lgb_path = f"trained_models/efficient_m1_lightgbm_{timestamp}.pkl"
    with open(lgb_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    
    trained_models['lightgbm'] = lgb_model
    training_results['lightgbm'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model_path': lgb_path,
        'feature_importance': dict(zip(feature_cols, lgb_model.feature_importances_))
    }
    
    print(f"   ✅ LightGBM: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # 2. Simple Neural Network (no LSTM to save memory)
    print("\n🧠 Training Simple Neural Network...")
    
    # Build simple dense model
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = nn_model.fit(
        X_train_scaled, y_train_5m,
        validation_data=(X_test_scaled, y_test_5m),
        epochs=20,
        batch_size=512,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Evaluate
    train_loss, train_acc = nn_model.evaluate(X_train_scaled, y_train_5m, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test_5m, verbose=0)
    
    # Save
    nn_path = f"trained_models/efficient_m1_neural_{timestamp}.keras"
    nn_model.save(nn_path)
    
    trained_models['neural_network'] = nn_model
    training_results['neural_network'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'model_path': nn_path
    }
    
    print(f"   ✅ Neural Network: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Save scaler
    scaler_path = f"trained_models/efficient_m1_scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save training results
    results_path = f"training_results/efficient_m1_training_{timestamp}.json"
    os.makedirs('training_results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"\n📊 EFFICIENT M1 TRAINING COMPLETED!")
    print(f"   💾 Results saved: {results_path}")
    
    return trained_models, training_results, scaler

def create_detailed_signal_analysis():
    """Tạo hệ thống phân tích signal chi tiết"""
    print("\n🔍 CREATING DETAILED SIGNAL ANALYSIS...")
    print("=" * 60)
    
    # Simulate 18 specialists system
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
    
    return specialists_data

def simulate_signal_generation(specialists_data, market_scenarios):
    """Simulate signal generation với 18 specialists"""
    print("\n📊 SIMULATING SIGNAL GENERATION...")
    print("=" * 60)
    
    all_signal_analyses = []
    specialist_performance = {}
    
    # Initialize performance tracking
    for category, specialists in specialists_data.items():
        for name, data in specialists.items():
            specialist_performance[name] = {
                'category': category,
                'accuracy': data['accuracy'],
                'bias': data['bias'],
                'total_signals': 0,
                'correct_predictions': 0,
                'vote_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                'confidence_scores': [],
                'category_agreements': 0,
                'overall_agreements': 0
            }
    
    # Generate signals for each scenario
    for scenario_idx, scenario in enumerate(market_scenarios):
        print(f"\n🎯 Analyzing Scenario {scenario_idx + 1}: {scenario['name']}")
        
        # Simulate market data
        market_data = scenario['data']
        actual_direction = scenario['actual_direction']
        
        # Generate votes from each specialist
        specialist_votes = {}
        category_votes = {}
        
        for category, specialists in specialists_data.items():
            category_votes[category] = []
            
            for specialist_name, specialist_info in specialists.items():
                # Simulate vote based on specialist bias and market scenario
                vote = generate_specialist_vote(specialist_info, scenario, market_data)
                
                specialist_votes[specialist_name] = vote
                category_votes[category].append(vote)
                
                # Update performance tracking
                perf = specialist_performance[specialist_name]
                perf['total_signals'] += 1
                perf['vote_distribution'][vote['decision']] += 1
                perf['confidence_scores'].append(vote['confidence'])
                
                # Check if prediction was correct
                if vote['decision'] == actual_direction:
                    perf['correct_predictions'] += 1
        
        # Analyze category consensus
        category_consensus = {}
        for category, votes in category_votes.items():
            decisions = [v['decision'] for v in votes]
            decision_counts = {d: decisions.count(d) for d in set(decisions)}
            majority_decision = max(decision_counts, key=decision_counts.get)
            consensus_strength = decision_counts[majority_decision] / len(decisions)
            
            category_consensus[category] = {
                'majority_decision': majority_decision,
                'consensus_strength': consensus_strength,
                'vote_distribution': decision_counts,
                'total_specialists': len(decisions)
            }
        
        # Overall consensus
        all_decisions = [v['decision'] for v in specialist_votes.values()]
        all_confidences = [v['confidence'] for v in specialist_votes.values()]
        
        decision_counts = {d: all_decisions.count(d) for d in set(all_decisions)}
        final_decision = max(decision_counts, key=decision_counts.get)
        consensus_strength = decision_counts[final_decision] / len(all_decisions)
        
        # Weighted decision (by confidence)
        weighted_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = sum(all_confidences)
        
        for vote in specialist_votes.values():
            weight = vote['confidence'] / total_confidence if total_confidence > 0 else 1/len(specialist_votes)
            weighted_scores[vote['decision']] += weight
        
        weighted_decision = max(weighted_scores, key=weighted_scores.get)
        weighted_confidence = weighted_scores[weighted_decision]
        
        # Signal quality metrics
        signal_quality = {
            'consensus_strength': consensus_strength,
            'weighted_confidence': weighted_confidence,
            'category_agreement': len([c for c in category_consensus.values() 
                                    if c['majority_decision'] == final_decision]) / len(category_consensus),
            'high_confidence_specialists': len([v for v in specialist_votes.values() 
                                              if v['confidence'] > 0.7]),
            'unanimous_categories': len([c for c in category_consensus.values() 
                                       if c['consensus_strength'] == 1.0])
        }
        
        # Store analysis
        signal_analysis = {
            'scenario': scenario,
            'specialist_votes': specialist_votes,
            'category_consensus': category_consensus,
            'final_decision': final_decision,
            'consensus_strength': consensus_strength,
            'weighted_decision': weighted_decision,
            'weighted_confidence': weighted_confidence,
            'signal_quality': signal_quality,
            'actual_direction': actual_direction,
            'prediction_correct': final_decision == actual_direction
        }
        
        all_signal_analyses.append(signal_analysis)
        
        # Print results
        print(f"   📊 Specialist votes: {decision_counts}")
        print(f"   🎯 Final decision: {final_decision} (consensus: {consensus_strength:.1%})")
        print(f"   ⚖️ Weighted decision: {weighted_decision} (confidence: {weighted_confidence:.1%})")
        print(f"   ✅ Actual direction: {actual_direction}")
        print(f"   🎯 Prediction correct: {final_decision == actual_direction}")
        
        # Update category agreement tracking
        for specialist_name, vote in specialist_votes.values():
            if vote['decision'] == category_consensus[specialist_performance[specialist_name]['category']]['majority_decision']:
                specialist_performance[specialist_name]['category_agreements'] += 1
            if vote['decision'] == final_decision:
                specialist_performance[specialist_name]['overall_agreements'] += 1
    
    # Calculate final performance metrics
    for name, perf in specialist_performance.items():
        if perf['total_signals'] > 0:
            perf['accuracy_rate'] = perf['correct_predictions'] / perf['total_signals']
            perf['avg_confidence'] = np.mean(perf['confidence_scores'])
            perf['category_agreement_rate'] = perf['category_agreements'] / perf['total_signals']
            perf['overall_agreement_rate'] = perf['overall_agreements'] / perf['total_signals']
            
            # Vote distribution percentages
            total_votes = perf['total_signals']
            perf['vote_percentages'] = {
                vote: count/total_votes*100 
                for vote, count in perf['vote_distribution'].items()
            }
    
    return all_signal_analyses, specialist_performance

def generate_specialist_vote(specialist_info, scenario, market_data):
    """Generate vote từ specialist dựa trên bias và scenario"""
    bias = specialist_info['bias']
    accuracy = specialist_info['accuracy']
    scenario_type = scenario['type']
    
    # Base probabilities
    if scenario_type == 'bullish':
        base_probs = [0.6, 0.2, 0.2]  # [BUY, SELL, HOLD]
    elif scenario_type == 'bearish':
        base_probs = [0.2, 0.6, 0.2]
    elif scenario_type == 'sideways':
        base_probs = [0.25, 0.25, 0.5]
    else:  # volatile
        base_probs = [0.4, 0.4, 0.2]
    
    # Adjust based on specialist bias
    if bias == 'contrarian':
        # Contrarian specialists go against the trend
        if scenario_type == 'bullish':
            base_probs = [0.3, 0.5, 0.2]
        elif scenario_type == 'bearish':
            base_probs = [0.5, 0.3, 0.2]
    elif bias == 'trend_following':
        # Trend followers amplify the trend
        if scenario_type == 'bullish':
            base_probs = [0.7, 0.1, 0.2]
        elif scenario_type == 'bearish':
            base_probs = [0.1, 0.7, 0.2]
    elif bias == 'risk_averse':
        # Risk averse specialists prefer HOLD
        base_probs = [p * 0.7 for p in base_probs[:2]] + [base_probs[2] + 0.3]
    
    # Generate vote
    decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=base_probs)
    
    # Generate confidence based on accuracy
    base_confidence = np.random.uniform(0.3, 0.9)
    confidence = base_confidence * accuracy + np.random.uniform(-0.1, 0.1)
    confidence = max(0.1, min(0.9, confidence))
    
    # Generate reasoning
    reasoning = f"{bias.replace('_', ' ').title()} analysis suggests {decision} " \
                f"based on {scenario_type} market conditions"
    
    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'technical_data': {
            'bias': bias,
            'scenario_type': scenario_type,
            'base_confidence': base_confidence
        }
    }

def create_market_test_scenarios():
    """Tạo scenarios để test signal generation"""
    scenarios = [
        {
            'name': 'Strong Bullish Breakout',
            'type': 'bullish',
            'data': pd.DataFrame({
                'Close': [2000, 2010, 2025, 2040, 2055],
                'Volume': [1000, 1500, 2000, 2500, 3000]
            }),
            'actual_direction': 'BUY'
        },
        {
            'name': 'Sharp Bearish Decline',
            'type': 'bearish',
            'data': pd.DataFrame({
                'Close': [2055, 2040, 2020, 1995, 1970],
                'Volume': [3000, 2500, 2000, 1500, 1000]
            }),
            'actual_direction': 'SELL'
        },
        {
            'name': 'Sideways Consolidation',
            'type': 'sideways',
            'data': pd.DataFrame({
                'Close': [2000, 2005, 1998, 2003, 1999],
                'Volume': [1000, 1100, 950, 1050, 975]
            }),
            'actual_direction': 'HOLD'
        },
        {
            'name': 'High Volatility Whipsaw',
            'type': 'volatile',
            'data': pd.DataFrame({
                'Close': [2000, 2030, 1970, 2040, 1960],
                'Volume': [2000, 3000, 2500, 3500, 2800]
            }),
            'actual_direction': 'HOLD'
        },
        {
            'name': 'Bullish Reversal',
            'type': 'bullish',
            'data': pd.DataFrame({
                'Close': [1950, 1945, 1960, 1980, 2000],
                'Volume': [1200, 1000, 1400, 1800, 2200]
            }),
            'actual_direction': 'BUY'
        },
        {
            'name': 'Bearish Reversal',
            'type': 'bearish',
            'data': pd.DataFrame({
                'Close': [2050, 2055, 2040, 2020, 2000],
                'Volume': [2200, 1800, 1400, 1000, 1200]
            }),
            'actual_direction': 'SELL'
        }
    ]
    
    return scenarios

def run_comprehensive_m1_analysis():
    """Chạy phân tích toàn diện M1"""
    print("🚀 COMPREHENSIVE M1 ANALYSIS WITH SIGNAL TRACKING")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load M1 sample data (memory efficient)
    df = load_m1_sample(sample_size=100000)  # 100K samples for testing
    if df is None:
        print("❌ Failed to load M1 data. Exiting...")
        return
    
    # 2. Prepare features
    X, y, df_clean, feature_cols = prepare_m1_features_efficient(df)
    
    # 3. Train models
    trained_models, training_results, scaler = train_efficient_models(X, y, feature_cols)
    
    # 4. Create specialist system
    specialists_data = create_detailed_signal_analysis()
    
    # 5. Create test scenarios
    scenarios = create_market_test_scenarios()
    
    # 6. Simulate signal generation
    signal_analyses, specialist_performance = simulate_signal_generation(specialists_data, scenarios)
    
    # 7. Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory
    os.makedirs('m1_comprehensive_results', exist_ok=True)
    
    # Save signal analyses
    signal_path = f"m1_comprehensive_results/signal_analyses_{timestamp}.json"
    with open(signal_path, 'w') as f:
        json.dump(signal_analyses, f, indent=2, default=str)
    
    # Save specialist performance
    perf_path = f"m1_comprehensive_results/specialist_performance_{timestamp}.json"
    with open(perf_path, 'w') as f:
        json.dump(specialist_performance, f, indent=2, default=str)
    
    # Create comprehensive summary
    summary = create_comprehensive_summary(
        training_results, signal_analyses, specialist_performance, X.shape
    )
    
    summary_path = f"m1_comprehensive_results/comprehensive_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print results
    print_comprehensive_results(summary, signal_analyses, specialist_performance)
    
    return summary

def create_comprehensive_summary(training_results, signal_analyses, specialist_performance, data_shape):
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
            category_performance[category] = {
                'specialists': [],
                'avg_accuracy': 0,
                'avg_confidence': 0
            }
        category_performance[category]['specialists'].append({
            'name': name,
            'accuracy': perf.get('accuracy_rate', 0),
            'confidence': perf.get('avg_confidence', 0)
        })
    
    # Calculate category averages
    for category, data in category_performance.items():
        specialists = data['specialists']
        data['avg_accuracy'] = np.mean([s['accuracy'] for s in specialists])
        data['avg_confidence'] = np.mean([s['confidence'] for s in specialists])
        data['specialist_count'] = len(specialists)
    
    # Top performers
    top_specialists = sorted(
        specialist_performance.items(),
        key=lambda x: x[1].get('accuracy_rate', 0),
        reverse=True
    )[:5]
    
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'data_summary': {
            'samples_processed': data_shape[0],
            'features_used': data_shape[1],
            'memory_efficient': True
        },
        'training_summary': {
            'models_trained': len(training_results),
            'best_model': {
                'name': best_model[0],
                'test_accuracy': best_model[1].get('test_accuracy', 0)
            },
            'all_models': {name: results.get('test_accuracy', 0) 
                          for name, results in training_results.items()}
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
        ],
        'signal_quality_metrics': {
            'avg_consensus_strength': np.mean([
                analysis['consensus_strength'] for analysis in signal_analyses
            ]),
            'avg_weighted_confidence': np.mean([
                analysis['weighted_confidence'] for analysis in signal_analyses
            ]),
            'high_quality_signals': len([
                analysis for analysis in signal_analyses 
                if analysis['signal_quality']['consensus_strength'] > 0.6
            ])
        }
    }
    
    return summary

def print_comprehensive_results(summary, signal_analyses, specialist_performance):
    """In kết quả toàn diện"""
    print(f"\n🎉 M1 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 80)
    
    # Data summary
    print(f"📊 DATA PROCESSED:")
    print(f"   • Samples: {summary['data_summary']['samples_processed']:,}")
    print(f"   • Features: {summary['data_summary']['features_used']}")
    
    # Training results
    print(f"\n🎯 TRAINING RESULTS:")
    print(f"   • Models trained: {summary['training_summary']['models_trained']}")
    print(f"   • Best model: {summary['training_summary']['best_model']['name']}")
    print(f"   • Best accuracy: {summary['training_summary']['best_model']['test_accuracy']:.4f}")
    
    # Signal analysis results
    print(f"\n📊 SIGNAL ANALYSIS RESULTS:")
    print(f"   • Scenarios tested: {summary['signal_analysis_summary']['scenarios_tested']}")
    print(f"   • Overall signal accuracy: {summary['signal_analysis_summary']['overall_signal_accuracy']:.1%}")
    print(f"   • Total specialists: {summary['signal_analysis_summary']['total_specialists']}")
    
    # Category performance
    print(f"\n🏆 CATEGORY PERFORMANCE:")
    for category, perf in summary['category_performance'].items():
        print(f"   • {category}: {perf['avg_accuracy']:.1%} accuracy, {perf['avg_confidence']:.3f} confidence")
    
    # Top specialists
    print(f"\n⭐ TOP PERFORMING SPECIALISTS:")
    for specialist in summary['top_specialists']:
        print(f"   • {specialist['name']}: {specialist['accuracy']:.1%} accuracy ({specialist['category']})")
    
    # Signal quality
    print(f"\n📈 SIGNAL QUALITY METRICS:")
    quality = summary['signal_quality_metrics']
    print(f"   • Average consensus: {quality['avg_consensus_strength']:.1%}")
    print(f"   • Average confidence: {quality['avg_weighted_confidence']:.1%}")
    print(f"   • High quality signals: {quality['high_quality_signals']}/{len(signal_analyses)}")
    
    print(f"\n💾 Results saved in: m1_comprehensive_results/")

if __name__ == "__main__":
    run_comprehensive_m1_analysis() 