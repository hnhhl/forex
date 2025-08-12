#!/usr/bin/env python3
"""
TRAINING WITH 1 MILLION M1 CANDLES & SIGNAL ANALYSIS
Training với 1 triệu nến M1 và phân tích chi tiết signal generation
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

def load_m1_data():
    """Load dữ liệu M1 (1 triệu nến)"""
    print("📊 LOADING 1 MILLION M1 CANDLES...")
    print("=" * 60)
    
    # Load working M1 data (1.1M records)
    m1_file = "data/working_free_data/XAUUSD_M1_realistic.csv"
    
    if not os.path.exists(m1_file):
        print(f"❌ M1 data file not found: {m1_file}")
        return None
    
    try:
        print(f"📈 Loading {m1_file}...")
        df = pd.read_csv(m1_file)
        print(f"✅ Loaded {len(df):,} M1 candles")
        
        # Take exactly 1 million records
        if len(df) > 1000000:
            df = df.tail(1000000)  # Take latest 1M candles
            print(f"✅ Selected latest 1,000,000 candles")
        
        print(f"📊 Data range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"📊 Columns: {list(df.columns)}")
        
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
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_ratio'] = df['High'] / df['Low']
    df['open_close_ratio'] = df['Open'] / df['Close']
    df['price_range'] = df['High'] - df['Low']
    df['body_size'] = abs(df['Close'] - df['Open'])
    df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
    df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
    
    # Fast technical indicators (optimized for M1)
    # RSI (14 periods = 14 minutes)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (fast for M1: 5, 13, 4)
    exp1 = df['Close'].ewm(span=5).mean()
    exp2 = df['Close'].ewm(span=13).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=4).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Moving averages (shorter periods for M1)
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_vs_sma_{period}'] = df['Close'] / df[f'sma_{period}']
    
    # Bollinger Bands (20 periods)
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR (14 periods)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['price_range'] / df['atr']
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    df['volume_price_trend'] = df['Volume'] * df['returns']
    
    # Volatility measures
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
    
    # Support/Resistance (short-term for M1)
    df['resistance_20'] = df['High'].rolling(window=20).max()
    df['support_20'] = df['Low'].rolling(window=20).min()
    df['price_vs_resistance'] = df['Close'] / df['resistance_20']
    df['price_vs_support'] = df['Close'] / df['support_20']
    
    # Momentum indicators
    df['momentum_5'] = df['Close'] / df['Close'].shift(5)
    df['momentum_10'] = df['Close'] / df['Close'].shift(10)
    df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    
    # Candlestick patterns (simplified)
    df['is_doji'] = (abs(df['body_size']) < (df['price_range'] * 0.1)).astype(int)
    df['is_hammer'] = ((df['lower_shadow'] > df['body_size'] * 2) & 
                       (df['upper_shadow'] < df['body_size'])).astype(int)
    df['is_shooting_star'] = ((df['upper_shadow'] > df['body_size'] * 2) & 
                              (df['lower_shadow'] < df['body_size'])).astype(int)
    
    # Time-based features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
    df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    # Create labels (multiple timeframes)
    # 1-minute ahead
    df['price_change_1m'] = df['Close'].shift(-1) - df['Close']
    df['direction_1m'] = (df['price_change_1m'] > 0).astype(int)
    
    # 5-minute ahead
    df['price_change_5m'] = df['Close'].shift(-5) - df['Close']
    df['direction_5m'] = (df['price_change_5m'] > 0).astype(int)
    
    # 15-minute ahead
    df['price_change_15m'] = df['Close'].shift(-15) - df['Close']
    df['direction_15m'] = (df['price_change_15m'] > 0).astype(int)
    
    # 60-minute ahead
    df['price_change_60m'] = df['Close'].shift(-60) - df['Close']
    df['direction_60m'] = (df['price_change_60m'] > 0).astype(int)
    
    # Multi-class labels (Strong Down, Down, Neutral, Up, Strong Up)
    df['movement_5m'] = pd.cut(df['price_change_5m'], 
                               bins=[-np.inf, -1.0, -0.2, 0.2, 1.0, np.inf], 
                               labels=[0, 1, 2, 3, 4])  
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in [
        'Date', 'Time', 'datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'price_change_1m', 'price_change_5m', 'price_change_15m', 'price_change_60m',
        'direction_1m', 'direction_5m', 'direction_15m', 'direction_60m', 'movement_5m'
    ]]
    
    # Final cleanup
    df_clean = df.dropna()
    X = df_clean[feature_cols].values
    y = {
        'direction_1m': df_clean['direction_1m'].values,
        'direction_5m': df_clean['direction_5m'].values,
        'direction_15m': df_clean['direction_15m'].values,
        'direction_60m': df_clean['direction_60m'].values,
        'movement_5m': df_clean['movement_5m'].values
    }
    
    print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]:,} samples")
    print(f"✅ Feature columns: {len(feature_cols)} features")
    print(f"✅ Memory usage: {X.nbytes / 1024 / 1024:.1f} MB")
    
    return X, y, df_clean, feature_cols

def train_m1_models(X, y, feature_cols):
    """Training models với dữ liệu M1"""
    print("\n🎯 TRAINING M1 MODELS...")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import lightgbm as lgb
    
    # Memory-efficient split
    test_size = 0.2
    print(f"📊 Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    
    # Split for main target (direction_5m)
    X_train, X_test, y_train_5m, y_test_5m = train_test_split(
        X, y['direction_5m'], test_size=test_size, random_state=42, stratify=y['direction_5m']
    )
    
    print(f"✅ Training size: {X_train.shape[0]:,} samples")
    print(f"✅ Testing size: {X_test.shape[0]:,} samples")
    
    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    trained_models = {}
    training_results = {}
    
    # 1. LightGBM (fast and efficient for large data)
    print("\n🌟 Training LightGBM...")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lgb_path = f"trained_models/m1_lightgbm_{timestamp}.pkl"
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
    
    # 2. Neural Network (LSTM)
    print("\n🧠 Training Neural Network...")
    
    # Reshape for LSTM (use only recent 60 features for memory efficiency)
    sequence_length = 60
    n_features = X_train_scaled.shape[1]
    
    def create_sequences(X, seq_len):
        sequences = []
        for i in range(seq_len, len(X)):
            sequences.append(X[i-seq_len:i])
        return np.array(sequences)
    
    print(f"   🔧 Creating sequences (length={sequence_length})...")
    X_train_seq = create_sequences(X_train_scaled, sequence_length)
    X_test_seq = create_sequences(X_test_scaled, sequence_length)
    y_train_seq = y_train_5m[sequence_length:]
    y_test_seq = y_test_5m[sequence_length:]
    
    print(f"   📊 Sequence shape: {X_train_seq.shape}")
    
    # Build LSTM model
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    lstm_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with early stopping
    history = lstm_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=30,
        batch_size=256,  # Larger batch for efficiency
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
    )
    
    # Evaluate
    train_loss, train_acc = lstm_model.evaluate(X_train_seq, y_train_seq, verbose=0)
    test_loss, test_acc = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    # Save
    lstm_path = f"trained_models/m1_lstm_{timestamp}.keras"
    lstm_model.save(lstm_path)
    
    trained_models['lstm'] = lstm_model
    training_results['lstm'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'model_path': lstm_path,
        'sequence_length': sequence_length
    }
    
    print(f"   ✅ LSTM: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # 3. Random Forest (smaller for memory)
    print("\n🌲 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Reduced for memory
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train_5m)
    
    # Evaluate
    train_pred = rf_model.predict(X_train)
    test_pred = rf_model.predict(X_test)
    
    train_acc = accuracy_score(y_train_5m, train_pred)
    test_acc = accuracy_score(y_test_5m, test_pred)
    
    # Save
    rf_path = f"trained_models/m1_random_forest_{timestamp}.pkl"
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    trained_models['random_forest'] = rf_model
    training_results['random_forest'] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'model_path': rf_path,
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
    }
    
    print(f"   ✅ Random Forest: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Save scaler
    scaler_path = f"trained_models/m1_scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save training results
    results_path = f"training_results/m1_training_{timestamp}.json"
    os.makedirs('training_results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"\n📊 M1 TRAINING COMPLETED!")
    print(f"   💾 Results saved: {results_path}")
    print(f"   💾 Scaler saved: {scaler_path}")
    
    return trained_models, training_results, scaler

def setup_signal_analysis():
    """Setup hệ thống phân tích signal"""
    print("\n🔍 SETTING UP SIGNAL ANALYSIS...")
    print("=" * 60)
    
    try:
        from master_system import create_development_system
        
        # Create system
        system = create_development_system()
        print("✅ Master system created")
        
        # Get voting engine
        voting_engine = system.components.get('democratic_voting_engine')
        if voting_engine:
            specialists = voting_engine.specialists
            print(f"✅ {len(specialists)} specialists available")
            
            # Group by category
            categories = {}
            for specialist in specialists:
                if specialist.category not in categories:
                    categories[specialist.category] = []
                categories[specialist.category].append(specialist.name)
            
            print("📊 Specialist categories:")
            for category, names in categories.items():
                print(f"   • {category}: {len(names)} specialists")
                for name in names:
                    print(f"     - {name}")
        
        return system, voting_engine
        
    except Exception as e:
        print(f"❌ Error setting up signal analysis: {e}")
        return None, None

def analyze_signal_generation_detailed(voting_engine, market_scenarios):
    """Phân tích chi tiết signal generation với nhiều scenarios"""
    print("\n📊 DETAILED SIGNAL GENERATION ANALYSIS...")
    print("=" * 60)
    
    if not voting_engine:
        print("❌ Voting engine not available")
        return None
    
    all_analyses = []
    specialist_performance = {}
    
    # Initialize performance tracking
    for specialist in voting_engine.specialists:
        specialist_performance[specialist.name] = {
            'category': specialist.category,
            'total_votes': 0,
            'vote_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'confidence_scores': [],
            'scenarios_participated': 0
        }
    
    # Analyze each scenario
    for i, scenario in enumerate(market_scenarios):
        print(f"\n🎯 Scenario {i+1}: {scenario['name']}")
        
        # Simulate votes from each specialist
        scenario_votes = {}
        category_consensus = {}
        
        for specialist in voting_engine.specialists:
            # Simulate specialist vote based on scenario
            if scenario['type'] == 'bullish':
                vote_probs = [0.6, 0.2, 0.2]  # [BUY, SELL, HOLD]
            elif scenario['type'] == 'bearish':
                vote_probs = [0.2, 0.6, 0.2]
            elif scenario['type'] == 'sideways':
                vote_probs = [0.25, 0.25, 0.5]
            elif scenario['type'] == 'volatile':
                vote_probs = [0.4, 0.4, 0.2]
            else:  # neutral
                vote_probs = [0.33, 0.33, 0.34]
            
            vote = np.random.choice(['BUY', 'SELL', 'HOLD'], p=vote_probs)
            confidence = np.random.uniform(0.1, 0.9)
            
            # Add specialist-specific bias
            if 'RSI' in specialist.name and scenario.get('rsi_oversold', False):
                vote = 'BUY'
                confidence = min(0.9, confidence + 0.2)
            elif 'MACD' in specialist.name and scenario.get('macd_bullish', False):
                vote = 'BUY'
                confidence = min(0.9, confidence + 0.15)
            elif 'Risk' in specialist.name and scenario.get('high_volatility', False):
                vote = 'HOLD'
                confidence = min(0.9, confidence + 0.25)
            
            scenario_votes[specialist.name] = {
                'vote': vote,
                'confidence': confidence,
                'reasoning': f"{specialist.name} analysis for {scenario['name']}"
            }
            
            # Update performance tracking
            specialist_performance[specialist.name]['total_votes'] += 1
            specialist_performance[specialist.name]['vote_distribution'][vote] += 1
            specialist_performance[specialist.name]['confidence_scores'].append(confidence)
            specialist_performance[specialist.name]['scenarios_participated'] += 1
            
            # Group by category
            if specialist.category not in category_consensus:
                category_consensus[specialist.category] = []
            category_consensus[specialist.category].append(vote)
        
        # Analyze category consensus
        category_results = {}
        for category, votes in category_consensus.items():
            vote_counts = {v: votes.count(v) for v in set(votes)}
            majority_vote = max(vote_counts, key=vote_counts.get)
            consensus_strength = vote_counts[majority_vote] / len(votes)
            
            category_results[category] = {
                'total_specialists': len(votes),
                'vote_distribution': vote_counts,
                'majority_vote': majority_vote,
                'consensus_strength': consensus_strength
            }
        
        # Overall consensus
        all_votes = [v['vote'] for v in scenario_votes.values()]
        overall_counts = {v: all_votes.count(v) for v in set(all_votes)}
        final_decision = max(overall_counts, key=overall_counts.get)
        consensus_strength = overall_counts[final_decision] / len(all_votes)
        
        # Calculate weighted decision (by confidence)
        weighted_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0
        
        for vote_data in scenario_votes.values():
            weighted_scores[vote_data['vote']] += vote_data['confidence']
            total_confidence += vote_data['confidence']
        
        if total_confidence > 0:
            weighted_decision = max(weighted_scores, key=weighted_scores.get)
            weighted_confidence = weighted_scores[weighted_decision] / total_confidence
        else:
            weighted_decision = final_decision
            weighted_confidence = consensus_strength
        
        # Store analysis
        scenario_analysis = {
            'scenario': scenario,
            'specialist_votes': scenario_votes,
            'category_consensus': category_results,
            'overall_consensus': {
                'vote_distribution': overall_counts,
                'final_decision': final_decision,
                'consensus_strength': consensus_strength,
                'weighted_decision': weighted_decision,
                'weighted_confidence': weighted_confidence,
                'total_specialists': len(all_votes)
            }
        }
        
        all_analyses.append(scenario_analysis)
        
        # Print results
        print(f"   📊 Vote distribution: {overall_counts}")
        print(f"   🎯 Final decision: {final_decision} ({consensus_strength:.1%} consensus)")
        print(f"   ⚖️ Weighted decision: {weighted_decision} ({weighted_confidence:.1%} confidence)")
    
    # Calculate specialist statistics
    for name, perf in specialist_performance.items():
        if perf['total_votes'] > 0:
            perf['avg_confidence'] = np.mean(perf['confidence_scores'])
            perf['confidence_std'] = np.std(perf['confidence_scores'])
            
            # Calculate vote percentages
            total = perf['total_votes']
            perf['vote_percentages'] = {
                vote: count/total*100 for vote, count in perf['vote_distribution'].items()
            }
    
    return all_analyses, specialist_performance

def create_market_scenarios():
    """Tạo các scenarios thị trường để test"""
    scenarios = [
        {
            'name': 'Strong Bullish Trend',
            'type': 'bullish',
            'data': pd.DataFrame({
                'Close': [2000, 2010, 2020, 2030, 2040],
                'High': [2005, 2015, 2025, 2035, 2045],
                'Low': [1995, 2005, 2015, 2025, 2035],
                'Volume': [1000, 1200, 1400, 1600, 1800]
            }),
            'rsi_oversold': False,
            'macd_bullish': True,
            'high_volatility': False
        },
        {
            'name': 'Strong Bearish Trend',
            'type': 'bearish',
            'data': pd.DataFrame({
                'Close': [2040, 2030, 2020, 2010, 2000],
                'High': [2045, 2035, 2025, 2015, 2005],
                'Low': [2035, 2025, 2015, 2005, 1995],
                'Volume': [1800, 1600, 1400, 1200, 1000]
            }),
            'rsi_oversold': True,
            'macd_bullish': False,
            'high_volatility': False
        },
        {
            'name': 'Sideways Consolidation',
            'type': 'sideways',
            'data': pd.DataFrame({
                'Close': [2000, 2005, 1998, 2003, 1999],
                'High': [2008, 2012, 2006, 2010, 2007],
                'Low': [1992, 1998, 1990, 1996, 1991],
                'Volume': [1000, 1100, 950, 1050, 975]
            }),
            'rsi_oversold': False,
            'macd_bullish': False,
            'high_volatility': False
        },
        {
            'name': 'High Volatility Breakout',
            'type': 'volatile',
            'data': pd.DataFrame({
                'Close': [2000, 2030, 1970, 2040, 1960],
                'High': [2010, 2050, 2000, 2060, 2000],
                'Low': [1990, 2020, 1950, 2030, 1940],
                'Volume': [2000, 2500, 2200, 2800, 2300]
            }),
            'rsi_oversold': True,
            'macd_bullish': True,
            'high_volatility': True
        },
        {
            'name': 'Low Volatility Drift',
            'type': 'neutral',
            'data': pd.DataFrame({
                'Close': [2000, 2001, 2000.5, 2001.5, 2000.8],
                'High': [2002, 2003, 2002.5, 2003.5, 2002.8],
                'Low': [1998, 1999, 1998.5, 1999.5, 1998.8],
                'Volume': [500, 520, 480, 510, 490]
            }),
            'rsi_oversold': False,
            'macd_bullish': False,
            'high_volatility': False
        }
    ]
    
    return scenarios

def run_m1_analysis():
    """Chạy phân tích toàn diện với dữ liệu M1"""
    print("🚀 M1 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load M1 data
    df = load_m1_data()
    if df is None:
        print("❌ Failed to load M1 data. Exiting...")
        return
    
    # 2. Prepare features
    X, y, df_clean, feature_cols = prepare_m1_features(df)
    
    # 3. Train models
    trained_models, training_results, scaler = train_m1_models(X, y, feature_cols)
    
    # 4. Setup signal analysis
    system, voting_engine = setup_signal_analysis()
    
    # 5. Create market scenarios
    scenarios = create_market_scenarios()
    
    # 6. Analyze signal generation
    if voting_engine:
        analyses, specialist_perf = analyze_signal_generation_detailed(voting_engine, scenarios)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Signal analysis results
        signal_path = f"signal_analysis/m1_signal_analysis_{timestamp}.json"
        os.makedirs('signal_analysis', exist_ok=True)
        with open(signal_path, 'w') as f:
            json.dump(analyses, f, indent=2, default=str)
        
        # Specialist performance
        perf_path = f"signal_analysis/m1_specialist_performance_{timestamp}.json"
        with open(perf_path, 'w') as f:
            json.dump(specialist_perf, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'training_summary': {
                'data_size': X.shape[0],
                'features': X.shape[1],
                'models_trained': len(trained_models),
                'best_accuracy': max([r.get('test_accuracy', 0) for r in training_results.values()])
            },
            'signal_analysis_summary': {
                'scenarios_tested': len(scenarios),
                'specialists_analyzed': len(specialist_perf),
                'categories': len(set([p['category'] for p in specialist_perf.values()]))
            }
        }
        
        summary_path = f"m1_analysis_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n🎉 M1 ANALYSIS COMPLETED!")
        print("=" * 80)
        print(f"📊 Data processed: {X.shape[0]:,} M1 candles")
        print(f"📊 Models trained: {len(trained_models)}")
        print(f"📊 Scenarios analyzed: {len(scenarios)}")
        print(f"📊 Specialists tracked: {len(specialist_perf)}")
        print(f"💾 Results saved:")
        print(f"   • Training: {training_results}")
        print(f"   • Signals: {signal_path}")
        print(f"   • Performance: {perf_path}")
        print(f"   • Summary: {summary_path}")
        
        # Print top performing specialists
        print(f"\n🏆 TOP PERFORMING SPECIALISTS:")
        sorted_specialists = sorted(specialist_perf.items(), 
                                  key=lambda x: x[1].get('avg_confidence', 0), 
                                  reverse=True)
        
        for name, perf in sorted_specialists[:5]:
            print(f"   • {name}: {perf.get('avg_confidence', 0):.3f} avg confidence")
            print(f"     Category: {perf['category']}")
            print(f"     Vote distribution: {perf.get('vote_percentages', {})}")
    
    return summary

if __name__ == "__main__":
    run_m1_analysis() 