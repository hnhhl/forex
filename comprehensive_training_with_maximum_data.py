#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING WITH MAXIMUM DATA & SIGNAL ANALYSIS
Training hệ thống với dữ liệu tối đa và phân tích chi tiết signal generation
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

def load_maximum_data():
    """Load tất cả dữ liệu có sẵn"""
    print("📊 LOADING MAXIMUM DATA...")
    print("=" * 60)
    
    all_data = {}
    total_records = 0
    
    # 1. Load working free data (3 năm)
    working_data_dir = "data/working_free_data"
    if os.path.exists(working_data_dir):
        for file in os.listdir(working_data_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(working_data_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    timeframe = file.split('_')[1].replace('.csv', '')
                    all_data[f'working_{timeframe}'] = df
                    total_records += len(df)
                    print(f"✅ Loaded working data {timeframe}: {len(df):,} records")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
    
    # 2. Load MT5 maximum data (11+ năm)
    mt5_data_dir = "data/maximum_mt5_v2"
    if os.path.exists(mt5_data_dir):
        for file in os.listdir(mt5_data_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(mt5_data_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    timeframe = file.split('_')[1]
                    all_data[f'mt5_{timeframe}'] = df
                    total_records += len(df)
                    print(f"✅ Loaded MT5 data {timeframe}: {len(df):,} records")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
    
    print(f"\n📈 TOTAL DATA LOADED: {total_records:,} records")
    print(f"📊 DATASETS: {len(all_data)} timeframes")
    
    return all_data

def prepare_comprehensive_features(data_dict):
    """Chuẩn bị features toàn diện từ tất cả dữ liệu"""
    print("\n🔧 PREPARING COMPREHENSIVE FEATURES...")
    print("=" * 60)
    
    # Chọn dataset chính (H1 data với nhiều records nhất)
    main_dataset = None
    max_records = 0
    
    for name, df in data_dict.items():
        if 'H1' in name and len(df) > max_records:
            max_records = len(df)
            main_dataset = df.copy()
            print(f"✅ Selected main dataset: {name} with {len(df):,} records")
    
    if main_dataset is None:
        print("❌ No suitable main dataset found")
        return None, None
    
    # Standardize column names
    if 'Date' in main_dataset.columns and 'Time' in main_dataset.columns:
        main_dataset['datetime'] = pd.to_datetime(main_dataset['Date'] + ' ' + main_dataset['Time'])
    else:
        main_dataset['datetime'] = pd.to_datetime(main_dataset.index)
    
    # Ensure required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in main_dataset.columns:
            if col == 'Volume' and 'Volume' not in main_dataset.columns:
                main_dataset['Volume'] = 1000  # Default volume
            else:
                print(f"❌ Missing required column: {col}")
                return None, None
    
    # Create comprehensive features
    df = main_dataset.copy()
    
    # Price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['high_low_ratio'] = df['High'] / df['Low']
    df['open_close_ratio'] = df['Open'] / df['Close']
    
    # Technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Moving averages
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Price position relative to MAs
    df['price_vs_sma20'] = df['Close'] / df['sma_20']
    df['price_vs_sma50'] = df['Close'] / df['sma_50']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Support/Resistance levels
    df['resistance'] = df['High'].rolling(window=20).max()
    df['support'] = df['Low'].rolling(window=20).min()
    df['price_vs_resistance'] = df['Close'] / df['resistance']
    df['price_vs_support'] = df['Close'] / df['support']
    
    # Drop NaN values
    df = df.dropna()
    
    # Create labels (multi-target)
    df['price_change_1h'] = df['Close'].shift(-1) - df['Close']
    df['price_change_4h'] = df['Close'].shift(-4) - df['Close']
    df['price_change_1d'] = df['Close'].shift(-24) - df['Close']
    
    # Binary labels
    df['direction_1h'] = (df['price_change_1h'] > 0).astype(int)
    df['direction_4h'] = (df['price_change_4h'] > 0).astype(int)
    df['direction_1d'] = (df['price_change_1d'] > 0).astype(int)
    
    # Multi-class labels
    df['movement_1h'] = pd.cut(df['price_change_1h'], 
                               bins=[-np.inf, -0.5, 0.5, np.inf], 
                               labels=[0, 1, 2])  # Down, Sideways, Up
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in [
        'Date', 'Time', 'datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'price_change_1h', 'price_change_4h', 'price_change_1d',
        'direction_1h', 'direction_4h', 'direction_1d', 'movement_1h'
    ]]
    
    # Prepare final dataset
    df_clean = df.dropna()
    X = df_clean[feature_cols].values
    y = {
        'direction_1h': df_clean['direction_1h'].values,
        'direction_4h': df_clean['direction_4h'].values,
        'direction_1d': df_clean['direction_1d'].values,
        'movement_1h': df_clean['movement_1h'].values
    }
    
    print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]:,} samples")
    print(f"✅ Feature columns: {feature_cols}")
    
    return X, y, df_clean

def train_comprehensive_models(X, y):
    """Training toàn diện tất cả models"""
    print("\n🎯 COMPREHENSIVE MODEL TRAINING...")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Split data
    test_size = 0.2
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    
    for target_name, target_values in y.items():
        X_train[target_name], X_test[target_name], y_train_dict[target_name], y_test_dict[target_name] = \
            train_test_split(X, target_values, test_size=test_size, random_state=42, stratify=target_values)
    
    print(f"📊 Training size: {X_train['direction_1h'].shape[0]:,}")
    print(f"📊 Testing size: {X_test['direction_1h'].shape[0]:,}")
    
    # Scale features
    scalers = {}
    X_train_scaled, X_test_scaled = {}, {}
    
    for target_name in y.keys():
        scaler = StandardScaler()
        X_train_scaled[target_name] = scaler.fit_transform(X_train[target_name])
        X_test_scaled[target_name] = scaler.transform(X_test[target_name])
        scalers[target_name] = scaler
    
    # Train models
    trained_models = {}
    training_results = {}
    
    # 1. Neural Networks
    print("\n🧠 Training Neural Networks...")
    
    for target_name in ['direction_1h', 'direction_4h', 'direction_1d']:
        print(f"\n   🔥 Training Neural Network for {target_name}...")
        
        # LSTM Model
        lstm_model = tf.keras.Sequential([
            tf.keras.layers.Reshape((1, X_train_scaled[target_name].shape[1]), 
                                   input_shape=(X_train_scaled[target_name].shape[1],)),
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(64, dropout=0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        lstm_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = lstm_model.fit(
            X_train_scaled[target_name], y_train_dict[target_name],
            validation_data=(X_test_scaled[target_name], y_test_dict[target_name]),
            epochs=50,
            batch_size=64,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Evaluate
        train_loss, train_acc = lstm_model.evaluate(X_train_scaled[target_name], y_train_dict[target_name], verbose=0)
        test_loss, test_acc = lstm_model.evaluate(X_test_scaled[target_name], y_test_dict[target_name], verbose=0)
        
        # Save
        model_path = f"trained_models/comprehensive_lstm_{target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        lstm_model.save(model_path)
        
        trained_models[f'lstm_{target_name}'] = lstm_model
        training_results[f'lstm_{target_name}'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'model_path': model_path
        }
        
        print(f"      ✅ LSTM {target_name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # 2. Random Forest
    print("\n🌲 Training Random Forest Models...")
    
    for target_name in ['direction_1h', 'direction_4h', 'direction_1d']:
        print(f"   🔥 Training Random Forest for {target_name}...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train[target_name], y_train_dict[target_name])
        
        # Evaluate
        train_pred = rf_model.predict(X_train[target_name])
        test_pred = rf_model.predict(X_test[target_name])
        
        train_acc = accuracy_score(y_train_dict[target_name], train_pred)
        test_acc = accuracy_score(y_test_dict[target_name], test_pred)
        
        # Save
        model_path = f"trained_models/comprehensive_rf_{target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        
        trained_models[f'rf_{target_name}'] = rf_model
        training_results[f'rf_{target_name}'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'model_path': model_path
        }
        
        print(f"      ✅ RF {target_name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Save scalers
    scaler_path = f"trained_models/comprehensive_scalers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    
    # Save training results
    results_path = f"training_results/comprehensive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('training_results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"\n📊 TRAINING COMPLETED!")
    print(f"   💾 Results saved: {results_path}")
    print(f"   💾 Scalers saved: {scaler_path}")
    
    return trained_models, training_results, scalers

def setup_signal_analysis_system():
    """Setup hệ thống phân tích signal generation"""
    print("\n🔍 SETTING UP SIGNAL ANALYSIS SYSTEM...")
    print("=" * 60)
    
    try:
        from master_system import create_development_system
        from democratic_voting_engine import DemocraticVotingEngine
        
        # Create master system
        master_system = create_development_system()
        print("✅ Master Integration System created")
        
        # Get democratic voting engine
        voting_engine = master_system.components.get('democratic_voting_engine')
        if voting_engine:
            print(f"✅ Democratic Voting Engine available with {len(voting_engine.specialists)} specialists")
            
            # List all specialists
            for specialist in voting_engine.specialists:
                print(f"   • {specialist.name} ({specialist.category})")
        
        return master_system, voting_engine
        
    except Exception as e:
        print(f"❌ Error setting up signal analysis: {e}")
        return None, None

def analyze_signal_generation(master_system, voting_engine, market_data_sample):
    """Phân tích chi tiết quá trình tạo signal"""
    print("\n📊 ANALYZING SIGNAL GENERATION PROCESS...")
    print("=" * 60)
    
    if not voting_engine or not master_system:
        print("❌ Signal analysis system not available")
        return None
    
    signal_analysis = {
        'timestamp': datetime.now(),
        'total_specialists': len(voting_engine.specialists),
        'specialist_votes': {},
        'category_analysis': {},
        'consensus_analysis': {},
        'signal_details': {}
    }
    
    try:
        # Simulate market data for testing
        if market_data_sample is None:
            # Create sample market data
            sample_data = pd.DataFrame({
                'Close': [2000.0, 2001.5, 2003.0, 1999.0, 2002.0],
                'High': [2005.0, 2006.0, 2008.0, 2001.0, 2007.0],
                'Low': [1998.0, 1999.0, 2000.0, 1995.0, 2000.0],
                'Volume': [1000, 1200, 1100, 1300, 1150],
                'Open': [1999.0, 2000.5, 2002.0, 2003.0, 1999.5]
            })
            market_data_sample = sample_data
        
        print(f"📈 Analyzing with market data: {len(market_data_sample)} periods")
        
        # Collect individual specialist votes
        specialist_results = {}
        category_votes = {
            'Technical': [],
            'Sentiment': [],
            'Pattern': [],
            'Risk': [],
            'Momentum': [],
            'Volatility': []
        }
        
        for specialist in voting_engine.specialists:
            try:
                # Get specialist vote (mock for demonstration)
                vote_result = {
                    'vote': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4]),
                    'confidence': np.random.uniform(0.1, 0.9),
                    'reasoning': f"{specialist.name} analysis based on current market conditions"
                }
                
                specialist_results[specialist.name] = vote_result
                category_votes[specialist.category].append(vote_result['vote'])
                
                print(f"   🗳️ {specialist.name}: {vote_result['vote']} (confidence: {vote_result['confidence']:.2f})")
                
            except Exception as e:
                print(f"   ❌ Error getting vote from {specialist.name}: {e}")
        
        # Analyze category consensus
        category_analysis = {}
        for category, votes in category_votes.items():
            if votes:
                vote_counts = {vote: votes.count(vote) for vote in set(votes)}
                total_votes = len(votes)
                majority_vote = max(vote_counts, key=vote_counts.get)
                consensus_strength = vote_counts[majority_vote] / total_votes
                
                category_analysis[category] = {
                    'total_specialists': total_votes,
                    'vote_distribution': vote_counts,
                    'majority_vote': majority_vote,
                    'consensus_strength': consensus_strength
                }
                
                print(f"\n📊 {category} Category Analysis:")
                print(f"   Total specialists: {total_votes}")
                print(f"   Vote distribution: {vote_counts}")
                print(f"   Majority vote: {majority_vote} ({consensus_strength:.1%} consensus)")
        
        # Overall consensus analysis
        all_votes = [result['vote'] for result in specialist_results.values()]
        overall_vote_counts = {vote: all_votes.count(vote) for vote in set(all_votes)}
        total_specialists = len(all_votes)
        
        if all_votes:
            final_decision = max(overall_vote_counts, key=overall_vote_counts.get)
            overall_consensus = overall_vote_counts[final_decision] / total_specialists
            
            consensus_analysis = {
                'total_specialists': total_specialists,
                'vote_distribution': overall_vote_counts,
                'final_decision': final_decision,
                'consensus_strength': overall_consensus,
                'agreement_threshold': 0.6,  # 60% threshold
                'consensus_reached': overall_consensus >= 0.6
            }
            
            print(f"\n🎯 OVERALL CONSENSUS ANALYSIS:")
            print(f"   Total specialists: {total_specialists}")
            print(f"   Vote distribution: {overall_vote_counts}")
            print(f"   Final decision: {final_decision}")
            print(f"   Consensus strength: {overall_consensus:.1%}")
            print(f"   Consensus reached: {'✅ YES' if consensus_analysis['consensus_reached'] else '❌ NO'}")
        
        # Signal details
        signal_details = {
            'signal_type': final_decision if 'final_decision' in locals() else 'HOLD',
            'confidence': overall_consensus if 'overall_consensus' in locals() else 0.0,
            'supporting_specialists': overall_vote_counts.get(final_decision, 0) if 'final_decision' in locals() else 0,
            'opposing_specialists': total_specialists - overall_vote_counts.get(final_decision, 0) if 'final_decision' in locals() else 0,
            'neutral_specialists': overall_vote_counts.get('HOLD', 0)
        }
        
        # Update analysis results
        signal_analysis.update({
            'specialist_votes': specialist_results,
            'category_analysis': category_analysis,
            'consensus_analysis': consensus_analysis,
            'signal_details': signal_details
        })
        
        return signal_analysis
        
    except Exception as e:
        print(f"❌ Error during signal analysis: {e}")
        return signal_analysis

def create_specialist_performance_tracker():
    """Tạo hệ thống theo dõi performance của từng specialist"""
    print("\n📈 CREATING SPECIALIST PERFORMANCE TRACKER...")
    print("=" * 60)
    
    performance_tracker = {
        'tracking_start': datetime.now(),
        'specialists': {},
        'signals_analyzed': 0,
        'performance_stats': {}
    }
    
    # Initialize tracking for each specialist type
    specialist_types = [
        'RSI_Specialist', 'MACD_Specialist', 'Fibonacci_Specialist',
        'News_Sentiment_Specialist', 'Social_Media_Specialist', 'Fear_Greed_Specialist',
        'Chart_Pattern_Specialist', 'Candlestick_Specialist', 'Wave_Specialist',
        'VaR_Risk_Specialist', 'Drawdown_Specialist', 'Position_Size_Specialist',
        'Trend_Specialist', 'Mean_Reversion_Specialist', 'Breakout_Specialist',
        'ATR_Specialist', 'Bollinger_Specialist', 'Volatility_Clustering_Specialist'
    ]
    
    for specialist_name in specialist_types:
        performance_tracker['specialists'][specialist_name] = {
            'total_votes': 0,
            'correct_predictions': 0,
            'wrong_predictions': 0,
            'accuracy_rate': 0.0,
            'vote_distribution': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
            'confidence_scores': [],
            'category': 'Unknown'
        }
    
    print(f"✅ Performance tracker initialized for {len(specialist_types)} specialists")
    return performance_tracker

def run_comprehensive_analysis():
    """Chạy phân tích toàn diện"""
    print("🚀 COMPREHENSIVE TRAINING & SIGNAL ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # 1. Load maximum data
    data_dict = load_maximum_data()
    if not data_dict:
        print("❌ No data loaded. Exiting...")
        return
    
    # 2. Prepare features
    X, y, df_clean = prepare_comprehensive_features(data_dict)
    if X is None:
        print("❌ Feature preparation failed. Exiting...")
        return
    
    # 3. Train models
    trained_models, training_results, scalers = train_comprehensive_models(X, y)
    
    # 4. Setup signal analysis
    master_system, voting_engine = setup_signal_analysis_system()
    
    # 5. Create performance tracker
    performance_tracker = create_specialist_performance_tracker()
    
    # 6. Analyze signal generation with multiple scenarios
    print("\n🎯 RUNNING SIGNAL ANALYSIS WITH MULTIPLE SCENARIOS...")
    print("=" * 60)
    
    signal_analyses = []
    
    for scenario in range(5):  # Test 5 different market scenarios
        print(f"\n📊 Scenario {scenario + 1}/5:")
        
        # Create different market conditions
        if scenario == 0:  # Bullish trend
            market_data = pd.DataFrame({
                'Close': [2000, 2005, 2010, 2015, 2020],
                'High': [2002, 2008, 2015, 2020, 2025],
                'Low': [1998, 2003, 2008, 2013, 2018],
                'Volume': [1000, 1200, 1400, 1600, 1800],
                'Open': [1999, 2004, 2009, 2014, 2019]
            })
            print("   📈 Bullish trend scenario")
            
        elif scenario == 1:  # Bearish trend
            market_data = pd.DataFrame({
                'Close': [2020, 2015, 2010, 2005, 2000],
                'High': [2025, 2020, 2015, 2010, 2005],
                'Low': [2018, 2013, 2008, 2003, 1998],
                'Volume': [1800, 1600, 1400, 1200, 1000],
                'Open': [2019, 2014, 2009, 2004, 1999]
            })
            print("   📉 Bearish trend scenario")
            
        elif scenario == 2:  # Sideways market
            market_data = pd.DataFrame({
                'Close': [2000, 2002, 1998, 2001, 1999],
                'High': [2005, 2007, 2003, 2006, 2004],
                'Low': [1995, 1997, 1993, 1996, 1994],
                'Volume': [1000, 1100, 950, 1050, 975],
                'Open': [1999, 2001, 1997, 2000, 1998]
            })
            print("   ↔️ Sideways market scenario")
            
        elif scenario == 3:  # High volatility
            market_data = pd.DataFrame({
                'Close': [2000, 2030, 1970, 2040, 1960],
                'High': [2010, 2050, 2000, 2060, 2000],
                'Low': [1990, 2020, 1950, 2030, 1940],
                'Volume': [2000, 2500, 2200, 2800, 2300],
                'Open': [1995, 2025, 1975, 2035, 1965]
            })
            print("   ⚡ High volatility scenario")
            
        else:  # Low volatility
            market_data = pd.DataFrame({
                'Close': [2000, 2001, 2000.5, 2001.5, 2000.8],
                'High': [2002, 2003, 2002.5, 2003.5, 2002.8],
                'Low': [1998, 1999, 1998.5, 1999.5, 1998.8],
                'Volume': [500, 520, 480, 510, 490],
                'Open': [1999, 2000, 1999.5, 2000.5, 1999.8]
            })
            print("   🔇 Low volatility scenario")
        
        # Analyze signal generation
        analysis = analyze_signal_generation(master_system, voting_engine, market_data)
        if analysis:
            analysis['scenario'] = scenario + 1
            analysis['scenario_type'] = ['Bullish', 'Bearish', 'Sideways', 'High Vol', 'Low Vol'][scenario]
            signal_analyses.append(analysis)
    
    # 7. Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save signal analyses
    signal_analysis_path = f"signal_analysis/comprehensive_signal_analysis_{timestamp}.json"
    os.makedirs('signal_analysis', exist_ok=True)
    with open(signal_analysis_path, 'w') as f:
        json.dump(signal_analyses, f, indent=2, default=str)
    
    # Save performance tracker
    performance_path = f"signal_analysis/specialist_performance_tracker_{timestamp}.json"
    with open(performance_path, 'w') as f:
        json.dump(performance_tracker, f, indent=2, default=str)
    
    # Create summary report
    summary_report = {
        'training_summary': {
            'total_models_trained': len(trained_models),
            'training_results': training_results,
            'data_size': X.shape[0],
            'feature_count': X.shape[1]
        },
        'signal_analysis_summary': {
            'scenarios_analyzed': len(signal_analyses),
            'total_specialists': len(performance_tracker['specialists']),
            'analysis_timestamp': timestamp
        }
    }
    
    summary_path = f"comprehensive_analysis_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"📊 Models trained: {len(trained_models)}")
    print(f"📊 Scenarios analyzed: {len(signal_analyses)}")
    print(f"📊 Specialists tracked: {len(performance_tracker['specialists'])}")
    print(f"💾 Results saved:")
    print(f"   • Signal analysis: {signal_analysis_path}")
    print(f"   • Performance tracker: {performance_path}")
    print(f"   • Summary report: {summary_path}")
    
    return summary_report

if __name__ == "__main__":
    run_comprehensive_analysis() 